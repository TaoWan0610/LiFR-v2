"""Expand depth training, distillation and evaluation presets.

Each section declares its defaults once. Fully expanded configs pass through.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

from model.modules.memory_utils import normalize_memory_type

_DATASETS = {"shf_emergy", "dsec", "m3ed-falcon", "m3ed-spot"}
_BACKBONES = {"vits", "vitb"}
_DATASET_ALIASES = {
    "carla": "shf_emergy", "m3ed": "m3ed-spot", "m3ed_d": "m3ed-falcon",
    "m3ed_falcon": "m3ed-falcon", "m3ed_spot": "m3ed-spot",
}


def normalize_dataset_name(name):
    return _DATASET_ALIASES.get(str(name), str(name))


def _dataset_spec(name: str) -> Dict[str, Any]:
    name = normalize_dataset_name(name)
    overrides = {
        "shf_emergy": {
            "display_name": "SHF-Emerge",
            # CARLA renders the far plane at very large depths.  The metric
            # benchmark is explicitly defined on finite, non-sky pixels in
            # (0, 100] metres.
            "min_depth": 0.0,
            "max_depth": 100.0,
            "semantic": "semantic.png",
            "sky_id": 0,
            "metric_depth_ranges": [[0.0, 50.0], [50.0, 100.0]],
            "default_iterations": {"backbone": 40_000, "anytime": 40_000, "refine": 40_000},
        },
        "dsec": {
            "display_name": "DSEC",
            "clamp_max_depth": 1000.0,
            "default_iterations": {"backbone": 50_000, "anytime": 50_000, "refine": 50_000},
        },
        "m3ed-spot": {
            "display_name": "M3ED-Spot",
            "event_layout": "continuous",
        },
        "m3ed-falcon": {
            "display_name": "M3ED-Falcon",
            "continuous_cross_split": True,
        },
    }
    if name not in overrides:
        raise ValueError(f"dataset must be one of {sorted(_DATASETS)}, got {name!r}")
    m3ed = name.startswith("m3ed-")
    env_name = f"MOGE_{name.upper().replace('-', '_')}_ROOT"
    return {
        "root": os.environ.get(env_name) or str(
            Path(os.environ.get("LIFR_DATA_ROOT", "data")) / name / "moge2"
        ),
        "width": 640,
        "height": 360 if m3ed else 440,
        "clamp_max_depth": 120.0,
        "frame_step_ms": 40 if m3ed else 50,
        "event_layout": "dsec",
        "sparsity_aware": m3ed,
        "workers": (4, 8, 8) if m3ed else (8, 16, 16),
        "default_iterations": {"backbone": 20_000, "anytime": 50_000, "refine": 50_000},
        **overrides[name],
    }


def _check_keys(values, allowed, name):
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"Unsupported {name} keys: {sorted(unknown)}")


def _section(raw, name, defaults, *, optional=(), nonnegative=(), unrestricted=()):
    """Merge defaults, convert scalar values and validate numeric options."""
    supplied = raw.get(name, {})
    if not isinstance(supplied, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    _check_keys(supplied, set(defaults) | set(optional), name)
    values = {**copy.deepcopy(defaults), **supplied}
    for key, default in defaults.items():
        kind = type(default)
        if kind in (str, bool, int, float, dict):
            values[key] = kind(values[key])
        if kind in (int, float) and key not in unrestricted:
            value = values[key]
            if value < 0 or (value == 0 and key not in nonnegative):
                bound = "non-negative" if key in nonnegative else "positive"
                raise ValueError(f"{name}.{key} must be {bound}, got {value}")
    return values


def _positive(value, name, kind=int):
    value = kind(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _experiment(raw, mode):
    sections = {
        "train": {"train", "optimizer", "lr_scheduler", "anytime", "refine"},
        "eval": {"eval", "anytime"},
        "distill": {"train", "optimizer", "loss"},
    }
    _check_keys(raw, {"preset", "dataset", "backbone", "paths", "data"} | sections[mode], mode)
    presets = {"distill"} if mode == "distill" else {"backbone", "anytime", "refine"}
    if mode == "eval":
        presets.remove("refine")
    preset = str(raw.get("preset", ""))
    dataset = normalize_dataset_name(raw.get("dataset", ""))
    backbone = str(raw.get("backbone", "vits" if mode == "distill" else ""))
    for name, value, allowed in (("preset", preset, presets), ("dataset", dataset, _DATASETS),
                                 ("backbone", backbone, _BACKBONES)):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return preset, dataset, backbone


def default_checkpoint(backbone):
    return {"vits": "pretrained/moge-2-vits-normal.pt",
            "vitb": "pretrained/moge-2-vitb-normal/model.pt"}[backbone]


def _model_config(backbone, tokens, stream=None):
    dim = {"vits": 384, "vitb": 768}[backbone]
    features = [dim, 256, 128, 64, 32]

    def decoder(channels):
        return {
            "dim_in": features, "dim_out": [None, None, None, None, channels],
            "dim_res_blocks": features, "num_res_blocks": [0, 1, 1, 1, 0],
            "res_block_in_norm": "none", "res_block_hidden_norm": "none",
            "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
        }

    model = {
        "encoder": {"backbone": f"dinov2_{backbone}14", "intermediate_layers": [5, 11], "dim_out": dim},
        "neck": {**decoder(None), "dim_in": [dim + 2, 2, 2, 2, 2], "dim_out": None},
        "points_head": decoder(3), "scale_head": {"dims": [dim, dim, dim, 1]},
        "remap_output": "exp", "num_tokens_range": tokens,
    }
    if stream is None:
        model.update(mask_head=decoder(1), normal_head=decoder(3))
    else:
        model.update(anytime=stream, remap_exp_clip=6.0)
    return model


def _loss_config(*, sparsity_aware: bool, include_mask: bool) -> Dict[str, Any]:
    global_params: Dict[str, Any] = {"align_resolution": 48}
    local_params: Dict[str, Any] = {
        "level": 4, "align_resolution": 24, "num_patches": 16,
    }
    if sparsity_aware:
        global_params["sparsity_aware"] = True
        local_params["sparsity_aware"] = True
    losses = {
        "global": {
            "function": "affine_invariant_global_loss", "weight": 1.0,
            "params": global_params,
        },
        "patch_4": {
            "function": "affine_invariant_local_loss", "weight": 1.0,
            "params": local_params,
        },
        "metric_scale": {"function": "metric_scale_loss", "weight": 0.1},
    }
    if include_mask:
        losses["mask"] = {"function": "mask_bce_loss", "weight": 0.1}
    return {"invalid": {}, "C": losses}


def _precision_train_options(value):
    if value not in {"fp32", "amp", "bf16", "fp16"}:
        raise ValueError(f"Unsupported training precision: {value!r}")
    return {
        "enable_mixed_precision": value in {"amp", "bf16", "fp16"},
        "mixed_precision_dtype": "bfloat16" if value == "bf16" else "float16",
    }


def _data_options(raw, dataset, *, train):
    defaults = {
        "width": dataset["width"], "height": dataset["height"],
        "frame_interval_ms": dataset["frame_step_ms"],
    }
    if train:
        defaults.update(clamp_max_depth=dataset["clamp_max_depth"], num_tokens_range=[1200, 3600])
    data = _section(raw, "data", defaults, optional=("min_depth", "max_depth", "metric_depth_ranges"))
    minimum = float(data.get("min_depth", dataset.get("min_depth", 0.0)))
    maximum = _positive(data.get("max_depth", dataset.get("max_depth", data.get(
        "clamp_max_depth", dataset["clamp_max_depth"]))), "data.max_depth", float)
    if minimum < 0 or minimum >= maximum:
        raise ValueError("data.min_depth must satisfy 0 <= min_depth < max_depth")
    ranges = copy.deepcopy(data.get("metric_depth_ranges", dataset.get("metric_depth_ranges", [])))
    for index, bounds in enumerate(ranges):
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"data.metric_depth_ranges[{index}] must be [lower, upper]")
        lower, upper = map(float, bounds)
        if lower < minimum or upper <= lower or upper > maximum:
            raise ValueError(f"data.metric_depth_ranges[{index}] must lie within [{minimum}, {maximum}]")
    data.update(min_depth=minimum, max_depth=maximum, metric_depth_ranges=ranges)
    if train:
        tokens = data["num_tokens_range"]
        if not isinstance(tokens, (list, tuple)) or len(tokens) != 2:
            raise ValueError("data.num_tokens_range must contain [minimum, maximum]")
        data["num_tokens_range"] = [_positive(value, "data.num_tokens_range") for value in tokens]
        if data["num_tokens_range"][1] < data["num_tokens_range"][0]:
            raise ValueError("data.num_tokens_range maximum must be >= minimum")
    return data


def _semantic_filter(dataset, data):
    if "max_depth" not in dataset:
        return {}
    return {"min_depth": data["min_depth"], "max_depth": data["max_depth"],
            "semantic": str(dataset["semantic"]), "invalid_semantic_ids": [int(dataset["sky_id"])]}


def _event_data(paths, dataset, data, anytime):
    step = data["frame_interval_ms"]
    duration = anytime["duration_ms"]
    if duration <= 0 or duration % step:
        raise ValueError(f"anytime.duration_ms must be a positive multiple of {step}")
    result = {
        "requires_events": True, "event_bins": anytime["event_bins"],
        "event_root": paths["event_root"], "event_layout": dataset["event_layout"],
        "duration_ms": duration, "frame_step_ms": step,
    }
    if dataset.get("continuous_cross_split", False):
        result["continuous_cross_split"] = True
    return result


def _paths(raw, dataset, defaults):
    paths = _section(raw, "paths", {"dataset_root": dataset["root"], "event_root": None, **defaults})
    if not paths["dataset_root"]:
        raise ValueError("paths.dataset_root must not be empty")
    paths["event_root"] = str(paths["event_root"] if "event_root" in raw.get("paths", {})
                              else os.path.dirname(paths["dataset_root"]))
    return paths


def _optimizer_config(raw, preset, event_patterns, decoder_patterns, scale_head):
    anytime = preset != "backbone"
    options = _section(raw, "optimizer", {
        "lr": 1e-4 if anytime else 1e-5, "weight_decay": 1e-6 if anytime else 1e-2,
    }, optional=("backbone_lr", "anytime_lr", "flow_lr", "softsplat_lr", "memory_lr",
                 "neck_lr", "head_lr", "event_encoder_lr", "decoder_lr", "scale_head_lr"),
        nonnegative=("weight_decay",))
    forbidden = "backbone_lr" if anytime else "anytime_lr"
    if forbidden in options:
        raise ValueError(f"optimizer.{forbidden} is not used by preset={preset!r}")
    groups = []

    def add(patterns, key="lr", default=None, exclude=()):
        if not patterns:
            return
        params = {"include": patterns}
        if exclude:
            params["exclude"] = list(exclude)
        groups.append({"params": params, "lr": _positive(options.get(key, default), f"optimizer.{key}", float),
                       "weight_decay": options["weight_decay"]})

    if preset == "backbone":
        add(["*"], exclude=["*.backbone.*"])
        add(["*.backbone.*"], "backbone_lr", 1e-6)
    elif preset == "anytime":
        secondary = _positive(options.get("anytime_lr", 1e-4), "optimizer.anytime_lr", float)
        add(["neck.*"], "neck_lr", 1e-4)
        add(["points_head.*", "scale_head.*"], "head_lr", 1e-5)
        for patterns, key in ((["flow_net.*"], "flow_lr"), (["softsplat_net.*"], "softsplat_lr"),
                              (["fusion_attens.*", "hrm.*"], "memory_lr")):
            add(patterns, key, secondary, exclude=["encoder.*", "neck.*"])
    else:
        add(["event_refine.*"], exclude=["event_refine.event_encoder.*", "event_refine.event_adapter.*"])
        add(event_patterns, "event_encoder_lr", 1e-5)
        add(decoder_patterns, "decoder_lr", 1e-5)
        if scale_head:
            add(["scale_head.*"], "scale_head_lr", 1e-5)
    return {"type": "AdamW", "params": groups}


def _scheduler_config(raw, iterations, anytime):
    options = _section(raw, "lr_scheduler", {
        "warmup_start": 1000, "warmup_end": 2000,
        "step_size": max(1, iterations // 2), "gamma": 0.5,
    }, nonnegative=("warmup_start",))
    start, end = options["warmup_start"], options["warmup_end"]
    if end <= start:
        raise ValueError("lr_scheduler.warmup_end must be greater than warmup_start")
    warmup = f"max(0.0, min(1.0, (epoch - {start}) / {end - start}))"
    return {"type": "SequentialLR", "params": {
        "schedulers": [
            {"type": "LambdaLR", "params": {"lr_lambda": warmup if anytime else ["1.0", warmup]}},
            {"type": "StepLR", "params": {"step_size": options["step_size"], "gamma": options["gamma"]}},
        ], "milestones": [end],
    }}


def expand_train_config(raw):
    if "preset" not in raw:
        return copy.deepcopy(dict(raw))
    if raw["preset"] == "distill":
        return expand_distill_config(raw)
    preset, dataset_name, backbone = _experiment(raw, "train")
    dataset = _dataset_spec(dataset_name)
    anytime, refine = preset != "backbone", preset == "refine"
    paths = _paths(raw, dataset, {
        "train_index": ".train.txt", "val_index": ".val.txt",
        "workspace": f"workspace/{dataset_name}_moge2_{backbone}_{preset}",
        "init_checkpoint": default_checkpoint(backbone), "resume_checkpoint": None,
        "flow_checkpoint": os.environ.get("MOGE_FLOW_CHECKPOINT", str(
            Path(os.environ.get("LIFR_WEIGHTS_ROOT", "weights")) / "flownet/dsec.tar")),
        "event_checkpoint": "",
    })
    data = _data_options(raw, dataset, train=True)
    memory_type = normalize_memory_type(raw.get("anytime", {}).get(
        "memory_type", "hrm" if refine else "memory_v1",
    ))
    stream = _section(raw, "anytime", {
        "duration_ms": 2 * data["frame_interval_ms"], "event_bins": 20, "event_reduce_factor": 5,
        "memory_type": "hrm" if refine else "memory_v1", "memory_layers": 1, "num_memory_slots": 5,
        "flow_iters": 2 if memory_type == "hrm" else 12,
        "warp_refine_identity_init": False, "warp_refine_max_scale": 1.0,
    })
    completion = _section(raw, "refine", {
        "event_encoder": "metadepth-cnn", "metric_scale_weight": 0.5,
    }, nonnegative=("metric_scale_weight",))
    if not anytime and raw.get("anytime"):
        raise ValueError("anytime options require an anytime/refine preset")
    if not refine and raw.get("refine"):
        raise ValueError("refine options require the refine preset")
    stream["memory_type"] = normalize_memory_type(stream["memory_type"])
    if stream["event_bins"] % stream["event_reduce_factor"]:
        raise ValueError("anytime.event_reduce_factor must divide anytime.event_bins")

    load_workers, process_workers, buffer_size = dataset["workers"]
    train = _section(raw, "train", {
        "num_iterations": dataset["default_iterations"][preset], "batch_size_forward": 4,
        "gradient_accumulation_steps": 1, "precision": "amp",
        "num_load_workers": load_workers, "num_process_workers": process_workers,
        "dataloader_buffer_size": buffer_size, "find_unused_parameters": refine,
        "validate_every": 2000, "validation_batch_size": 1,
        "best_metric": "rmse", "log_every": 10,
        "gate_start_step": 1000 if dataset_name.startswith("m3ed-") else 5000,
    }, nonnegative=("gate_start_step",))
    if not refine:
        train.pop("gate_start_step")
    validation = {
        "enabled": True, "every": train.pop("validate_every"), "batch_size": train.pop("validation_batch_size"),
        "metric": train.pop("best_metric"), "mode": "min",
    }
    train.update(_precision_train_options(train.pop("precision")))
    train.update(workspace=paths["workspace"], seed=0)
    if paths["resume_checkpoint"] is not None:
        train["checkpoint"] = str(paths["resume_checkpoint"])

    dataset_config = {
        "name": f"{dataset['display_name']}_{preset}", "format": "folder", "path": paths["dataset_root"],
        "label_type": "C", "weight": 1.0, "index": paths["train_index"], "depth": "depth.png",
        "depth_unit": 1.0, "finite_depth_mask": "only_known",
        **_semantic_filter(dataset, data),
    }
    model_stream = None
    if anytime:
        dataset_config.update(_event_data(paths, dataset, data, stream))
        model_stream = {**stream, "enabled": True, "num_warps": stream["duration_ms"] // data["frame_interval_ms"],
                        "memory_num_heads": 8, "memory_layer_norm": "WithBias",
                        "memory_ffn_expansion_factor": 2.66, "memory_bias": False}
        del model_stream["duration_ms"]
        model_stream["flow"] = {"iters": model_stream.pop("flow_iters"), "checkpoint": paths["flow_checkpoint"]}

    event_patterns, frozen_event, decoder_patterns = [], [], []
    config = {
        "data": {"image_sizes": [[data["width"], data["height"]]], "clamp_max_depth": data["clamp_max_depth"],
                 "center_augmentation": 0.0, "fov_range_absolute": [1, 179], "fov_range_relative": [1.0, 1.0],
                 "image_augmentation": ["jittering", "jpeg_loss", "blurring"], "datasets": [dataset_config]},
        "train": train, "init_checkpoint": {"path": paths["init_checkpoint"]}, "model_version": "v2",
        "lr_scheduler": _scheduler_config(raw, train["num_iterations"], anytime),
        "low_resolution_training_steps": 0,
        "loss": _loss_config(sparsity_aware=dataset["sparsity_aware"], include_mask=not anytime),
        "validation": validation,
    }
    if refine:
        encoder = completion["event_encoder"].strip().lower().replace("_", "-")
        if encoder in {"cnn", "metadepth"}:
            encoder = "metadepth-cnn"
        if encoder != "metadepth-cnn":
            raise ValueError("refine.event_encoder must be 'metadepth-cnn'")
        if not paths["event_checkpoint"]:
            raise ValueError("paths.event_checkpoint is required for the refine preset")
        model_stream["refine"] = {"enabled": True, "event_encoder": encoder, "event_checkpoint": paths["event_checkpoint"]}
        decoder_patterns = ['neck.*', 'points_head.*']
        config.update(
            freeze={"include": ["encoder.*", "flow_net.*", "softsplat_net.*", "hrm.*",
                                "event_refine.event_encoder.*", "event_refine.event_adapter.*"]},
            event_encoder_frozen=True,
            depth_loss={"min_depth": max(data["min_depth"], 1e-3), "max_depth": data["max_depth"]},
        )
        config["loss"]["C"]["metric_scale"]["weight"] = completion["metric_scale_weight"]
    elif anytime:
        config["freeze"] = {"include": ["encoder.*"]}
    config["model"] = _model_config(backbone, data["num_tokens_range"], model_stream)
    config["train"]["stage"] = "completion" if refine else "propagation" if anytime else "backbone"
    config["optimizer"] = _optimizer_config(raw, preset, event_patterns, decoder_patterns, refine)
    return config


def expand_distill_config(raw):
    if "preset" not in raw:
        return copy.deepcopy(dict(raw))
    _, dataset_name, _ = _experiment(raw, "distill")
    dataset = _dataset_spec(dataset_name)
    paths = _paths(raw, dataset, {
        "train_index": ".train.txt", "teacher_checkpoint": "", "resume_checkpoint": None,
        "workspace": f"output_distill/{dataset_name}", "raw_event_h5_root": None,
    })
    if not paths["teacher_checkpoint"]:
        raise ValueError("paths.teacher_checkpoint is required for distillation")
    if "raw_event_h5_root" in raw.get("paths", {}):
        paths["raw_event_h5_root"] = str(paths["raw_event_h5_root"])
    else:
        paths.pop("raw_event_h5_root")
    load_workers, process_workers, buffer_size = dataset["workers"]
    train = _section(raw, "train", {
        "num_iterations": 50_000, "batch_size_forward": 4, "precision": "bf16",
        "num_load_workers": load_workers, "num_process_workers": process_workers,
        "dataloader_buffer_size": buffer_size, "grad_clip_norm": 1.0,
        "validate_every": 2000, "log_every": 20, "seed": 3407,
    }, optional=("amp",), unrestricted=("seed",))
    if train["precision"] not in {"no", "fp16", "bf16"}:
        raise ValueError("distill train.precision must be one of: no, fp16, bf16")
    if "amp" in train:
        train["amp"] = bool(train["amp"])
    data = _section(raw, "data", {
        "width": dataset["width"], "height": dataset["height"], "frame_interval_ms": dataset["frame_step_ms"],
        "event_bins": 20, "event_layout": dataset["event_layout"],
        "clamp_max_depth": dataset["clamp_max_depth"], "num_tokens_range": [1200, 1200],
    })
    data["num_tokens_range"] = list(data["num_tokens_range"])
    loss = _section(raw, "loss", {}, optional=("use_event_mask", "event_tau", "intra_weight",
                                               "cross_weight", "similarity_threshold"))
    optimizer = _section(raw, "optimizer", {"lr": 1e-4, "weight_decay": 1e-4, "power": 0.9},
                         nonnegative=("weight_decay",))
    return {"dataset": dataset_name, "paths": paths, "data": data, "train": train, "loss": loss,
            "optimizer": optimizer, "model": {"distiller": "metadepth_cnn_aggregate", "event_encoder": "metadepth-cnn"}}


def expand_eval_config(raw):
    if "preset" not in raw:
        return copy.deepcopy(dict(raw))
    preset, dataset_name, _ = _experiment(raw, "eval")
    dataset = _dataset_spec(dataset_name)
    paths = _paths(raw, dataset, {"val_index": ".val.txt", "checkpoint": None, "output": None})
    data = _data_options(raw, dataset, train=False)
    options = _section(raw, "eval", {
        "precision": "fp32", "num_load_workers": 4, "num_process_workers": 8,
        "drop_max_depth": dataset["clamp_max_depth"], "resolution_level": 9,
    }, optional=("num_tokens", "batch_size"))
    if options["precision"] not in {"fp32", "fp16"}:
        raise ValueError("eval.precision must be fp32 or fp16")
    if options.get("num_tokens") is not None:
        _positive(options["num_tokens"], "eval.num_tokens")
    anytime = _section(raw, "anytime", {"duration_ms": 2 * data["frame_interval_ms"], "event_bins": 20})
    if preset == "backbone" and raw.get("anytime"):
        raise ValueError("anytime options require the anytime preset")
    benchmark = {
        "path": paths["dataset_root"], "width": data["width"], "height": data["height"],
        "split": paths["val_index"], "depth_unit": 1.0, "drop_max_depth": options["drop_max_depth"],
        "num_load_workers": options["num_load_workers"], "num_process_workers": options["num_process_workers"],
        **_semantic_filter(dataset, data),
    }
    if data["metric_depth_ranges"]:
        benchmark["depth_metric_ranges"] = data["metric_depth_ranges"]
    if preset == "anytime":
        benchmark.update(_event_data(paths, dataset, data, anytime))
        name = f"{dataset['display_name']}_anytime{anytime['duration_ms']}_val"
    else:
        name = f"{dataset['display_name']}_val"
    return {name: benchmark}


def read_config(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Configuration root must be a JSON object: {path}")
    return value


def load_train_config(path):
    return expand_train_config(read_config(path))


def load_distill_config(path):
    return expand_distill_config(read_config(path))
