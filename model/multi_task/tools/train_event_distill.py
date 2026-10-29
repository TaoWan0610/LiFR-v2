from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import Tensor, nn
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


MTMAMBA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MTMAMBA_ROOT.parent.parent
for root in (str(MTMAMBA_ROOT), str(REPO_ROOT), str(REPO_ROOT / "model/seg")):
    if root not in sys.path:
        sys.path.insert(0, root)

from model.modules.egcm.multi_task.distillation import SwinB0FourStageDistiller  # noqa: E402
from semseg.datasets.distillation import EventDistillationDataset  # noqa: E402
from utils.common_config import get_model  # noqa: E402
from utils.config import create_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Swin to four-stage Event SegFormer-B0 distillation",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def seed_everything(seed: int, rank: int) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_state(checkpoint: object) -> dict[str, Tensor]:
    while isinstance(checkpoint, dict):
        nested = next(
            (
                checkpoint[key]
                for key in ("model", "state_dict", "model_state_dict")
                if isinstance(checkpoint.get(key), dict)
            ),
            None,
        )
        if nested is None:
            break
        checkpoint = nested
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(value, Tensor)
    }


def load_teacher(config_path: str, checkpoint_path: str) -> nn.Module:
    config = create_config(
        config_path,
        {
            "run_mode": "infer",
            "use_ddp": False,
        },
    )
    config.load_pretrained = False
    model = get_model(config)
    state = extract_state(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False),
    )
    model.load_state_dict(state, strict=True)
    teacher = model.backbone
    teacher.requires_grad_(False).eval()
    return teacher


def load_mit_b0_pretrained(encoder: nn.Module, checkpoint_path: str) -> None:
    source = extract_state(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False),
    )
    source = {
        key.removeprefix("backbone."): value for key, value in source.items()
    }
    target = encoder.state_dict()
    mapped = {}
    for source_key, value in source.items():
        match = re.match(r"patch_embed([1-4])\.(.+)", source_key)
        if match:
            target_key = f"patch_embeds.{int(match.group(1)) - 1}.{match.group(2)}"
        else:
            match = re.match(r"block([1-4])\.(.+)", source_key)
            if match:
                target_key = f"blocks.{int(match.group(1)) - 1}.{match.group(2)}"
            else:
                match = re.match(r"norm([1-4])\.(.+)", source_key)
                if not match:
                    continue
                target_key = f"norms.{int(match.group(1)) - 1}.{match.group(2)}"

        if target_key == "patch_embeds.0.proj.weight" and value.shape[1] == 3:
            event_bins = target[target_key].shape[1]
            value = value.mean(1, keepdim=True).repeat(1, event_bins, 1, 1)
            value = value * (3.0 / event_bins)
        if target_key in target and value.shape == target[target_key].shape:
            mapped[target_key] = value

    missing = sorted(set(target) - set(mapped))
    if missing:
        raise RuntimeError(
            f"MiT-B0 checkpoint does not cover the event encoder: {missing[:16]}"
        )
    encoder.load_state_dict(mapped, strict=True)
    print(
        f"[student-init] loaded {len(mapped)} MiT-B0 tensors; "
        f"event_bins={target['patch_embeds.0.proj.weight'].shape[1]}",
        flush=True,
    )


def build_dataset(config: dict, split: str) -> EventDistillationDataset:
    data = config["data"]
    return EventDistillationDataset(
        root=data["root"],
        split=split,
        dataset_type=data.get("type", "dsec"),
        frame_index_stride=int(data.get("frame_index_stride", 1)),
        image_size=tuple(data.get("image_size", (440, 640))),
        event_scale=float(data.get("event_scale", 255.0)),
        horizontal_flip=split == "train" and bool(data.get("horizontal_flip", False)),
        exclude_sequences=data.get(
            "train_exclude_sequences" if split == "train" else "val_exclude_sequences",
        ),
        include_semantic=False,
    )


def build_loader(
    dataset: EventDistillationDataset,
    batch_size: int,
    workers: int,
    rank: int,
    world_size: int,
    training: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = (
        DistributedSampler(dataset, world_size, rank, shuffle=training)
        if world_size > 1 else None
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        drop_last=training,
    )
    return loader, sampler


def reduce_metrics(values: Tensor, world_size: int) -> dict[str, float]:
    if world_size > 1:
        dist.all_reduce(values)
    count = values[-1].clamp_min(1)
    names = (
        "loss", "stage", "intra", "cross",
        "stage1", "stage2", "stage3", "stage4",
    )
    return {
        name: float((values[index] / count).item())
        for index, name in enumerate(names)
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    world_size: int,
    amp_dtype: torch.dtype,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_clip: float = 1.0,
    log_every: int = 20,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    statistics = torch.zeros(9, device=device)
    primary = not dist.is_initialized() or dist.get_rank() == 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, batch in enumerate(loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            event = batch["event"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=amp_dtype):
                output = model(image, event)
            if training:
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    grad_clip,
                )
                optimizer.step()
                scheduler.step()

            batch_size = image.shape[0]
            batch_values = torch.stack(
                (
                    output.loss.detach(),
                    output.stage_loss.detach(),
                    output.intra_loss.detach(),
                    output.cross_loss.detach(),
                    *(loss.detach() for loss in output.stage_losses),
                )
            )
            statistics[:8] += batch_values * batch_size
            statistics[8] += batch_size
            if training and primary and batch_index % log_every == 0:
                print(
                    f"[train] batch={batch_index}/{len(loader)} "
                    f"loss={output.loss.item():.6f} "
                    f"stage={output.stage_loss.item():.6f} "
                    f"intra={output.intra_loss.item():.6f} "
                    f"cross={output.cross_loss.item():.6f}",
                    flush=True,
                )
    return reduce_metrics(statistics, world_size)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_loss: float,
    metrics: dict[str, float],
) -> None:
    unwrapped = model.module if isinstance(model, DDP) else model
    payload = {
        **unwrapped.checkpoint_state(),
        "epoch": epoch,
        "best_val_loss": best_loss,
        "validation_metrics": metrics,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rank, local_rank, world_size, device = setup_distributed()
    primary = rank == 0
    seed_everything(int(config.get("seed", 3407)), rank)
    torch.set_float32_matmul_precision("high")

    train_dataset = build_dataset(config, "train")
    val_dataset = build_dataset(config, "val")
    if not train_dataset or not val_dataset:
        raise RuntimeError(
            f"No synchronized distillation pairs: train={len(train_dataset)} "
            f"val={len(val_dataset)}"
        )
    training = config["train"]
    batch_size = int(training.get("batch_size", 4))
    workers = int(training.get("num_workers", 8))
    train_loader, train_sampler = build_loader(
        train_dataset, batch_size, workers, rank, world_size, True,
    )
    val_loader, _ = build_loader(
        val_dataset,
        int(training.get("val_batch_size", batch_size)),
        workers,
        rank,
        world_size,
        False,
    )

    teacher = load_teacher(
        config["teacher"]["config"],
        config["teacher"]["checkpoint"],
    )
    loss = config["loss"]
    model = SwinB0FourStageDistiller(
        teacher=teacher,
        event_bins=int(config["data"].get("event_bins", 20)),
        teacher_channels=tuple(int(value) for value in teacher.num_features),
        stage_weight=float(loss.get("stage_weight", 1.0)),
        intra_weight=float(loss.get("intra_weight", 10.0)),
        cross_weight=float(loss.get("cross_weight", 4.0)),
        similarity_threshold=float(loss.get("similarity_threshold", 0.1)),
        relation_grid=int(loss.get("relation_grid", 32)),
    )
    load_mit_b0_pretrained(
        model.event_encoder,
        config["student"]["checkpoint"],
    )
    model.to(device)

    optimizer_config = config["optimizer"]
    patch_parameters = list(model.event_encoder.patch_embeds.parameters())
    patch_ids = {id(parameter) for parameter in patch_parameters}
    other_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in patch_ids
    ]
    lr = float(optimizer_config.get("lr", 5e-5))
    optimizer = torch.optim.AdamW(
        (
            {
                "params": patch_parameters,
                "lr": lr * float(optimizer_config.get("patch_lr_multiplier", 1.0)),
            },
            {"params": other_parameters, "lr": lr},
        ),
        weight_decay=float(optimizer_config.get("weight_decay", 1e-4)),
    )
    epochs = int(training.get("epochs", 100))
    total_steps = epochs * len(train_loader)
    warmup_steps = int(training.get("warmup_epochs", 1)) * len(train_loader)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1.0 - progress, 0.0) ** float(training.get("poly_power", 0.9))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start_epoch = 0
    best_loss = math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.event_adapter.load_distillation_checkpoint(checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        best_loss = float(checkpoint.get("best_val_loss", math.inf))

    output = Path(config["output"])
    if primary:
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(
            f"[distill] world_size={world_size} per_gpu_batch={batch_size} "
            f"global_batch={batch_size * world_size} train={len(train_dataset)} "
            f"val={len(val_dataset)} losses=Lstage+10Lintra_s3+4Lcross_s3",
            flush=True,
        )
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    amp_dtype = (
        torch.bfloat16
        if str(training.get("precision", "bf16")).lower() in {"bf16", "bfloat16"}
        else torch.float16
    )
    validate_every = int(training.get("validate_every_epochs", 1))
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            world_size,
            amp_dtype,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=float(training.get("grad_clip", 1.0)),
            log_every=int(training.get("log_every", 20)),
        )
        should_validate = (epoch + 1) % validate_every == 0 or epoch + 1 == epochs
        val_metrics = (
            run_epoch(
                model,
                val_loader,
                device,
                world_size,
                amp_dtype,
            )
            if should_validate else None
        )
        if primary:
            record = {
                "epoch": epoch + 1,
                "lr": optimizer.param_groups[-1]["lr"],
                "train": train_metrics,
                "val": val_metrics,
                "best_val_loss": best_loss,
                "seconds": time.time() - start_time,
            }
            if val_metrics is not None:
                improved = val_metrics["loss"] < best_loss
                best_loss = min(best_loss, val_metrics["loss"])
                record["best_val_loss"] = best_loss
            else:
                improved = False
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")
            save_checkpoint(
                output / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_loss,
                val_metrics or train_metrics,
            )
            if improved:
                save_checkpoint(
                    output / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    best_loss,
                    val_metrics,
                )
            print(json.dumps(record), flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
