from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moge.config import load_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MoGe2 training checkpoints and keep the lowest-RMSE model."
    )
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--output-workspace", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--step", type=int, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_eval_config(train_config_path: Path, checkpoint: Path, output: Path) -> tuple[dict, dict]:
    compact = load_json(train_config_path)
    if compact.get("preset") not in {"backbone", "anytime", "refine"}:
        eval_config_path = train_config_path.with_name(f"{train_config_path.stem}_eval.json")
        if not eval_config_path.is_file():
            eval_config_path = train_config_path.parent / "eval_config.json"
        if not eval_config_path.is_file():
            raise ValueError(
                "Expanded training configs require a sibling <config_stem>_eval.json "
                "or eval_config.json"
            )
        eval_config = load_json(eval_config_path)
        eval_config.setdefault("paths", {})["checkpoint"] = str(checkpoint)
        eval_config["paths"]["output"] = str(output)
        compact = {
            "preset": "refine" if compact.get("model", {}).get("anytime", {}).get("refine", {}).get("enabled", False) else eval_config["preset"],
            "dataset": eval_config["dataset"],
            "backbone": eval_config["backbone"],
        }
        return compact, eval_config
    expanded = load_train_config(train_config_path)
    dataset = expanded["data"]["datasets"][0]
    width, height = expanded["data"]["image_sizes"][0]
    anytime = compact.get("anytime", {})
    anytime_enabled = compact["preset"] in {"anytime", "refine"}
    paths = {
        "dataset_root": dataset["path"],
        "val_index": compact.get("paths", {}).get("val_index", ".val.txt"),
        "checkpoint": str(checkpoint),
        "output": str(output),
    }
    data = {"width": width, "height": height}
    for key in ("min_depth", "max_depth", "metric_depth_ranges"):
        if key in compact.get("data", {}):
            data[key] = compact["data"][key]
    if anytime_enabled:
        paths["event_root"] = dataset["event_root"]
        data["frame_interval_ms"] = dataset["frame_step_ms"]
    eval_config = {
        "preset": "anytime" if anytime_enabled else "backbone",
        "dataset": compact["dataset"],
        "backbone": compact["backbone"],
        "paths": paths,
        "data": data,
        "eval": {
            "precision": "fp32",
            "num_load_workers": expanded["train"]["num_load_workers"],
            "num_process_workers": expanded["train"]["num_process_workers"],
            "drop_max_depth": expanded["data"]["clamp_max_depth"],
            "batch_size": int(expanded["validation"].get("batch_size", 1)),
            "num_tokens": None,
            "resolution_level": 9,
        },
    }
    if anytime_enabled:
        eval_config["anytime"] = {
            key: anytime[key] for key in ("duration_ms", "event_bins") if key in anytime
        }
    return compact, eval_config


def read_existing_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    payload = load_json(path)
    records = payload.get("records", [])
    return records if isinstance(records, list) else []


def best_record(records: list[dict], metric: str, mode: str) -> dict:
    if any(metric not in record for record in records):
        raise KeyError(f"Validation metric {metric!r} is absent from one or more records")
    if mode == "max":
        return max(records, key=lambda item: (item[metric], -item["step"]))
    return min(records, key=lambda item: (item[metric], item["step"]))


def write_metric_history(path: Path, records: list[dict], primary_metric: str, mode: str) -> None:
    best_step = best_record(records, primary_metric, mode)["step"]
    range_suffixes = sorted({
        key[len("rmse_"):]
        for record in records
        for key in record
        if key.startswith("rmse_") and key != "rmse_log"
    })
    columns = ["step", "rmse", "abs_rel", "delta1", "log_rmse"]
    for suffix in range_suffixes:
        columns.extend([
            f"rmse_{suffix}", f"abs_rel_{suffix}",
            f"delta1_{suffix}", f"log_rmse_{suffix}",
        ])
    columns.extend(["inference_time_s", "is_best"])
    lines = ["\t".join(columns)]
    for record in sorted(records, key=lambda item: item["step"]):
        values = []
        for column in columns:
            if column == "is_best":
                values.append(str(int(record["step"] == best_step)))
            elif column == "step":
                values.append(str(record[column]))
            else:
                value = record.get(column)
                values.append("-" if value is None else f"{value:.9g}")
        lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_record(result: dict, step: int, checkpoint: Path) -> tuple[dict, str]:
    names = [name for name in result if name != "mean"]
    if len(names) != 1:
        raise RuntimeError(f"Expected one validation benchmark, got {names}")
    benchmark_name = names[0]
    benchmark = result[benchmark_name]
    metric = benchmark["depth_metric"]
    record = {
        "step": step,
        "rmse": float(metric["rmse"]),
        "abs_rel": float(metric["rel"]),
        "delta1": float(metric["delta1"]),
        "log_rmse": float(metric["rmselog"]),
        "inference_time_s": float(benchmark["inference_time"]),
        "source_checkpoint": str(checkpoint),
    }
    for name, range_metric in benchmark.items():
        if not name.startswith("depth_metric_"):
            continue
        suffix = name[len("depth_metric_"):]
        record.update({
            f"rmse_{suffix}": float(range_metric["rmse"]),
            f"abs_rel_{suffix}": float(range_metric["rel"]),
            f"delta1_{suffix}": float(range_metric["delta1"]),
            f"log_rmse_{suffix}": float(range_metric["rmselog"]),
        })
    numeric = [record[key] for key in ("rmse", "abs_rel", "delta1", "log_rmse", "inference_time_s")]
    numeric.extend(
        record[key]
        for key in record
        if key.startswith(("rmse_", "abs_rel_", "delta1_", "log_rmse_"))
        and key != "rmse_log"
    )
    if not all(math.isfinite(value) for value in numeric):
        raise RuntimeError(f"Non-finite validation metric: {record}")
    return record, benchmark_name


def best_checkpoint_path(workspace: Path, step: int) -> Path:
    return workspace / f"{step:06d}.pt"


def copy_checkpoint(source: Path, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def remove_replaced_best(workspace: Path, current_best: Path, previous_summary: dict) -> None:
    previous = previous_summary.get("best", {}).get("checkpoint")
    if not previous:
        return
    previous_path = Path(previous)
    if not previous_path.is_absolute():
        previous_path = workspace / previous_path
    try:
        previous_path.relative_to(workspace)
    except ValueError:
        return
    is_managed_checkpoint = previous_path.stem.isdigit() or previous_path.name == "best.pt"
    if previous_path != current_best and previous_path.suffix == ".pt" and is_managed_checkpoint:
        previous_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    train_config_path = resolve_path(args.train_config)
    compact = load_json(train_config_path)
    expanded = load_train_config(train_config_path)
    primary_metric = str(expanded["validation"].get("metric", "rmse"))
    primary_mode = str(expanded["validation"].get("mode", "min"))
    source_workspace = resolve_path(expanded["train"]["workspace"])
    output_workspace = resolve_path(args.output_workspace) if args.output_workspace else source_workspace
    output_workspace.mkdir(parents=True, exist_ok=True)

    checkpoint = resolve_path(args.checkpoint)
    step = args.step

    eval_dir = output_workspace / "checkpoint_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_workspace / "metrics.json"
    previous_summary = load_json(metrics_path) if metrics_path.is_file() else {}
    records = read_existing_records(metrics_path)

    result_path = eval_dir / f"{step:08d}.json"
    compact, eval_config = build_eval_config(train_config_path, checkpoint, result_path)
    eval_config_path = output_workspace / "eval_config.json"
    write_json(eval_config_path, eval_config)
    eval_command = [
            sys.executable,
        ]
    if args.num_devices > 1:
        eval_command.extend([
            "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={args.num_devices}",
        ])
    eval_command.extend([
            "tools/val.py",
            "--config",
            str(eval_config_path),
            "--output",
            str(result_path),
            "--pretrained",
            str(checkpoint),
            "--device",
            args.device,
        ])
    subprocess.run(
        eval_command,
        cwd=PROJECT_ROOT,
        check=True,
    )
    record, _ = metric_record(load_json(result_path), step, checkpoint)
    range_suffixes = sorted(
        key[len("rmse_"):]
        for key in record
        if key.startswith("rmse_") and key != "rmse_log"
    )
    for suffix in range_suffixes:
        print(
            f"Validation step={step} {suffix.replace('_', '-')}: "
            f"RMSE={record[f'rmse_{suffix}']:.4f}m, "
            f"AbsRel={record[f'abs_rel_{suffix}']:.4f}, "
            f"delta1={record[f'delta1_{suffix}']:.4f}, "
            f"LogRMSE={record[f'log_rmse_{suffix}']:.4f}",
            flush=True,
        )
    records = [item for item in records if int(item["step"]) != step]
    records.append(record)
    records.sort(key=lambda item: int(item["step"]))
    best = best_record(records, primary_metric, primary_mode)

    best_path = best_checkpoint_path(output_workspace, int(best['step']))
    if best['step'] == step:
        copy_checkpoint(checkpoint, best_path)
        remove_replaced_best(output_workspace, best_path, previous_summary)
    elif not best_path.is_file():
        raise FileNotFoundError(f'Best checkpoint is missing: {best_path}')

    summary = {
        "task": f"MoGe2 {compact['preset']} online validation",
        "dataset": compact["dataset"],
        "evaluation_protocol": "full validation set, metric depth, FP32",
        "primary_metric": primary_metric,
        "primary_mode": primary_mode,
        "records": records,
        "best": {**best, "checkpoint": str(best_path)},
    }
    write_json(metrics_path, summary)
    previous_summary = summary
    write_metric_history(
        output_workspace / "validation_metrics.txt",
        records,
        primary_metric,
        primary_mode,
    )
    print("EVALUATED", json.dumps(record, sort_keys=True), flush=True)

    best = best_record(records, primary_metric, primary_mode)
    best_path = best_checkpoint_path(output_workspace, int(best["step"]))
    print(
        "VALIDATION_COMPLETE",
        json.dumps({**best, "best_checkpoint": str(best_path)}, sort_keys=True),
    )


if __name__ == "__main__":
    main()
