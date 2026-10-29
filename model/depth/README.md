# LiFR depth model

Use the repository root's `run.py` with the depth configurations. Install
`model/depth/requirements.txt`; this task is part of LiFR, not a standalone MoGe package.

```text
v1.py / v2.py     LiFR models using v1 memory / HRM + EGCM
tools/
  train.py       Backbone and LiFR training
  train_event_distill.py  CNN event-encoder distillation
  val.py         Full validation split, optionally distributed
  infer.py       Single RGB frame plus chronological event voxels
moge/
  model/         MoGe v2 RGB backbone and DINOv2
  train/         Training loop, data loading, losses and resumable checkpoints
  evaluation/    Validation data, metrics and best-checkpoint selection
  utils/         Data I/O, geometry and augmentation
```

Shared flow, propagation, HRM and EGCM implementations are under `model/modules/`.
Only the final depth prediction is supervised with the configured MoGe main
losses. Independent CNN distillation retains its own objective.

Training writes `checkpoint/latest.pt` (model, optimizer, scheduler and step),
`train_metrics.jsonl`, and validation history. Scheduled validation keeps the
best model according to the configured metric. Resume with `--checkpoint latest`
or a checkpoint path. Model-only checkpoints restore weights and step with a fresh optimizer.

Validation computes RMSE, AbsRel, delta1 and natural-log RMSE without scale/shift
alignment, averaging per image. The GT validity mask and configured depth ranges
are preserved; empty masks do not contribute scores. SHF-Emerge near-field scores
use `1 < depth <= 10` m. DSEC keeps its own configured range.

Inference reads the architecture from the checkpoint. Pass `--events` in time
order: one context voxel before the RGB frame, then one voxel per forward warp.
RGB-only checkpoints require no events. It saves `depth.npy`; `--visualize` also
saves `depth_vis.png`.

See [workflows](../../docs/workflows.md), [dataset preparation](../../docs/data.md)
and [model versions](../../docs/models.md).
