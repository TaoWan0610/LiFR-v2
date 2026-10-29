# Dataset layout

Supply prepared datasets outside the repository and set `LIFR_DATA_ROOT` to their
location. No data files are included.

## Temporal convention

For the CARLA/DSEC 50 ms grid, `event[N]` contains events from RGB frame `N`
to RGB frame `N+1`. Offline event distillation pairs this voxel with **RGB[N+1]**.
Propagation starts from RGB[N], uses the past voxel for flow context and the
future voxel for the query. Multi-step horizons require subsequent voxels.
M3ED uses the same endpoint convention on its 40 ms grid.

```text
RGB[N]             RGB[N+1]
  |------ event[N] ------|
  t0                  t0 + 50 ms
```

Do not infer the temporal horizon from the file index alone: `FRAME_INDEX_STRIDE`,
`FRAME_INTERVAL_MS`, and the dataset metadata determine it. The DSEC depth main
model uses two 50 ms warps while its event encoder is distilled on 50 ms windows.

## Expected roots

Configuration directories use the following dataset names:

| Task | Dataset directories under `configs/<task>/` |
|---|---|
| `seg` | `dsec`, `m3ed-falcon`, `m3ed-spot`, `shf-dsec`, `shf-emergy` |
| `depth` | `shf_emergy`, `dsec`, `m3ed-falcon`, `m3ed-spot` |
| `multi_task` | `shf_emergy`, `dsec`, `m3ed-falcon`, `m3ed-spot` |

The SHF-Emerge semantic configuration directory is `shf-emergy`; all three
tasks share the prepared data root `shf_emergy`. SHF-DSEC has its own semantic
configurations and data root. `DATASET.ID` records one of these five datasets;
`DATASET.NAME` selects its adapter (`DSEC`, `SHFDSEC`, `M3ED`, or `SHFEmerge`).
The internal `TYPE: carla` value retains the SHF label and event conventions.

Semantic dataset source files are limited to:

| File under `model/seg/semseg/datasets/` | Purpose |
|---|---|
| `dsec.py` | DSEC and the shared aligned temporal loading implementation |
| `shf_dsec.py` | Explicit SHF-DSEC entry using the shared temporal loader |
| `m3ed.py` | Shared M3ED-Falcon / M3ED-Spot 40 ms adapter |
| `shf_emergy.py` | SHF-Emerge train/val semantic samples and distillation |
| `distillation.py` | Event-window / endpoint-RGB pairs for distillation |
| `sampling.py` | Training-length padding |
| `__init__.py` | Explicit dataset registry and exports |

```text
data/
  shf_emergy/
  shf-dsec/          # 50 ms windows
  dsec/
  m3ed-falcon/
  m3ed-spot/
```

For CARLA/DSEC, the semantic adapters expect:

```text
<dataset>/images/{train,val}/<sequence>/<index>.png
<dataset>/semantic/{train,val}/<sequence>/<index>.png
<dataset>/event_t0_t1/event_20/{train,val}/<sequence>/<index>.npy
```

CARLA validation RGB may be under `images/val/val_semantic/`. M3ED uses `image/`
and supports the continuous `image/all`, `event_t0/all` layout. Past/future event
aliases follow the dataset adapter's index convention; do not blindly duplicate
or shift event files. Events are signed CHW tensors with 20 temporal bins.

MoGe and multi-task depth use `<dataset>/moge2/.train.txt` and `.val.txt`. Each
manifest row identifies a sample folder with image, depth, intrinsics and
`meta.json`. Keep depth units, timestamp indices and event metadata consistent
with the dataset adapters.

## SHF-Emerge / SHF-DSEC

Both datasets are generated in CARLA. SHF-Emerge uses the Fast layout;
SHF-DSEC uses the continuous semantic layout and its explicit 11-class mapping.

The SHF-Emerge recipes load all sequences in the prepared `train` and `val`
directories. No activity CSV, fixed town list or activity-sample cap is required.
Semantic samples are indexed by endpoint GT and retained when their RGB and
past/future event inputs exist. Distillation loads each event window with its
endpoint RGB; it does not require semantic GT.

For example, the 50 ms semantic recipe uses RGB at index S, context events ending
at S, forward events starting at S, and GT at S+1. Frame alignment and missing-input
checks remain in place. Put the desired train/val split in the corresponding
directories; the loader does not randomly split or move files. Validation RGB
can use `images/val/<sequence>/` or the existing `images/val/val_semantic/<sequence>/`.

These full-directory splits differ from historical activity-filtered experiments;
report metrics using the split actually evaluated.

Keep the 11-class mapping with its checkpoint: CARLA includes road lines at
class 9; DSEC's class 9 is wall. Person/rider maps to person, and bicycles,
motorcycles and larger vehicles share the vehicle class. Preserve ignore ID 255.

## DSEC

RGB images and depth GT must be aligned to the event-camera view using the
provided calibration. Event voxels must use rectified coordinates and the
configured temporal interval. Keep the official train/validation split.

## M3ED

Align RGB, semantic labels, depth and event windows by timestamp. Supply the
MoGe sample metadata and train/validation manifests for depth and joint-task
training. Use the configured Spot/Falcon split and depth-validity range.
