# Training and evaluation

Run commands from the repository root after setting the data, weights and
output roots described in the [README](../README.md). Configurations are grouped
by task, dataset and stage.

| Stage | Configuration | Trainable components |
|---|---|---|
| RGB backbone | `backbone` | RGB encoder and task decoder |
| V1 propagation | `lifr_v1` | E-RAFT, propagation, v1 memory and decoder |
| Event distillation | `distill` | Event student; RGB teacher frozen |
| V2 propagation | `lifr_v2_hrm` | E-RAFT, propagation, HRM and decoder |
| V2 completion | `lifr_v2` | ConvFusion, gate and decoder |

Propagation freezes the RGB encoder. Completion loads the trained propagation
model and distilled event encoder, freezing propagation and both encoders.
The completion gate starts after 50 epochs for segmentation, 10 epochs for
multi-task, and 5,000 updates for DSEC/SHF-Emerge depth (1,000 for M3ED).

For DSEC segmentation:

```bash
python run.py --task seg --action train --config configs/seg/dsec/backbone.yaml --gpus 4
python run.py --task seg --action distill --config configs/seg/dsec/distill.yaml --gpus 4
python run.py --task seg --action train --config configs/seg/dsec/lifr_v2_hrm.yaml --gpus 4
python run.py --task seg --action train --config configs/seg/dsec/lifr_v2.yaml --gpus 4
```

Between stages, place the selected weights at the paths described in
[checkpoints](checkpoints.md). For v1, train `lifr_v1.yaml` after the backbone.
For depth, use `--task depth` and `configs/depth/<dataset>/*.json`;
for multi-task, use `--task multi_task` and `configs/multi_task/<dataset>/*.yaml`.
Depth training normally validates every 1,000 optimizer updates.

## Evaluation and inference

```bash
python run.py --task seg --action eval \
  --config configs/seg/dsec/lifr_v2.yaml --weights weights/seg/lifr2_dsec.pt
python run.py --task depth --action eval \
  --config configs/depth/dsec/lifr_v2_eval.json --weights weights/depth/lifr2_dsec.pt --gpus 4
python run.py --task multi_task --action eval \
  --config configs/multi_task/dsec/lifr_v2.yaml --weights /path/to/joint_model.pt --gpus 4
```

Segmentation reports endpoint mIoU and class IoU. Depth reports metric RMSE,
AbsRel, delta1 and log-RMSE. SHF-Emerge depth uses the 1–10 m GT range; DSEC uses
its own validity range. MoGe resolution levels select token counts within
1,200–3,600; keep the same setting when comparing checkpoints.
For single-sample depth inference, use `--action infer` with arguments shown by
`python run.py --task depth --action infer --weights weights/depth/lifr2_dsec.pt -- --help`.
