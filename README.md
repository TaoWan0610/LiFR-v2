# LiFR v2

**Tao Wan**, **Xiaoshan Wu**, Yifei Yu, Bo Wang, Xiaoyang Lyu,  
Muxin Liu, Aoxuan Pan, Zhongrui Wang, Xiaojuan Qi

RGB–event feature propagation and completion for semantic segmentation,
monocular depth, and joint semantic/depth prediction.

**LiFR v2: Completion-Augmented Event Propagation for High-Frequency Dense Prediction**

![LiFR v2 teaser: events reveal newly visible objects between RGB observations.](docs/assets/teaser.png)

Events provide evidence of scene changes between RGB observations, supporting
semantic and depth prediction before the next RGB frame.

## Method

![LiFR v2 architecture: RGB feature encoding, event-guided propagation, HRM, EGCM, memory bank and task decoder.](docs/assets/framework.png)

An RGB keyframe provides anchor features. Event-guided propagation, HRM and
EGCM update these features for target-time predictions.

LiFR v1 uses its memory mechanism without completion. LiFR v2 uses SA+CA HRM followed
by ConvFusion EGCM and a learned gate. Segmentation and multi-task use four
attention layers; depth uses one SA+CA block. See [models](docs/models.md).

## Qualitative results

![Semantic segmentation, depth and multi-task comparisons on M3ED, DSEC and SHF-Emerge.](docs/assets/qualitative_results.png)

Representative comparisons across the three tasks. Target-time RGB images and
HFR outputs serve as visual references.

## Getting started

Install Python 3.10, CUDA-enabled PyTorch, and the requirements for your task:

```bash
pip install -r model/seg/requirements.txt
# For depth: pip install -r model/depth/requirements.txt
# For multi-task: pip install -r model/multi_task/requirements.txt
```

Follow [installation](docs/installation.md) for CuPy and multi-task extensions.
Prepare the [datasets](docs/data.md) and [checkpoints](docs/checkpoints.md):

```bash
export LIFR_DATA_ROOT=/path/to/prepared/datasets
export LIFR_WEIGHTS_ROOT="$PWD/weights"
export LIFR_OUTPUT_ROOT=/path/to/results

python run.py --task seg --action eval \
  --config configs/seg/dsec/lifr_v2.yaml --weights weights/seg/lifr2_dsec.pt
python run.py --task depth --action eval \
  --config configs/depth/dsec/lifr_v2_eval.json --weights weights/depth/lifr2_dsec.pt
```

[Training instructions](docs/workflows.md) cover the RGB backbone, event
encoder distillation, propagation and completion stages. Use `--gpus N` for
multi-GPU training. Arguments after `--` are forwarded to the task entry point.

## Pretrained weights

Download the [DSEC checkpoints](https://github.com/TaoWan0610/LiFR-v2/releases/tag/v2.0-weights)
and place them at the paths below:

| Component | Download | Local path |
|---|---|---|
| Semantic LiFR v2 | [Checkpoint](https://github.com/TaoWan0610/LiFR-v2/releases/download/v2.0-weights/seg_lifr2_dsec.pt) | `weights/seg/lifr2_dsec.pt` |
| Semantic RGB backbone | [Checkpoint](https://github.com/TaoWan0610/LiFR-v2/releases/download/v2.0-weights/seg_backbone_dsec.pt) | `weights/seg/backbone_dsec.pt` |
| Depth LiFR v2 | [Checkpoint](https://github.com/TaoWan0610/LiFR-v2/releases/download/v2.0-weights/depth_lifr2_dsec.pt) | `weights/depth/lifr2_dsec.pt` |
| Depth RGB backbone | [Checkpoint](https://github.com/TaoWan0610/LiFR-v2/releases/download/v2.0-weights/depth_backbone_dsec.pt) | `weights/depth/backbone_dsec.pt` |
| E-RAFT | [Checkpoint](https://github.com/TaoWan0610/LiFR-v2/releases/download/v2.0-weights/eraft_dsec.tar) | `weights/flownet/dsec.tar` |

See the [release metadata](https://github.com/TaoWan0610/LiFR-v2/releases/download/v2.0-weights/manifest.json)
for architectures and metric provenance.

## Repository layout

```text
model/
  seg/          Semantic segmentation
  depth/        MoGe depth estimation
  multi_task/   MTMamba joint prediction
  modules/      E-RAFT, SoftSplat, memory, HRM and task-specific EGCM
configs/        Dataset and training-stage configurations for each task
docs/           Guides and paper figures under assets/
tools/          Portable configuration loading
weights/        Downloaded checkpoints (excluded from Git)
run.py          Common training, distillation, validation and inference launcher
```

Data, checkpoint binaries and generated outputs are excluded from source control.
Original LiFR contributions use the [MIT license](LICENSE). Third-party code
retains its original terms; see the [bundled licenses](docs/third_party/).
