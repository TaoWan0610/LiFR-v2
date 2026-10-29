# Installation

Use Python 3.10 and install a CUDA-enabled PyTorch build before the task
requirements. Run from the repository root:

```bash
pip install -r model/seg/requirements.txt
# Depth: pip install -r model/depth/requirements.txt
# Multi-task: pip install -r model/multi_task/requirements.txt
```

SoftSplat needs CuPy and CUDA NVRTC. Install the CuPy wheel matching your CUDA
runtime, for example `cupy-cuda12x` for CUDA 12. Keep the CUDA shared libraries
available to the dynamic linker.

Multi-task additionally requires MMCV, Mamba and causal-conv1d extensions
compatible with its PyTorch/CUDA build. Use a separate environment for this
stack. Depth installs `utils3d` and `pipeline` from the pinned upstream commits
in its requirements.

Set `LIFR_DATA_ROOT`, `LIFR_WEIGHTS_ROOT` and `LIFR_OUTPUT_ROOT`, or use the
launcher's `--data-root`, `--weights-root` and `--output-root` options.
The launcher sets task imports and writes rendered configurations under the
output directory. See [data](data.md) and [training](workflows.md).
