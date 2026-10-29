# Checkpoints

Set `LIFR_WEIGHTS_ROOT` or pass `--weights-root`. Checkpoints use a flat layout
within each task directory:

```text
weights/
  seg/backbone_dsec.pt
  seg/lifr2_dsec.pt
  depth/backbone_dsec.pt
  depth/lifr2_dsec.pt
  flownet/dsec.tar
  manifest.json
```

`manifest.json` records checkpoint architecture and metric provenance. The
current depth v2 checkpoint uses the single-block SA+CA `DepthHRM` and is loaded
strictly by the public model loader. Full inference checkpoints include all
components and do not require their original training initialization files.

For training, the templates use `backbone_<dataset>.pt`,
`hrm_<dataset>.pt`, and `event_encoder_<dataset>.pt` under each task's directory.
Copy the selected result of each preceding stage to these names, or update the
configuration to its actual path. Dataset hyphens become underscores in these
filenames, for example `backbone_m3ed_falcon.pt`. Intermediate checkpoints and
weights for additional datasets must be supplied or trained separately.
Semantic and multi-task distillation also use a MiT-B0 initialization checkpoint
at `seg/mit_b0.pth`; obtain it from the [SegFormer project](https://github.com/NVlabs/SegFormer).
E-RAFT initialization is `flownet/dsec.tar` from [E-RAFT](https://github.com/uzh-rpg/E-RAFT).

Backbone training also expects `seg/mit_b2.pth` from SegFormer,
`depth/moge2_vits.pt` from [MoGe](https://github.com/microsoft/MoGe), and
`pretrained/mmseg_swin_tiny_patch4_window7_224.pth` for MTMamba. These upstream
initializers are separate from the task-trained DSEC backbones listed above.

Initialization starts a new optimizer and stage counter. Resuming a training
checkpoint restores the optimizer, scheduler and progress. Keep architecture,
event bins, temporal horizon and label mapping consistent with the checkpoint.
V1 memory, depth single-block HRM and segmentation's stacked HRM weights are
not interchangeable.

Checkpoint binaries are ignored by Git; publish them separately as release
assets or through a model hosting service. Keep `manifest.json` with the source.
