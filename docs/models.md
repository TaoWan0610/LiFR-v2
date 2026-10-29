# Model versions

| Task | RGB backbone | LiFR v2 HRM | EGCM event encoder |
|---|---|---|---|
| Segmentation | MiT-B2 + SegFormer head | Four SA+CA layers on Stage 3 | MiT-B0 |
| Depth | DINOv2 ViT-S + MoGe decoder | One SA+CA block on the fused 384-channel feature | MetaDepth CNN |
| Multi-task | Swin-T + MTMamba decoder | Four SA+CA layers on Stage 3 | MiT-B0 |

V1 uses `model/modules/memory_v1/`, without EGCM. E-RAFT defaults to 12
iterations in v1 and 2 in v2. V2 uses a five-state memory bank followed by
ConvFusion completion with a learned spatial gate. Gates use average/max
pooling; task decoders run in the top-level models.

Segmentation and multi-task use `model/modules/hrm/hrm.py`: stacked SA+CA
attention with spatial/temporal positions and rotary attention. Depth uses
`model/modules/hrm/depth.py`: one self-attention residual, one memory
cross-attention residual, and a gated convolutional FFN. Its learned temporal
position table has seven entries; the runtime bank retains five states.
These implementations share the memory-retrieval design but have different
attention and FFN parameterizations.

Model entry points are `model/seg/semseg/models/lifr_v1.py` and `lifr_v2.py`,
and `model/{depth,multi_task}/v1.py` and `v2.py`. Shared propagation modules
live under `model/modules/`. Each `egcm/{seg,depth,multi_task}/` package contains:

- `encoder.py`: event encoder and feature projections.
- `completion.py`: ConvFusion and gating.
- `distillation.py`: teacher/student training objectives.

Training supervises final task outputs. Candidate, gate and person-specific
auxiliary losses are disabled. Independent event distillation retains its own
objective; depth retains the MoGe main-loss combination and multi-task retains
both semantic and depth losses. See [training stages](workflows.md).
