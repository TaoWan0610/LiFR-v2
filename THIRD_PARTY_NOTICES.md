# Third-party notices

Third-party components retain their original copyright and license terms.
The MIT license for LiFR-authored additions does not replace these terms or grant
rights to datasets, pretrained weights or components without a license.

| Component / source | Location in this repository | License / notice |
|---|---|---|
| [E-RAFT](https://github.com/uzh-rpg/E-RAFT) | `model/modules/eraft/` | [MIT](docs/third_party/ERAFT.LICENSE) |
| [RAFT](https://github.com/princeton-vl/RAFT) | Flow utilities and correlation code | [BSD-3-Clause](docs/third_party/RAFT.LICENSE) |
| [Softmax Splatting](https://github.com/sniklaus/softmax-splatting) | `model/modules/softsplat/` | Upstream restricts use to academic purposes; retain attribution |
| [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) | Semantic encoder and framework adaptations | [MIT](docs/third_party/CMX.LICENSE) |
| [SegFormer](https://github.com/NVlabs/SegFormer) | MiT encoders and SegFormer head adaptations | [NVIDIA source license](docs/third_party/SegFormer.LICENSE), non-commercial use |
| [MoGe](https://github.com/microsoft/MoGe) | `model/depth/moge/` | [MIT and included dependency notices](docs/third_party/MoGe.LICENSE) |
| [DINOv2](https://github.com/facebookresearch/dinov2) | `model/depth/moge/model/dinov2/` | [Apache-2.0](docs/third_party/DINOv2.LICENSE) |
| [SAM 2](https://github.com/facebookresearch/sam2) | HRM attention and position encoding adaptations | [Apache-2.0](docs/third_party/SAM2.LICENSE) |
| [Restormer](https://github.com/swz30/Restormer) | V1 memory and depth HRM FFN adaptations | [MIT](docs/third_party/Restormer.LICENSE) |
| MetaDepth | `model/modules/egcm/depth/encoder.py` | [FAIR Noncommercial Research License](docs/third_party/MetaDepth.LICENSE) |
| [VMamba](https://github.com/MzeroMiko/VMamba) | Multi-task scan and state-space adaptations | [MIT](docs/third_party/VMamba.LICENSE) |
| [timm](https://github.com/huggingface/pytorch-image-models) | Shared DropPath helpers | [Apache-2.0](docs/third_party/timm.LICENSE) |
| [ASTMT](https://github.com/facebookresearch/astmt) | Multi-task utilities, losses and evaluation adaptations | [CC BY-NC 4.0](docs/third_party/ASTMT.LICENSE) |
| [MTMamba](https://github.com/EnVision-Research/MTMamba) | `model/multi_task/` | [Attribution and license status](docs/third_party/MTMamba.upstream.md) |

Some MTMamba-derived utilities, losses and evaluation files carry
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) notices; those
notices remain in the source. MTMamba's upstream repository does not supply an
overall license. Permission for remaining portions must be established before
public redistribution. Do not assume that the MIT licenses of its dependencies
license the whole multi-task implementation.

LiFR adaptations include task integration, event encoding, propagation,
memory/completion modules, staged training, portable configurations and entry
points. Third-party license texts are retained under `docs/third_party/`.
