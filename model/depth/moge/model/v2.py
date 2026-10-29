from typing import Any, Dict, IO, List, Literal, Optional, Sequence, Union
from numbers import Number
from pathlib import Path
import ctypes
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils3d

from ..utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift
from .modules import DINOv2Encoder, MLP, ConvStack
from model.modules.hrm import DepthHRM
from model.modules.memory_v1.depth import MemoryV1
from model.modules.memory_utils import normalize_memory_type


def _load_anytime_classes():
    nvrtc_candidates = sorted(
        Path(sys.prefix).glob("lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12")
    )
    if nvrtc_candidates:
        ctypes.CDLL(str(nvrtc_candidates[0]), mode=ctypes.RTLD_GLOBAL)

    from model.modules.eraft.eraft import ERAFT
    from model.modules.softsplat.frame_synthesis import Synthesis

    return ERAFT, Synthesis


class MoGeModel(nn.Module):
    encoder: DINOv2Encoder
    neck: ConvStack
    points_head: ConvStack
    mask_head: ConvStack
    scale_head: MLP
    def __init__(self,
        encoder: Dict[str, Any],                                         # DINOv2 image encoder configuration
        neck: Dict[str, Any],                                            # Shared multiscale feature decoding and fusion network
        points_head: Dict[str, Any] = None,                              # 3D point-map prediction head; None disables it
        mask_head: Dict[str, Any] = None,                                # Valid-pixel mask prediction head; None disables it
        normal_head: Dict[str, Any] = None,                              # Surface-normal prediction head; None disables it
        scale_head: Dict[str, Any] = None,                               # Global metric-scale prediction head; None disables it
        anytime: Optional[Dict[str, Any]] = None,                        # Anytime event propagation and feature fusion configuration
        remap_output: Literal['linear', 'sinh', 'exp', 'sinh_exp'] = 'linear',  # Numerical mapping applied to raw point-map predictions
        remap_exp_clip: Optional[float] = None,                          # Upper clipping bound before exp to prevent overflow
        num_tokens_range: List[int] = [1200, 3600],                      # ViT token range for training and default inference
    ):
        super(MoGeModel, self).__init__()
        self.remap_output = remap_output
        self.remap_exp_clip = remap_exp_clip
        self.num_tokens_range = num_tokens_range
        self.anytime_config = anytime or {}
        self.anytime_enabled = bool(self.anytime_config.get('enabled', False))
        self.anytime_refine_enabled = False

        self.encoder = DINOv2Encoder(**encoder)         # Fixed encoder implementation
        self.neck = ConvStack(**neck)
        if points_head is not None:         # Required by this recipe
            self.points_head = ConvStack(**points_head)
        if mask_head is not None:           # Not required by this recipe
            self.mask_head = ConvStack(**mask_head)
        if normal_head is not None:         # Not required by this recipe
            self.normal_head = ConvStack(**normal_head)
        if scale_head is not None:          # Required by this recipe
            self.scale_head = MLP(**scale_head)
        if self.anytime_enabled:
            self._init_anytime_modules()

    def _init_anytime_modules(self):
        try:
            ERAFT, Synthesis = _load_anytime_classes()
        except Exception as e:
            raise ImportError(
                "Anytime MoGe requires the shared E-RAFT/SoftSplat modules under model.modules. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e

        # Anytime propagation is applied to the fused ViT feature before ConvNeck.
        # ViT-S: [B,384,Hp,Wp] -> warp/memory -> ConvNeck five-level features.
        self.anytime_feature_dim = self.encoder.output_projections[0].out_channels
        self.anytime_memory_type = normalize_memory_type(self.anytime_config.get('memory_type', 'memory_v1'))
        self.anytime_config['memory_type'] = self.anytime_memory_type
        self.event_bins = int(self.anytime_config.get('event_bins', 20))
        self.event_reduce_factor = int(self.anytime_config.get('event_reduce_factor', 5))
        if self.event_bins % self.event_reduce_factor != 0:
            raise ValueError(
                f"event_bins ({self.event_bins}) must be divisible by event_reduce_factor ({self.event_reduce_factor})"
            )

        flow = self.anytime_config.get('flow', {})
        self.num_warps = int(self.anytime_config.get('num_warps', 2))
        if self.num_warps <= 0:
            raise ValueError(f'num_warps must be positive, got {self.num_warps}')
        self.flow_net = ERAFT(
            n_first_channels=self.event_bins // self.event_reduce_factor,
            iters=int(flow.get('iters', 2 if self.anytime_memory_type == 'hrm' else 12)),
        )
        self.softsplat_net = Synthesis(
            [self.anytime_feature_dim],
            activation=self.anytime_config.get('activation', 'PReLU'),
            warp_refine_identity_init=bool(
                self.anytime_config.get('warp_refine_identity_init', False)
            ),
            warp_refine_max_scale=float(
                self.anytime_config.get('warp_refine_max_scale', 1.0)
            ),
        )
        # ERAFT and the historical SoftSplat score-metric branch do not
        # necessarily use the same event-time channel count. In particular,
        # official MVSEC 20 Hz E-RAFT consumes 15 bins, while Softmetric was
        # trained with four broad event-time bins.
        self.softmetric_event_channels = int(
            self.softsplat_net.netSoftmetric.netEventInput.in_channels
        )
        memory_num_heads = int(self.anytime_config.get('memory_num_heads', 8))
        if self.anytime_memory_type == 'hrm':
            self.hrm = DepthHRM(
                feature_dim=self.anytime_feature_dim,
                num_heads=memory_num_heads,
                num_memory_slots=int(self.anytime_config.get('num_memory_slots', 5)),
                position_slots=int(self.anytime_config.get('position_slots', 7)),
                ffn_expansion_factor=float(self.anytime_config.get('memory_ffn_expansion_factor', 2.66)),
                detach_memory=bool(self.anytime_config.get('memory_detach', False)),
            )
        else:
            self.fusion_attens = MemoryV1(
                dim=self.anytime_feature_dim, num_heads=memory_num_heads,
                layer_norm=self.anytime_config.get('memory_layer_norm', 'WithBias'),
                ffn_expansion_factor=float(self.anytime_config.get('memory_ffn_expansion_factor', 2.66)),
                bias=bool(self.anytime_config.get('memory_bias', False)),
            )

        flow_checkpoint = flow.get('checkpoint', '')
        if flow_checkpoint:
            self._load_flow_checkpoint(flow_checkpoint)

        refine = self.anytime_config.get('refine', {})
        self.anytime_refine_enabled = bool(refine.get('enabled', False))
        if self.anytime_refine_enabled:
            from model.modules.egcm.depth.completion import MoGeEventRefine

            if refine.get('event_encoder', 'metadepth-cnn') != 'metadepth-cnn':
                raise ValueError('Depth completion uses the MetaDepth CNN event encoder')

            self.event_refine = MoGeEventRefine(
                feature_channels=self.anytime_feature_dim,
                event_bins=self.event_bins,
            )
            event_checkpoint = str(refine.get('event_checkpoint', ''))
            if event_checkpoint:
                self.event_refine.load_event_checkpoint(event_checkpoint)

    @staticmethod
    def _load_state_dict_compatible(path: Union[str, Path], map_location=torch.device('cpu')) -> Dict[str, torch.Tensor]:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(ckpt, dict):
            for key in ('model', 'state_dict', 'model_state_dict'):
                if key in ckpt and isinstance(ckpt[key], dict):
                    ckpt = ckpt[key]
                    break
        if not isinstance(ckpt, dict):
            raise TypeError(f"Unsupported checkpoint format for {path}: {type(ckpt)}")

        cleaned = {}
        for key, value in ckpt.items():
            key = key.replace('module.', '', 1) if key.startswith('module.') else key
            key = key.replace('flow_net.', '', 1) if key.startswith('flow_net.') else key
            cleaned[key] = value
        return cleaned

    def _load_flow_checkpoint(self, checkpoint_path: Union[str, Path]):
        flow_state = self._load_state_dict_compatible(checkpoint_path)
        target_state = self.flow_net.state_dict()
        flow_state = {
            key: value for key, value in flow_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        msg = self.flow_net.load_state_dict(flow_state, strict=False)
        print(f'[flow] loaded {len(flow_state)} compatible keys from {checkpoint_path}')
        print(msg)

    def _bin_event_voxel(self, event_voxel: torch.Tensor) -> torch.Tensor:
        # Keep the historical FP32 event path by default.  The profiling script
        # enables force_fp16_inference together with half weights/inputs when a
        # true FP16 deployment profile is requested.
        event_dtype = self.dtype if getattr(self, 'force_fp16_inference', False) else torch.float32
        event_voxel = torch.nan_to_num(
            event_voxel.to(dtype=event_dtype), nan=0.0, posinf=0.0, neginf=0.0
        )
        channels = event_voxel.shape[1]
        if channels % self.event_reduce_factor != 0:
            raise ValueError(
                f'Event voxel channel count {channels} is not divisible by event_reduce_factor {self.event_reduce_factor}'
            )
        return torch.cat(
            [
                event_voxel[:, self.event_reduce_factor * i:self.event_reduce_factor * (i + 1)].mean(1, keepdim=True)
                for i in range(channels // self.event_reduce_factor)
            ],
            dim=1,
        )

    @staticmethod
    def _reduce_event_channels(event_voxel: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Average contiguous event-time groups without changing ERAFT input.

        ``torch.tensor_split`` keeps groups non-overlapping when the source
        count is not divisible by the target count (for example 15 -> 4).
        This matches the historical broad-bin averaging semantics and lets
        the four-channel Softmetric weights remain fully loadable.
        """
        channels = int(event_voxel.shape[1])
        target_channels = int(target_channels)
        if channels == target_channels:
            return event_voxel
        if channels < target_channels or target_channels <= 0:
            raise ValueError(
                f'Cannot reduce event channels from {channels} to {target_channels}'
            )
        return torch.cat(
            [chunk.mean(dim=1, keepdim=True) for chunk in torch.tensor_split(
                event_voxel, target_channels, dim=1
            )],
            dim=1,
        )

    def _apply_anytime(
        self,
        encoded: torch.Tensor,
        event_sequence: Sequence[torch.Tensor],
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Dict[str, Any]]]:
        event_sequence = list(event_sequence)
        expected_events = self.num_warps + 1
        if len(event_sequence) != expected_events or any(value is None for value in event_sequence):
            raise ValueError(
                f'Anytime MoGe with num_warps={self.num_warps} requires '
                f'{expected_events} chronological event tensors (one context + '
                f'{self.num_warps} forward windows), got {len(event_sequence)}.'
            )

        binned_events = [self._bin_event_voxel(value) for value in event_sequence]
        flows = []
        for step in range(self.num_warps):
            flows.append(self.flow_net(binned_events[step], binned_events[step + 1]))

        # SoftSplat propagates a list containing the fused ViT feature.
        features = [encoded]
        memory_module = self.hrm if self.anytime_memory_type == 'hrm' else self.fusion_attens
        memory_input = encoded if self.anytime_memory_type == 'hrm' else features
        memory_bank = memory_module.append_memory([], memory_input)

        warped_features = features
        for step, flow in enumerate(flows):
            softmetric_event = self._reduce_event_channels(
                binned_events[step + 1], self.softmetric_event_channels
            )
            warped_features = self.softsplat_net(
                tenEncone=warped_features,
                tenForward=flow,
                event_voxel=softmetric_event,
            )
            if self.anytime_memory_type == 'hrm':
                warped_features[0] = self.hrm(warped_features[0], memory_bank)
            else:
                warped_features = self.fusion_attens.attend(warped_features, memory_bank)
            if step < len(flows) - 1:
                memory_input = warped_features[0] if self.anytime_memory_type == 'hrm' else warped_features
                memory_bank = memory_module.append_memory(memory_bank, memory_input)

        if self.anytime_refine_enabled:
            return warped_features[0], {
                'event_voxel': event_sequence[-1],
            }
        return warped_features[0]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], **hf_kwargs) -> 'MoGeModel':
        """Build the checkpoint's model and load its complete state strictly."""
        from . import load_model
        return load_model(pretrained_model_name_or_path, **hf_kwargs)

    def init_weights(self):
        self.encoder.init_weights()

    def enable_gradient_checkpointing(self):
        self.encoder.enable_gradient_checkpointing()
        self.neck.enable_gradient_checkpointing()
        for head in ['points_head', 'normal_head', 'mask_head']:
            if hasattr(self, head):
                getattr(self, head).enable_gradient_checkpointing()

    def enable_pytorch_native_sdpa(self):
        self.encoder.enable_pytorch_native_sdpa()

    def _remap_points(self, points: torch.Tensor) -> torch.Tensor:
        if self.remap_output == 'linear':
            pass
        elif self.remap_output =='sinh':
            points = torch.sinh(points)
        elif self.remap_output == 'exp':
            xy, z = points.split([2, 1], dim=-1)
            if self.remap_exp_clip is not None:
                z = z.clamp(max=self.remap_exp_clip)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output =='sinh_exp':
            xy, z = points.split([2, 1], dim=-1)
            if self.remap_exp_clip is not None:
                z = z.clamp(max=self.remap_exp_clip)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        return points

    @staticmethod
    def _token_grid(
        image: torch.Tensor,
        num_tokens: Union[int, torch.LongTensor],
    ) -> tuple[Union[int, torch.Tensor], Union[int, torch.Tensor], float]:
        _, _, image_height, image_width = image.shape
        aspect_ratio = image_width / image_height
        base_h, base_w = (num_tokens / aspect_ratio) ** 0.5, (num_tokens * aspect_ratio) ** 0.5
        if isinstance(base_h, torch.Tensor):
            base_h, base_w = base_h.round().long(), base_w.round().long()
        else:
            base_h, base_w = round(base_h), round(base_w)
        return base_h, base_w, aspect_ratio

    def _neck_features(
        self,
        encoded: torch.Tensor,
        image: torch.Tensor,
        base_h: Union[int, torch.Tensor],
        base_w: Union[int, torch.Tensor],
        aspect_ratio: float,
    ) -> List[torch.Tensor]:
        batch_size = encoded.shape[0]
        features = [encoded, None, None, None, None]
        for level in range(5):
            uv = normalized_view_plane_uv(
                width=base_w * 2 ** level,
                height=base_h * 2 ** level,
                aspect_ratio=aspect_ratio,
                dtype=image.dtype,
                device=image.device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            features[level] = uv if features[level] is None else torch.cat([features[level], uv], dim=1)
        return self.neck(features)

    def encode_image_features(
        self,
        image: torch.Tensor,
        num_tokens: Union[int, torch.LongTensor],
    ) -> tuple[List[torch.Tensor], torch.Tensor]:
        base_h, base_w, aspect_ratio = self._token_grid(image, num_tokens)
        encoded, cls_token = self.encoder(image, base_h, base_w, return_class_token=True)
        return self._neck_features(encoded, image, base_h, base_w, aspect_ratio), cls_token

    def decode_points(self, features: List[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        with torch.autocast(device_type=features[0].device.type, enabled=False):
            features = [feature.float() for feature in features]
            points = self.points_head(features)[-1]
            points = F.interpolate(points, output_size, mode='bilinear', align_corners=False, antialias=False)
            return self._remap_points(points.permute(0, 2, 3, 1))

    def decode_features(self, features: List[torch.Tensor], cls_token: torch.Tensor,
                        output_size: tuple[int, int],
                        points: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if points is None and hasattr(self, 'points_head'):
            points = self.decode_points(features, output_size)
        normal = self.normal_head(features)[-1] if hasattr(self, 'normal_head') else None
        mask = self.mask_head(features)[-1] if hasattr(self, 'mask_head') else None
        if normal is not None:
            normal = F.interpolate(normal, output_size, mode='bilinear', align_corners=False, antialias=False)
            normal = F.normalize(normal.permute(0, 2, 3, 1), dim=-1)
        if mask is not None:
            mask = F.interpolate(mask, output_size, mode='bilinear', align_corners=False, antialias=False)
            mask = mask.squeeze(1).sigmoid()
        metric_scale = None
        if hasattr(self, 'scale_head'):
            with torch.autocast(device_type=cls_token.device.type, enabled=False):
                metric_scale = self.scale_head(cls_token.float()).squeeze(1).exp()
        output = {'points': points, 'normal': normal, 'mask': mask, 'metric_scale': metric_scale}
        return {key: value for key, value in output.items() if value is not None}

    def forward(
        self,
        image: torch.Tensor,
        num_tokens: Union[int, torch.LongTensor],
        event_sequence: Optional[Sequence[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        output_size = image.shape[-2:]
        refine_context = None
        base_h, base_w, aspect_ratio = self._token_grid(image, num_tokens)
        encoded, cls_token = self.encoder(
            image, base_h, base_w, return_class_token=True,
        )

        if self.anytime_enabled:
            anytime_output = self._apply_anytime(
                encoded,
                event_sequence=event_sequence,
            )
            if self.anytime_refine_enabled:
                encoded, refine_context = anytime_output
            else:
                encoded = anytime_output

        if refine_context is not None:
            encoded = self.event_refine(
                warped=encoded,
                event_voxel=refine_context['event_voxel'],
            )

        features = self._neck_features(
            encoded, image, base_h, base_w, aspect_ratio,
        )
        return self.decode_features(features, cls_token, output_size)

    @torch.inference_mode()
    def infer(
        self,
        image: torch.Tensor,
        num_tokens: int = None,
        resolution_level: int = 9,
        apply_mask: bool = True,
        fov_x: Optional[Union[Number, torch.Tensor]] = None,
        use_fp16: bool = True,
        event_sequence: Optional[Sequence[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        User-friendly inference function

        ### Parameters
        - `image`: input image tensor of shape (B, 3, H, W) or (3, H, W)
        - `num_tokens`: the number of base ViT tokens to use for inference, `'least'` or `'most'` or an integer. Suggested range: 1200 ~ 2500.
            More tokens will result in significantly higher accuracy and finer details, but slower inference time. Default: `'most'`.
        - `apply_mask`: if True, the output point map will be masked using the predicted mask. Default: True
        - `fov_x`: the horizontal camera FoV in degrees. If None, it will be inferred from the predicted point map. Default: None
        - `use_fp16`: if True, use mixed precision to speed up inference. Default: True

        ### Returns

        A dictionary containing the following keys:
        - `points`: output tensor of shape (B, H, W, 3) or (H, W, 3).
        - `depth`: tensor of shape (B, H, W) or (H, W) containing the depth map.
        - `intrinsics`: tensor of shape (B, 3, 3) or (3, 3) containing the camera intrinsics.
        """
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        image = image.to(dtype=self.dtype, device=self.device)

        original_height, original_width = image.shape[-2:]
        aspect_ratio = original_width / original_height

        # Determine the number of base tokens to use
        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        # Main outputs: this path uses only points and scale.
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_fp16 and self.dtype != torch.float16):
            output = self.forward(
                image,
                num_tokens=num_tokens,
                event_sequence=event_sequence,
            )
        points, normal, mask, metric_scale = (output.get(k, None) for k in ['points', 'normal', 'mask', 'metric_scale'])

        # Always process the output in fp32 precision
        points, normal, mask, metric_scale, fov_x = map(lambda x: x.float() if isinstance(x, torch.Tensor) else x, [points, normal, mask, metric_scale, fov_x])
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            if mask is not None:
                mask_binary = mask > 0.5
            else:
                mask_binary = None

            if points is not None:
                # Convert affine point map to camera-space. Recover depth and intrinsics from point map.
                # NOTE: Focal here is the focal length relative to half the image diagonal
                if fov_x is None:
                    # Recover focal and shift from predicted point map
                    focal, shift = recover_focal_shift(points, mask_binary)
                else:
                    # Focal is known, recover shift only
                    focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2))
                    if focal.ndim == 0:
                        focal = focal[None].expand(points.shape[0])
                    _, shift = recover_focal_shift(points, mask_binary, focal=focal)
                fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
                intrinsics = utils3d.pt.intrinsics_from_focal_center(fx, fy, torch.tensor(0.5, device=points.device, dtype=points.dtype), torch.tensor(0.5, device=points.device, dtype=points.dtype))
                points[..., 2] += shift[..., None, None]
                if mask_binary is not None:
                    mask_binary &= points[..., 2] > 0        # in case depth is contains negative values (which should never happen in practice)
                depth = points[..., 2].clone()
            else:
                depth, intrinsics = None, None

            if depth is not None:
                points = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics)

            # Apply metric scale
            if metric_scale is not None:
                if points is not None:
                    points *= metric_scale[:, None, None, None]
                if depth is not None:
                    depth *= metric_scale[:, None, None]

            # Apply mask
            if apply_mask and mask_binary is not None:
                points = torch.where(mask_binary[..., None], points, torch.inf) if points is not None else None
                depth = torch.where(mask_binary, depth, torch.inf) if depth is not None else None
                normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal)) if normal is not None else None

        return_dict = {
            'points': points,
            'intrinsics': intrinsics,
            'depth': depth,
            'mask': mask_binary,
            'normal': normal
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        if omit_batch_dim:
            return_dict = {k: v.squeeze(0) for k, v in return_dict.items()}

        return return_dict
