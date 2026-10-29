"""MiT-B0 event encoder and projections to the joint-task feature pyramid."""
from __future__ import annotations

import math
from types import MethodType
from collections.abc import Mapping, Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from model.modules.layers import DropPath


# MiT components adapted from CMX/MiT; see THIRD_PARTY_NOTICES.md.
class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio 
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            
        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)

class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)
        
    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))

class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, patch_size//2)    # padding=(ps[0]//2, ps[1]//2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


DEFAULT_EVENT_BINS = 20


def _inference_sdpa_attention(self, x: Tensor, height: int, width: int) -> Tensor:
    if self.training:
        return Attention.forward(self, x, height, width)
    batch, tokens, channels = x.shape
    q = self.q(x).reshape(batch, tokens, self.head, channels // self.head).permute(0, 2, 1, 3)
    if self.sr_ratio > 1:
        x = self.sr(x.permute(0, 2, 1).reshape(batch, channels, height, width))
        x = self.norm(x.reshape(batch, channels, -1).permute(0, 2, 1))
    k, v = self.kv(x).reshape(batch, -1, 2, self.head, channels // self.head).permute(2, 0, 3, 1, 4)
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    return self.proj(x.transpose(1, 2).reshape(batch, tokens, channels))


class MiTB0PyramidEncoder(nn.Module):
    """MiT-B0 event pyramid shared by distillation/refinement (3 or 4 stages)."""

    all_channels = (32, 64, 160, 256)
    all_depths = (2, 2, 2, 2)
    all_heads = (1, 2, 5, 8)
    all_sr_ratios = (8, 4, 2, 1)

    def __init__(
        self,
        drop_path_rate: float = 0.1,
        event_bins: int = DEFAULT_EVENT_BINS,
        num_stages: int = 4,
    ) -> None:
        super().__init__()
        self.event_bins = int(event_bins)
        self.num_stages = int(num_stages)
        if self.num_stages not in (3, 4):
            raise ValueError(f"MiTB0 num_stages must be 3 or 4, got {self.num_stages}")
        self.channels = self.all_channels[:self.num_stages]
        self.depths = self.all_depths[:self.num_stages]
        self.heads = self.all_heads[:self.num_stages]
        self.sr_ratios = self.all_sr_ratios[:self.num_stages]
        patch_embeds = [
            PatchEmbed(self.event_bins, self.channels[0], patch_size=7, stride=4),
            PatchEmbed(self.channels[0], self.channels[1], patch_size=3, stride=2),
            PatchEmbed(self.channels[1], self.channels[2], patch_size=3, stride=2),
        ]
        if self.num_stages == 4:
            patch_embeds.append(
                PatchEmbed(self.channels[2], self.channels[3], patch_size=3, stride=2)
            )
        self.patch_embeds = nn.ModuleList(patch_embeds)

        drop_paths = torch.linspace(0, drop_path_rate, sum(self.depths)).tolist()
        cursor = 0
        self.blocks = nn.ModuleList()
        for channels, depth, heads, sr_ratio in zip(self.channels, self.depths, self.heads, self.sr_ratios):
            stage = [Block(channels, heads, sr_ratio, drop_paths[cursor + index]) for index in range(depth)]
            self.blocks.append(nn.ModuleList(stage))
            cursor += depth

        self.norms = nn.ModuleList(nn.LayerNorm(channels) for channels in self.channels)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            kernel_area = module.kernel_size[0] * module.kernel_size[1]
            fan_out = kernel_area * module.out_channels / module.groups
            nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, event: Tensor) -> tuple[Tensor, ...]:
        if event.ndim != 4 or event.shape[1] != self.event_bins:
            message = f"MiTB0 expects {self.event_bins} event channels, got {tuple(event.shape)}"
            raise ValueError(message)

        x = event
        outputs = []
        batch = x.shape[0]
        for patch_embed, blocks, norm in zip(self.patch_embeds, self.blocks, self.norms):
            x, height, width = patch_embed(x)
            for block in blocks:
                x = block(x, height, width)
            x = norm(x).reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
            outputs.append(x)
        return tuple(outputs)


class SegFormerB0EventEncoder(MiTB0PyramidEncoder):
    """Native multi-bin SegFormer/MiT-B0 event encoder."""

    channels = (32, 64, 160, 256)

    def __init__(self, event_bins: int = 20) -> None:
        super().__init__(event_bins=event_bins, num_stages=4)
        self.inference_stages = None

    def configure_fast_inference(self, *, sdpa: bool = False, stages: int | None = None) -> None:
        """Opt-in execution changes only; state_dict and training math stay unchanged."""
        if stages not in (None, 3, 4):
            raise ValueError('Inference stages must be 3, 4, or None')
        self.inference_stages = stages
        for blocks in self.blocks:
            for block in blocks:
                block.attn.forward = MethodType(_inference_sdpa_attention if sdpa else Attention.forward, block.attn)

    def forward(self, event: Tensor) -> tuple[Tensor, ...]:
        if self.training or self.inference_stages is None:
            return super().forward(event)
        if event.ndim != 4 or event.shape[1] != self.event_bins:
            raise ValueError(f'Expected [B,{self.event_bins},H,W], got {tuple(event.shape)}')
        x, outputs = event, []
        count = self.inference_stages
        for patch, blocks, norm in zip(self.patch_embeds[:count], self.blocks[:count], self.norms[:count]):
            x, height, width = patch(x)
            for block in blocks:
                x = block(x, height, width)
            x = norm(x).reshape(event.shape[0], height, width, -1).permute(0, 3, 1, 2).contiguous()
            outputs.append(x)
        return tuple(outputs)


class SwinEventAdapter(nn.Module):
    """Project MiT-B0 stages to the corresponding Swin pyramid channels."""

    def __init__(
        self,
        event_bins: int = 20,
        target_channels: Sequence[int] = (96, 192, 384, 768),
    ) -> None:
        super().__init__()
        if len(target_channels) != 4:
            raise ValueError("SwinEventAdapter requires four target channels")
        self.encoder = SegFormerB0EventEncoder(event_bins=event_bins)
        self.target_channels = tuple(int(channels) for channels in target_channels)
        self.projections = nn.ModuleList(
            nn.Conv2d(source, target, 1)
            for source, target in zip(self.encoder.channels, self.target_channels)
        )
        for projection in self.projections:
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(
        self,
        event_voxel: Tensor,
        target_features: Sequence[Tensor] | None = None,
    ) -> tuple[Tensor, ...]:
        encoded = self.encoder(event_voxel)
        features = tuple(
            projection(feature)
            for projection, feature in zip(
                self.projections[:len(encoded)], encoded, strict=True,
            )
        )
        if target_features is None:
            return features
        if len(target_features) != 4:
            raise ValueError("Expected four Swin target features")
        aligned = []
        for event, target in zip(features, target_features[:len(features)], strict=True):
            if event.shape[-2:] != target.shape[-2:]:
                event = F.interpolate(
                    event,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(event)
        return tuple(aligned)

    def load_distillation_checkpoint(self, checkpoint: Mapping[str, object]) -> None:
        encoder_state = checkpoint.get("event_encoder")
        adapter_state = checkpoint.get("event_adapter")
        if not isinstance(encoder_state, Mapping):
            raise KeyError("Distillation checkpoint is missing event_encoder")
        if not isinstance(adapter_state, Mapping):
            # Compatibility with the older SemSeg four-stage checkpoint name.
            adapter_state = checkpoint.get("student_projections")
        if not isinstance(adapter_state, Mapping):
            raise KeyError("Distillation checkpoint is missing event_adapter")
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.projections.load_state_dict(adapter_state, strict=True)
