"""Four-stage MiT-B0 event encoder, channel projections and distilled-weight loading."""
from __future__ import annotations
from collections.abc import Mapping
import math
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


class MiTB0PyramidEncoder(nn.Module):
    """Fixed four-stage, 20-bin MiT-B0 shared by distillation and EGCM."""

    all_channels = (32, 64, 160, 256)
    all_depths = (2, 2, 2, 2)
    all_heads = (1, 2, 5, 8)
    all_sr_ratios = (8, 4, 2, 1)

    def __init__(self) -> None:
        super().__init__()
        self.event_bins = 20
        self.channels = self.all_channels
        self.depths = self.all_depths
        self.heads = self.all_heads
        self.sr_ratios = self.all_sr_ratios
        self.patch_embeds = nn.ModuleList([
            PatchEmbed(20, 32, patch_size=7, stride=4),
            PatchEmbed(32, 64, patch_size=3, stride=2),
            PatchEmbed(64, 160, patch_size=3, stride=2),
            PatchEmbed(160, 256, patch_size=3, stride=2),
        ])
        drop_path_rate = 0.1
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


class EventAdapter(nn.Module):
    channels = (64, 128, 320, 512)

    def __init__(self) -> None:
        super().__init__()
        self.encoder = MiTB0PyramidEncoder()
        self.projections = nn.ModuleList([
            nn.Conv2d(cin, cout, 1)
            for cin, cout in zip(self.encoder.channels, self.channels)
        ])

    def load_distillation_checkpoint(
        self,
        checkpoint: Mapping[str, object],
        *,
        strict: bool = True,
    ) -> dict[str, object]:
        """Load a distilled Event B0, optionally accepting compatible tensors only."""
        if "event_encoder" not in checkpoint:
            raise KeyError("Event checkpoint is missing the 'event_encoder' state dict")

        source_encoder = checkpoint["event_encoder"]
        if not isinstance(source_encoder, Mapping):
            raise TypeError("'event_encoder' must be a state-dict mapping")

        if strict:
            self.encoder.load_state_dict(source_encoder, strict=True)
            source_projections = checkpoint["student_projections"]
            if not isinstance(source_projections, Mapping):
                raise TypeError("'student_projections' must be a state-dict mapping")
            self.projections.load_state_dict(source_projections, strict=True)
            return {
                "encoder_loaded": len(source_encoder),
                "projection_loaded": len(source_projections),
                "encoder_skipped": (),
                "projection_skipped": (),
            }

        def compatible_state(
            target: Mapping[str, Tensor],
            source: Mapping[str, object],
        ) -> tuple[dict[str, Tensor], tuple[str, ...]]:
            loaded = {
                key: value
                for key, value in source.items()
                if key in target
                and isinstance(value, Tensor)
                and value.shape == target[key].shape
            }
            skipped = tuple(
                key
                for key, value in source.items()
                if key not in target
                or not isinstance(value, Tensor)
                or value.shape != target[key].shape
            )
            return loaded, skipped

        encoder_state, encoder_skipped = compatible_state(
            self.encoder.state_dict(), source_encoder
        )
        if not encoder_state:
            raise RuntimeError(
                "Event checkpoint has no name-and-shape-compatible encoder tensors"
            )
        self.encoder.load_state_dict(encoder_state, strict=False)

        source_projections = checkpoint.get(
            "student_projections", checkpoint.get("student_projection", {})
        )
        if not isinstance(source_projections, Mapping):
            source_projections = {}
        projection_state, projection_skipped = compatible_state(
            self.projections.state_dict(), source_projections
        )
        if projection_state:
            self.projections.load_state_dict(projection_state, strict=False)

        return {
            "encoder_loaded": len(encoder_state),
            "projection_loaded": len(projection_state),
            "encoder_skipped": encoder_skipped,
            "projection_skipped": projection_skipped,
        }

    def forward(self, event: Tensor) -> tuple[Tensor, ...]:
        return tuple(
            projection(feature)
            for projection, feature in zip(self.projections, self.encoder(event))
        )
