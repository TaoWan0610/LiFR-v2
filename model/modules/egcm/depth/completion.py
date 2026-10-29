"""CNN event-guided depth completion with ConvFusion and a learned gate."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from model.modules.egcm.schedule import ScheduledGate

from .encoder import build_cnn_encoder


def _gn(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _block(in_channels: int, out_channels: int, kernel_size: int = 1,
           groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, groups=groups, bias=False,
        ),
        _gn(out_channels),
        nn.GELU(),
    )


class ConvFusion(nn.Module):
    def __init__(self, channels: int = 384) -> None:
        super().__init__()
        self.warp_proj = _block(channels, channels)
        self.event_proj = _block(channels, channels)
        self.body = nn.Sequential(
            _block(2 * channels, channels),
            _block(channels, channels, 3, channels),
            _block(channels, channels),
        )
        self.head = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, warped: Tensor, event: Tensor) -> Tensor:
        # [B,384,Hp,Wp] x 2 -> [B,768,Hp,Wp] -> delta [B,384,Hp,Wp]
        x = torch.cat((self.warp_proj(warped), self.event_proj(event)), dim=1)
        return self.head(self.body(x))


class PoolGate(nn.Module):
    def __init__(self, channels: int = 384, hidden_channels: int = 16) -> None:
        super().__init__()
        self.proj = _block(3 * channels, channels)
        self.head = nn.Sequential(
            _block(2, hidden_channels, 5),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, warped: Tensor, event: Tensor) -> tuple[Tensor, Tensor]:
        # [Fw, Fe, Fe-Fw]: 1152 -> 384 -> channel avg/max -> spatial heatmap
        x = self.proj(torch.cat((warped, event, event - warped), dim=1))
        x = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), dim=1)
        logit = self.head(x)
        return logit.sigmoid(), logit


class MoGeEventRefine(ScheduledGate, nn.Module):
    event_encoder_type = "metadepth-cnn"

    def __init__(self, feature_channels: int, event_bins: int = 20) -> None:
        super().__init__()
        self.init_gate_schedule()
        self.event_encoder = build_cnn_encoder("large", dim_in=event_bins)
        channels = sum(shape.channels for shape in self.event_encoder.output_shapes.values())
        self.event_adapter = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, feature_channels, 3, padding=1),
        )
        self.fusion = ConvFusion(feature_channels)
        self.gate = PoolGate(feature_channels)

    def load_event_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_type = checkpoint.get("event_encoder_type", self.event_encoder_type)
        if checkpoint_type != self.event_encoder_type:
            raise ValueError(f"Expected a CNN event checkpoint, got {checkpoint_type!r}")
        self.event_encoder.load_state_dict(checkpoint["event_encoder"], strict=True)
        self.event_adapter.load_state_dict(checkpoint["student_projection"], strict=True)

    def _event_feature(self, event_voxel: Tensor, output_size: tuple[int, int]) -> Tensor:
        stages = self.event_encoder(event_voxel).values()
        stages = [F.interpolate(
            feature, output_size, mode="bilinear", align_corners=False,
        ) for feature in stages]
        return self.event_adapter(torch.cat(stages, dim=1))

    def forward(self, warped: Tensor, event_voxel: Tensor) -> Tensor:
        event = self._event_feature(event_voxel, warped.shape[-2:])
        delta = self.fusion(warped, event)
        if not self._use_gate:
            return warped + delta
        gate, _ = self.gate(warped.detach(), event.detach())
        return warped + self.effective_gate(gate) * delta
