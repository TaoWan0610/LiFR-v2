"""Four-stage ConvFusion, pooled spatial gates for feature completion."""
from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from model.modules.egcm.schedule import ScheduledGate
from .encoder import EventAdapter


def build_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Choose a GroupNorm group count that divides the channel count."""
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvGNAct(nn.Module):     # Convolution + GroupNorm + activation
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 groups: int = 1, activation: bool = True) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            build_group_norm(out_channels),
        ]
        if activation:
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConvFusion(nn.Module):
    def __init__(self, channels: int, hidden: int = 512) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(channels * 2, hidden, kernel_size=1),
            ConvGNAct(hidden, hidden, kernel_size=3, groups=hidden),
            ConvGNAct(hidden, hidden, kernel_size=1),
        )
        self.head = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, warp: Tensor, event: Tensor) -> Tensor:
        return self.head(self.body(torch.cat([warp, event], dim=1)))


class PoolGate(nn.Module):
    """Predict a spatial gate from channel statistics and local Avg/Max pooling.

    The Bx1xHxW gate is shared across feature channels. Pooling preserves H/W.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        reduced = max(channels // 4, 16)
        hidden = max(reduced // 2, 16)
        self.reduce = ConvGNAct(channels * 2, reduced, kernel_size=1)
        self.head = nn.Sequential(
            ConvGNAct(4, hidden, kernel_size=5),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, warp: Tensor, event: Tensor) -> tuple[Tensor, Tensor]:
        feature = self.reduce(torch.cat([warp, event], dim=1))
        statistics = torch.cat([
            feature.mean(dim=1, keepdim=True),
            feature.amax(dim=1, keepdim=True),
        ], dim=1)
        pooled = torch.cat([
            F.avg_pool2d(statistics, 3, stride=1, padding=1),
            F.max_pool2d(statistics, 3, stride=1, padding=1),
        ], dim=1)
        logit = self.head(pooled)
        return torch.sigmoid(logit), logit


class GateRefine4Stage(ScheduledGate, nn.Module):
    """Four-stage ConvFusion completion with learned residual gates."""
    channels = (64, 128, 320, 512)
    hidden_channels = (512, 512, 960, 1024)

    def __init__(self) -> None:
        super().__init__()
        self.init_gate_schedule()
        self.event_encoder = EventAdapter()
        self.fusions = nn.ModuleList(
            ConvFusion(channels, hidden)
            for channels, hidden in zip(self.channels, self.hidden_channels)
        )
        self.gates = nn.ModuleList(PoolGate(channels) for channels in self.channels)

    def forward(
        self,
        warped_features: Sequence[Tensor],
        event_voxel: Tensor,
    ) -> dict[str, tuple[Tensor, ...]]:
        event_features = self.event_encoder(event_voxel)
        aligned_events = tuple(
            F.interpolate(event, size=warp.shape[-2:], mode="bilinear", align_corners=False)
            for warp, event in zip(warped_features, event_features)
        )

        fusion_outputs = tuple(
            fusion(warp, event)
            for fusion, warp, event in zip(self.fusions, warped_features, aligned_events)
        )
        deltas = fusion_outputs
        gates = tuple(
            self.effective_gate(gate(warp.detach(), event.detach())[0])
            for gate, warp, event in zip(self.gates, warped_features, aligned_events)
        ) if self._use_gate else tuple(torch.ones_like(warp[:, :1]) for warp in warped_features)
        refined = tuple(
            warp + gate * delta
            for warp, gate, delta in zip(warped_features, gates, deltas)
        )
        return {
            "refined_features": refined,
            "gates": gates,
        }
