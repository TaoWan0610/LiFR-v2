"""Single-block depth HRM: self-attention, memory retrieval and convolutional FFN.

The gated FFN is adapted from Restormer; see THIRD_PARTY_NOTICES.md.
LiFR adds temporal retrieval, self-attention and a residual output head.
"""
from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).square().mean(1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight[:, None, None] + self.bias[:, None, None]


class PositionEmbeddingSine(nn.Module):
    def __init__(self, dim: int, temperature: int = 10000):
        super().__init__()
        self.num_pos_feats = dim // 2
        self.temperature = temperature

    def forward(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        y = torch.arange(1, height + 1, device=x.device, dtype=torch.float32)
        x_positions = torch.arange(1, width + 1, device=x.device, dtype=torch.float32)
        y = y[:, None].expand(height, width) / height * 2 * math.pi
        x_positions = x_positions[None, :].expand(height, width) / width * 2 * math.pi
        scales = torch.arange(self.num_pos_feats, device=x.device, dtype=torch.float32)
        scales = self.temperature ** (2 * (scales // 2) / self.num_pos_feats)
        px, py = x_positions[..., None] / scales, y[..., None] / scales
        px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), -1).flatten(2)
        py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), -1).flatten(2)
        position = torch.cat((py, px), -1).permute(2, 0, 1)
        return position[None].expand(batch, -1, -1, -1).to(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion_factor: float):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=False)
        self.depthwise = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1,
                                   groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        first, second = self.depthwise(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(first) * second)


class DepthHRM(nn.Module):
    """Retrieve encoded history at the fused MoGe feature resolution."""
    num_layers = 1

    def __init__(self, feature_dim: int = 384, num_heads: int = 8,
                 num_memory_slots: int = 5, position_slots: int = 7,
                 ffn_expansion_factor: float = 2.66, detach_memory: bool = False):
        super().__init__()
        self.num_memory_slots = num_memory_slots
        self.detach_memory = detach_memory
        self.encoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, groups=feature_dim, bias=False),
            LayerNorm2d(feature_dim), nn.GELU(),
        )
        self.pos_encoder = PositionEmbeddingSine(feature_dim)
        self.self_norm = nn.LayerNorm(feature_dim)
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.q_norm = nn.LayerNorm(feature_dim)
        self.mem_norm = nn.LayerNorm(feature_dim)
        self.cross_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.ffn_norm = LayerNorm2d(feature_dim)
        self.ffn = FeedForward(feature_dim, ffn_expansion_factor)
        self.temporal_pos = nn.Parameter(torch.randn(position_slots, 1, feature_dim) * 0.02)
        self.delta_head = nn.Conv2d(feature_dim, feature_dim, 1)
        nn.init.zeros_(self.self_attn.out_proj.weight)
        nn.init.zeros_(self.self_attn.out_proj.bias)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def append_memory(self, memory_bank: Sequence[Tensor], feature: Tensor) -> list[Tensor]:
        memory = self.encoder(feature)
        if self.detach_memory:
            memory = memory.detach()
        return [*memory_bank, memory][-self.num_memory_slots:]

    def read_memory(self, current: Tensor, memory_bank: Sequence[Tensor]) -> Tensor:
        memory_bank = memory_bank[-self.num_memory_slots:]
        batch, channels, height, width = current.shape
        position = self.pos_encoder(current).flatten(2).transpose(1, 2)
        query = self.q_norm(current.flatten(2).transpose(1, 2)) + position
        memory = torch.cat([entry.flatten(2).transpose(1, 2) for entry in memory_bank], dim=1)
        # Position zero represents the newest state; the bank is stored oldest first.
        ages = self.temporal_pos[:len(memory_bank)].flip(0)
        memory_position = torch.cat([position + age for age in ages], dim=1)
        memory = self.mem_norm(memory)
        retrieved, _ = self.cross_attn(query, memory + memory_position, memory, need_weights=False)
        return retrieved.transpose(1, 2).reshape(batch, channels, height, width)

    def forward(self, feature: Tensor, memory_bank: Sequence[Tensor]) -> Tensor:
        current = self.encoder(feature)
        batch, channels, height, width = current.shape
        tokens = current.flatten(2).transpose(1, 2)
        normalized = self.self_norm(tokens)
        position = self.pos_encoder(current).flatten(2).transpose(1, 2)
        attended, _ = self.self_attn(normalized + position, normalized + position,
                                     normalized, need_weights=False)
        current = (tokens + attended).transpose(1, 2).reshape(batch, channels, height, width)
        fused = current + self.read_memory(current, memory_bank)
        fused = fused + self.ffn(self.ffn_norm(fused))
        return feature + self.delta_head(fused)
