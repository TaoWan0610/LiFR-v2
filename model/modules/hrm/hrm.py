"""Task-independent historical retrieval for BCHW features."""
import math
from typing import Sequence

import torch
from torch import Tensor, nn

from .attention import MemoryAttention


class PositionEncoding(nn.Module):
    """Normalized two-dimensional sine/cosine position encoding."""

    def __init__(self, dim: int, temperature: int = 10000):
        super().__init__()
        assert dim % 4 == 0
        self.num_pos_feats = dim // 2
        self.temperature = temperature

    def forward(self, feature: Tensor) -> Tensor:
        batch, _, height, width = feature.shape
        y = torch.arange(1, height + 1, device=feature.device, dtype=torch.float32)
        x = torch.arange(1, width + 1, device=feature.device, dtype=torch.float32)
        y = y[:, None].expand(height, width) / height * (2 * math.pi)
        x = x[None, :].expand(height, width) / width * (2 * math.pi)
        scale = torch.arange(self.num_pos_feats, device=feature.device, dtype=torch.float32)
        scale = self.temperature ** (2 * (scale // 2) / self.num_pos_feats)
        px, py = x[..., None] / scale, y[..., None] / scale
        px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), -1).flatten(2)
        py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), -1).flatten(2)
        position = torch.cat((py, px), -1).permute(2, 0, 1)
        return position[None].expand(batch, -1, -1, -1).to(feature.dtype)


class HRM(nn.Module):
    """Read a bounded feature bank using self-attention, cross-attention and FFN.

    The caller selects the feature to update and explicitly writes each memory
    snapshot. HRM reads and returns BCHW tensors without modifying the bank.
    """

    def __init__(self, dim: int, num_heads: int = 8, num_memory_slots: int = 5,
                 num_layers: int = 4, ffn_dim: int = 2048, dropout: float = 0.1,
                 detach_memory: bool = False):
        super().__init__()
        if num_layers < 1 or num_memory_slots < 1:
            raise ValueError('num_layers and num_memory_slots must be positive')
        self.num_layers = num_layers
        self.num_memory_slots = num_memory_slots
        self.detach_memory = detach_memory
        self.position_encoding = PositionEncoding(dim)
        self.memory_attention = MemoryAttention(
            dim=dim, num_heads=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, num_layers=num_layers,
        )
        self.temporal_position = nn.Parameter(torch.empty(num_memory_slots, 1, dim))
        nn.init.normal_(self.temporal_position, std=0.02)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Keep older position tables intact while enforcing the configured bank capacity.
        position = state_dict.get(prefix + 'temporal_position')
        if (position is not None and position.shape[0] >= self.num_memory_slots
                and position.shape[1:] == self.temporal_position.shape[1:]
                and position.shape != self.temporal_position.shape):
            with torch.no_grad():
                self.temporal_position.set_(self.temporal_position.new_empty(position.shape))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def append_memory(self, memory_bank: Sequence[Tensor], feature: Tensor) -> list[Tensor]:
        memory = feature.detach() if self.detach_memory else feature
        return [*memory_bank, memory][-self.num_memory_slots:]

    def forward(self, current: Tensor, memory_bank: Sequence[Tensor]) -> Tensor:
        if not memory_bank:
            raise ValueError('HRM requires at least one previous memory')
        memory_bank = memory_bank[-self.num_memory_slots:]
        _, channels, height, width = current.shape
        tokens = current.flatten(2).transpose(1, 2)
        position = self.position_encoding(current).flatten(2).transpose(1, 2)
        memory = torch.cat([entry.flatten(2).transpose(1, 2) for entry in memory_bank], dim=1)
        time = self.temporal_position[-len(memory_bank):]
        memory_position = torch.cat([position + offset for offset in time], dim=1)
        output = self.memory_attention(
            current=tokens, memory=memory, current_position=position,
            memory_position=memory_position, spatial_size=(height, width),
        )
        return output.transpose(1, 2).reshape(-1, channels, height, width)
