"""SAM 2 memory attention adapted for LiFR's feature shapes and temporal bank.

SAM 2 copyright Meta Platforms, Inc. and affiliates, Apache-2.0.
See THIRD_PARTY_NOTICES.md for the original license.
"""

import copy
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _axial_rotary_frequencies(
    head_dim: int,
    height: int,
    width: int,
    device: torch.device,
    theta: float,
) -> Tensor:
    """Return the complex 2-D axial RoPE frequencies for one feature map."""
    assert head_dim % 4 == 0
    frequencies = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 4, device=device, dtype=torch.float32)
            / head_dim
        )
    )
    positions = torch.arange(height * width, device=device, dtype=torch.float32)
    x_positions = positions.remainder(width)
    y_positions = torch.div(positions, width, rounding_mode="floor")
    x_angles = torch.outer(x_positions, frequencies)
    y_angles = torch.outer(y_positions, frequencies)
    return torch.cat(
        (
            torch.polar(torch.ones_like(x_angles), x_angles),
            torch.polar(torch.ones_like(y_angles), y_angles),
        ),
        dim=-1,
    )


def _apply_rotary_encoding(tokens: Tensor, frequencies: Tensor) -> Tensor:
    complex_tokens = torch.view_as_complex(
        tokens.float().reshape(*tokens.shape[:-1], -1, 2)
    )
    frequencies = frequencies.view(1, 1, frequencies.shape[0], frequencies.shape[1])
    rotated = torch.view_as_real(complex_tokens * frequencies).flatten(-2)
    return rotated.to(dtype=tokens.dtype)


class RotaryAttention(nn.Module):
    """Multi-head scaled-dot-product attention with SAM 2 axial RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        repeat_key_rope: bool,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        assert (dim // num_heads) % 4 == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.repeat_key_rope = repeat_key_rope
        self.rope_theta = rope_theta

        self.q_projection = nn.Linear(dim, dim)
        self.k_projection = nn.Linear(dim, dim)
        self.v_projection = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)
        self.register_buffer(
            "_rotary_frequencies",
            torch.empty(0, dtype=torch.complex64),
            persistent=False,
        )
        self._rotary_spatial_size = (-1, -1)

    def _split_heads(self, tokens: Tensor) -> Tensor:
        batch, length, _ = tokens.shape
        return tokens.reshape(
            batch,
            length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _get_rotary_frequencies(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tensor:
        spatial_size = (height, width)
        if (
            self._rotary_spatial_size != spatial_size
            or self._rotary_frequencies.device != device
        ):
            self._rotary_frequencies = _axial_rotary_frequencies(
                self.head_dim,
                height,
                width,
                device,
                self.rope_theta,
            )
            self._rotary_spatial_size = spatial_size
        return self._rotary_frequencies

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        spatial_size: Tuple[int, int],
    ) -> Tensor:
        query = self._split_heads(self.q_projection(query))
        key = self._split_heads(self.k_projection(key))
        value = self._split_heads(self.v_projection(value))

        height, width = spatial_size
        spatial_tokens = height * width
        assert query.shape[-2] == spatial_tokens
        assert key.shape[-2] % spatial_tokens == 0

        frequencies = self._get_rotary_frequencies(height, width, query.device)
        query = _apply_rotary_encoding(query, frequencies)
        if self.repeat_key_rope:
            frequencies = frequencies.repeat(key.shape[-2] // spatial_tokens, 1)
        else:
            assert key.shape[-2] == spatial_tokens
        key = _apply_rotary_encoding(key, frequencies)

        dropout = self.dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout,
        )
        batch, _, length, _ = output.shape
        output = output.transpose(1, 2).reshape(batch, length, -1)
        return self.output_projection(output)


class MemoryAttentionLayer(nn.Module):
    """Pre-norm self-attention, memory cross-attention, then FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = RotaryAttention(
            dim,
            num_heads,
            dropout,
            repeat_key_rope=False,
        )
        self.cross_attention = RotaryAttention(
            dim,
            num_heads,
            dropout,
            repeat_key_rope=True,
        )
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.activation = nn.ReLU()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        current: Tensor,
        memory: Tensor,
        memory_position: Tensor,
        spatial_size: Tuple[int, int],
    ) -> Tensor:
        normalized = self.norm1(current)
        current = current + self.dropout1(
            self.self_attention(
                normalized,
                normalized,
                normalized,
                spatial_size,
            )
        )

        normalized = self.norm2(current)
        current = current + self.dropout2(
            self.cross_attention(
                normalized,
                memory + memory_position,
                memory,
                spatial_size,
            )
        )

        normalized = self.norm3(current)
        feedforward = self.linear2(
            self.ffn_dropout(self.activation(self.linear1(normalized)))
        )
        return current + self.dropout3(feedforward)


class MemoryAttention(nn.Module):
    """Configurable SAM 2 memory-attention stack with final normalization."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        layer = MemoryAttentionLayer(
            dim=dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        current: Tensor,
        memory: Tensor,
        current_position: Tensor,
        memory_position: Tensor,
        spatial_size: Tuple[int, int],
    ) -> Tensor:
        output = current + 0.1 * current_position
        for layer in self.layers:
            output = layer(
                current=output,
                memory=memory,
                memory_position=memory_position,
                spatial_size=spatial_size,
            )
        return self.norm(output)
