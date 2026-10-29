"""V1 memory for depth/joint prediction with initial and previous feature fusion."""
from .blocks import MultiAttentionBlock


class MemoryV1(MultiAttentionBlock):
    def __init__(self, dim, num_heads=8, feature_index=0,
                 layer_norm='WithBias', ffn_expansion_factor=2.66, bias=False):
        super().__init__(dim, num_heads, layer_norm, ffn_expansion_factor, bias, is_DA=True)
        self.feature_index = feature_index

    def append_memory(self, memory_bank, features, *, initialize=False):
        return [*memory_bank, features[self.feature_index]]

    def attend(self, features, memory_bank):
        output = list(features)
        output[self.feature_index] = super().forward(
            Fw=output[self.feature_index], F0_c=memory_bank[0],
            Kd=memory_bank[-1] if len(memory_bank) > 1 else None,
        )
        return output
