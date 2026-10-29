"""V1 semantic memory: fuse Stage 4 with the fixed t0 feature."""
from .blocks import MultiAttentionBlock


class MemoryV1(MultiAttentionBlock):
    def __init__(self, dim=512, num_heads=8):
        super().__init__(dim, num_heads, 'WithBias', 2.66, False, is_DA=False)

    def append_memory(self, memory_bank, features, *, initialize=False):
        return [features[-1]] if initialize or not memory_bank else list(memory_bank)

    def attend(self, warped_features, memory_bank):
        output = list(warped_features)
        output[-1] = super().forward(Fw=output[-1], F0_c=memory_bank[0])
        return output
