"""LiFR v1: RGB propagation with memory of the fixed t0 feature, without EGCM."""
from semseg.models.base import TemporalModel
from model.modules.memory_v1 import MemoryV1


class LiFRV1(TemporalModel):
    version = 'v1'

    def __init__(self, num_classes=11, dataset='dsec'):
        super().__init__(num_classes, dataset)
        self.fusion_attens = MemoryV1(dim=self.backbone.channels[-1], num_heads=8)
        self.apply(self._init_weights)

    @property
    def memory(self):
        return self.fusion_attens
