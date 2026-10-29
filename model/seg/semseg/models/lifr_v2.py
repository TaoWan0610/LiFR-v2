"""LiFR v2: fixed HRM propagation and four-stage ConvFusion EGCM."""
from semseg.models.base import TemporalModel
from model.modules.hrm import HRM
from model.modules.egcm.seg.completion import GateRefine4Stage


class LiFRV2(TemporalModel):
    version = 'v2'

    def __init__(self, num_classes=11, dataset='dsec', hrm_layers=4):
        super().__init__(num_classes, dataset)
        self.hrm_memory = HRM(
            dim=self.backbone.channels[-2], num_heads=8,
            num_memory_slots=5, num_layers=hrm_layers,
        )
        self.apply(self._init_weights)
        self.mask_refiner = GateRefine4Stage()
        self.completion_enabled = True

    @property
    def memory(self):
        return self.hrm_memory

    @property
    def HRM(self):
        return self.hrm_memory

    def _append_memory(self, bank, features):
        return self.hrm_memory.append_memory(bank, features[-2])

    def _attend_memory(self, features, bank):
        features = list(features)
        features[-2] = self.hrm_memory(features[-2], bank)
        return features

    def _complete(self, features, event):
        if not self.completion_enabled:
            return features, None
        if event.ndim != 4 or event.shape[1] != 20:
            raise ValueError('EGCM requires a 20-bin event voxel.')
        aux = self.mask_refiner(
            warped_features=features,
            event_voxel=event * (255.0 / self.event_scale),
        )
        return list(aux['refined_features']), aux
