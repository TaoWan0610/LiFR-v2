"""LiFR v1 joint semantic/depth model with its memory mechanism."""
from copy import deepcopy

from model._imports import use_task_packages

use_task_packages('multi_task')
from models.MTMamba_plus_anytime import MTMamba_plus_anytime


class Model(MTMamba_plus_anytime):
    version = 'v1'

    def __init__(self, p, backbone, **kwargs):
        p = deepcopy(p)
        p['memory_type'] = 'memory_v1'
        p['event_refine_enabled'] = False
        super().__init__(p, backbone, **kwargs)

    @property
    def memory(self):
        return self.fusion_attens
