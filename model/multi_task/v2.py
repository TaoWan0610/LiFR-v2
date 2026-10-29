"""LiFR v2 joint semantic/depth model with HRM."""
from copy import deepcopy

from model._imports import use_task_packages

use_task_packages('multi_task')
from models.MTMamba_plus_anytime import MTMamba_plus_anytime


class Model(MTMamba_plus_anytime):
    version = 'v2'

    def __init__(self, p, backbone, **kwargs):
        p = deepcopy(p)
        p['memory_type'] = 'hrm'
        p['event_refine_enabled'] = p.get('training_stage', 'completion') == 'completion'
        p['event_refine_gate_enabled'] = True
        p.setdefault('memory_layers', 4)
        super().__init__(p, backbone, **kwargs)

    @property
    def HRM(self):
        return self.hrm
