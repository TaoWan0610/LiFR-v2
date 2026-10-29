"""LiFR v2 depth with single-block SA+CA HRM and CNN-guided completion."""
from copy import deepcopy

from model._imports import use_task_packages

use_task_packages('depth')
from moge.model.v2 import MoGeModel


class Model(MoGeModel):
    version = 'v2'

    def __init__(self, *, anytime=None, **kwargs):
        anytime = deepcopy(anytime or {})
        anytime['enabled'] = True
        anytime['memory_type'] = 'hrm'
        anytime['memory_layers'] = 1
        anytime.setdefault('refine', {})['enabled'] = True
        super().__init__(anytime=anytime, **kwargs)

    @property
    def HRM(self):
        return getattr(self, 'hrm', None)
