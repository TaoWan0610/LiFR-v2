"""LiFR v1 depth with its memory mechanism, on the MoGe v2 RGB backbone."""
from copy import deepcopy

from model._imports import use_task_packages

use_task_packages('depth')
from moge.model.v2 import MoGeModel


class Model(MoGeModel):
    version = 'v1'

    def __init__(self, *, anytime=None, **kwargs):
        anytime = deepcopy(anytime or {})
        anytime['enabled'] = True
        anytime['memory_type'] = 'memory_v1'
        anytime['refine'] = {'enabled': False}
        super().__init__(anytime=anytime, **kwargs)

    @property
    def memory(self):
        return getattr(self, 'fusion_attens', None)
