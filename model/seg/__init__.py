"""Public semantic versions; implementations live under semseg/models."""
from importlib import import_module
from model._imports import use_task_packages

use_task_packages('seg')
__all__ = ['LiFRV1', 'LiFRV2']


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module('semseg.models'), name)
