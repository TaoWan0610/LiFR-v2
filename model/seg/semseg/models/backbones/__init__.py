"""Load model implementations only when requested."""
from importlib import import_module

_MODULES = {'Encoder': 'encoder'}
__all__ = list(_MODULES)

def __getattr__(name):
    if name not in _MODULES: raise AttributeError(name)
    value = getattr(import_module("." + _MODULES[name], __name__), name)
    globals()[name] = value
    return value
