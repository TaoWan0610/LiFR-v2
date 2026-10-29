"""Adapters for the five public semantic datasets."""
from .dsec import DSEC
from .shf_dsec import SHFDSEC
from .m3ed import M3ED
from .shf_emergy import SHFEmerge, SHFEmergeEventDistillationDataset
from .distillation import EventDistillationDataset
from .sampling import RepeatDataset

DATASETS = {
    'dsec': DSEC,
    'shf-dsec': SHFDSEC,
    'shf-emergy': SHFEmerge,
    'm3ed-falcon': M3ED,
    'm3ed-spot': M3ED,
}
_ADAPTERS = {'DSEC': DSEC, 'SHFDSEC': SHFDSEC, 'SHFEmerge': SHFEmerge, 'M3ED': M3ED}


def get_dataset_class(name):
    """Resolve a canonical dataset ID or one of its explicit layout adapters."""
    if name in DATASETS:
        return DATASETS[name]
    if name in _ADAPTERS:
        return _ADAPTERS[name]
    raise ValueError(f'Unknown semantic dataset {name!r}; choose from {tuple(DATASETS)}')


def get_distillation_dataset_class(dataset_id):
    if dataset_id not in DATASETS:
        raise ValueError(f'Unknown distillation dataset: {dataset_id!r}')
    return SHFEmergeEventDistillationDataset if dataset_id == 'shf-emergy' else EventDistillationDataset


__all__ = [
    'DSEC', 'SHFDSEC', 'M3ED', 'SHFEmerge', 'EventDistillationDataset',
    'SHFEmergeEventDistillationDataset', 'RepeatDataset',
    'DATASETS', 'get_dataset_class', 'get_distillation_dataset_class',
]
