"""SHF-Emerge samples from the prepared train/val directories."""
from .dsec import DSEC
from .distillation import EventDistillationDataset


class SHFEmerge(DSEC):
    """Load aligned RGB, events and endpoint GT with the SHF label convention."""

    def __init__(self, *args, dataset_type='carla', **kwargs):
        if dataset_type != 'carla':
            raise ValueError('SHFEmerge requires DATASET.TYPE=carla')
        super().__init__(*args, dataset_type=dataset_type, **kwargs)


class SHFEmergeEventDistillationDataset(EventDistillationDataset):
    """Load event windows and their endpoint RGB from the requested split."""

    def __init__(self, *args, dataset_type='carla', **kwargs):
        if dataset_type != 'carla':
            raise ValueError('SHFEmergeEventDistillationDataset requires dataset_type=carla')
        super().__init__(*args, dataset_type=dataset_type, **kwargs)
