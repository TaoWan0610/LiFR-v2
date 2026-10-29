"""SHF-DSEC dataset entry, sharing the aligned temporal loader with DSEC."""
from .dsec import DSEC


class SHFDSEC(DSEC):
    """SHF-DSEC RGB, events and semantic labels in the prepared SHF layout."""

    def __init__(self, *args, dataset_type='carla', **kwargs):
        if dataset_type != 'carla':
            raise ValueError('SHFDSEC requires DATASET.TYPE=carla')
        super().__init__(*args, dataset_type=dataset_type, **kwargs)
