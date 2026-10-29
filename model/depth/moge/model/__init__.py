"""Construct the RGB backbone or LiFR depth model from its configuration."""
from copy import deepcopy
from pathlib import Path


def build_model(config):
    from model.modules.memory_utils import normalize_memory_type
    config = deepcopy(config)
    anytime = config.get('anytime', {})
    memory = normalize_memory_type(anytime.get('memory_type', 'memory_v1'))
    if not anytime.get('enabled', False):
        from .v2 import MoGeModel as Model
    elif memory == 'hrm' and anytime.get('refine', {}).get('enabled', False):
        from model.depth.v2 import Model
    elif memory == 'memory_v1' and not anytime.get('refine', {}).get('enabled', False):
        from model.depth.v1 import Model
    else:
        # HRM propagation checkpoints can initialize the completion stage.
        from .v2 import MoGeModel as Model
    return Model(**config)


def load_model(path, **hf_kwargs):
    import torch
    if not Path(path).is_file():
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=str(path), filename='model.pt', **hf_kwargs)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = deepcopy(checkpoint['model_config'])
    anytime = config.get('anytime', {})
    anytime.get('flow', {}).pop('checkpoint', None)
    anytime.get('refine', {}).pop('event_checkpoint', None)
    model = build_model(config)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model
