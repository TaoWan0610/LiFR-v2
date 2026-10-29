"""Fixed semantic model versions and the training/inference factory."""
from copy import deepcopy
from importlib import import_module
from pathlib import Path

_MODULES = {'RGBBackbone': 'base', 'LiFRV1': 'lifr_v1', 'LiFRV2': 'lifr_v2'}
__all__ = [*_MODULES, 'build_model', 'prepare_model_config']


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module('.' + _MODULES[name], __name__), name)
    globals()[name] = value
    return value


def _dataset_id(config):
    from .base import DATASETS, DATASET_ALIASES
    data = config['DATASET']
    if data.get('ID'):
        return data['ID']
    for part in reversed(Path(data.get('ROOT', '')).parts):
        part = DATASET_ALIASES.get(part, part)
        if part in DATASETS:
            return part
    return DATASET_ALIASES.get(data.get('TYPE', 'dsec'), data.get('TYPE', 'dsec'))


def prepare_model_config(config):
    """Validate the version and dataset; architecture is defined by the model class."""
    from .base import dataset_name
    config = deepcopy(config)
    options = config['MODEL']
    name = options['NAME']
    if name not in _MODULES:
        raise ValueError(f'Unknown semantic model: {name!r}; choose RGBBackbone, LiFRV1 or LiFRV2')
    allowed = {
        'NAME', 'PRETRAINED', 'RESUME', 'RESUME_FLOWNET',
        'INIT_CHECKPOINT', 'INIT_CHECKPOINT_STRICT', 'INIT_CHECKPOINT_EXCLUDE_PREFIXES',
        'BACKBONE_CHECKPOINT', 'BASE_CHECKPOINT', 'EVENT_ENCODER_CHECKPOINT', 'EVENT_CHECKPOINT_STRICT', 'HRM_LAYERS',
    }
    unknown = options.keys() - allowed
    if unknown:
        raise ValueError(f'Unsupported MODEL settings: {sorted(unknown)}; architecture is fixed by NAME')
    train = config.setdefault('TRAIN', {})
    stage = train.setdefault('STAGE', {'RGBBackbone': 'backbone', 'LiFRV1': 'propagation', 'LiFRV2': 'completion'}[name])
    valid = {'RGBBackbone': {'backbone'}, 'LiFRV1': {'propagation'}, 'LiFRV2': {'propagation', 'completion'}}
    if stage not in valid[name]:
        raise ValueError(f'TRAIN.STAGE={stage!r} is incompatible with {name}')
    if stage == 'completion':
        from model.modules.egcm.schedule import gate_alpha
        gate_alpha(0, train.get('GATE_SCHEDULE', 'fixed'), train.get('GATE_ALPHA', 0.01))
    config['DATASET']['ID'] = dataset_name(_dataset_id(config))
    return config


def build_model(config, num_classes=11, *, duration_ms=None, event_interval_ms=None, load_event_weights=False):
    """Construct one fixed version; runtime timestamps and weight loading are separate."""
    config = prepare_model_config(config)
    options = config['MODEL']
    cls = __getattr__(options['NAME'])
    kwargs = {'hrm_layers': int(options.get('HRM_LAYERS', 4))} if options['NAME'] == 'LiFRV2' else {}
    model = cls(num_classes=num_classes, dataset=config['DATASET']['ID'], **kwargs)
    model.set_runtime(
        duration_ms=int(duration_ms if duration_ms is not None else config.get('EXPERIMENT', {}).get('DURATION_MS', model.duration_ms)),
        event_interval_ms=int(event_interval_ms if event_interval_ms is not None else config['DATASET'].get('FRAME_INTERVAL_MS', model.event_interval_ms)),
    )
    if model.version == 'v2':
        model.completion_enabled = config['TRAIN']['STAGE'] == 'completion'
    checkpoint = options.get('EVENT_ENCODER_CHECKPOINT')
    if model.version == 'v2' and model.completion_enabled and load_event_weights and checkpoint:
        import torch
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.mask_refiner.event_encoder.load_distillation_checkpoint(
            state, strict=bool(options.get('EVENT_CHECKPOINT_STRICT', True)),
        )
    return model
