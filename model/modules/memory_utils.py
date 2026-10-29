"""Memory version names and historical checkpoint prefix conversion."""
from collections import OrderedDict
from collections.abc import Mapping


def normalize_memory_type(value):
    aliases = {'legacy': 'memory_v1', 'memory': 'memory_v1',
               'memory_v1': 'memory_v1', 'pyramid': 'hrm', 'hrm': 'hrm'}
    return aliases[str(value).strip().lower()]


def remap_memory_state(state, hrm_prefix='hrm'):
    """Map historical HRM names and discard the unused zero-multiplied parameter."""
    if not isinstance(state, Mapping):
        return state

    def rename(key):
        parts = key.split('.')
        for index, part in enumerate(parts):
            if part not in {'pyramid_memory', 'hrm', 'hrm_memory'}:
                continue
            if part == 'pyramid_memory':
                parts[index] = hrm_prefix
            if index + 1 < len(parts):
                child = parts[index + 1]
                if child == 'no_memory':
                    return None
                parts[index + 1] = {
                    'attention': 'memory_attention', 'time_position': 'temporal_position',
                    'position': 'position_encoding',
                }.get(child, child)
        return '.'.join(parts)

    result = OrderedDict()
    for key, value in state.items():
        parts = key.split('.')
        if any(part in {'hrm', 'hrm_memory'} and parts[index + 1] in
               {'cross_attn', 'cross_attention', 'q_norm', 'current_norm'}
               for index, part in enumerate(parts[:-1])):
            raise RuntimeError('This checkpoint uses the retired CA-only HRM; use SA+CA pyramid-memory weights for LiFR v2.')
        target = rename(key)
        if target is None:
            continue
        if target not in result or target == key:
            result[target] = value
    metadata = getattr(state, '_metadata', None)
    if metadata is not None:
        result._metadata = OrderedDict((rename(key), value) for key, value in metadata.items()
                                       if rename(key) is not None)
    return result


def remap_memory_optimizer_state(state, optimizer, model):
    """Remove the old HRM's unused no_memory slot when resuming training."""
    core = getattr(model, 'module', model)
    memory = getattr(core, 'hrm_memory', getattr(core, 'hrm', None))
    if memory is None:
        return state
    for group_index, (saved, current) in enumerate(zip(state['param_groups'], optimizer.param_groups)):
        if len(saved['params']) != len(current['params']) + 1:
            continue
        for index, parameter in enumerate(current['params']):
            if parameter is not memory.temporal_position:
                continue
            # Old HRM registered no_memory directly after temporal_position.
            groups = [dict(group, params=list(group['params'])) for group in state['param_groups']]
            removed = groups[group_index]['params'].pop(index + 1)
            if 'param_names' in groups[group_index]:
                names = list(groups[group_index]['param_names'])
                names.pop(index + 1)
                groups[group_index]['param_names'] = names
            slots = dict(state['state'])
            slots.pop(removed, None)
            return dict(state, state=slots, param_groups=groups)
    return state


class MemoryCheckpointMixin:
    hrm_state_prefix = 'hrm'

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        converted = remap_memory_state(state_dict, self.hrm_state_prefix)
        state_dict.clear()
        state_dict.update(converted)
        metadata = getattr(state_dict, '_metadata', None)
        if metadata is not None and hasattr(converted, '_metadata'):
            metadata.clear()
            metadata.update(converted._metadata)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
