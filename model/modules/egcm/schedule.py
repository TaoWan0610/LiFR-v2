"""Gate strength shared by completion training and checkpointed inference."""
import torch


def gate_alpha(progress, schedule='fixed', alpha=0.01):
    """Interpolate (0, .01), (.5, .5), (1, 1) or keep a fixed strength."""
    if schedule == 'fixed':
        value = float(alpha)
    elif schedule == 'progressive':
        progress = min(max(float(progress), 0.0), 1.0)
        value = 0.01 + 0.98 * progress if progress <= 0.5 else progress
    else:
        raise ValueError(f'Unknown gate schedule: {schedule!r}; use fixed or progressive')
    if not 0.0 <= value <= 1.0:
        raise ValueError('Gate alpha must be between 0 and 1')
    return value


class ScheduledGate:
    def init_gate_schedule(self):
        self.register_buffer('_gate_active', torch.tensor(True))
        self.register_buffer('_gate_alpha', torch.tensor(0.01))
        self._use_gate = True

    def update_gate_alpha(self, progress, schedule='fixed', alpha=0.01):
        value = gate_alpha(progress, schedule, alpha)
        self._gate_alpha.fill_(value)
        if hasattr(self, '_inference_graph'):
            self._inference_graph = None
        return value

    def effective_gate(self, gate):
        alpha = self._gate_alpha.to(dtype=gate.dtype)
        return (1 - alpha) + alpha * gate

    def set_gate_active(self, active):
        active = bool(active)
        if active != self._use_gate and hasattr(self, '_inference_graph'):
            self._inference_graph = None
        self._use_gate = active
        self._gate_active.fill_(active)

    def train_gate(self, active):
        gates = getattr(self, 'gates', None)
        if gates is None:
            gates = self.gate
        gates.requires_grad_(active)
        gates.train(active)
        self.set_gate_active(active)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Preserve the predictions of checkpoints trained before alpha blending.
        key = prefix + '_gate_active'
        state_dict.setdefault(key, torch.tensor(True))
        state_dict.setdefault(prefix + '_gate_alpha', torch.tensor(1.0))
        self._use_gate = bool(state_dict[key].item())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
