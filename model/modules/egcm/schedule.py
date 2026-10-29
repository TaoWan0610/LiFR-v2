"""Checkpointed gate state for completion-stage warm-up."""
import torch


class ScheduledGate:
    def init_gate_schedule(self):
        self.register_buffer('_gate_active', torch.tensor(True))
        self._use_gate = True

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
        # Older released checkpoints always evaluated their learned gates.
        key = prefix + '_gate_active'
        state_dict.setdefault(key, torch.tensor(True))
        self._use_gate = bool(state_dict[key].item())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
