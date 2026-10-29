"""Shared RGB prediction and temporal execution for the fixed LiFR versions."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from semseg.models.backbones import Encoder
from semseg.models.heads import SegFormerHead
from model.modules.layers import trunc_normal_
from model.modules.memory_utils import MemoryCheckpointMixin, remap_memory_state

# Dataset conventions, independent of the model architecture.
DATASETS = {
    'dsec': ('dsec', 50, 255.0),
    'shf-emergy': ('carla', 50, 1.0),
    'shf-dsec': ('carla', 50, 1.0),
    'm3ed-falcon': ('m3ed', 40, 1.0),
    'm3ed-spot': ('m3ed', 40, 255.0),
}
DATASET_ALIASES = {'carla': 'shf-emergy', 'shf_emergy': 'shf-emergy', 'm3ed': 'm3ed-spot'}


def dataset_name(name):
    name = DATASET_ALIASES.get(name, name)
    if name not in DATASETS:
        raise ValueError(f'Unknown semantic dataset: {name!r}')
    return name


class BaseModel(MemoryCheckpointMixin, nn.Module):
    """RGB encoder and semantic decoder shared by LiFR v1 and v2."""
    hrm_state_prefix = 'hrm_memory'
    BACKBONE = 'Encoder-B2'

    def __init__(self, num_classes=11, dataset='dsec'):
        super().__init__()
        self.dataset = dataset_name(dataset)
        self.dataset_type, self.event_interval_ms, self.event_scale = DATASETS[self.dataset]
        self.duration_ms = self.event_interval_ms
        self.modals = ['img']
        self.backbone_name = self.BACKBONE
        self.backbone = Encoder('B2', ['img'])
        self.decode_head = SegFormerHead(self.backbone.channels, 512, num_classes)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels // module.groups
            nn.init.normal_(module.weight, mean=0, std=(2.0 / fan_out) ** .5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _decode_features(self, features, output_size):
        logits = self.decode_head(features)
        return [F.interpolate(logits, size=output_size, mode='bilinear', align_corners=False)]

    def forward(self, inputs):
        return self._decode_features(self.backbone(inputs), inputs[0].shape[-2:])

    def init_pretrained(self, path=None):
        if not path:
            return
        state = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(state, dict):
            for key in ('model_state_dict', 'state_dict', 'model'):
                if isinstance(state.get(key), dict):
                    state = state[key]
                    break
        if not isinstance(state, dict):
            raise TypeError(f'Unsupported checkpoint: {path}')
        state = remap_memory_state({k.removeprefix('module.'): v for k, v in state.items() if torch.is_tensor(v)}, 'hrm_memory')
        target = self.state_dict()
        # A standalone RGB encoder checkpoint has no backbone prefix.
        if not any(k.startswith('backbone.') for k in state):
            state = {'backbone.' + k: v for k, v in state.items()}
        matched = {k: v for k, v in state.items() if k in target and target[k].shape == v.shape}
        if not matched:
            raise RuntimeError(f'No compatible RGB initialization weights in {path}')
        return self.load_state_dict(matched, strict=False)



class RGBBackbone(BaseModel):
    """Same-time RGB segmentation, before temporal propagation is trained."""
    version = 'backbone'

    def __init__(self, num_classes=11, dataset='dsec'):
        super().__init__(num_classes, dataset)
        self.duration_ms = 0
        self.apply(self._init_weights)

    def set_runtime(self, *, duration_ms, event_interval_ms):
        if duration_ms != 0:
            raise ValueError('RGB backbone training and evaluation require duration=0.')
        self.event_interval_ms = int(event_interval_ms)
        return self


class TemporalModel(BaseModel):
    """Common execution; subclasses construct their memory and completion modules."""

    def __init__(self, num_classes=11, dataset='dsec'):
        super().__init__(num_classes, dataset)
        from model.modules.eraft.eraft import ERAFT
        from model.modules.softsplat.frame_synthesis import Synthesis
        self.flow_net = ERAFT(n_first_channels=4, iters=12 if self.version == 'v1' else 2)
        self.softsplat_net = Synthesis(self.backbone.channels, activation='PReLU', warp_refine_identity_init=False)

    def set_runtime(self, *, duration_ms, event_interval_ms):
        """Set the prediction horizon and spacing of the input event windows."""
        if duration_ms <= 0 or event_interval_ms <= 0 or duration_ms % event_interval_ms:
            raise ValueError('The horizon must be a positive multiple of the event interval.')
        self.duration_ms = int(duration_ms)
        self.event_interval_ms = int(event_interval_ms)
        return self

    @staticmethod
    def _reduce_dsec_event(event):
        if event.ndim != 4 or event.shape[1] != 20:
            raise ValueError(f'Expected Bx20xHxW events, got {tuple(event.shape)}')
        b, _, h, w = event.shape
        return event.reshape(b, 4, 5, h, w).mean(dim=2)

    def _complete(self, features, event):
        # v1 has no completion module. v2 overrides this operation.
        return features, None

    def _append_memory(self, bank, features):
        return self.memory.append_memory(bank, features)

    def _attend_memory(self, features, bank):
        return self.memory.attend(features, bank)

    def forward(self, inputs, return_completion_aux=False):
        if len(inputs) < 3:
            raise ValueError('LiFR expects [RGB, future_event, past_event, ...].')
        if return_completion_aux and self.version != 'v2':
            raise ValueError('Completion auxiliary output belongs to LiFRV2.')
        features = self.backbone(inputs)
        previous = self._reduce_dsec_event(inputs[2])
        events = [inputs[1], *inputs[3:]]
        bank = self._append_memory([], features)
        self.memory_bank = bank
        for step, event in enumerate(events):
            current = self._reduce_dsec_event(event)
            flow = self.flow_net(previous, current)
            warped = self.softsplat_net(
                tenEncone=features, tenForward=flow,
                event_voxel=current, second_anytime_flag=False,
            )
            features = self._attend_memory(warped, bank)
            if step == len(events) - 1:
                completed, aux = self._complete(features, event)
            bank = self._append_memory(bank, features)
            self.memory_bank = bank
            previous = current
        logits = self._decode_features(completed, inputs[0].shape[-2:])
        return (logits, aux) if return_completion_aux else logits
