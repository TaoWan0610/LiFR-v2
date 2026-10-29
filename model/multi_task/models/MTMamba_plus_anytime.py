import sys
from pathlib import Path

import torch

from .MTMamba_plus import MTMamba_plus


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.modules.eraft.eraft import ERAFT
from model.modules.memory_v1.depth import MemoryV1
from model.modules.memory_utils import MemoryCheckpointMixin, normalize_memory_type
from model.modules.hrm import HRM
from model.modules.softsplat.frame_synthesis import Synthesis

from model.modules.egcm.multi_task.completion import SwinStage123EventRefine


def _load_state_dict_compatible(path, map_location=torch.device('cpu')):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ('model', 'state_dict', 'model_state_dict'):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format for {path}: {type(ckpt)}")

    cleaned = {}
    for key, value in ckpt.items():
        key = key.replace('module.', '', 1) if key.startswith('module.') else key
        key = key.replace('flow_net.', '', 1) if key.startswith('flow_net.') else key
        cleaned[key] = value
    return cleaned


class MTMamba_plus_anytime(MemoryCheckpointMixin, MTMamba_plus):
    def __init__(self, p, backbone, d_state=16, dt_rank="auto", ssm_ratio=2, mlp_ratio=0):
        super().__init__(p, backbone, d_state=d_state, dt_rank=dt_rank, ssm_ratio=ssm_ratio, mlp_ratio=mlp_ratio)

        self.time_window_steps = int(getattr(p, 'time_window_steps', 2))
        self.event_bins = int(getattr(p, 'event_bins', 20))
        self.event_reduce_factor = int(getattr(p, 'event_reduce_factor', 5))
        if self.time_window_steps not in (0, 1, 2):
            raise NotImplementedError(
                f'MTMamba_plus_anytime currently supports time_window_steps in {{0, 1, 2}}, got {self.time_window_steps}'
            )
        if self.event_bins % self.event_reduce_factor != 0:
            raise ValueError(
                f'event_bins ({self.event_bins}) must be divisible by event_reduce_factor ({self.event_reduce_factor})'
            )

        self.memory_type = normalize_memory_type(getattr(p, 'memory_type', 'memory_v1'))
        self.flow_iters = int(getattr(p, 'flow_iters', 2 if self.memory_type == 'hrm' else 12))
        self.warm_start_flow = bool(getattr(p, 'warm_start_flow', False))
        self.anytime_flag = bool(getattr(p, 'anytime_flag', False))
        self.n_first_channels = self.event_bins // self.event_reduce_factor

        self.flow_net = ERAFT(
            n_first_channels=self.n_first_channels,
            iters=self.flow_iters,
        )
        self.softsplat_net = Synthesis(
            list(self.feature_channel),
            activation='PReLU',
            warp_refine_identity_init=True,
        )
        if self.memory_type == 'hrm':
            self.hrm = HRM(
                dim=int(self.feature_channel[-2]),
                num_heads=int(getattr(p, 'memory_num_heads', 8)),
                num_memory_slots=int(getattr(p, 'memory_num_slots', 5)),
                num_layers=int(getattr(p, 'memory_layers', 4)),
            )
        else:
            self.fusion_attens = MemoryV1(
                dim=int(self.feature_channel[-1]), num_heads=int(getattr(p, 'memory_num_heads', 8)),
                feature_index=-1,
            )

        self.event_refine_enabled = bool(getattr(p, 'event_refine_enabled', False))
        self.event_refine_input_multiplier = float(
            getattr(p, 'event_refine_input_multiplier', 1.0)
        )
        if self.event_refine_input_multiplier <= 0:
            raise ValueError('event_refine_input_multiplier must be positive')
        self.event_refine = None
        if self.event_refine_enabled:
            self.event_refine = SwinStage123EventRefine(
                event_bins=self.event_bins,
                stage_channels=tuple(int(channels) for channels in self.feature_channel),
                stage3_prior=float(getattr(p, 'event_refine_stage3_prior', 0.5)),
                stage2_prior=float(getattr(p, 'event_refine_stage2_prior', 0.25)),
                gate_enabled=bool(getattr(p, 'event_refine_gate_enabled', True)),
            )
            checkpoint = str(getattr(p, 'event_refine_checkpoint', ''))
            if checkpoint:
                self.event_refine.load_event_checkpoint(checkpoint)

        resume_flow_net = getattr(p, 'resume_flow_net', '')
        if resume_flow_net:
            self._load_flow_checkpoint(resume_flow_net)

    def _load_flow_checkpoint(self, checkpoint_path):
        flow_state = _load_state_dict_compatible(checkpoint_path)
        target_state = self.flow_net.state_dict()
        flow_state = {
            key: value for key, value in flow_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        if not flow_state:
            raise RuntimeError(
                f'No compatible E-RAFT tensors were found in {checkpoint_path}'
            )
        msg = self.flow_net.load_state_dict(flow_state, strict=False)
        print(f'[flow] loaded {len(flow_state)} compatible keys from {checkpoint_path}')
        print(msg)

    def _bin_event_voxel(self, event_voxel):
        channels = event_voxel.shape[1]
        if channels % self.event_reduce_factor != 0:
            raise ValueError(
                f'Event voxel channel count {channels} is not divisible by event_reduce_factor {self.event_reduce_factor}'
            )
        return torch.cat(
            [
                event_voxel[:, self.event_reduce_factor * i:self.event_reduce_factor * (i + 1)].mean(1, keepdim=True)
                for i in range(channels // self.event_reduce_factor)
            ],
            dim=1,
        )

    def _decode_from_features(self, selected_fea):
        x_dict = None
        for stage in range(len(self.feature_channel) - 1):
            x_dict = self._forward_expand(x_dict, selected_fea, stage)
            x_dict = self._forward_block_stm(x_dict, stage)
            x_dict = self._forward_block_ctm(x_dict, stage)
            x_dict = {t: xx.permute(0, 3, 1, 2) for t, xx in x_dict.items()}

        out = {}
        for t in self.tasks:
            z = self.final_expand[t](x_dict[t])
            z = self.final_project[t](z)
            z = torch.nn.functional.interpolate(
                z.float(), self.img_size, mode='bilinear', align_corners=False
            )
            out[t] = torch.nn.functional.softplus(z) if t == 'depth' else z
        return out

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            selected_fea = self.backbone(batch)
            return self._decode_from_features(selected_fea)

        images = batch['image']
        selected_fea = self.backbone(images)

        time_window_steps = self.time_window_steps
        if 'event_after' in batch:
            time_window_steps = 2

        if time_window_steps == 0:
            return self._decode_from_features(selected_fea)

        event_sequence = [batch['event_before'], batch['event']]
        if time_window_steps == 2:
            event_sequence.append(batch['event_after'])
        binned_events = [self._bin_event_voxel(event) for event in event_sequence]

        flow_state = None
        memory_module = self.hrm if self.memory_type == 'hrm' else self.fusion_attens
        memory_input = selected_fea[-2] if self.memory_type == 'hrm' else selected_fea
        memory_bank = memory_module.append_memory([], memory_input)
        warped_features = selected_fea
        for step in range(time_window_steps):
            if self.warm_start_flow:
                flow_state, flow = self.flow_net(
                    binned_events[step],
                    binned_events[step + 1],
                    flow_init=flow_state,
                    return_state=True,
                )
                flow_state = flow_state.detach()
            else:
                flow = self.flow_net(binned_events[step], binned_events[step + 1])

            feature_dtype = warped_features[0].dtype
            with torch.cuda.amp.autocast(enabled=False):
                warped_features = self.softsplat_net(
                    tenEncone=[feature.float() for feature in warped_features],
                    tenForward=flow.float(),
                    event_voxel=binned_events[step + 1].float(),
                    second_anytime_flag=self.anytime_flag and step > 0,
                )
            warped_features = [feature.to(feature_dtype) for feature in warped_features]
            if self.memory_type == 'hrm':
                warped_features[-2] = self.hrm(warped_features[-2], memory_bank)
            else:
                warped_features = self.fusion_attens.attend(warped_features, memory_bank)
            if step < time_window_steps - 1:
                memory_input = warped_features[-2] if self.memory_type == 'hrm' else warped_features
                memory_bank = memory_module.append_memory(memory_bank, memory_input)

        if self.event_refine is not None:
            refine_output = self.event_refine(
                warped_features,
                event_sequence[-1] * self.event_refine_input_multiplier,
                return_aux=False,
            )
            warped_features = refine_output['refined_features']

        return self._decode_from_features(warped_features)
