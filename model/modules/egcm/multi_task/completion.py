"""Joint-task ConvFusion completion, gates and inference execution."""
from __future__ import annotations

from collections.abc import Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from model.modules.egcm.schedule import ScheduledGate
from .encoder import SwinEventAdapter


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvGNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            _group_norm(out_channels),
            nn.GELU(),
        )


class ConvFusion(nn.Module):
    """Predict a residual correction from aligned warp/event features."""

    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(channels * 2, hidden_channels, 1),
            ConvGNAct(hidden_channels, hidden_channels, 3, groups=hidden_channels),
            ConvGNAct(hidden_channels, hidden_channels, 1),
        )
        self.head = nn.Conv2d(hidden_channels, channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, warp: Tensor, event: Tensor) -> Tensor:
        return self.head(self.body(torch.cat((warp, event), dim=1)))


class DetailGate(nn.Module):
    """Non-pooled local gate used by the SemSeg Stage1-3 refine model."""

    def __init__(self, channels: int, projection_channels: int = 32) -> None:
        super().__init__()
        self.warp_projection = ConvGNAct(channels, projection_channels, 1)
        self.event_projection = ConvGNAct(channels, projection_channels, 1)
        self.head = nn.Sequential(
            ConvGNAct(6, 16, 3),
            nn.Conv2d(16, 1, 1),
        )

    def forward_logits(self, warp: Tensor, event: Tensor) -> Tensor:
        warp = self.warp_projection(warp)
        event = self.event_projection(event)
        difference = (event - warp).abs()
        statistics = torch.cat(
            (
                warp.mean(1, keepdim=True),
                warp.amax(1, keepdim=True),
                event.mean(1, keepdim=True),
                event.amax(1, keepdim=True),
                difference.mean(1, keepdim=True),
                difference.amax(1, keepdim=True),
            ),
            dim=1,
        )
        return self.head(statistics)

    def forward(self, warp: Tensor, event: Tensor) -> tuple[Tensor, Tensor]:
        logit = self.forward_logits(warp, event)
        return torch.sigmoid(logit), logit


class _EventInferenceGraph:
    """Fixed-shape, one-query graph. Outputs are cloned to preserve history."""
    def __init__(self, owner, warps, event):
        if not event.is_cuda or any(not x.is_cuda or x.device != event.device for x in warps):
            raise ValueError('CUDA-graph inference requires inputs on one CUDA device')
        self.signature = self.input_signature(warps, event)
        self.warps = tuple(x.detach().clone() for x in warps)
        self.event = event.detach().clone()
        self.graph = torch.cuda.CUDAGraph()
        current = torch.cuda.current_stream(event.device)
        stream = torch.cuda.Stream(device=event.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                owner._forward_impl(self.warps, self.event, return_aux=False)
        current.wait_stream(stream)
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = owner._forward_impl(self.warps, self.event, return_aux=False)

    @staticmethod
    def input_signature(warps, event):
        autocast = torch.is_autocast_enabled()
        return (tuple((tuple(x.shape), x.dtype, x.device) for x in (*warps, event)),
                autocast, torch.get_autocast_dtype('cuda') if autocast else None)

    def replay(self, warps, event):
        for destination, source in zip(self.warps, warps, strict=True):
            destination.copy_(source)
        self.event.copy_(event)
        self.graph.replay()
        # A later replay must not overwrite preceding queries' memory states.
        return {'refined_features': tuple(x.clone() for x in self.output['refined_features'])}


class SwinStage123EventRefine(ScheduledGate, nn.Module):
    """Refine Swin stages 1-3 after warping and memory attention.

    Stage 4 is passed through unchanged.  When gating is enabled, Stage-3 gate
    logits provide a detached prior to Stage 2, then Stage 2 guides Stage 1.
    Gate-disabled mode directly adds each predicted residual to its warp.
    """

    def __init__(
        self,
        event_bins: int = 20,
        stage_channels: Sequence[int] = (96, 192, 384, 768),
        hidden_channels: Sequence[int] = (128, 512, 960),
        stage3_prior: float = 0.5,
        stage2_prior: float = 0.25,
        gate_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.init_gate_schedule()
        if len(stage_channels) != 4 or len(hidden_channels) != 3:
            raise ValueError("Expected four stage channels and three hidden channels")
        self.stage_channels = tuple(int(channels) for channels in stage_channels)
        self.event_adapter = SwinEventAdapter(event_bins, self.stage_channels)
        self.fusions = nn.ModuleList(
            ConvFusion(channels, hidden)
            for channels, hidden in zip(
                self.stage_channels[:3], hidden_channels, strict=True,
            )
        )
        self.gate_enabled = bool(gate_enabled)
        self.gates = nn.ModuleList(
            DetailGate(channels) for channels in self.stage_channels[:3]
        ) if self.gate_enabled else nn.ModuleList()
        self.stage3_prior = float(stage3_prior)
        self.stage2_prior = float(stage2_prior)
        self.force_gate_one = False
        self.fast_inference = False
        self.cuda_graph_inference = False
        self._inference_graph = None

    def enable_fast_inference(self, enabled: bool = True, *, sdpa: bool = True,
                              lean: bool = True, cuda_graph: bool = False):
        """Opt-in eval path. Explicit return_aux=True preserves diagnostic outputs.

        Training retains the original attention; auxiliary outputs can be omitted.
        Existing checkpoint keys/shapes are unchanged; no weights are reinitialized.
        """
        self.fast_inference = bool(enabled and lean)
        if cuda_graph and not lean:
            raise ValueError('CUDA graph requires the lean inference output path')
        self.cuda_graph_inference = bool(enabled and cuda_graph)
        self._inference_graph = None
        self.event_adapter.encoder.configure_fast_inference(
            sdpa=bool(enabled and sdpa), stages=3 if enabled and lean else None,
        )
        return self

    def train(self, mode: bool = True):
        self._inference_graph = None
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        # Device/dtype changes invalidate graph-captured parameter pointers.
        self._inference_graph = None
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self._inference_graph = None
        return super()._load_from_state_dict(*args, **kwargs)

    def load_event_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.event_adapter.load_distillation_checkpoint(checkpoint)

    def forward(
        self,
        warped_features: Sequence[Tensor],
        event_voxel: Tensor,
        *,
        return_aux: bool | None = None,
    ) -> dict[str, object]:
        use_graph = (self.cuda_graph_inference and not self.training and not torch.is_grad_enabled()
                     and return_aux is not True and event_voxel.is_cuda)
        if use_graph:
            signature = _EventInferenceGraph.input_signature(warped_features, event_voxel)
            if self._inference_graph is None or self._inference_graph.signature != signature:
                self._inference_graph = _EventInferenceGraph(self, warped_features, event_voxel)
            return self._inference_graph.replay(warped_features, event_voxel)
        return self._forward_impl(warped_features, event_voxel, return_aux=return_aux)

    def _forward_impl(
        self, warped_features: Sequence[Tensor], event_voxel: Tensor,
        *, return_aux: bool | None = None,
    ) -> dict[str, object]:
        if len(warped_features) != 4:
            raise ValueError("SwinStage123EventRefine expects four warped stages")
        warps = tuple(warped_features)
        lean = self.fast_inference and not self.training
        if return_aux is None:
            return_aux = not lean
        events = self.event_adapter(event_voxel, warps)
        deltas = tuple(
            fusion(warp, event)
            for fusion, warp, event in zip(
                self.fusions, warps[:3], events[:3], strict=True,
            )
        )
        if self.gate_enabled and self._use_gate:
            local_logits = tuple(
                (gate.forward_logits(warp.detach(), event.detach()) if lean
                 else gate(warp.detach(), event.detach())[1])
                for gate, warp, event in zip(
                    self.gates, warps[:3], events[:3], strict=True,
                )
            )
            logit3 = local_logits[2]
            # Torch 2.0 has no CUDA BF16 bilinear upsampling kernel.  Execute
            # only this detached prior resize in FP32, then restore the dtype.
            stage3_prior = F.interpolate(
                logit3.detach().float(), size=local_logits[1].shape[-2:],
                mode="bilinear", align_corners=False,
            ).to(dtype=local_logits[1].dtype)
            logit2 = local_logits[1] + self.stage3_prior * stage3_prior
            stage2_prior = F.interpolate(
                logit2.detach().float(), size=local_logits[0].shape[-2:],
                mode="bilinear", align_corners=False,
            ).to(dtype=local_logits[0].dtype)
            logit1 = local_logits[0] + self.stage2_prior * stage2_prior
            logits = (logit1, logit2, logit3)
            gates = tuple(self.effective_gate(torch.sigmoid(logit)) for logit in logits)
            if self.force_gate_one:
                gates = tuple(torch.ones_like(gate) for gate in gates)
        else:
            local_logits = tuple(torch.zeros_like(warp[:, :1]) for warp in warps[:3])
            logits = local_logits
            gates = tuple(torch.ones_like(logit) for logit in logits)

        candidate = list(warps) if return_aux else None
        refined = list(warps)
        for stage in range(3):
            if return_aux:
                candidate[stage] = warps[stage] + deltas[stage]
            refined[stage] = (
                warps[stage] + gates[stage] * deltas[stage]
                if self.gate_enabled else warps[stage] + deltas[stage]
            )

        if not return_aux:
            return {"refined_features": tuple(refined)}
        return {
            "refined_features": tuple(refined),
            "candidate_features": tuple(candidate),
            "warped_features": warps,
            "event_features": events,
            "delta_features": deltas,
            "gates": gates,
            "gate_logits": logits,
            "gate_local_logits": local_logits,
        }
