"""Distill frozen RGB MoGe features into the CNN event encoder."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encoder import build_cnn_encoder


@dataclass
class CNNDistillationOutput:
    loss: Tensor
    loss_l1: Tensor
    loss_intra: Tensor
    loss_cross: Tensor
    mask_coverage: Tensor


def build_event_mask(activity: Tensor, out_hw: tuple[int, int], tau: float = 64.0) -> Tensor:
    # activity: [B,1,H,W] raw H5 event counts -> mask: [B,1,Ht,Wt]
    height, width = activity.shape[-2:]
    out_h, out_w = out_hw
    density = F.adaptive_avg_pool2d(activity, out_hw)
    density = density * float(height * width) / float(out_h * out_w)
    return density.ge(tau).to(activity.dtype)


class ScaleEventLoss(nn.Module):
    def __init__(
        self,
        intra_weight: float = 10.0,
        cross_weight: float = 4.0,
        similarity_threshold: float = 0.1,
        use_event_mask: bool = False,
    ) -> None:
        super().__init__()
        self.intra_weight = intra_weight
        self.cross_weight = cross_weight
        self.similarity_threshold = similarity_threshold

    def forward(self, event: Tensor, teacher: Tensor, mask: Tensor) -> CNNDistillationOutput:
        # [B,D,H,W] -> [B,N,D], masked by raw-event activity
        event = event.float().flatten(2).transpose(1, 2)
        teacher = teacher.detach().float().flatten(2).transpose(1, 2)
        token_mask = mask.float().flatten(2).transpose(1, 2)
        event = event * token_mask
        teacher = teacher * token_mask

        loss_l1 = F.l1_loss(event, teacher)
        event = F.normalize(event, dim=-1)
        teacher = F.normalize(teacher, dim=-1)
        teacher_sim = (teacher @ teacher.transpose(-1, -2)).clamp_min_(0)
        event_sim = (event @ event.transpose(-1, -2)).clamp_min_(0)
        cross_sim = (teacher @ event.transpose(-1, -2)).clamp_min_(0)
        relation_mask = teacher_sim.gt(self.similarity_threshold)
        teacher_sim = teacher_sim.masked_fill(~relation_mask, 0)
        event_sim = event_sim.masked_fill(~relation_mask, 0)
        cross_sim = cross_sim.masked_fill(~relation_mask, 0)
        loss_intra = F.mse_loss(event_sim, teacher_sim)
        loss_cross = F.mse_loss(cross_sim, teacher_sim)
        loss = loss_l1 + self.intra_weight * loss_intra + self.cross_weight * loss_cross
        return CNNDistillationOutput(loss, loss_l1, loss_intra, loss_cross, mask.mean())


class MetaDepthCNNDistiller(nn.Module):
    event_encoder_type = "metadepth-cnn"

    def __init__(
        self,
        teacher: nn.Module,
        event_bins: int = 20,
        teacher_dim: int = 384,
        event_tau: float = 64.0,
        intra_weight: float = 10.0,
        cross_weight: float = 4.0,
        similarity_threshold: float = 0.1,
        use_event_mask: bool = False,
    ) -> None:
        super().__init__()
        self.teacher = teacher.requires_grad_(False)
        self.event_bins = event_bins
        self.teacher_dim = teacher_dim
        self.event_encoder = build_cnn_encoder("large", dim_in=event_bins)
        channels = sum(shape.channels for shape in self.event_encoder.output_shapes.values())
        self.student_projection = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, teacher_dim, 3, padding=1),
        )
        self.loss_fn = ScaleEventLoss(intra_weight, cross_weight, similarity_threshold)
        self.event_tau = event_tau
        self.use_event_mask = bool(use_event_mask)
        self.teacher.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def _teacher_feature(self, image: Tensor, num_tokens: int) -> Tensor:
        token_h, token_w, _ = self.teacher._token_grid(image, num_tokens)
        return self.teacher.encoder(image, token_h, token_w, return_class_token=False)

    def forward(
        self,
        target_image: Tensor,
        event: Tensor,
        activity: Tensor | None,
        num_tokens: int,
    ) -> CNNDistillationOutput:
        with torch.no_grad():
            teacher = self._teacher_feature(target_image, num_tokens)

        # C0..C4 -> [B,576,Ht,Wt] -> [B,384,Ht,Wt]
        stages = self.event_encoder(event)
        stages = [
            F.interpolate(x, teacher.shape[-2:], mode="bilinear", align_corners=False)
            for x in stages.values()
        ]
        event_feature = self.student_projection(torch.cat(stages, dim=1))
        mask = (
            build_event_mask(activity, teacher.shape[-2:], self.event_tau)
            if self.use_event_mask
            else teacher.new_ones((teacher.shape[0], 1, *teacher.shape[-2:]))
        )
        return self.loss_fn(event_feature, teacher, mask)
