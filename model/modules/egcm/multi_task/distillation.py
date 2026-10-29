"""Distill a frozen RGB Swin encoder into the joint-task event encoder."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .encoder import SwinEventAdapter


@dataclass
class DistillationOutput:
    loss: Tensor
    stage_loss: Tensor
    intra_loss: Tensor
    cross_loss: Tensor
    stage_losses: tuple[Tensor, ...]
    student_features: tuple[Tensor, ...]
    teacher_features: tuple[Tensor, ...]


class Stage3RelationLoss(nn.Module):
    """Stage-3 self-relation and teacher/student cross-relation losses."""

    def __init__(self, similarity_threshold: float = 0.1, relation_grid: int = 32) -> None:
        super().__init__()
        self.similarity_threshold = float(similarity_threshold)
        self.relation_grid = int(relation_grid)

    def forward(self, student: Tensor, teacher: Tensor) -> tuple[Tensor, Tensor]:
        size = (
            min(self.relation_grid, teacher.shape[-2]),
            min(self.relation_grid, teacher.shape[-1]),
        )
        student = F.adaptive_avg_pool2d(student.float(), size)
        teacher = F.adaptive_avg_pool2d(teacher.detach().float(), size)
        student = F.normalize(student.flatten(2).transpose(1, 2), dim=-1)
        teacher = F.normalize(teacher.flatten(2).transpose(1, 2), dim=-1)

        teacher_similarity = (teacher @ teacher.transpose(-1, -2)).clamp_min(0)
        valid = teacher_similarity.gt(self.similarity_threshold)
        teacher_similarity = teacher_similarity.masked_fill(~valid, 0)
        student_similarity = (
            student @ student.transpose(-1, -2)
        ).clamp_min(0).masked_fill(~valid, 0)
        cross_similarity = (
            teacher @ student.transpose(-1, -2)
        ).clamp_min(0).masked_fill(~valid, 0)
        return (
            F.mse_loss(student_similarity, teacher_similarity),
            F.mse_loss(cross_similarity, teacher_similarity),
        )


class SwinB0FourStageDistiller(nn.Module):
    """Distill a frozen Swin encoder into a four-stage Event MiT-B0.

    Every adapted stage receives feature L1 supervision.  Intra- and
    cross-relation supervision are applied only at Stage 3.
    """

    def __init__(
        self,
        teacher: nn.Module,
        event_bins: int = 20,
        teacher_channels: tuple[int, int, int, int] = (96, 192, 384, 768),
        stage_weight: float = 1.0,
        intra_weight: float = 10.0,
        cross_weight: float = 4.0,
        similarity_threshold: float = 0.1,
        relation_grid: int = 32,
    ) -> None:
        super().__init__()
        self.teacher = teacher.requires_grad_(False)
        self.event_adapter = SwinEventAdapter(event_bins, teacher_channels)
        self.stage_weight = float(stage_weight)
        self.intra_weight = float(intra_weight)
        self.cross_weight = float(cross_weight)
        self.relation = Stage3RelationLoss(similarity_threshold, relation_grid)
        self.teacher.eval()

    @property
    def event_encoder(self) -> nn.Module:
        return self.event_adapter.encoder

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, target_rgb: Tensor, event_voxel: Tensor) -> DistillationOutput:
        with torch.no_grad():
            teacher_features = tuple(
                feature.detach() for feature in self.teacher(target_rgb)
            )
        student_features = self.event_adapter(event_voxel, teacher_features)
        stage_losses = tuple(
            F.l1_loss(student.float(), teacher.float())
            for student, teacher in zip(
                student_features, teacher_features, strict=True,
            )
        )
        stage_loss = torch.stack(stage_losses).mean()
        intra_loss, cross_loss = self.relation(
            student_features[2], teacher_features[2],
        )
        loss = (
            self.stage_weight * stage_loss
            + self.intra_weight * intra_loss
            + self.cross_weight * cross_loss
        )
        return DistillationOutput(
            loss=loss,
            stage_loss=stage_loss,
            intra_loss=intra_loss,
            cross_loss=cross_loss,
            stage_losses=stage_losses,
            student_features=student_features,
            teacher_features=teacher_features,
        )

    def checkpoint_state(self) -> dict[str, object]:
        return {
            "event_encoder": self.event_adapter.encoder.state_dict(),
            "event_adapter": self.event_adapter.projections.state_dict(),
        }
