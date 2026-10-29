"""RGB-to-event feature and relation distillation, unmasked by default."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .encoder import MiTB0PyramidEncoder


@dataclass
class DistillationOutput:
    loss: Tensor
    feature_loss: Tensor
    intra_loss: Tensor
    cross_loss: Tensor
    mask_coverage: Tensor
    stage_losses: tuple[Tensor, ...] = ()


class EventActivationMask(nn.Module):
    """Build a valid-token mask for distillation from the full 20-bin event voxel."""

    def __init__(self, density_threshold: float = 0.0, event_epsilon: float = 0.0) -> None:
        super().__init__()
        self.density_threshold = float(density_threshold)
        self.event_epsilon = float(event_epsilon)

    def forward(self, event: Tensor, output_size: tuple[int, int]) -> Tensor:
        active_bins = event.abs().gt(self.event_epsilon).float()
        density = F.adaptive_avg_pool2d(active_bins.mean(dim=1, keepdim=True), output_size)
        return density.gt(self.density_threshold).to(event.dtype)


class ScaleEventStructureLoss(nn.Module):
    """Feature and structural relation losses with optional activity masking."""

    def __init__(self, feature_weight: float = 1.0, intra_weight: float = 10.0, cross_weight: float = 4.0, similarity_threshold: float = 0.1) -> None:
        super().__init__()
        self.feature_weight = float(feature_weight)
        self.intra_weight = float(intra_weight)
        self.cross_weight = float(cross_weight)
        self.similarity_threshold = float(similarity_threshold)

    def forward(self, teacher_feature: Tensor, student_feature: Tensor, token_mask: Tensor | None = None) -> DistillationOutput:
        teacher = teacher_feature.flatten(2).transpose(1, 2)
        student = student_feature.flatten(2).transpose(1, 2)
        coverage = student.new_ones(())
        if token_mask is not None:
            mask = token_mask.flatten(2).transpose(1, 2)
            teacher = teacher * mask
            student = student * mask
            coverage = mask.mean()
        feature_loss = F.l1_loss(student, teacher)
        if self.intra_weight == 0.0 and self.cross_weight == 0.0:
            zero = feature_loss.new_zeros(())
            loss = self.feature_weight * feature_loss
            return DistillationOutput(loss, feature_loss, zero, zero, coverage)

        teacher = F.normalize(teacher, dim=-1)
        student = F.normalize(student, dim=-1)
        teacher_similarity = torch.matmul(teacher, teacher.transpose(-1, -2)).clamp_min(0)
        relation_mask = teacher_similarity.gt(self.similarity_threshold)
        teacher_similarity = teacher_similarity * relation_mask
        if self.intra_weight:
            student_similarity = torch.matmul(student, student.transpose(-1, -2)).clamp_min(0)
            intra_loss = F.mse_loss(student_similarity * relation_mask, teacher_similarity)
        else:
            intra_loss = feature_loss.new_zeros(())
        if self.cross_weight:
            cross_similarity = torch.matmul(teacher, student.transpose(-1, -2)).clamp_min(0)
            cross_loss = F.mse_loss(cross_similarity * relation_mask, teacher_similarity)
        else:
            cross_loss = feature_loss.new_zeros(())
        loss = self.feature_weight * feature_loss + self.intra_weight * intra_loss + self.cross_weight * cross_loss
        return DistillationOutput(loss, feature_loss, intra_loss, cross_loss, coverage)


class RGBEventDistiller(nn.Module):
    """Freeze the RGB B2 teacher and distill all four Event MiT-B0 stages."""

    def __init__(self, teacher: nn.Module, density_threshold: float = 0.0, event_epsilon: float = 0.0, feature_weights: tuple[float, float, float, float] = (0.1, 0.25, 1.0, 1.0), intra_weights: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0), cross_weights: tuple[float, float, float, float] = (0.0, 0.0, 4.0, 4.0), similarity_threshold: float = 0.1, use_event_mask: bool = False) -> None:
        super().__init__()
        if not (len(feature_weights) == len(intra_weights) == len(cross_weights) == 4):
            raise ValueError("Four-stage distillation requires exactly four weights per loss")
        self.teacher = teacher.requires_grad_(False)
        self.student = MiTB0PyramidEncoder()
        self.student_projections = nn.ModuleList([
            nn.Conv2d(student_channels, teacher_channels, 1)
            for student_channels, teacher_channels
            in zip(self.student.channels, self.teacher.channels[:4])
        ])
        self.activation_mask = EventActivationMask(density_threshold, event_epsilon) if use_event_mask else None
        self.criteria = nn.ModuleList([ScaleEventStructureLoss(feature, intra, cross, similarity_threshold) for feature, intra, cross in zip(feature_weights, intra_weights, cross_weights)])
        for projection in self.student_projections:
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.teacher.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, target_rgb: Tensor, event: Tensor) -> DistillationOutput:
        with torch.no_grad():
            teacher_features = self.teacher([target_rgb])[:4]
        student_features = self.student(event)
        stage_outputs = []
        for teacher_feature, student_feature, projection, criterion in zip(teacher_features, student_features, self.student_projections, self.criteria):
            student_feature = projection(student_feature)
            token_mask = self.activation_mask(event, student_feature.shape[-2:]).float() if self.activation_mask is not None else None
            stage_outputs.append(criterion(teacher_feature.float(), student_feature.float(), token_mask))
        stage_losses = tuple(output.loss for output in stage_outputs)
        loss = torch.stack(stage_losses).sum()
        feature_loss = torch.stack([criterion.feature_weight * output.feature_loss for criterion, output in zip(self.criteria, stage_outputs)]).sum()
        intra_loss = torch.stack([criterion.intra_weight * output.intra_loss for criterion, output in zip(self.criteria, stage_outputs)]).sum()
        cross_loss = torch.stack([criterion.cross_weight * output.cross_loss for criterion, output in zip(self.criteria, stage_outputs)]).sum()
        mask_coverage = torch.stack([output.mask_coverage for output in stage_outputs]).mean()
        return DistillationOutput(loss, feature_loss, intra_loss, cross_loss, mask_coverage, stage_losses)
