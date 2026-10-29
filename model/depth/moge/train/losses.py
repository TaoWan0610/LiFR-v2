from typing import *
import math

import torch
import torch.nn.functional as F
import utils3d

from ..utils.geometry_torch import weighted_mean, harmonic_mean
from ..utils.alignment import align_points_scale_z_shift, align_points_scale_xyz_shift


def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


def affine_invariant_global_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    mask: torch.Tensor,
    align_resolution: int = 64,
    beta: float = 0.0,
    trunc: float = 1.0,
    sparsity_aware: bool = False
):
    pred_points = pred_points.float()
    gt_points = gt_points.float()
    device = pred_points.device

    mask = mask.bool()
    if not mask.any():
        batch_shape = pred_points.shape[:-3]
        zero = torch.nan_to_num(pred_points).sum() * 0.0
        loss = zero.expand(batch_shape) if batch_shape else zero
        scale = torch.zeros(batch_shape, dtype=pred_points.dtype, device=device)
        return loss, {'truncated_error': 0.0, 'delta': 0.0, 'skipped_no_valid_anchor': 1.0}, scale
    gt_points = torch.where(mask[..., None], gt_points, 1)
    pred_points = torch.where(mask[..., None], pred_points, 0)

    # Align
    pred_points_lr, gt_points_lr, lr_mask = utils3d.pt.masked_nearest_resize(pred_points, gt_points, mask=mask, size=(align_resolution, align_resolution))
    if not lr_mask.any():
        batch_shape = pred_points.shape[:-3]
        zero = pred_points.sum() * 0.0
        loss = zero.expand(batch_shape) if batch_shape else zero
        scale = torch.zeros(batch_shape, dtype=pred_points.dtype, device=device)
        return loss, {'truncated_error': 0.0, 'delta': 0.0, 'skipped_no_valid_anchor': 1.0}, scale
    scale, shift = align_points_scale_z_shift(pred_points_lr.flatten(-3, -2), gt_points_lr.flatten(-3, -2), lr_mask.flatten(-2, -1) / gt_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2), trunc=trunc)
    valid = torch.isfinite(scale) & torch.isfinite(shift).all(dim=-1) & (scale > 0)
    scale, shift = torch.where(valid, scale, 0), torch.where(valid[..., None], shift, 0)

    pred_points = scale[..., None, None, None] * pred_points + shift[..., None, None, :]

    # Compute loss
    weight = (valid[..., None, None] & mask).float() / gt_points[..., 2].clamp_min(1e-5)
    weight = weight.clamp_max(10.0 * weighted_mean(weight, mask, dim=(-2, -1), keepdim=True))   # In case your data contains extremely small depth values
    loss = _smooth((pred_points - gt_points).abs() * weight[..., None], beta=beta).mean(dim=(-3, -2, -1))

    if sparsity_aware:
        # Reweighting improves performance on sparse depth data. NOTE: this is not used in MoGe-1.
        sparsity = mask.float().mean(dim=(-2, -1)) / lr_mask.float().mean(dim=(-2, -1))
        loss = loss / (sparsity + 1e-7)

    err = torch.nan_to_num((pred_points.detach() - gt_points).norm(dim=-1) / gt_points[..., 2], nan=0.0, posinf=1.0, neginf=1.0)

    # Record any scalar metric
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1.0), mask).item(),
        'delta': weighted_mean((err < 1).float(), mask).item()
    }

    return loss, misc, scale.detach()


def monitoring(points: torch.Tensor):
    return {
        'std': points.std().item(),
    }


def compute_anchor_sampling_weight(
    points: torch.Tensor,
    mask: torch.Tensor,
    radius_2d: torch.Tensor,
    radius_3d: torch.Tensor,
    num_test: int = 64
) -> torch.Tensor:
    # Importance sampling to balance the sampled probability of fine strutures.
    # NOTE: MoGe-1 uses uniform random sampling instead of importance sampling.
    #       This is an incremental trick introduced later than the publication of MoGe-1 paper.

    height, width = points.shape[-3:-1]

    pixel_i, pixel_j = torch.meshgrid(
        torch.arange(height, device=points.device),
        torch.arange(width, device=points.device),
        indexing='ij'
    )

    test_delta_i = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
    test_delta_j = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
    test_i, test_j = pixel_i[..., None] + test_delta_i, pixel_j[..., None] + test_delta_j                       # [height, width, num_test]
    test_mask = (test_i >= 0) & (test_i < height) & (test_j >= 0) & (test_j < width)                            # [height, width, num_test]
    test_i, test_j = test_i.clamp(0, height - 1), test_j.clamp(0, width - 1)                                    # [height, width, num_test]
    test_mask = test_mask & mask[..., test_i, test_j]                                                           # [..., height, width, num_test]
    test_points = points[..., test_i, test_j, :]                                                                # [..., height, width, num_test, 3]
    test_dist = (test_points - points[..., None, :]).norm(dim=-1)                                               # [..., height, width, num_test]

    weight = 1 / ((test_dist <= radius_3d[..., None]) & test_mask).float().sum(dim=-1).clamp_min(1)
    weight = torch.where(mask, weight, 0)
    weight = weight / weight.sum(dim=(-2, -1), keepdim=True).add(1e-7)                                          # [..., height, width]
    return weight


def affine_invariant_local_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    mask: torch.Tensor,
    focal: torch.Tensor,
    global_scale: torch.Tensor,
    level: Literal[4, 16, 64],
    align_resolution: int = 32,
    num_patches: int = 16,
    beta: float = 0.0,
    trunc: float = 1.0,
    sparsity_aware: bool = False
):
    pred_points = pred_points.float()
    gt_points = gt_points.float()
    focal = focal.float()
    if global_scale is not None:
        global_scale = global_scale.float()
    device, dtype = pred_points.device, pred_points.dtype
    *batch_shape, height, width, _ = pred_points.shape
    batch_size = math.prod(batch_shape)

    gt_mask = mask.bool()
    if not gt_mask.any():
        zero = torch.nan_to_num(pred_points).sum() * 0.0
        return zero, {'skipped_no_valid_anchor': 1.0}
    gt_points = torch.where(gt_mask[..., None], gt_points, 1)
    pred_points = torch.where(gt_mask[..., None], pred_points, 0)
    pred_points, gt_points, gt_mask, focal, global_scale = pred_points.reshape(-1, height, width, 3), gt_points.reshape(-1, height, width, 3), gt_mask.reshape(-1, height, width), focal.reshape(-1), global_scale.reshape(-1) if global_scale is not None else None

    # Sample patch anchor points indices [num_total_patches]
    radius_2d = math.ceil(0.5 / level * (height ** 2 + width ** 2) ** 0.5)
    radius_3d = 0.5 / level / focal * gt_points[..., 2]
    anchor_sampling_weights = compute_anchor_sampling_weight(gt_points, gt_mask, radius_2d, radius_3d, num_test=64)
    where_mask = torch.where(gt_mask)
    random_selection = torch.multinomial(anchor_sampling_weights[where_mask], num_patches * batch_size, replacement=True)
    patch_batch_idx, patch_anchor_i, patch_anchor_j = [indices[random_selection] for indices in where_mask]     # [num_total_patches]

    # Get patch indices [num_total_patches, patch_h, patch_w]
    patch_i, patch_j = torch.meshgrid(
        torch.arange(-radius_2d, radius_2d + 1, device=device),
        torch.arange(-radius_2d, radius_2d + 1, device=device),
        indexing='ij'
    )
    patch_i, patch_j = patch_i + patch_anchor_i[:, None, None], patch_j + patch_anchor_j[:, None, None]
    patch_mask = (patch_i >= 0) & (patch_i < height) & (patch_j >= 0) & (patch_j < width)
    patch_i, patch_j = patch_i.clamp(0, height - 1), patch_j.clamp(0, width - 1)

    # Get patch mask and gt patch points
    gt_patch_anchor_points = gt_points[patch_batch_idx, patch_anchor_i, patch_anchor_j]
    gt_patch_radius_3d = 0.5 / level / focal[patch_batch_idx] * gt_patch_anchor_points[:, 2]
    gt_patch_points = gt_points[patch_batch_idx[:, None, None], patch_i, patch_j]
    gt_patch_dist = (gt_patch_points - gt_patch_anchor_points[:, None, None, :]).norm(dim=-1)
    patch_mask &= gt_mask[patch_batch_idx[:, None, None], patch_i, patch_j]
    patch_mask &= gt_patch_dist <= gt_patch_radius_3d[:, None, None]

    # Pick only non-empty patches
    MINIMUM_POINTS_PER_PATCH = 32
    nonempty = torch.where(patch_mask.sum(dim=(-2, -1)) >= MINIMUM_POINTS_PER_PATCH)
    num_nonempty_patches = nonempty[0].shape[0]
    if num_nonempty_patches == 0:
        return torch.tensor(0.0, dtype=dtype, device=device), {}

    # Finalize all patch variables
    patch_batch_idx, patch_i, patch_j = patch_batch_idx[nonempty], patch_i[nonempty], patch_j[nonempty]
    patch_mask = patch_mask[nonempty]                                   # [num_nonempty_patches, patch_h, patch_w]
    gt_patch_points = gt_patch_points[nonempty]                         # [num_nonempty_patches, patch_h, patch_w, 3]
    gt_patch_radius_3d = gt_patch_radius_3d[nonempty]                   # [num_nonempty_patches]
    gt_patch_anchor_points = gt_patch_anchor_points[nonempty]           # [num_nonempty_patches, 3]
    pred_patch_points = pred_points[patch_batch_idx[:, None, None], patch_i, patch_j]

    # Align patch points
    pred_patch_points_lr, gt_patch_points_lr, patch_lr_mask = utils3d.pt.masked_nearest_resize(pred_patch_points, gt_patch_points, mask=patch_mask, size=(align_resolution, align_resolution))
    local_scale, local_shift = align_points_scale_xyz_shift(pred_patch_points_lr.flatten(-3, -2), gt_patch_points_lr.flatten(-3, -2), patch_lr_mask.flatten(-2) / gt_patch_radius_3d[:, None].add(1e-7), trunc=trunc)
    if global_scale is not None:
        scale_differ = local_scale / global_scale[patch_batch_idx]
        patch_valid = (
            torch.isfinite(local_scale)
            & torch.isfinite(local_shift).all(dim=-1)
            & torch.isfinite(scale_differ)
            & (scale_differ > 0.1)
            & (scale_differ < 10.0)
            & (global_scale[patch_batch_idx] > 0)
        )
    else:
        patch_valid = torch.isfinite(local_scale) & torch.isfinite(local_shift).all(dim=-1) & (local_scale > 0)
    local_scale, local_shift = torch.where(patch_valid, local_scale, 0), torch.where(patch_valid[:, None], local_shift, 0)
    patch_mask &= patch_valid[:, None, None]

    pred_patch_points = local_scale[:, None, None, None] * pred_patch_points + local_shift[:, None, None, :]                   # [num_patches_nonempty, patch_h, patch_w, 3]

    # Compute loss
    gt_mean = harmonic_mean(gt_points[..., 2], gt_mask, dim=(-2, -1))
    patch_weight = patch_mask.float() / gt_patch_points[..., 2].clamp_min(0.1 * gt_mean[patch_batch_idx, None, None])          # [num_patches_nonempty, patch_h, patch_w]
    loss = _smooth((pred_patch_points - gt_patch_points).abs() * patch_weight[..., None], beta=beta).mean(dim=(-3, -2, -1))    # [num_patches_nonempty]

    if sparsity_aware:
        # Reweighting improves performance on sparse depth data. NOTE: this is not used in MoGe-1.
        sparsity = patch_mask.float().mean(dim=(-2, -1)) / patch_lr_mask.float().mean(dim=(-2, -1))
        loss = loss / (sparsity + 1e-7)
    # pred_points may be FP16 while GT-driven arithmetic promotes the patch
    # loss to FP32. scatter_reduce requires self/src to have identical dtypes.
    loss = torch.scatter_reduce(
        torch.zeros(batch_size, dtype=loss.dtype, device=device),
        dim=0,
        index=patch_batch_idx,
        src=loss,
        reduce='sum',
    ) / num_patches
    loss = loss.reshape(batch_shape)

    err = torch.nan_to_num(
        (pred_patch_points.detach() - gt_patch_points).norm(dim=-1) / gt_patch_radius_3d[..., None, None],
        nan=0.0,
        posinf=1.0,
        neginf=1.0,
    )

    # Record any scalar metric
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1), patch_mask).item(),
        'delta': weighted_mean((err < 1).float(), patch_mask).item()
    }

    return loss, misc


def mask_bce_loss(pred_mask_prob: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
    # Compute BCE in FP32 for numerical stability when the model-facing mask
    # prediction is FP16. Gradients still flow back through the cast.
    loss = (gt_mask_pos | gt_mask_neg) * F.binary_cross_entropy(
        pred_mask_prob.float(), gt_mask_pos.float(), reduction='none'
    )
    loss = loss.mean(dim=(-2, -1))
    return loss, {}


def metric_scale_loss(scale_pred: torch.Tensor, scale_gt: torch.Tensor):
    scale_pred_fp32 = scale_pred.float()
    scale_gt_fp32 = scale_gt.float()
    valid = (
        torch.isfinite(scale_gt_fp32)
        & (scale_gt_fp32 > 0)
    )
    safe_gt = torch.where(valid, scale_gt_fp32, 1.0)
    log_error = scale_pred_fp32.clamp_min(1e-6).log() - safe_gt.log()
    loss = torch.where(valid, log_error.square(), 0.0)
    return loss, {
        'log_ratio': (
            log_error[valid].detach().abs().mean().item()
            if valid.any()
            else 0.0
        ),
        'valid': valid.float().mean().item(),
    }


