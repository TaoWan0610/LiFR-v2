"""Metric depth scores, averaged per image by the evaluator."""
import torch


def rel_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    rel = (torch.abs(pred - gt) / (gt + eps)).mean()
    return rel.item()


def rmse_depth(pred: torch.Tensor, gt: torch.Tensor):
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2))
    return rmse.item()


def rmse_log_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    """Root mean squared error in natural-log metric depth.

    Ground-truth validity is selected by the dataset mask before this function is
    called.  Clamp non-positive/underflowed predictions to a small positive
    depth so they receive a large finite error instead of aborting evaluation.
    """
    pred = pred.clamp_min(eps)
    gt = gt.clamp_min(eps)
    rmse_log = torch.sqrt(torch.mean((torch.log(pred) - torch.log(gt)) ** 2))
    return rmse_log.item()


def delta1_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    # Invalid non-positive predictions must count as failures, not accidentally
    # satisfy the ratio test through two negative ratios.
    valid_pred = pred > 0
    pred_safe = pred.clamp_min(eps)
    gt_safe = gt.clamp_min(eps)
    delta1 = (valid_pred & (torch.maximum(gt_safe / pred_safe, pred_safe / gt_safe) < 1.25)).float().mean()
    return delta1.item()


def compute_metrics(pred, gt):
    """Evaluate valid GT pixels without scale or shift alignment."""
    mask = gt['depth_mask']
    pred_depth, gt_depth = pred['depth_metric'], gt['depth']
    if not gt['is_metric']:
        raise ValueError('Metric depth evaluation requires metric ground truth')

    def scores(valid):
        p, g = pred_depth[valid], gt_depth[valid]
        return {'rmse': rmse_depth(p, g), 'rel': rel_depth(p, g),
                'delta1': delta1_depth(p, g), 'rmselog': rmse_log_depth(p, g)}

    metrics = {'depth_metric': scores(mask)} if mask.any() else {}
    for lower, upper in gt.get('depth_metric_ranges', ()):
        valid = mask & (gt_depth > lower) & (gt_depth <= upper)
        if valid.any():
            suffix = f'{lower:g}_{upper:g}m'.replace('.', 'p')
            metrics[f'depth_metric_{suffix}'] = scores(valid)
    return metrics
