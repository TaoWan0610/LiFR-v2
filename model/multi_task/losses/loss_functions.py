# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np
import math

class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss with ignore regions.
    """
    def __init__(self, ignore_index=255, class_weight=None, balanced=False):
        super().__init__()
        self.ignore_index = ignore_index
        if balanced:
            assert class_weight is None
        self.balanced = balanced
        if class_weight is not None:
            self.register_buffer('class_weight', class_weight)
        else:
            self.class_weight = None

    def forward(self, out, label, reduction='mean'):
        label = torch.squeeze(label, dim=1).long()
        if self.balanced:
            mask = (label != self.ignore_index)
            masked_label = torch.masked_select(label, mask)
            assert torch.max(masked_label) < 2  # binary
            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            w_pos = num_labels_neg / num_total
            class_weight = torch.stack((1. - w_pos, w_pos), dim=0)
            loss = nn.functional.cross_entropy(
                out, label, weight=class_weight, ignore_index=self.ignore_index, reduction='none')
        else:
            loss = nn.functional.cross_entropy(out,
                                               label,
                                               weight=self.class_weight,
                                               ignore_index=self.ignore_index,
                                               reduction='none')
        if reduction == 'mean':
            n_valid = (label != self.ignore_index).sum()
            return (loss.sum() / max(n_valid, 1)).float()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss


class OhemCrossEntropy(nn.Module):
    """
    Online hard example mining cross entropy loss with ignore regions.
    This is primarily intended for semantic segmentation.
    """

    def __init__(self, ignore_index=255, class_weight=None, thresh=0.7, aux_weights=(1.0, 0.0)):
        super().__init__()
        self.ignore_index = ignore_index
        self.aux_weights = tuple(aux_weights)
        if not (0.0 < float(thresh) < 1.0):
            raise ValueError(f'thresh should be in (0, 1), got {thresh}')
        self.thresh = float(-math.log(float(thresh)))
        if class_weight is not None:
            self.register_buffer('class_weight', class_weight)
        else:
            self.class_weight = None
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weight,
            ignore_index=self.ignore_index,
            reduction='none',
        )

    def _forward(self, preds, labels):
        labels = torch.squeeze(labels, dim=1).long()
        valid_mask = labels != self.ignore_index
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return preds.sum() * 0.0

        n_min = max(valid_count // 16, 1)
        loss = self.criterion(preds, labels)
        loss = loss[valid_mask].view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard = loss.topk(n_min, sorted=False)[0]

        return loss_hard.mean()

    def forward(self, preds, labels, reduction='mean'):
        if reduction != 'mean':
            raise NotImplementedError('OhemCrossEntropy currently supports mean reduction only')

        if isinstance(preds, (tuple, list)):
            weights = self.aux_weights
            if len(weights) < len(preds):
                weights = weights + (weights[-1],) * (len(preds) - len(weights))
            return sum(w * self._forward(pred, labels) for pred, w in zip(preds, weights) if w != 0)

        return self._forward(preds, labels)

class BalancedBinaryCrossEntropyLoss(nn.Module):
    """
    Balanced binary cross entropy loss with ignore regions.
    """
    def __init__(self, pos_weight=None, ignore_index=255):
        super().__init__()
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

    def forward(self, output, label, reduction='mean'):

        mask = (label != self.ignore_index)
        masked_label = torch.masked_select(label, mask)
        masked_output = torch.masked_select(output, mask)

        # weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            w = num_labels_neg / num_total
            if w == 1.0:
                return 0
        else:
            w = torch.as_tensor(self.pos_weight, device=output.device)
        factor = 1. / (1 - w)

        loss = nn.functional.binary_cross_entropy_with_logits(
            masked_output,
            masked_label,
            pos_weight=w*factor,
            reduction=reduction)
        loss /= factor
        return loss


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top


class NormalsLoss(Module):
    """
    L1 loss with ignore labels
    normalize: normalization for surface normals
    """
    def __init__(self, size_average=True, normalize=False, norm=1):
        super(NormalsLoss, self).__init__()

        self.size_average = size_average

        if normalize:
            self.normalize = Normalize()
        else:
            self.normalize = None

        if norm == 1:
            print('Using L1 loss for surface normals')
            self.loss_func = F.l1_loss
        elif norm == 2:
            print('Using L2 loss for surface normals')
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

    def forward(self, out, label, ignore_label=255):
        assert not label.requires_grad
        mask = (label != ignore_label)
        n_valid = torch.sum(mask).item()

        if self.normalize is not None:
            out_norm = self.normalize(out)
            loss = self.loss_func(torch.masked_select(out_norm, mask), torch.masked_select(label, mask), reduction='sum')
        else:
            loss = self.loss_func(torch.masked_select(out, mask), torch.masked_select(label, mask), reduction='sum')

        if self.size_average:
            if ignore_label:
                ret_loss = torch.div(loss, max(n_valid, 1e-6))
                return ret_loss
            else:
                ret_loss = torch.div(loss, float(np.prod(label.size())))
                return ret_loss

        return loss

class L1Loss(nn.Module):
    """
    from ATRC
    L1 loss with ignore regions.
    normalize: normalization for surface normals
    """
    def __init__(
        self,
        normalize=False,
        ignore_index=255,
        min_value=None,
        max_value=None,
        value_weight_ranges=(),
    ):
        super().__init__()
        self.normalize = normalize
        self.ignore_index = ignore_index
        self.min_value = min_value
        self.max_value = max_value
        self.value_weight_ranges = tuple(
            (float(upper), float(weight))
            for upper, weight in value_weight_ranges
        )

    def forward(self, out, label, reduction='mean'):

        if self.normalize:
            out = nn.functional.normalize(out, p=2, dim=1)

        mask = (
            (label != self.ignore_index)
            & torch.isfinite(label)
            & (label > 0)
        ).all(dim=1, keepdim=True)
        if self.min_value is not None:
            mask &= label > self.min_value
        if self.max_value is not None:
            mask &= label < self.max_value
        if not mask.any():
            return out.sum() * 0.0

        masked_out = torch.masked_select(out, mask)
        masked_label = torch.masked_select(label, mask)
        loss = nn.functional.l1_loss(masked_out, masked_label, reduction='none')

        if self.value_weight_ranges:
            weights = torch.zeros_like(label)
            lower = self.min_value or 0.0
            for upper, weight in self.value_weight_ranges:
                weights = torch.where(
                    (label >= lower) & (label < upper),
                    weights.new_tensor(weight),
                    weights,
                )
                lower = upper
            masked_weights = torch.masked_select(weights, mask)
            loss = loss * masked_weights
        else:
            masked_weights = torch.ones_like(loss)

        if reduction == 'mean':
            return loss.sum() / masked_weights.sum().clamp_min(1.0)
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
