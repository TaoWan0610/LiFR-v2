# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

from shutil import ignore_patterns
import warnings
import cv2
import os.path
import numpy as np
import glob
import torch
import json
import scipy.io as sio

class DepthMeter(object):
    def __init__(self, ignore_index=255, max_depth=None, min_depth=None):
        self.total_rmses = 0.0
        self.total_log_rmses = 0.0
        self.n_valid = 0.0
        self.ignore_index = ignore_index
        self.max_depth = max_depth
        self.min_depth = min_depth

        self.abs_rel = 0.0
        self.sq_rel = 0.0

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()
        mask = torch.isfinite(gt) & (gt > 0) & (gt != self.ignore_index)
        if self.max_depth is not None and self.min_depth is not None:
            mask &= (gt < self.max_depth) & (gt > self.min_depth)
        valid_pixels = mask.float().sum().item()
        self.n_valid += valid_pixels # Valid pixels per image
        if valid_pixels == 0:
            return
        
        pred = torch.nan_to_num(pred, nan=1e-9, posinf=1e9, neginf=1e-9).clamp_min(1e-9)

        # Per pixel rmse and log-rmse.
        log_rmse_tmp = torch.pow(torch.log(gt[mask]) - torch.log(pred[mask]), 2)
        self.total_log_rmses += log_rmse_tmp.sum().item()

        rmse_tmp = torch.pow(gt[mask] - pred[mask], 2)
        self.total_rmses += rmse_tmp.sum().item()

        # abs rel
        self.abs_rel += (torch.abs(gt[mask] - pred[mask]) / gt[mask]).sum().item()
        # sq_rel
        self.sq_rel += (((gt[mask] - pred[mask]) ** 2) / gt[mask]).sum().item()

    def reset(self):
        self.rmses = []
        self.log_rmses = []
        
    def get_score(self, verbose=True):
        eval_result = dict()
        if self.n_valid == 0:
            eval_result['rmse'] = float('inf')
            eval_result['log_rmse'] = float('inf')
            eval_result['abs_rel'] = float('inf')
            eval_result['sq_rel'] = float('inf')
            if verbose:
                print('Results for depth prediction')
                print('No valid depth pixels were found during evaluation; returning inf metrics.')
            return eval_result
        eval_result['rmse'] = np.sqrt(self.total_rmses / self.n_valid)
        eval_result['log_rmse'] = np.sqrt(self.total_log_rmses / self.n_valid)
        eval_result['abs_rel'] = self.abs_rel / self.n_valid
        eval_result['sq_rel'] = self.sq_rel / self.n_valid

        if verbose:
            print('Results for depth prediction')
            for x in eval_result:
                spaces = ''
                for j in range(0, 15 - len(x)):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_result[x]))

        return eval_result
