# This code is referenced from MTI-Net
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import cv2
import imageio
import numpy as np
import json
import torch
import scipy.io as sio
from utils.utils import get_output
import pdb

DSEC_JOINT_REF = {
    'semseg': {
        'mIoU': 0.7215969809528099,
    },
    'depth': {
        'rmse': 4.484561558367569,
        'log_rmse': 0.569663921247184,
        'abs_rel': 0.12052780778213863,
        'sq_rel': 0.7661232128246488,
    },
}

DSEC_DEPTH_QUALITY_WEIGHTS = {
    'rmse': 0.20,
    'log_rmse': 0.30,
    'abs_rel': 0.40,
    'sq_rel': 0.10,
}


def _finite_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _safe_ratio(numerator, denominator, eps=1e-6):
    numerator = _finite_float(numerator)
    denominator = _finite_float(denominator)
    if numerator <= 0:
        return 0.0
    return numerator / max(denominator, eps)


def _dsec_joint_score(eval_results):
    semseg_result = eval_results.get('semseg', {})
    depth_result = eval_results.get('depth', {})

    seg_quality = _safe_ratio(
        semseg_result.get('mIoU'),
        DSEC_JOINT_REF['semseg']['mIoU'],
    )

    depth_quality = 0.0
    for metric, weight in DSEC_DEPTH_QUALITY_WEIGHTS.items():
        depth_quality += weight * _safe_ratio(
            DSEC_JOINT_REF['depth'][metric],
            depth_result.get(metric),
        )

    return 0.5 * seg_quality + 0.5 * depth_quality


def count_improvement(dataset_name, eval_results, task_list):
    if dataset_name == 'NYUD':
        base_result_dict = {'semseg': 0.55, 'depth': 0.49, 'normals': 18.40, 'edge': 0.0465}
        task_metric = {'semseg': 'mIoU', 'depth': 'rmse', 'normals': 'mean', 'edge': 'loss'}
        weight_dict = {'semseg': 1, 'depth': 0, 'normals': 0, 'edge': 0}
    elif dataset_name == 'PASCALContext':
        base_result_dict = {'semseg': 80.89, 'human_parts': 71.71, 'sal': 85.28, 'normals': 13.47, 'edge': 0.04}
        task_metric = {'semseg': 'mIoU', 'human_parts': 'mIoU', 'sal': 'maxF', 'normals': 'mean', 'edge': 'loss'}
        weight_dict = {'semseg': 1, 'human_parts': 1, 'sal': 1, 'normals': 0, 'edge': 0}
    elif dataset_name == 'Cityscapes':
        base_result_dict = {'semseg': 80, 'depth': 1}
        task_metric = {'semseg': 'mIoU', 'depth': 'rmse'}
        weight_dict = {'semseg': 1, 'depth': 0}
    elif dataset_name == 'DSEC':
        if 'semseg' in task_list and 'depth' in task_list:
            return _dsec_joint_score(eval_results)
        if 'semseg' in task_list:
            return eval_results['semseg']['mIoU']
        if 'depth' in task_list:
            return -eval_results['depth']['rmse']
        return 0.0
    elif dataset_name in {'M3ED', 'shf_emergy'}:
        # M3ED does not have a fixed cross-task reference baseline here.
        # Keep semantic segmentation as the primary checkpoint criterion while
        # still evaluating and recording all enabled depth metrics.
        if 'semseg' in task_list:
            return eval_results['semseg']['mIoU']
        if 'depth' in task_list:
            return -eval_results['depth']['rmse']
        return 0.0
    else:
        raise ValueError

    base_result, new_result, weight = [], [], []
    for tname in task_list:
        base_result.append(base_result_dict[tname])
        new_result.append(eval_results[tname][task_metric[tname]])
        weight.append(weight_dict[tname])

    assert len(weight) == len(base_result) == len(new_result)
    improvement = (((-1)**np.array(weight))*\
                    (np.array(base_result)-np.array(new_result))/\
                     np.array(base_result)).mean()
    return improvement


class PerformanceMeter(object):
    """ A general performance meter which shows performance across one or more tasks """
    def __init__(self, p, tasks):
        self.database = p['train_db_name']
        self.tasks = tasks
        self.meters = {t: get_single_task_meter(p, self.database, t) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        for t in self.tasks:
            self.meters[t].update(pred[t], gt[t])

    def get_score(self, verbose=True):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score(verbose)

        return eval_dict

def get_single_task_meter(p, database, task):
    """ Retrieve a meter to measure the single-task performance """

    # ignore index based on transforms.AddIgnoreRegions
    if task == 'semseg':
        if database in {'DSEC', 'M3ED', 'shf_emergy'}:
            from evaluation.eval_semseg_dsec import DSECSemsegMeter
            return DSECSemsegMeter(
                n_classes=p.TASKS.NUM_OUTPUT['semseg'],
                class_names=getattr(p, 'semseg_class_names', None),
                ignore_idx=p.ignore_index,
            )
        from evaluation.eval_semseg import SemsegMeter
        return SemsegMeter(database, ignore_idx=p.ignore_index)

    elif task == 'human_parts':
        from evaluation.eval_human_parts import HumanPartsMeter
        return HumanPartsMeter(database, ignore_idx=p.ignore_index)

    elif task == 'normals':
        from evaluation.eval_normals import NormalsMeter
        return NormalsMeter(ignore_index=p.ignore_index) 

    elif task == 'sal':
        from evaluation.eval_sal import  SaliencyMeter
        return SaliencyMeter(ignore_index=p.ignore_index, threshold_step=0.05, beta_squared=0.3)

    elif task == 'depth':
        from evaluation.eval_depth import DepthMeter
        return DepthMeter(ignore_index=p.ignore_index, max_depth=p.TASKS.depth_max, min_depth=p.TASKS.depth_min) 

    elif task == 'edge': # just for reference
        from evaluation.eval_edge import EdgeMeter
        return EdgeMeter(pos_weight=p['edge_w'], ignore_index=p.ignore_index)

    else:
        raise NotImplementedError

@torch.no_grad()
def save_model_pred_for_one_task(p, sample, output, save_dirs, task=None, epoch=None):
    """ Save model predictions for one task"""

    inputs, meta = sample['image'].cuda(non_blocking=True), sample['meta']
    output_task = get_output(output[task], task)

    for jj in range(int(inputs.size()[0])):
        if len(sample[task][jj].unique()) == 1 and sample[task][jj].unique() == p.ignore_index:
            continue
        fname = meta['img_name'][jj]

        im_height = meta['img_size'][jj][0]
        im_width = meta['img_size'][jj][1]
        pred = output_task[jj] # (H, W) or (H, W, C)
        # if we used padding on the input, we crop the prediction accordingly
        if (im_height, im_width) != pred.shape[:2]:
            delta_height = max(pred.shape[0] - im_height, 0)
            delta_width = max(pred.shape[1] - im_width, 0)
            if delta_height > 0 or delta_width > 0:
                # deprecated by python
                # height_location = [delta_height // 2,
                #                    (delta_height // 2) + im_height]
                # width_location = [delta_width // 2,
                #                   (delta_width // 2) + im_width]
                height_begin = torch.div(delta_height, 2, rounding_mode="trunc")
                height_location = [height_begin, height_begin + im_height]
                width_begin =torch.div(delta_width, 2, rounding_mode="trunc")
                width_location = [width_begin, width_begin + im_width]
                pred = pred[height_location[0]:height_location[1],
                            width_location[0]:width_location[1]]
        assert pred.shape[:2] == (im_height, im_width)
        if pred.ndim == 3:
            raise
        result = pred.cpu().numpy()
        if task == 'depth':
            sio.savemat(os.path.join(save_dirs[task], fname + '.mat'), {'depth': result})
        else:
            imageio.imwrite(os.path.join(save_dirs[task], fname + '.png'), result.astype(np.uint8))
