import numpy as np
import torch


class DSECSemsegMeter(object):
    def __init__(self, n_classes, class_names=None, ignore_idx=255):
        self.n_classes = int(n_classes)
        self.class_names = class_names or [f'class_{i}' for i in range(self.n_classes)]
        self.ignore_idx = ignore_idx
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes

    @torch.no_grad()
    def update(self, pred, gt):
        pred = pred.squeeze()
        gt = gt.squeeze()
        valid = (gt != self.ignore_idx)

        for class_idx in range(self.n_classes):
            gt_mask = (gt == class_idx)
            pred_mask = (pred == class_idx)
            self.tp[class_idx] += torch.sum(gt_mask & pred_mask & valid).item()
            self.fp[class_idx] += torch.sum(~gt_mask & pred_mask & valid).item()
            self.fn[class_idx] += torch.sum(gt_mask & ~pred_mask & valid).item()

    def reset(self):
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes

    def get_score(self, verbose=True):
        jac = [0] * self.n_classes
        for class_idx in range(self.n_classes):
            denom = float(self.tp[class_idx] + self.fp[class_idx] + self.fn[class_idx])
            jac[class_idx] = float(self.tp[class_idx]) / max(denom, 1e-8)

        eval_result = {'mIoU': np.mean(jac)}

        if verbose:
            print('\nSemantic Segmentation mIoU: {0:.4f}\n'.format(100 * eval_result['mIoU']))
            for class_name, class_iou in zip(self.class_names, jac):
                spaces = ' ' * max(0, 20 - len(class_name))
                print('{0:s}{1:s}{2:.4f}'.format(class_name, spaces, 100 * class_iou))

        return eval_result
