"""Dataset construction, temporal supervision checks and distributed loaders."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, Sampler, DistributedSampler, RandomSampler
from semseg import datasets
from semseg.datasets import RepeatDataset
from semseg.augmentations import (get_mask_refine_train_augmentation,
                                     get_train_augmentation, get_val_augmentation)

def _seed_data_worker(worker_id: int) -> None:
    np.random.seed(3407 + worker_id)

def _loader_runtime_kwargs(num_workers: int, prefetch_factor: int = 2) -> dict:
    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    if num_workers > 0:
        kwargs.update(
            persistent_workers=True,
            worker_init_fn=_seed_data_worker,
            prefetch_factor=max(1, int(prefetch_factor)),
        )
    return kwargs

class DistributedEvalSampler(Sampler):
    """Shard validation without padding or duplicating samples."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.world_size - 1) // self.world_size)

def build_datasets(cfg, duration):
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    classes = 11
    semantic_layout_dataset = dataset_cfg['NAME'] in {'DSEC', 'SHFDSEC', 'SHFEmerge'}
    traintransform = (
        get_mask_refine_train_augmentation()
        if train_cfg.get('STAGE') == 'completion'
        else get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    )
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    dataset_extra_kwargs = {}
    if semantic_layout_dataset:
        dataset_extra_kwargs['frame_interval_ms'] = int(
            dataset_cfg.get('FRAME_INTERVAL_MS', 50)
        )
        dataset_extra_kwargs['frame_index_stride'] = int(
            dataset_cfg.get('FRAME_INDEX_STRIDE', 1)
        )
        dataset_extra_kwargs['anchor_modulo'] = int(
            dataset_cfg.get('ANCHOR_MODULO', 0)
        )
        dataset_extra_kwargs['anchor_remainders'] = dataset_cfg.get(
            'ANCHOR_REMAINDERS'
        )

    train_dataset_extra_kwargs = dict(dataset_extra_kwargs)
    val_dataset_extra_kwargs = dict(dataset_extra_kwargs)
    if semantic_layout_dataset:
        train_dataset_extra_kwargs['sequence_names'] = dataset_cfg.get('TRAIN_SEQUENCES')
        val_dataset_extra_kwargs['sequence_names'] = dataset_cfg.get('VAL_SEQUENCES')
        train_dataset_extra_kwargs['exclude_sequence_names'] = dataset_cfg.get(
            'TRAIN_EXCLUDE_SEQUENCES'
        )
        val_dataset_extra_kwargs['exclude_sequence_names'] = dataset_cfg.get(
            'VAL_EXCLUDE_SEQUENCES'
        )

    trainset = datasets.get_dataset_class(dataset_cfg['NAME'])(
        dataset_cfg['ROOT'].replace("${DURATION}", str(duration)),
        'train',
        classes,
        traintransform,
        dataset_cfg['MODALS'],
        duration=duration,
        dataset_type=dataset_cfg['TYPE'],
        sample_gap=int(dataset_cfg.get('SAMPLE_GAP', 0)),
        **train_dataset_extra_kwargs,
    )
    if len(trainset) == 0:
        raise RuntimeError(
            "Training dataset is empty after causal input filtering. "
            f"Check RGB/event/semantic paths for root={dataset_cfg['ROOT']!r}, "
            f"duration={duration}ms."
        )
    # Compute the target length after padding.
    if len(trainset) % train_cfg['BATCH_SIZE'] != 0:
        num_batches = math.ceil(len(trainset) / train_cfg['BATCH_SIZE'])
        target_length = num_batches * train_cfg['BATCH_SIZE']
        trainset = RepeatDataset(trainset, target_length)
    valset = datasets.get_dataset_class(dataset_cfg['NAME'])(
        dataset_cfg['ROOT'].replace("${DURATION}", str(duration)),
        'val',
        classes,
        valtransform,
        dataset_cfg['MODALS'],
        duration=duration,
        dataset_type=dataset_cfg['TYPE'],
        sample_gap=int(dataset_cfg.get('VAL_SAMPLE_GAP', 0)),
        **val_dataset_extra_kwargs,
    )
    if len(valset) == 0:
        raise RuntimeError(
            "Validation dataset is empty after causal input filtering. "
            f"Check RGB/event/semantic paths for root={dataset_cfg['ROOT']!r}, "
            f"duration={duration}ms."
        )
    frame_interval_ms = int(getattr(trainset, 'frame_interval_ms', 0) or 0)
    if frame_interval_ms <= 0:
        raise RuntimeError(f"Invalid dataset frame interval: {frame_interval_ms} ms")
    expected_supervision_offsets = [duration]
    temporal_datasets = [('train', trainset), ('val', valset)]
    for split_name, dataset in temporal_datasets:
        actual_offsets = list(
            getattr(dataset, 'supervision_offsets_ms', [])
        )
        if actual_offsets != expected_supervision_offsets:
            raise RuntimeError(
                f"{split_name} temporal supervision mismatch: "
                f"duration={duration}ms, "
                f"expected_offsets={expected_supervision_offsets}, "
                f"dataset_offsets={actual_offsets}"
            )
    if (not train_cfg['DDP']) or torch.distributed.get_rank() == 0:
        print(
            f"[temporal_supervision] RGB=t0 duration={duration}ms "
            f"frame_interval={frame_interval_ms}ms "
            f"GT_offsets_ms={expected_supervision_offsets}",
            flush=True,
        )
    return trainset, valset


@dataclass
class Loaders:
    train: DataLoader
    val: DataLoader


def build_loaders(cfg, duration):
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    trainset, valset = build_datasets(cfg, duration)
    distributed = bool(train_cfg.get('DDP', False))
    train_sampler = DistributedSampler(trainset, shuffle=True) if distributed else RandomSampler(trainset)
    def validation_loader(dataset):
        return DataLoader(dataset, batch_size=eval_cfg['BATCH_SIZE'],
                          sampler=DistributedEvalSampler(dataset) if distributed else None,
                          **_loader_runtime_kwargs(int(eval_cfg.get('NUM_WORKERS', 0)),
                                                  int(eval_cfg.get('PREFETCH_FACTOR', 2))))
    return Loaders(
        DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], drop_last=True,
                   sampler=train_sampler,
                   **_loader_runtime_kwargs(int(train_cfg.get('NUM_WORKERS', 2)),
                                           int(train_cfg.get('PREFETCH_FACTOR', 2)))),
        validation_loader(valset),
    )
