"""Shared setup for MTMamba/LiFR training and formal validation."""
import argparse
import datetime
import json
import os
import pprint
import sys
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter

from utils.common_config import (
    get_criterion, get_model, get_optimizer, get_test_dataloaders,
    get_train_dataloader, get_train_dataset, get_transformations,
)
from utils.config import create_config
from utils.logger import Logger
from utils.test_utils import test_phase
from utils.train_utils import train_phase


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _match_module_prefix(state_dict, model):
    """Match checkpoints saved with or without DistributedDataParallel."""
    model_uses_module = any(key.startswith('module.') for key in model.state_dict())
    checkpoint_uses_module = any(key.startswith('module.') for key in state_dict)
    if checkpoint_uses_module and not model_uses_module:
        return {key.removeprefix('module.'): value for key, value in state_dict.items()}
    if model_uses_module and not checkpoint_uses_module:
        return {f'module.{key}': value for key, value in state_dict.items()}
    return state_dict


def run(default_mode):
    parser = argparse.ArgumentParser(description='LiFR multi-task training and validation')
    parser.add_argument('--config_exp', required=True, help='Experiment configuration')
    parser.add_argument('--local-rank', '--local_rank', dest='local_rank', default=0, type=int)
    parser.add_argument('--run_mode', choices=['train', 'infer'], default=default_mode)
    parser.add_argument('--trained_model', help='Model or resumable training checkpoint')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    rank = int(os.environ.get('RANK', '0'))
    training = args.run_mode == 'train'
    primary = rank == 0
    distributed = training and int(os.environ.get('WORLD_SIZE', '1')) > 1
    # Validation uses the full split on rank zero, including separate DSEC task loaders.
    if not training and not primary:
        return

    torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True
    cv2.setNumThreads(0)
    start_time = time.time()
    set_seed(args.seed)
    if distributed:
        dist.init_process_group('nccl', init_method='env://', timeout=datetime.timedelta(hours=2))

    p = create_config(args.config_exp, {
        'run_mode': args.run_mode,
        'use_ddp': distributed,
    })
    precision = str(p.get('precision', 'fp32')).lower()
    if precision not in {'fp32', 'bf16'}:
        raise ValueError(f'precision must be fp32 or bf16, got {precision}')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('The selected CUDA device does not support BF16')
    if primary:
        sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))
        pprint.pprint(p)
        print(f'Precision: {precision}')

    model = get_model(p).cuda()
    if primary:
        print(f'Model Param Size: {sum(param.numel() for param in model.parameters())}')
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True,
        )
    train_transforms, val_transforms = get_transformations(p)
    test_dataloader = get_test_dataloaders(p, val_transforms)
    if training:
        train_dataset = get_train_dataset(p, train_transforms)
        train_sampler = (
            torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
            if distributed else RandomSampler(train_dataset)
        )
        train_dataloader = get_train_dataloader(p, train_dataset, train_sampler)
        criterion = get_criterion(p).cuda()
        scheduler, optimizer = get_optimizer(p, model, iters_per_epoch=len(train_dataloader))

    checkpoint_path = args.trained_model or p['checkpoint']
    start_epoch, iter_count, best_imp = 0, 0, -100
    if args.trained_model or not training or os.path.exists(checkpoint_path):
        if primary:
            print(f'Use checkpoint {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(_match_module_prefix(checkpoint['model'], model))
        iter_count = checkpoint.get('iter_count', 0)
        if training:
            if 'optimizer' in checkpoint:
                from model.modules.memory_utils import remap_memory_optimizer_state
                optimizer.load_state_dict(remap_memory_optimizer_state(
                    checkpoint['optimizer'], optimizer, model,
                ))
            if scheduler is not None and 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_imp = checkpoint.get('best_imp', best_imp)
            # Older model-only checkpoints store their training position in best.txt.
            best_txt = p['checkpoint'].replace('checkpoint.pth.tar', 'best.txt')
            if 'best_imp' not in checkpoint and args.trained_model is None and os.path.exists(best_txt):
                with open(best_txt) as file:
                    metadata = json.load(file)
                best_imp = metadata.get('best_imp', best_imp)
                if 'epoch' not in checkpoint:
                    start_epoch = metadata.get('epoch', -1) + 1
                if 'iter_count' not in checkpoint:
                    iter_count = start_epoch * len(train_dataloader)
    elif primary:
        print('Fresh start...')

    writer = None
    try:
        if training:
            if primary:
                writer = SummaryWriter(os.path.join(p.root_dir, 'tb_dir', 'train'))
            for epoch in range(start_epoch, p['epochs']):
                if distributed:
                    train_sampler.set_epoch(epoch)
                if primary:
                    print(f'Epoch {epoch + 1}/{p["epochs"]}')
                end_signal, iter_count, best_imp = train_phase(
                    p, args, train_dataloader, test_dataloader, model, criterion,
                    optimizer, scheduler, epoch, writer, iter_count, best_imp,
                )
                if end_signal:
                    break
        if primary:
            eval_model = model.module if hasattr(model, 'module') else model
            results = test_phase(
                p, test_dataloader, eval_model, iter_count,
                save_edge='edge' in p.TASKS.NAMES,
            )
            print('Validation results:')
            print(results)
            print(f'Total running time: {(time.time() - start_time) / 3600} h.')
    finally:
        if writer is not None:
            writer.close()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
