from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from semseg.datasets import EventDistillationDataset, get_distillation_dataset_class
from semseg.config import apply_overrides, load_config
from semseg.models.backbones import Encoder
from model.modules.egcm.seg.distillation import RGBEventDistiller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Encoder-B2 to Event MiT-B0 structure-aware distillation')
    parser.add_argument('--cfg', default='configs/event_distill.yaml')
    parser.add_argument('--resume', default='')
    parser.add_argument('--teacher-checkpoint', default='')
    parser.add_argument('--dataset-root', default='')
    parser.add_argument('--save-dir', default='')
    parser.add_argument('--set', dest='config_overrides', nargs='*', default=(), metavar='KEY=VALUE')
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size > 1:
        dist.init_process_group('nccl')
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device('cuda', local_rank)


def seed_everything(seed: int, rank: int) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_state(checkpoint: dict) -> dict:
    for key in ('model_state_dict', 'model', 'state_dict'):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    return checkpoint


def load_teacher(teacher: Encoder, checkpoint_path: str) -> None:
    source = extract_state(torch.load(checkpoint_path, map_location='cpu', weights_only=False))
    source = {key.removeprefix('module.'): value for key, value in source.items()}
    target = teacher.state_dict()
    compatible = {}
    for key, value in source.items():
        key = key.removeprefix('model.').removeprefix('backbone.')
        if key in target:
            compatible[key] = value
    teacher.load_state_dict(compatible, strict=True)


def load_mit_b0_pretrained(student: torch.nn.Module, checkpoint_path: str) -> None:
    source = extract_state(torch.load(checkpoint_path, map_location='cpu', weights_only=False))
    source = {key.removeprefix('module.').removeprefix('backbone.'): value for key, value in source.items()}
    target = student.state_dict()
    mapped = {}
    for source_key, value in source.items():
        match = re.match(r'patch_embed([1-4])\.(.+)', source_key)
        if match:
            target_key = f'patch_embeds.{int(match.group(1)) - 1}.{match.group(2)}'
        else:
            match = re.match(r'block([1-4])\.(.+)', source_key)
            if match:
                target_key = f'blocks.{int(match.group(1)) - 1}.{match.group(2)}'
            else:
                match = re.match(r'norm([1-4])\.(.+)', source_key)
                if not match:
                    continue
                target_key = f'norms.{int(match.group(1)) - 1}.{match.group(2)}'
        if (
            target_key == 'patch_embeds.0.proj.weight'
            and value.shape[1] == 3
            and target[target_key].shape[1] != 3
        ):
            event_channels = target[target_key].shape[1]
            value = value.mean(dim=1, keepdim=True).repeat(1, event_channels, 1, 1)
            value = value * (3.0 / event_channels)
        mapped[target_key] = value
    student.load_state_dict(mapped, strict=True)
    print(
        f'[student-init] loaded all {len(mapped)} Stage1/2/3/4 tensors from {checkpoint_path}; '
        f'event input channels={target["patch_embeds.0.proj.weight"].shape[1]}',
        flush=True,
    )


def build_dataset(cfg: dict, split: str) -> Dataset:
    dataset_cfg, train_cfg = cfg['DATASET'], cfg['TRAIN']
    dataset_class = get_distillation_dataset_class(dataset_cfg['ID'])
    return dataset_class(
        root=dataset_cfg['ROOT'],
        split=split,
        dataset_type=dataset_cfg.get('TYPE', 'dsec'),
        frame_index_stride=dataset_cfg['FRAME_INDEX_STRIDE'],
        image_size=tuple(train_cfg['IMAGE_SIZE']),
        event_scale=dataset_cfg.get('EVENT_SCALE', 255.0),
        horizontal_flip=split == 'train' and bool(train_cfg.get('HORIZONTAL_FLIP', False)),
        exclude_sequences=(
            dataset_cfg.get('TRAIN_EXCLUDE_SEQUENCES')
            if split == 'train'
            else dataset_cfg.get('VAL_EXCLUDE_SEQUENCES')
        ),
        include_semantic=False,
    )


def build_loader(
    dataset: EventDistillationDataset,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    training: bool,
    prefetch_factor: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=training) if world_size > 1 else None
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = max(1, int(prefetch_factor))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        drop_last=training,
        **loader_kwargs,
    )
    return loader, sampler


def reduce_statistics(statistics: Tensor, world_size: int) -> dict[str, float]:
    if world_size > 1:
        dist.all_reduce(statistics)
    count = statistics[-1].clamp_min(1)
    names = (
        'loss', 'loss_feature', 'loss_intra', 'loss_cross', 'mask_coverage',
        'loss_stage1', 'loss_stage2', 'loss_stage3', 'loss_stage4',
    )
    return {name: float((statistics[index] / count).item()) for index, name in enumerate(names)}


def run_epoch(
    model: torch.nn.Module | DDP,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    world_size: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: GradScaler | None = None,
    grad_clip: float = 0.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    statistics = torch.zeros(10, device=device)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            image = batch['image'].to(device, non_blocking=True)
            event = batch['event'].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=amp_enabled, dtype=amp_dtype):
                output = model(image, event)
            if training:
                scaler.scale(output.loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad), grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            batch_size = image.shape[0]
            values = torch.stack([
                output.loss.detach(),
                output.feature_loss.detach(),
                output.intra_loss.detach(),
                output.cross_loss.detach(),
                output.mask_coverage.detach(),
                *(stage_loss.detach() for stage_loss in output.stage_losses),
            ])
            statistics[:9] += values * batch_size
            statistics[9] += batch_size
    return reduce_statistics(statistics, world_size)


def student_checkpoint(
    model: torch.nn.Module | DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_loss: float,
    validation_metrics: dict[str, float],
) -> dict:
    model = model.module if isinstance(model, DDP) else model
    state = {
        'epoch': epoch,
        'best_val_loss': best_loss,
        'validation_metrics': validation_metrics,
        'event_encoder': model.student.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    state['student_projections'] = model.student_projections.state_dict()
    return state


def save_checkpoint_atomic(
    state: dict,
    checkpoint_path: Path,
    selection_metric: str,
    selection_value: float | None,
) -> None:
    payload = dict(state)
    payload['selection_metric'] = selection_metric
    payload['selection_value'] = selection_value
    temporary_path = checkpoint_path.with_name(
        f'.{checkpoint_path.name}.tmp.{os.getpid()}'
    )
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def restore_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpoint_path: str,
) -> tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.student.load_state_dict(checkpoint['event_encoder'])
    model.student_projections.load_state_dict(checkpoint['student_projections'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return int(checkpoint['epoch']), float(checkpoint.get('best_val_loss', math.inf))


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.cfg), args.config_overrides)
    if args.teacher_checkpoint:
        cfg['MODEL']['TEACHER_CHECKPOINT'] = args.teacher_checkpoint
    if args.dataset_root:
        cfg['DATASET']['ROOT'] = args.dataset_root
    if args.save_dir:
        cfg['SAVE_DIR'] = args.save_dir
    rank, local_rank, world_size, device = setup_distributed()
    seed_everything(int(cfg.get('SEED', 3407)), rank)
    primary = rank == 0

    train_dataset = build_dataset(cfg, 'train')
    val_dataset = build_dataset(cfg, 'val')
    if not train_dataset or not val_dataset:
        raise RuntimeError(f'No valid synchronized pairs: train={len(train_dataset)} val={len(val_dataset)}')
    train_cfg = cfg['TRAIN']
    train_loader, train_sampler = build_loader(
        train_dataset, train_cfg['BATCH_SIZE'], train_cfg['NUM_WORKERS'], rank, world_size, True,
        int(train_cfg.get('PREFETCH_FACTOR', 2)),
    )
    val_loader, _ = build_loader(
        val_dataset, cfg['EVAL']['BATCH_SIZE'], cfg['EVAL']['NUM_WORKERS'], rank, world_size, False,
        int(cfg['EVAL'].get('PREFETCH_FACTOR', 2)),
    )

    teacher_variant = str(cfg['MODEL'].get('TEACHER_BACKBONE', 'B2')).upper()
    teacher = Encoder(teacher_variant, ['img'], with_events=False)
    load_teacher(teacher, cfg['MODEL']['TEACHER_CHECKPOINT'])
    loss_cfg = cfg['LOSS']
    model = RGBEventDistiller(
        teacher=teacher,
        use_event_mask=bool(loss_cfg.get('USE_EVENT_MASK', False)),
        density_threshold=loss_cfg.get('MASK_DENSITY_THRESHOLD', 0.0),
        event_epsilon=loss_cfg.get('EVENT_EPSILON', 0.0),
        similarity_threshold=loss_cfg['SIMILARITY_THRESHOLD'],
        feature_weights=tuple(loss_cfg['FEATURE_WEIGHTS']),
        intra_weights=tuple(loss_cfg['INTRA_WEIGHTS']),
        cross_weights=tuple(loss_cfg['CROSS_WEIGHTS']),
    )
    load_mit_b0_pretrained(model.student, cfg['MODEL']['STUDENT_CHECKPOINT'])
    model.to(device)

    optim_cfg = cfg['OPTIMIZER']
    patch_parameters = [parameter for parameter in model.student.patch_embeds.parameters() if parameter.requires_grad]
    patch_parameter_ids = {id(parameter) for parameter in patch_parameters}
    remaining_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in patch_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {'params': patch_parameters, 'lr': optim_cfg['LR'] * optim_cfg.get('PATCH_EMBED_LR_MULT', 0.5)},
            {'params': remaining_parameters, 'lr': optim_cfg['LR']},
        ],
        weight_decay=optim_cfg['WEIGHT_DECAY'],
    )
    epochs = int(train_cfg.get('DISTILL_EPOCHS', train_cfg['EPOCHS']))
    total_steps = epochs * len(train_loader)
    warmup_steps = int(train_cfg.get('WARMUP_EPOCHS', 0)) * len(train_loader)

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1.0 - progress, 0.0) ** float(train_cfg.get('POLY_POWER', 0.9))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_dtype = torch.bfloat16 if str(train_cfg['AMP_DTYPE']).lower() in {'bf16', 'bfloat16'} else torch.float16
    amp_enabled = bool(train_cfg['AMP'])
    scaler = GradScaler('cuda', enabled=amp_enabled and amp_dtype == torch.float16)
    start_epoch = 0
    best_loss = math.inf
    resume_path = args.resume or str(cfg['MODEL'].get('DISTILL_RESUME', ''))
    if resume_path:
        start_epoch, best_loss = restore_training(model, optimizer, scheduler, resume_path)

    save_dir = Path(cfg['SAVE_DIR'])
    if primary:
        save_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / 'run_config.yaml').open('w', encoding='utf-8') as file:
            yaml.safe_dump(cfg, file, sort_keys=False, allow_unicode=True)
        print(
            f'[distill] world_size={world_size} per_gpu_batch={train_cfg["BATCH_SIZE"]} '
            f'train_pairs={len(train_dataset)} val_pairs={len(val_dataset)} '
            f'teacher=MiT-{teacher_variant} '
            'event20bins -> native20 '
            '-> target RGB endpoint stages=1,2,3,4 semantic_supervision=off',
            flush=True,
        )
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    for epoch in range(start_epoch, epochs):
        start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, device, amp_enabled, amp_dtype, world_size,
            optimizer, scheduler, scaler, float(train_cfg['GRAD_CLIP_NORM']),
        )
        val_metrics = run_epoch(
            model, val_loader, device, amp_enabled, amp_dtype, world_size,
        )
        improved = val_metrics['loss'] < best_loss
        best_loss = min(best_loss, val_metrics['loss'])
        if primary:
            record = {
                'epoch': epoch + 1,
                'lr': optimizer.param_groups[-1]['lr'],
                'train': train_metrics,
                'val': val_metrics,
                'best_val_loss': best_loss,
                'seconds': time.time() - start,
            }
            with open(save_dir / 'metrics.jsonl', 'a', encoding='utf-8') as file:
                file.write(json.dumps(record) + '\n')
            state = student_checkpoint(
                model, optimizer, scheduler, epoch + 1, best_loss, val_metrics
            )
            save_checkpoint_atomic(
                state, save_dir / 'last.pth', 'last_epoch', float(epoch + 1)
            )
            if improved:
                save_checkpoint_atomic(
                    state, save_dir / 'best.pth',
                    'val_four_stage_kd_loss', val_metrics['loss'],
                )
            print(json.dumps(record), flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
