"""Train the CNN event encoder against frozen endpoint-RGB MoGe features."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

if (_package_root := str(Path(__file__).absolute().parents[1])) not in sys.path:
    sys.path.insert(0, _package_root)
_project_root = Path(_package_root)
if str(_project_root.parents[1]) not in sys.path:
    sys.path.insert(0, str(_project_root.parents[1]))

import click
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm

from moge.config import load_distill_config
from moge.model.v2 import MoGeModel
from model.modules.egcm.depth.distillation import MetaDepthCNNDistiller
from moge.train.dataloader import TrainDataLoaderPipeline


def _load_checkpoint(path: str) -> tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = _project_root / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or 'model_config' not in checkpoint:
        raise KeyError(f'Teacher checkpoint must contain model_config: {path}')
    state = checkpoint.get('model', checkpoint.get('state_dict'))
    if not isinstance(state, dict):
        raise KeyError(f'Teacher checkpoint must contain model weights: {path}')
    return (dict(checkpoint['model_config']), state)


def _build_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    paths, data = config['paths'], config['data']
    dataset = {
        'name': f"{config['dataset']}_event_distill",
        'format': 'folder',
        'path': paths['dataset_root'],
        'index': paths.get('train_index', '.train.txt'),
        'label_type': 'C',
        'weight': 1.0,
        'depth': 'depth.png',
        'depth_unit': 1.0,
        'finite_depth_mask': 'only_known',
        'image_augmentation': [],
        'center_augmentation': 0.0,
        'fov_range_relative': [1.0, 1.0],
        'requires_events': True,
        'event_bins': int(data.get('event_bins', 20)),
        'event_root': paths['event_root'],
        'event_layout': data.get('event_layout', 'continuous'),
        'duration_ms': int(data['frame_interval_ms']),
        'frame_step_ms': int(data['frame_interval_ms']),
        'event_keys': ['event'],
        'load_target_image': True,
    }
    use_event_mask = bool(config.get('loss', {}).get('use_event_mask', False))
    if use_event_mask and 'raw_event_h5_root' in paths:
        dataset['raw_event_h5_root'] = paths['raw_event_h5_root']
        dataset['load_raw_event_activity'] = True
    return {
        'image_sizes': [[int(data['width']), int(data['height'])]],
        'clamp_max_depth': float(data.get('clamp_max_depth', 1000.0)),
        'center_augmentation': 0.0,
        'fov_range_absolute': [1, 179],
        'fov_range_relative': [1.0, 1.0],
        'image_augmentation': [],
        'datasets': [dataset],
    }


def _save_checkpoint(path: Path, model: torch.nn.Module, step: int,
                     config: Dict[str, Any], optimizer: torch.optim.Optimizer,
                     scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'step': step,
        'event_encoder_type': model.event_encoder_type,
        'event_encoder': model.event_encoder.state_dict(),
        'event_encoder_config': {
            'arch': 'large',
            'event_bins': model.event_bins,
            'teacher_dim': model.teacher_dim,
        },
        'student_projection': model.student_projection.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'config': config,
    }
    torch.save(checkpoint, path)


def _write_aggregate_history(path: Path, records: list[tuple[int, ...]]) -> None:
    best_step = min(records, key=lambda item: (item[1], item[0]))[0]
    lines = ['step\ttotal\tl1\tintra\tcross\tmask_coverage\tis_best']
    lines.extend(
        '\t'.join([str(step), *(f'{value:.9g}' for value in values), str(int(step == best_step))])
        for step, *values in records
    )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


@click.command()
@click.option('--config', 'config_path', default='configs/train/distill.json', show_default=True)
@click.option('--teacher-checkpoint', type=str, default=None,
              help='Override paths.teacher_checkpoint from the JSON config.')
@click.option('--workspace', type=str, default=None, help='Override paths.workspace from the JSON config.')
def main(config_path: str, teacher_checkpoint: str | None, workspace: str | None) -> None:
    config_file = Path(config_path)
    if not config_file.is_file() and (not config_file.is_absolute()):
        config_file = _project_root / config_file
    config = load_distill_config(config_file)
    if teacher_checkpoint is not None:
        config['paths']['teacher_checkpoint'] = teacher_checkpoint
    if workspace is not None:
        config['paths']['workspace'] = workspace
    train = config['train']
    mixed_precision = str(
        train.get('precision', 'fp16' if bool(train.get('amp', True)) else 'no')
    ).lower()
    if mixed_precision not in {'no', 'fp16', 'bf16'}:
        raise ValueError('train.precision must be one of: no, fp16, bf16')
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    torch.manual_seed(int(train.get('seed', 0)) + accelerator.process_index)
    teacher_config, teacher_state = _load_checkpoint(config['paths']['teacher_checkpoint'])
    teacher_config.pop('temporal', None)
    teacher_config['anytime'] = {'enabled': False}
    teacher = MoGeModel(**teacher_config)
    teacher.load_state_dict(teacher_state, strict=False)
    model_options = config.get('model', {})
    if model_options.get('event_encoder', 'metadepth-cnn') != 'metadepth-cnn':
        raise ValueError('Depth distillation uses the MetaDepth CNN event encoder')
    if model_options.get('distiller', 'metadepth_cnn_aggregate') != 'metadepth_cnn_aggregate':
        raise ValueError('Depth distillation uses metadepth_cnn_aggregate')
    for head_name in ('points_head', 'normal_head', 'mask_head', 'scale_head'):
        if hasattr(teacher, head_name):
            delattr(teacher, head_name)
    loss_config = config.get('loss', {})
    use_event_mask = bool(loss_config.get('use_event_mask', False))
    model = MetaDepthCNNDistiller(
        teacher=teacher,
        event_bins=int(config['data'].get('event_bins', 20)),
        teacher_dim=int(teacher_config['encoder']['dim_out']),
        event_tau=float(loss_config.get('event_tau', 64.0)),
        intra_weight=float(loss_config.get('intra_weight', 10.0)),
        cross_weight=float(loss_config.get('cross_weight', 4.0)),
        similarity_threshold=float(loss_config.get('similarity_threshold', 0.1)),
        use_event_mask=use_event_mask,
    )
    optimizer_config = config['optimizer']
    learning_rate = float(optimizer_config['lr'])
    weight_decay = float(optimizer_config.get('weight_decay', 0.0001))
    parameter_groups = [{
        'params': [parameter for parameter in model.parameters() if parameter.requires_grad],
        'lr': learning_rate,
    }]
    optimizer = torch.optim.AdamW(parameter_groups, lr=learning_rate, weight_decay=weight_decay)
    total_steps = int(train['num_iterations'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: max(0.0, 1.0 - step / max(total_steps, 1)) ** float(optimizer_config.get('power', 0.9)),
    )
    start_step = 0
    resume_path = config['paths'].get('resume_checkpoint')
    if resume_path:
        resume = torch.load(resume_path, map_location='cpu', weights_only=True)
        if resume.get('event_encoder_type', 'metadepth-cnn') != model.event_encoder_type:
            raise ValueError('Resume checkpoint must contain a CNN event encoder')
        model.event_encoder.load_state_dict(resume['event_encoder'], strict=True)
        model.student_projection.load_state_dict(resume['student_projection'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        start_step = int(resume['step']) + 1
    model, optimizer = accelerator.prepare(model, optimizer)
    data_pipe = TrainDataLoaderPipeline(
        _build_data_config(config),
        max(4, int(train.get('batch_size_forward', 4))),
        num_load_workers=int(train.get('num_load_workers', 4)),
        num_process_workers=int(train.get('num_process_workers', 4)),
        buffer_size=int(train.get('dataloader_buffer_size', 4)),
    )
    workspace = Path(config['paths'].get('workspace', 'output_distill/event_distill'))
    if not workspace.is_absolute():
        workspace = _project_root / workspace
    if accelerator.is_main_process:
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    log_every = int(train.get('log_every', 20))
    validate_every = int(train.get('validate_every', train.get('save_every', 2000)))
    if validate_every <= 0:
        raise ValueError('train.validate_every must be positive')
    token_range = config['data'].get('num_tokens_range', [1200, 1200])
    best_loss = float('inf')
    running_loss = 0.0
    interval_loss = 0.0
    interval_coverage = 0.0
    aggregate_names = ('loss_l1', 'loss_intra', 'loss_cross')
    running_aggregate = [0.0] * len(aggregate_names)
    interval_aggregate = [0.0] * len(aggregate_names)
    interval_count = 0
    loss_history: list[tuple] = []
    model.train()
    with data_pipe, tqdm(range(start_step, total_steps), disable=not accelerator.is_main_process,
                         desc='Event distillation') as progress:
        for step in progress:
            batch = data_pipe.get()
            target_image = batch['target_image'].to(accelerator.device)
            event = batch['event'].to(accelerator.device)
            num_tokens = random.randint(int(token_range[0]), int(token_range[1]))
            with accelerator.autocast():
                activity = batch['event_activity'].to(accelerator.device) if use_event_mask else None
                output = model(target_image, event, activity, num_tokens)
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(output.loss)
            accelerator.clip_grad_norm_(model.parameters(), float(train.get('grad_clip_norm', 1.0)))
            optimizer.step()
            scheduler.step()
            reduced_loss = accelerator.reduce(output.loss.detach(), reduction='mean').item()
            reduced_coverage = accelerator.reduce(output.mask_coverage.detach(), reduction='mean').item()
            reduced_aggregate = [
                accelerator.reduce(getattr(output, name).detach(), reduction='mean').item()
                for name in aggregate_names
            ]
            running_aggregate = [
                total + value for total, value in zip(running_aggregate, reduced_aggregate)
            ]
            interval_aggregate = [
                total + value for total, value in zip(interval_aggregate, reduced_aggregate)
            ]
            running_loss += reduced_loss
            interval_loss += reduced_loss
            interval_coverage += reduced_coverage
            interval_count += 1
            if (step + 1) % log_every == 0:
                mean_loss = running_loss / log_every
                running_loss = 0.0
                if accelerator.is_main_process:
                    mean_aggregate = [value / log_every for value in running_aggregate]
                    progress.set_postfix(
                        total=f'{mean_loss:.4f}', l1=f'{mean_aggregate[0]:.4f}',
                        intra=f'{mean_aggregate[1]:.4f}', cross=f'{mean_aggregate[2]:.4f}',
                        mask=f'{reduced_coverage:.3f}',
                    )
                running_aggregate = [0.0] * len(aggregate_names)
            completed_step = step + 1
            if completed_step % validate_every == 0 or completed_step == total_steps:
                completed_count = interval_count
                mean_interval_loss = interval_loss / completed_count
                mean_interval_coverage = interval_coverage / completed_count
                interval_loss = 0.0
                interval_coverage = 0.0
                interval_count = 0
                if accelerator.is_main_process:
                    mean_aggregate = [value / completed_count for value in interval_aggregate]
                    loss_history.append((
                        completed_step, mean_interval_loss, *mean_aggregate, mean_interval_coverage,
                    ))
                    _write_aggregate_history(workspace / 'distillation_metrics.txt', loss_history)
                    if mean_interval_loss < best_loss:
                        best_loss = mean_interval_loss
                        _save_checkpoint(
                            workspace / 'best.pt', accelerator.unwrap_model(model),
                            step, config, optimizer, scheduler,
                        )
                interval_aggregate = [0.0] * len(aggregate_names)


if __name__ == '__main__':
    main()
