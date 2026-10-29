import os
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
import json
import random
import subprocess
import warnings
from typing import *
import fnmatch
from datetime import timedelta

import torch
import torch.nn as nn
import torch.version
import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
from accelerate.utils import set_seed
import utils3d
import click
from tqdm import tqdm
torch.backends.cudnn.benchmark = False      # Varying input size, make sure cudnn benchmark is disabled

from moge.train.dataloader import TrainDataLoaderPipeline
from moge.utils.temporal import event_keys_in_mapping
from moge.train.losses import (
    affine_invariant_global_loss,
    affine_invariant_local_loss,
    mask_bce_loss,
    metric_scale_loss,
    monitoring,
)
from moge.train.utils import build_optimizer, build_lr_scheduler
from moge.train.checkpoints import load_checkpoint, save_checkpoint
from moge.utils.tools import key_average, flatten_nested_dict
from moge.config import load_train_config

def any_match(name: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)

def load_checkpoint_model_state(checkpoint_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint type for {checkpoint_path}: {type(checkpoint)}")
    if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        return checkpoint['model']
    if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
        return checkpoint['state_dict']
    return checkpoint

def get_configured_train_option(ctx: click.Context, train_config: Dict[str, Any], param_name: str, current_value: Any, config_key: Optional[str] = None) -> Any:
    config_key = config_key or param_name
    source = ctx.get_parameter_source(param_name)
    if source == click.core.ParameterSource.DEFAULT and config_key in train_config:
        return train_config[config_key]
    return current_value

@click.command()
@click.option('--config', 'config_path', type=click.Path(exists=True), required=True)
@click.option('--workspace', type=str, default='outputs/depth', help='Path to the workspace')
@click.option('--checkpoint', 'checkpoint_path', type=str, default=None, help='Path to the checkpoint to load. "latest" to load latest checkpoint in workspace')
@click.option('--batch_size_forward', type=int, default=8, help='Batch size for each forward pass on each device')
@click.option('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
@click.option('--num_load_workers', type=int, default=4, help='Number of dataloader instance loading workers')
@click.option('--num_process_workers', type=int, default=8, help='Number of dataloader instance processing workers')
@click.option('--dataloader_buffer_size', type=int, default=8, help='Number of prepared batches buffered by the dataloader')
@click.option('--enable_mixed_precision', type=bool, default=False, help='Use mixed precision training. Backbone is converted to FP16')
@click.option('--mixed_precision_dtype', type=click.Choice(['float16', 'bfloat16']), default='float16', help='Autocast dtype used by mixed precision training')
@click.option('--num_iterations', type=int, default=1000000, help='Number of iterations to train the model')
@click.option('--save_every', type=int, default=10000, help='Save checkpoint every n iterations')
@click.option('--log_every', type=int, default=1000, help='Log metrics every n iterations')
@click.option('--seed', type=int, default=0, help='Random seed')
def main(
    config_path: str,
    workspace: str,
    checkpoint_path: str,
    batch_size_forward: int,
    gradient_accumulation_steps: int,
    num_load_workers: int,
    num_process_workers: int,
    dataloader_buffer_size: int,
    enable_mixed_precision: bool,
    mixed_precision_dtype: str,
    num_iterations: int,
    save_every: int,
    log_every: int,
    seed: Optional[int],
):
    # Load config
    config = load_train_config(config_path)

    train_config = config.get('train', {})
    ctx = click.get_current_context()
    workspace = get_configured_train_option(ctx, train_config, 'workspace', workspace)
    checkpoint_path = get_configured_train_option(ctx, train_config, 'checkpoint_path', checkpoint_path, 'checkpoint') or None
    batch_size_forward = get_configured_train_option(ctx, train_config, 'batch_size_forward', batch_size_forward)
    gradient_accumulation_steps = get_configured_train_option(ctx, train_config, 'gradient_accumulation_steps', gradient_accumulation_steps)
    num_load_workers = get_configured_train_option(ctx, train_config, 'num_load_workers', num_load_workers)
    num_process_workers = get_configured_train_option(ctx, train_config, 'num_process_workers', num_process_workers)
    dataloader_buffer_size = get_configured_train_option(ctx, train_config, 'dataloader_buffer_size', dataloader_buffer_size)
    enable_mixed_precision = get_configured_train_option(ctx, train_config, 'enable_mixed_precision', enable_mixed_precision)
    mixed_precision_dtype = get_configured_train_option(ctx, train_config, 'mixed_precision_dtype', mixed_precision_dtype)
    num_iterations = get_configured_train_option(ctx, train_config, 'num_iterations', num_iterations)
    save_every = get_configured_train_option(ctx, train_config, 'save_every', save_every)
    log_every = get_configured_train_option(ctx, train_config, 'log_every', log_every)
    seed = get_configured_train_option(ctx, train_config, 'seed', seed)
    find_unused_parameters = bool(train_config.get('find_unused_parameters', False) or
                                  config['model'].get('anytime', {}).get('refine', {}).get('enabled', False))
    validation_config = config.get('validation', {})
    online_validation = bool(validation_config.get('enabled', False))
    validation_every = int(validation_config.get('every', save_every))
    if online_validation and validation_every <= 0:
        raise ValueError('validation.every must be positive')
    config_path_resolved = Path(config_path).resolve()

    if mixed_precision_dtype not in {'float16', 'bfloat16'}:
        raise ValueError(f'Unsupported mixed precision dtype: {mixed_precision_dtype}')

    autocast_dtype = torch.bfloat16 if mixed_precision_dtype == 'bfloat16' else torch.float16
    accelerator_precision = 'bf16' if mixed_precision_dtype == 'bfloat16' else 'fp16'

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=accelerator_precision if enable_mixed_precision else None,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters),
            InitProcessGroupKwargs(timeout=timedelta(hours=2)),
        ]
    )
    device = accelerator.device
    if enable_mixed_precision and autocast_dtype == torch.bfloat16 and device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f'CUDA device {torch.cuda.get_device_name(device)} does not support BF16')
    if accelerator.is_main_process:
        print(f'Mixed precision: enabled={enable_mixed_precision}, dtype={mixed_precision_dtype}')
    if enable_mixed_precision:
        training_precision_label = f'amp_{mixed_precision_dtype}'
    else:
        training_precision_label = 'float32'

    # Log config
    if accelerator.is_main_process:
        Path(workspace).mkdir(parents=True, exist_ok=True)
        with Path(workspace).joinpath('config.json').open('w') as f:
            json.dump(config, f, indent=4)

    # Set seed
    if seed is not None:
        set_seed(seed, device_specific=True)

    # Initialize model
    print('Initialize model')
    with accelerator.local_main_process_first():
        from moge.model import build_model
        model = build_model(config['model'])
        anytime = config['model'].get('anytime', {})
        completion = anytime.get('refine', {}).get('enabled', False)
        model_stage = 'completion' if completion else 'propagation' if anytime.get('enabled', False) else 'backbone'
        stage = config['train'].get('stage', model_stage)
        if stage != model_stage:
            raise ValueError('Training stage does not match the depth model configuration')
        freeze_patterns = [] if stage == 'backbone' else ['encoder.*']
        if completion:
            if anytime['memory_type'] != 'hrm':
                raise ValueError('Completion training requires an HRM propagation checkpoint')
            freeze_patterns += ['flow_net.*', 'softsplat_net.*', 'hrm.*',
                                'event_refine.event_encoder.*', 'event_refine.event_adapter.*']
        config['freeze'] = {'include': freeze_patterns}
        init_checkpoint_path = config.get('init_checkpoint', {}).get('path') if checkpoint_path is None else None
        initialized_from_config = False
        if init_checkpoint_path:
            print(f'Initialize from config checkpoint: {init_checkpoint_path}')
            init_state = load_checkpoint_model_state(init_checkpoint_path)
            scale_key = 'softsplat_net.netWarp.refine_scales'
            if (
                scale_key in init_state
                and scale_key in model.state_dict()
                and init_state[scale_key].shape != model.state_dict()[scale_key].shape
                and model.state_dict()[scale_key].numel() == 1
            ):
                init_state[scale_key] = init_state[scale_key][:1].clone()
                print('Migrate four-branch SoftSplat scale to fused-feature branch 0')
            required_frozen_keys = [
                name for name, _ in model.named_parameters()
                if any_match(name, freeze_patterns) and not name.startswith('event_refine.')
            ]
            missing_frozen_keys = [name for name in required_frozen_keys if name not in init_state]
            if missing_frozen_keys:
                raise ValueError(
                    'Initialization checkpoint does not contain every parameter that must be frozen: '
                    f'{missing_frozen_keys}'
                )
            msg = model.load_state_dict(init_state, strict=False)
            print(msg)
            initialized_from_config = True
        if freeze_patterns:
            parameters = dict(model.named_parameters())
            unmatched_patterns = [
                pattern for pattern in freeze_patterns
                if not any(fnmatch.fnmatch(name, pattern) for name in parameters)
            ]
            if unmatched_patterns:
                raise ValueError(f'Freeze patterns matched no parameters: {unmatched_patterns}')
            frozen_count = 0
            frozen_params = 0
            for name, param in model.named_parameters():
                if any_match(name, freeze_patterns):
                    param.requires_grad_(False)
                    frozen_count += 1
                    frozen_params += param.numel()
            print(f'Freeze parameters: {frozen_count} tensors / {frozen_params} scalars')
            for pattern in freeze_patterns:
                matched = [name for name in parameters if fnmatch.fnmatch(name, pattern)]
                matched_scalars = sum(parameters[name].numel() for name in matched)
                print(f'- Frozen {pattern}: {len(matched)} tensors / {matched_scalars} scalars')
    count_total_parameters = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {count_total_parameters}')

    # Initialize optimizer and scheduler.
    optimizer = build_optimizer(model, config['optimizer'])
    parameter_names = {id(param): name for name, param in model.named_parameters()}
    frozen_in_optimizer = [
        parameter_names.get(id(param), '<unknown>')
        for group in optimizer.param_groups
        for param in group['params']
        if not param.requires_grad
    ]
    if frozen_in_optimizer:
        raise RuntimeError(f'Frozen parameters were added to the optimizer: {frozen_in_optimizer}')
    lr_scheduler = build_lr_scheduler(optimizer, config['lr_scheduler'])

    count_grouped_parameters = [sum(p.numel() for p in param_group['params'] if p.requires_grad) for param_group in optimizer.param_groups]
    for i, count in enumerate(count_grouped_parameters):
        print(f'- Group {i}: {count} parameters')

    with accelerator.local_main_process_first():
        checkpoint = load_checkpoint(checkpoint_path, workspace)

    if checkpoint is None:
        # Initialize model weights
        if initialized_from_config:
            print('Skip default model weight initialization because init_checkpoint was loaded from config')
        else:
            print('Initialize model weights')
            with accelerator.local_main_process_first():
                model.init_weights()
        initial_step = 0
    else:
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'step' in checkpoint:
            initial_step = checkpoint['step'] + 1
        else:
            initial_step = 0
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        elif checkpoint.get('_model_only_resume', False) and initial_step > 0:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                optimizer.step()
                for _ in range(initial_step):
                    lr_scheduler.step()
            print(
                f'Model-only resume at step {initial_step}: initialized a fresh optimizer '
                'and advanced the LR scheduler to the matching step'
            )

        del checkpoint

    model, optimizer = accelerator.prepare(model, optimizer)
    if torch.version.hip and isinstance(model, torch.nn.parallel.DistributedDataParallel):
        # Hacking potential gradient synchronization issue in ROCm backend
        from moge.model.utils import sync_ddp_hook
        model.register_comm_hook(None, sync_ddp_hook)

    # Initialize training data pipeline
    with accelerator.local_main_process_first():
        train_data_pipe = TrainDataLoaderPipeline(
            config['data'],
            batch_size_forward,
            num_load_workers=num_load_workers,
            num_process_workers=num_process_workers,
            buffer_size=dataloader_buffer_size,
        )

    # Ready to train
    records = []
    model.train()
    unwrapped_model = accelerator.unwrap_model(model)

    def resolve_module(path: str) -> nn.Module:
        module = unwrapped_model
        for name in path.split('.'):
            module = getattr(module, name)
        return module

    frozen_module_paths = [
        pattern[:-2] for pattern in freeze_patterns if pattern.endswith('.*')
    ]
    for module_path in frozen_module_paths:
        resolve_module(module_path).eval()
    if frozen_module_paths:
        print(f'Frozen modules kept in eval mode: {frozen_module_paths}')

    gate_start_step = int(config['train'].get('gate_start_step', 5000))
    if completion:
        unwrapped_model.event_refine.train_gate(initial_step >= gate_start_step)
        print(f'Completion gate: enable after {gate_start_step} completed optimizer steps')

    def zero_all_gradients():
        optimizer.zero_grad()

    def describe_sample(batch: Dict[str, Any], index: int) -> str:
        info = batch.get('info')
        try:
            if isinstance(info, (list, tuple)):
                value = info[index]
            elif isinstance(info, dict):
                value = {
                    key: item[index] if isinstance(item, (list, tuple)) and len(item) > index else item
                    for key, item in info.items()
                }
            else:
                value = info
            return repr(value).replace('\n', ' ')[:1500]
        except Exception as exc:
            return f'<unable to format batch info: {exc!r}>'

    def distributed_sum(value: int) -> int:
        flag = torch.tensor(value, dtype=torch.int32, device=device)
        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.SUM)
        return int(flag.item())

    def distributed_any(value: bool) -> bool:
        return distributed_sum(int(value)) > 0

    with (
        train_data_pipe,
        tqdm(initial=initial_step, total=num_iterations, desc='Training', disable=not accelerator.is_main_process) as pbar,
    ):
        # Reset seed to avoid training on the same data when resuming training
        if seed is not None:
            set_seed(seed + initial_step, device_specific=True)

        loss_min_depth = float(
            config.get('depth_loss', {}).get('min_depth', 0.0)
        )
        loss_max_depth = float(
            config.get('depth_loss', {}).get(
                'max_depth', config['data']['clamp_max_depth']
            )
        )

        # Training loop
        for i_step in range(initial_step, num_iterations):

            if stage == 'completion' and i_step == gate_start_step:
                unwrapped_model.event_refine.train_gate(True)
                print(f'Enabled completion gate at update {i_step + 1}')

            i_accumulate = 0
            while i_accumulate < gradient_accumulation_steps:
                # Load batch
                batch = train_data_pipe.get()
                image, gt_depth, gt_mask_fin, gt_mask_inf, gt_intrinsics, label_type, is_metric = batch['image'], batch['depth'], batch['depth_mask_fin'], batch['depth_mask_inf'], batch['intrinsics'], batch['label_type'], batch['is_metric']
                image_dtype = torch.float32
                image = image.to(device=device, dtype=image_dtype)
                gt_depth, gt_mask_fin, gt_mask_inf, gt_intrinsics = gt_depth.to(device), gt_mask_fin.to(device), gt_mask_inf.to(device), gt_intrinsics.to(device)
                current_batch_size = image.shape[0]
                if distributed_any(all(label == 'invalid' for label in label_type)):
                    continue            # NOTE: Skip all-invalid batches to avoid messing up the optimizer.

                gt_points_mask = (
                    gt_mask_fin
                    & torch.isfinite(gt_depth)
                    & (gt_depth > loss_min_depth)
                    & (gt_depth <= loss_max_depth)
                )
                safe_gt_depth = torch.where(
                    gt_points_mask, gt_depth, torch.ones_like(gt_depth)
                )
                gt_points = utils3d.pt.depth_map_to_point_map(
                    safe_gt_depth, intrinsics=gt_intrinsics
                )
                gt_points = torch.where(
                    gt_points_mask[..., None], gt_points, torch.nan
                )
                gt_focal = 1 / (1 / gt_intrinsics[..., 0, 0] ** 2 + 1 / gt_intrinsics[..., 1, 1] ** 2) ** 0.5

                with accelerator.accumulate(model):
                    # Forward
                    if i_step <= config.get('low_resolution_training_steps', 0):
                        num_tokens = config['model']['num_tokens_range'][0]
                    else:
                        num_tokens = accelerate.utils.broadcast_object_list([random.randint(*config['model']['num_tokens_range'])])[0]
                    with torch.autocast(device_type=accelerator.device.type, dtype=autocast_dtype, enabled=enable_mixed_precision):
                        model_kwargs = {}
                        event_keys = event_keys_in_mapping(batch)
                        if event_keys:
                            model_kwargs['event_sequence'] = [
                                batch[key].to(device=device, dtype=image_dtype)
                                for key in event_keys
                            ]
                        output = model(image, num_tokens=num_tokens, **model_kwargs)
                    pred_points, pred_mask, pred_metric_scale = (output.get(k) for k in ['points', 'mask', 'metric_scale'])
                    finite_output_keys = ['points', 'mask', 'normal', 'metric_scale']
                    outputs_finite = torch.ones((), dtype=torch.bool, device=device)
                    for key in finite_output_keys:
                        value = output.get(key)
                        if value is not None and torch.is_floating_point(value):
                            outputs_finite &= torch.isfinite(value).all()
                    bad_output = not outputs_finite.item()
                    if distributed_any(bad_output):
                        if accelerator.is_main_process:
                            pbar.write(
                                f'Non-finite model output at step {i_step}; '
                                'reject update on all ranks'
                            )
                        zero_all_gradients()
                        continue

                    # Compute loss (per instance)
                    loss_list = []
                    record_step = i_step == initial_step or i_step % log_every == 0
                    valid_targets = gt_points_mask.flatten(1).any(dim=1)
                    valid_targets = valid_targets.detach().cpu().tolist()
                    for i in range(current_batch_size):
                        if not valid_targets[i]:
                            pbar.write(
                                f'Skip target without valid points at step {i_step}, '
                                f'rank {accelerator.process_index}: sample={describe_sample(batch, i)}'
                            )
                            continue
                        gt_metric_scale = None
                        loss_dict, weight_dict, misc_dict = {}, {}, {}
                        if record_step:
                            misc_dict['monitoring'] = monitoring(pred_points[i])
                        for k, v in config['loss'][label_type[i]].items():
                            weight_dict[k] = v['weight']
                            if v['function'] == 'affine_invariant_global_loss':
                                loss_dict[k], misc_dict[k], gt_metric_scale = affine_invariant_global_loss(
                                    pred_points[i], gt_points[i], gt_points_mask[i], **v['params']
                                )
                            elif v['function'] == 'affine_invariant_local_loss':
                                loss_dict[k], misc_dict[k] = affine_invariant_local_loss(
                                    pred_points[i], gt_points[i], gt_points_mask[i],
                                    gt_focal[i], gt_metric_scale, **v['params']
                                )
                            elif v['function'] == 'mask_bce_loss':
                                loss_dict[k], misc_dict[k] = mask_bce_loss(pred_mask[i], gt_mask_fin[i], gt_mask_inf[i])
                            elif v['function'] == 'metric_scale_loss':
                                if is_metric[i] and pred_metric_scale is not None:
                                    loss_dict[k], misc_dict[k] = metric_scale_loss(pred_metric_scale[i], gt_metric_scale)
                            else:
                                raise ValueError(f'Undefined loss function: {v["function"]}')
                        weight_dict = {'.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()}
                        loss_dict = {'.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()}
                        loss_ = sum([weight_dict[k] * loss_dict[k] for k in loss_dict], start=torch.tensor(0.0, device=device))
                        loss_list.append(loss_)

                        if record_step:
                            misc_dict = {'.'.join(k): v for k, v in flatten_nested_dict(misc_dict).items()}
                            records.append({
                                **{k: v.item() for k, v in loss_dict.items()},
                                **misc_dict,
                            })

                    valid_processes = distributed_sum(int(bool(loss_list)))
                    if valid_processes == 0:
                        if accelerator.is_main_process:
                            pbar.write(f'No valid samples on any rank at step {i_step}; fetch another batch')
                        zero_all_gradients()
                        continue
                    if loss_list:
                        loss = sum(loss_list) / len(loss_list)
                    else:
                        loss = torch.nan_to_num(pred_points).sum() * 0.0
                    if distributed_any(not torch.isfinite(loss).item()):
                        if accelerator.is_main_process:
                            pbar.write(f'Non-finite aggregate loss at step {i_step}; skip update on all ranks')
                        zero_all_gradients()
                        continue

                    # Backward & update
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        bad_gradient = (~torch.isfinite(grad_norm)).to(torch.int32)
                        if accelerator.num_processes > 1:
                            torch.distributed.all_reduce(
                                bad_gradient, op=torch.distributed.ReduceOp.MAX
                            )
                        if bad_gradient.item():
                            if accelerator.is_main_process:
                                pbar.write(f'Non-finite gradients at step {i_step}; skip update on all ranks')
                            zero_all_gradients()
                            continue

                    optimizer.step()
                    zero_all_gradients()

                i_accumulate += 1

            lr_scheduler.step()

            # Log metrics
            if i_step == initial_step or i_step % log_every == 0:
                records = [key_average(records)]
                records = accelerator.gather_for_metrics(records, use_gather_object=True)
                if accelerator.is_main_process:
                    records = key_average(records)
                    record = {'step': i_step + 1, **records}
                    with Path(workspace, 'train_metrics.jsonl').open('a') as stream:
                        stream.write(json.dumps(record) + '\n')
                    pbar.write(json.dumps(record))
                records = []

            completed_step = i_step + 1
            should_validate = online_validation and (
                completed_step % validation_every == 0 or completed_step == num_iterations
            )
            should_save = should_validate or completed_step == num_iterations or completed_step % save_every == 0
            if should_save:
                accelerator.wait_for_everyone()
                failure = torch.zeros(1, dtype=torch.int32, device=device)
                error = None
                if accelerator.is_main_process:
                    candidate = Path(workspace, f'.validation_candidate_{os.getpid()}.pt')
                    try:
                        payload = {
                            'model_config': config['model'],
                            'training_precision': training_precision_label,
                            'step': i_step,
                            'model': accelerator.unwrap_model(model).state_dict(),
                        }
                        save_checkpoint(Path(workspace, 'checkpoint', 'latest.pt'), {
                            **payload, 'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                        })
                        if should_validate:
                            save_checkpoint(candidate, payload)
                            subprocess.run([
                                sys.executable, '-m', 'moge.evaluation.checkpoints',
                                '--train-config', str(config_path_resolved),
                                '--device', str(device), '--num-devices', str(accelerator.num_processes),
                                '--output-workspace', str(Path(workspace).resolve()),
                                '--checkpoint', str(candidate.resolve()), '--step', str(completed_step),
                            ], cwd=Path(__file__).resolve().parents[2], check=True, timeout=3600)
                    except Exception as exc:
                        failure.fill_(1)
                        error = exc
                    finally:
                        candidate.unlink(missing_ok=True)
                if accelerator.num_processes > 1:
                    torch.distributed.broadcast(failure, src=0)
                if failure.item():
                    raise RuntimeError(f'Checkpoint/validation failed at step {completed_step}') from error

            pbar.set_postfix({'loss': loss.item()}, refresh=False)
            pbar.update(1)
