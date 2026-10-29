"""Assemble the fixed semantic model and coordinate training and validation."""
from functools import partial
import torch
from torch import distributed as dist
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from semseg.losses import final_cross_entropy, get_loss
from semseg.optimizers import get_backbone_optimizer, get_mask_refine_optimizer, get_optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from semseg.models import build_model, prepare_model_config
from semseg.schedulers import get_scheduler
from .checkpoints import initialize_model, restore_training_state, TrainingRecord, _unwrap_model
from .data import build_loaders
from .evaluation import evaluate
from model.modules.egcm.schedule import gate_alpha


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def validate_training_config(cfg, duration):
    model, train = cfg['MODEL'], cfg.get('TRAIN', {})
    stage = train.get('STAGE', 'backbone' if model['NAME'] == 'RGBBackbone' else 'propagation')
    if (stage == 'backbone' and duration != 0) or (stage != 'backbone' and duration <= 0):
        raise ValueError('Use duration=0 for RGB backbone training and a positive horizon for anytime training')
    if int(train.get('LOG_INTERVAL', 20)) <= 0 or float(train.get('GRAD_CLIP_NORM', 10.0)) <= 0:
        raise ValueError('Log interval and gradient clipping norm must be positive')
    if int(train.get('MAX_CONSECUTIVE_NONFINITE', 3)) <= 0:
        raise ValueError('MAX_CONSECUTIVE_NONFINITE must be positive')


def configure_training(model, cfg, *, epoch=None):
    """Set the stage recipe with gates trainable throughout completion."""
    core = _unwrap_model(model)
    stage = cfg['TRAIN']['STAGE']
    core.train()
    core.requires_grad_(True)
    if stage == 'backbone':
        return
    core.backbone.requires_grad_(False)
    core.backbone.eval()
    if stage == 'propagation':
        if core.version == 'v2':
            core.mask_refiner.requires_grad_(False)
            core.mask_refiner.eval()
        return

    core.requires_grad_(False)
    core.eval()
    core.decode_head.requires_grad_(True)
    core.decode_head.train()
    refiner = core.mask_refiner
    refiner.fusions.requires_grad_(True)
    refiner.fusions.train()
    refiner.train_gate(True)
    options = cfg['TRAIN']
    refiner.update_gate_alpha(
        (epoch or 0) / max(int(options['EPOCHS']) - 1, 1),
        options.get('GATE_SCHEDULE', 'fixed'), options.get('GATE_ALPHA', 0.01),
    )


def build_optimizer(model, cfg, resume=None):
    options = cfg['OPTIMIZER']
    lr = float(options['LR'])
    legacy_layout = bool(isinstance(resume, dict) and 'optimizer_state_dict' in resume
                         and not resume.get('optimizer_excludes_frozen', False)
                         and model.version == 'v1')
    if cfg['TRAIN']['STAGE'] == 'backbone':
        optimizer = get_backbone_optimizer(model, options['NAME'], lr,
                                          float(options.get('DECODER_LR', lr)), options['WEIGHT_DECAY'])
    elif cfg['TRAIN']['STAGE'] == 'completion':
        optimizer = get_mask_refine_optimizer(
            model, options['NAME'], refine_lr=lr,
            event_encoder_lr=float(options.get('EVENT_ENCODER_LR', lr * .1)),
            decoder_lr=float(options.get('DECODER_LR', lr * .25)),
            gate_lr=float(options.get('GATE_LR', lr)),
            weight_decay=options['WEIGHT_DECAY'], event_encoder=model.mask_refiner.event_encoder.encoder,
        )
    else:
        optimizer = get_optimizer(model, options['NAME'], lr, options['WEIGHT_DECAY'], include_frozen=legacy_layout)
    return optimizer, legacy_layout


DSEC_CLASS_WEIGHTS = (0.8977, 0.8722, 0.9863, 0.9992, 0.9899, 0.7580,
                      0.9377, 0.7867, 0.9666, 0.9805, 0.9968)


class TrainingObjective:
    def __init__(self, cfg, dataset, device):
        if cfg['TRAIN']['STAGE'] == 'completion':
            self.semantic = partial(final_cross_entropy, ignore_index=dataset.ignore_label)
        else:
            weights = torch.tensor(DSEC_CLASS_WEIGHTS, device=device) if cfg['DATASET']['TYPE'] == 'dsec' else None
            self.semantic = get_loss(cfg['LOSS'].get('NAME', 'CrossEntropy'), dataset.ignore_label, weights)

    def __call__(self, model, sample, labels):
        if len(labels) != 1:
            raise RuntimeError('Training requires one GT at t0 + duration')
        core = _unwrap_model(model)
        if core.version == 'v2':
            logits, completion = model(sample, return_completion_aux=True)
        else:
            logits, completion = model(sample), None
        loss = self.semantic(logits[-1], labels[-1])
        metrics = {'loss_final': loss.detach()}
        if completion is not None:
            metrics['gate_mean'] = torch.stack([gate.detach().mean() for gate in completion['gates']]).mean()
        return loss, metrics


def train_one_epoch(model, loader, optimizer, scheduler, scaler, objective, cfg,
                    device, epoch, amp_dtype, writer, *, primary=True):
    train = cfg['TRAIN']
    configure_training(model, cfg, epoch=epoch)
    if isinstance(loader.sampler, torch.utils.data.DistributedSampler):
        loader.sampler.set_epoch(epoch)
    sums = {}
    successful_steps = 0
    nonfinite_steps = 0
    iterator = tqdm(loader, disable=not primary, desc=f'Epoch {epoch + 1}/{train["EPOCHS"]}')
    for index, (_, _, sample, labels) in enumerate(iterator):
        sample = [x.to(device, non_blocking=True) for x in sample]
        labels = [x.to(device, non_blocking=True) for x in labels]
        optimizer.zero_grad(set_to_none=True)
        with autocast(device.type, enabled=bool(train.get('AMP', False)), dtype=amp_dtype):
            loss, metrics = objective(model, sample, labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'Non-finite loss at epoch={epoch + 1}, batch={index + 1}')
        scaler.scale(loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train.get('GRAD_CLIP_NORM', 10.0)))
        if not bool(torch.isfinite(norm)):
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                nonfinite_steps += 1
                optimizer.zero_grad(set_to_none=True)
                if nonfinite_steps >= int(train.get('MAX_CONSECUTIVE_NONFINITE', 3)):
                    raise FloatingPointError('Repeated non-finite gradients after backward')
            if primary:
                iterator.write(f'Skipped non-finite gradient update at epoch={epoch + 1}, batch={index + 1}')
            continue
        nonfinite_steps = 0
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        successful_steps += 1
        metrics['loss'] = loss.detach()
        for name, value in metrics.items():
            sums[name] = sums.get(name, torch.zeros((), device=device)) + value
        if primary and ((index + 1) % int(train.get('LOG_INTERVAL', 20)) == 0 or index + 1 == len(loader)):
            iterator.set_postfix(loss=f'{(sums["loss"] / successful_steps).item():.4f}',
                                 lr=f'{optimizer.param_groups[0]["lr"]:.3g}')
    iterator.close()
    if not successful_steps:
        raise FloatingPointError('No valid optimizer updates in this epoch')
    for name, value in sums.items():
        # DDP reports the mean over actual successful updates on all ranks.
        count = value.new_tensor(successful_steps)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value)
            dist.all_reduce(count)
        sums[name] = (value / count).item()
        if primary:
            writer.add_scalar('train/' + name, sums[name], epoch)
    return sums['loss']


def train(cfg, gpu, save_dir, duration, logger):
    cfg = prepare_model_config(cfg)
    validate_training_config(cfg, duration)
    options = cfg['TRAIN']
    distributed = bool(options.get('DDP', False))
    primary = not distributed or dist.get_rank() == 0
    device = torch.device(cfg['DEVICE'])
    dtype_name = str(options.get('AMP_DTYPE', 'float16')).lower()
    dtypes = {'float16': torch.float16, 'fp16': torch.float16,
              'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16}
    if dtype_name not in dtypes:
        raise ValueError(f'Unsupported AMP dtype: {dtype_name!r}')
    dtype = dtypes[dtype_name]
    loaders = build_loaders(cfg, duration)
    objective = TrainingObjective(cfg, loaders.train.dataset, device)
    initialization = cfg['MODEL']
    complete_weights = initialization.get('RESUME') or (
        initialization.get('INIT_CHECKPOINT') and initialization.get('INIT_CHECKPOINT_STRICT', True))
    model = build_model(cfg, loaders.train.dataset.n_classes, duration_ms=duration,
                        event_interval_ms=loaders.train.dataset.frame_interval_ms,
                        load_event_weights=not complete_weights)
    checkpoint = initialize_model(model, cfg, logger)
    model.to(device)
    configure_training(model, cfg)
    optimizer, legacy_layout = build_optimizer(model, cfg, checkpoint)
    if distributed:
        model = DDP(model, device_ids=[gpu], output_device=gpu,
                    find_unused_parameters=bool(options.get('FIND_UNUSED_PARAMETERS', False)))
    schedule = cfg['SCHEDULER']
    epochs, iterations = int(options['EPOCHS']), len(loaders.train)
    scheduler = get_scheduler(schedule['NAME'], optimizer, (epochs + 1) * iterations,
                              schedule['POWER'], iterations * schedule['WARMUP'], schedule['WARMUP_RATIO'])
    start_epoch = restore_training_state(checkpoint, optimizer, scheduler, cfg, iterations, model)
    run_epochs = int(options.get('RUN_EPOCHS', epochs - start_epoch))
    if run_epochs <= 0:
        raise ValueError('No training epochs remain; check TRAIN.EPOCHS/RUN_EPOCHS and the resume checkpoint')
    writer = SummaryWriter(str(save_dir)) if primary and not options.get('COMPACT_OUTPUT', False) else NullWriter()
    record = TrainingRecord(cfg, save_dir, logger, writer, checkpoint)
    scaler = GradScaler(enabled=bool(options.get('AMP', False)) and dtype == torch.float16 and device.type == 'cuda')
    if primary:
        logger.info(f'Model={cfg["MODEL"]["NAME"]} trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    try:
        for epoch in range(start_epoch, min(epochs, start_epoch + run_epochs)):
            if primary and options['STAGE'] == 'completion':
                alpha = gate_alpha(epoch / max(epochs - 1, 1), options.get('GATE_SCHEDULE', 'fixed'),
                                   options.get('GATE_ALPHA', 0.01))
                logger.info(f'Completion epoch={epoch + 1} gate_alpha={alpha:.6f}')
            loss = train_one_epoch(model, loaders.train, optimizer, scheduler, scaler,
                                   objective, cfg, device, epoch, dtype, writer, primary=primary)
            should_evaluate = (((epoch + 1) % int(options['EVAL_INTERVAL']) == 0
                                and epoch + 1 > int(options.get('EVAL_START', 0))) or epoch + 1 == epochs)
            if should_evaluate:
                core = model.module if distributed else model
                result = evaluate(core, loaders.val, device)
                if primary:
                    record.validation(result, model, epoch)
            if primary:
                record.save_last(model, optimizer, scheduler, epoch, loss, legacy_layout)
            if distributed:
                dist.barrier()
        if primary:
            logger.info(f'Best {record.metric_name}={record.best_miou:.2f} at epoch {record.best_epoch}')
    finally:
        writer.close()
