"""Weight initialization, training-state restoration and best/last checkpoint saving."""
import os
from pathlib import Path
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from model.modules.memory_utils import remap_memory_state, remap_memory_optimizer_state


def _unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def _save_checkpoint_atomic(payload: dict, checkpoint_path: Path) -> None:
    """Replace the checkpoint only after a complete write in the same directory."""
    temporary_path = checkpoint_path.with_name(f'{checkpoint_path.name}.tmp')
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _extract_model_state(checkpoint, keys=('model_state_dict', 'model', 'state_dict')):
    if isinstance(checkpoint, dict):
        for key in keys:
            if isinstance(checkpoint.get(key), dict):
                return remap_memory_state(checkpoint[key], 'hrm_memory')
    return remap_memory_state(checkpoint, 'hrm_memory')


def _strip_module_prefix(state_dict):
    return remap_memory_state({key.removeprefix('module.'): value for key, value in state_dict.items()}, 'hrm_memory')


def _read_checkpoint(path, setting):
    if not Path(path).is_file():
        raise FileNotFoundError(f'{setting} not found: {path!r}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = _extract_model_state(checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f'{setting} must contain a model state dict: {path!r}')
    return checkpoint, _strip_module_prefix(state)


def _compatible_state(state, target):
    return {key: value for key, value in state.items()
            if torch.is_tensor(value) and key in target and (
                value.shape == target[key].shape or (
                    key.endswith('.temporal_position')
                    and value.shape[1:] == target[key].shape[1:]
                    and value.shape[0] >= target[key].shape[0]
                )
            )}


def _load_mask_refine_base_checkpoint(model, checkpoint_path):
    """Load HRM propagation weights, leaving only EGCM uninitialized."""
    _, source = _read_checkpoint(checkpoint_path, 'MODEL.BASE_CHECKPOINT')
    compatible = _compatible_state(
        {key: value for key, value in source.items() if not key.startswith('mask_refiner.')},
        model.state_dict(),
    )
    for prefix in ('backbone.', 'flow_net.', 'softsplat_net.', 'decode_head.', 'hrm_memory.'):
        if not any(key.startswith(prefix) for key in compatible):
            raise RuntimeError(f'Base checkpoint has no compatible {prefix} weights: {checkpoint_path}')
    message = model.load_state_dict(compatible, strict=False)
    missing = [key for key in message.missing_keys if not key.startswith('mask_refiner.')]
    if missing:
        raise RuntimeError(f'Base checkpoint is missing propagation weights: {missing[:32]}')
    return message, len(compatible), len(source)


def _load_flow_weights(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    source = _extract_model_state(checkpoint, keys=('model', 'state_dict', 'model_state_dict'))
    if not isinstance(source, dict):
        raise TypeError(f'Unsupported flow checkpoint: {path!r}')
    source = {key.removeprefix('module.').removeprefix('flow_net.'): value for key, value in source.items()}
    target = model.flow_net.state_dict()
    for prefix in ('fnet', 'cnet'):
        weight, bias = f'{prefix}.conv1.weight', f'{prefix}.conv1.bias'
        if weight in source and weight in target and source[weight].shape != target[weight].shape:
            source.pop(weight)
            source.pop(bias, None)
    compatible = _compatible_state(source, target)
    message = model.flow_net.load_state_dict(compatible, strict=False)
    return message, len(compatible), len(source)


def initialize_model(model, cfg, logger):
    options = cfg['MODEL']
    primary = not cfg['TRAIN']['DDP'] or torch.distributed.get_rank() == 0

    def report(message):
        if primary:
            logger.info(message)

    resume = str(options.get('RESUME', '') or '')
    initializer = str(options.get('INIT_CHECKPOINT', '') or '')
    if resume and initializer:
        raise ValueError('MODEL.RESUME and MODEL.INIT_CHECKPOINT are mutually exclusive')
    if resume:
        checkpoint, state = _read_checkpoint(resume, 'MODEL.RESUME')
        message = model.load_state_dict(state, strict=True)
        report(f'[resume] {resume}: {message}')
        return checkpoint
    if initializer:
        _, state = _read_checkpoint(initializer, 'MODEL.INIT_CHECKPOINT')
        excluded = tuple(str(prefix) for prefix in options.get('INIT_CHECKPOINT_EXCLUDE_PREFIXES', []))
        state = {key: value for key, value in state.items() if not key.startswith(excluded)}
        strict = bool(options.get('INIT_CHECKPOINT_STRICT', True))
        target = model.state_dict()
        compatible = state if strict else _compatible_state(state, target)
        message = model.load_state_dict(compatible, strict=strict)
        mismatched = [key for key, value in state.items()
                      if torch.is_tensor(value) and key in target and value.shape != target[key].shape]
        report(f'[init] {initializer}: strict={strict}, loaded={len(compatible)}/{len(state)}, '
               f'excluded={excluded}, shape_mismatch={mismatched}; {message}')
        return None
    if cfg['TRAIN']['STAGE'] == 'completion':
        path = str(options.get('BASE_CHECKPOINT', ''))
        message, loaded, total = _load_mask_refine_base_checkpoint(model, path)
        report(f'[base] {path}: loaded={loaded}/{total}; {message}')
        event_path = options.get('EVENT_ENCODER_CHECKPOINT', '')
        report(f'[EGCM] event encoder checkpoint: {event_path or "none (random initialization)"}')
        return None

    if cfg['TRAIN']['STAGE'] == 'backbone':
        model.init_pretrained(options.get('PRETRAINED'))
        return None
    path = options['BACKBONE_CHECKPOINT']
    _, source = _read_checkpoint(path, 'MODEL.BACKBONE_CHECKPOINT')
    state = {key: value for key, value in source.items() if key.startswith(('backbone.', 'decode_head.'))}
    message = model.load_state_dict(state, strict=False)
    missing = [key for key in message.missing_keys if key.startswith(('backbone.', 'decode_head.'))]
    if missing or message.unexpected_keys:
        raise RuntimeError(f'Incomplete RGB encoder/decoder checkpoint: {missing}, {message.unexpected_keys}')
    report(f'[RGB backbone] {path}: loaded encoder and decoder')
    flow_path = options.get('RESUME_FLOWNET', '')
    if flow_path:
        message, loaded, total = _load_flow_weights(model, flow_path)
        report(f'[flow] {flow_path}: loaded={loaded}/{total}; {message}')
    else:
        report('[flow] No external flow checkpoint; using current initialization.')
    return None


def restore_training_state(checkpoint, optimizer, scheduler, cfg, iterations, model):
    """Restore epoch/optimizer/scheduler while retaining configured LR overrides."""
    if not isinstance(checkpoint, dict):
        return 0
    epoch = int(checkpoint.get('epoch', 0) or 0)
    configured_lrs = [group.get('initial_lr', group['lr']) for group in optimizer.param_groups]
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(remap_memory_optimizer_state(
            checkpoint['optimizer_state_dict'], optimizer, model,
        ))
    reset = bool(cfg['TRAIN'].get('RESET_SCHEDULER_ON_RESUME', False))
    if 'scheduler_state_dict' in checkpoint and not reset:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if reset:
        scheduler.last_epoch = epoch * iterations
    if reset or cfg['OPTIMIZER'].get('OVERRIDE_RESUME_LR', False):
        scheduler.base_lrs = configured_lrs
        for group, lr in zip(optimizer.param_groups, configured_lrs):
            group['initial_lr'] = lr
        rates = scheduler.get_lr()
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate
        scheduler._last_lr = list(rates)
    return epoch


class TrainingRecord:
    def __init__(self, cfg, save_dir, logger, writer, checkpoint=None):
        self.cfg, self.save_dir = cfg, Path(save_dir)
        self.logger, self.writer = logger, writer
        state = checkpoint if isinstance(checkpoint, dict) else {}
        self.metric_name = 'miou'
        if state.get('best_metric_name', self.metric_name) != self.metric_name:
            raise ValueError('Resume checkpoint uses a different model-selection metric')
        self.best_miou = float(state.get('best_miou', 0.0))
        self.best_epoch = int(state.get('best_epoch', 0))
        self.best_path = None
        if self.best_epoch:
            dataset = cfg['DATASET'].get('EXPERIMENT_NAME', cfg['DATASET']['NAME'])
            previous = self.save_dir / f'{dataset}_epoch{self.best_epoch}_{self.best_miou:.2f}.pth'
            if previous.is_file():
                self.best_path = previous
        self.latest = {}

    def validation(self, result, model, epoch):
        score = float(result[-1])
        self.latest = {'final_miou': score, 'checkpoint_miou': score}
        self.writer.add_scalar('val/mIoU', score, epoch)
        model_state = _unwrap_model(model).state_dict()
        if score > self.best_miou:
            self.best_miou, self.best_epoch = float(score), epoch + 1
            dataset = self.cfg['DATASET'].get('EXPERIMENT_NAME', self.cfg['DATASET']['NAME'])
            path = self.save_dir / f'{dataset}_epoch{self.best_epoch}_{self.best_miou:.2f}.pth'
            _save_checkpoint_atomic(model_state, path)
            if self.cfg['TRAIN'].get('COMPACT_OUTPUT', False) and self.best_path and self.best_path != path:
                self.best_path.unlink(missing_ok=True)
            self.best_path = path
        self.logger.info(f'epoch={epoch + 1} mIoU={score:.2f} best={self.best_miou:.2f}')

    def save_last(self, model, optimizer, scheduler, epoch, loss, legacy_optimizer_layout):
        if not self.cfg['TRAIN'].get('SAVE_TRAINING_STATE', True):
            return
        payload = {
            'epoch': epoch + 1, 'model_state_dict': _unwrap_model(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
            'optimizer_excludes_frozen': not legacy_optimizer_layout, 'loss': loss,
            'best_miou': self.best_miou, 'best_epoch': self.best_epoch,
            'best_metric_name': self.metric_name, **self.latest,
        }
        _save_checkpoint_atomic(payload, self.save_dir / 'last_checkpoint.pth')
