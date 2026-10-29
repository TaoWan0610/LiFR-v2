"""Train the RGB backbone, LiFR v1 or LiFR v2 from a semantic configuration."""
import argparse
import os
from pathlib import Path
import yaml
import torch
from semseg.config import apply_overrides, config_duration_ms, load_config
from semseg.engine.trainer import train, validate_training_config
from semseg.models import prepare_model_config
from semseg.utils.utils import cleanup_ddp, fix_seeds, get_logger, setup_cudnn, setup_ddp


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--duration', type=int)
    for name in ('resume', 'init-checkpoint', 'save-dir', 'dataset-root',
                 'backbone-checkpoint', 'base-checkpoint', 'event-checkpoint'):
        parser.add_argument('--' + name)
    for name in ('batch-size', 'epochs'):
        parser.add_argument('--' + name, type=int)
    for name in ('decoder-lr', 'gate-alpha'):
        parser.add_argument('--' + name, type=float)
    parser.add_argument('--gate-schedule', choices=('fixed', 'progressive'))
    parser.add_argument('--set', dest='overrides', nargs='*', default=(), metavar='KEY=VALUE')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.cfg), args.overrides)
    validate_training_config(cfg, config_duration_ms(cfg, args.duration))
    overrides = {
        'save_dir': ('SAVE_DIR',), 'dataset_root': ('DATASET', 'ROOT'),
        'resume': ('MODEL', 'RESUME'), 'init_checkpoint': ('MODEL', 'INIT_CHECKPOINT'),
        'base_checkpoint': ('MODEL', 'BASE_CHECKPOINT'),
        'event_checkpoint': ('MODEL', 'EVENT_ENCODER_CHECKPOINT'),
        'batch_size': ('TRAIN', 'BATCH_SIZE'), 'epochs': ('TRAIN', 'EPOCHS'),
        'gate_alpha': ('TRAIN', 'GATE_ALPHA'),
        'gate_schedule': ('TRAIN', 'GATE_SCHEDULE'),
        'backbone_checkpoint': ('MODEL', 'BACKBONE_CHECKPOINT'),
        'decoder_lr': ('OPTIMIZER', 'DECODER_LR'),
    }
    for name, keys in overrides.items():
        value = getattr(args, name)
        if value is not None:
            target = cfg if len(keys) == 1 else cfg[keys[0]]
            target[keys[-1]] = value
    if args.resume and args.init_checkpoint is None:
        cfg['MODEL']['INIT_CHECKPOINT'] = ''
    cfg = prepare_model_config(cfg)
    cfg['TRAIN']['DDP'] = int(os.environ.get('WORLD_SIZE', '1')) > 1
    args.duration = config_duration_ms(cfg, args.duration)
    validate_training_config(cfg, args.duration)
    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp() if cfg['TRAIN']['DDP'] else 0
    try:
        save_dir = Path(cfg['SAVE_DIR'])
        resume = str(cfg['MODEL'].get('RESUME', '') or '')
        if os.path.isfile(resume) and cfg['TRAIN'].get('RESUME_IN_PLACE', True):
            save_dir = Path(resume).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        cfg['SAVE_DIR'] = str(save_dir)
        primary = not cfg['TRAIN']['DDP'] or torch.distributed.get_rank() == 0
        logger = get_logger(save_dir / 'epoch_metrics.log' if primary else None, file_mode='a')
        if primary:
            (save_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        train(cfg, gpu, save_dir, args.duration, logger)
    finally:
        if cfg['TRAIN']['DDP']:
            cleanup_ddp()


if __name__ == '__main__':
    main()
