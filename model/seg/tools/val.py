"""Validate a fixed LiFR semantic segmentation checkpoint."""
import argparse
import os
from pathlib import Path
import time
from semseg.config import load_config, config_duration_ms
from semseg.models import prepare_model_config
from semseg.engine.evaluation import run_validation
from semseg.utils.utils import setup_cudnn, get_logger

SCENE = 'day'
INPUT_TYPE = 'rgb'
NUM_CLASSES = 11


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--model-path', '--model_path', dest='model_path', required=True)
    parser.add_argument('--duration', type=int, default=None)
    parser.add_argument('--val-sequences', nargs='+', default=None)
    parser.add_argument('--val-exclude-sequences', nargs='+', default=None)
    parser.add_argument('--clear-val-excludes', action='store_true')
    parser.add_argument('--output-tag', default=None)
    parser.add_argument('--save-visualizations', action='store_true')
    args = parser.parse_args()

    cfg = prepare_model_config(load_config(args.cfg))
    args.duration = config_duration_ms(cfg, args.duration)
    if args.val_sequences is not None:
        cfg['DATASET']['VAL_SEQUENCES'] = args.val_sequences
    if args.clear_val_excludes:
        cfg['DATASET']['VAL_EXCLUDE_SEQUENCES'] = None
    elif args.val_exclude_sequences is not None:
        cfg['DATASET']['VAL_EXCLUDE_SEQUENCES'] = args.val_exclude_sequences
    if args.output_tag is not None:
        cfg['EVAL']['OUTPUT_TAG'] = args.output_tag
    if args.save_visualizations:
        cfg['EVAL']['SAVE_VISUALIZATIONS'] = True

    setup_cudnn()
    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    exp_name = '_'.join(['inference', cfg['DATASET']['NAME'], 'Encoder-B2', modals])
    save_dir = Path(cfg['SAVE_DIR'], exp_name)
    resume_path = str(cfg['MODEL'].get('RESUME', '') or '')
    if os.path.isfile(resume_path):
        save_dir = Path(os.path.dirname(resume_path))
    os.makedirs(save_dir, exist_ok=True)
    time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_logger(save_dir / f'{INPUT_TYPE}_{SCENE}_{NUM_CLASSES}_{time_}_eval.log')
    run_validation(cfg, args.model_path, args.duration, logger)


if __name__ == '__main__':
    main()
