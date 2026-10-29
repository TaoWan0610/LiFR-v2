#!/usr/bin/env python3
"""Launch a task with portable paths and an isolated Python import path."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from tools.config import ROOT, read_config, render_tree, roots


def build_command(task, action, config, gpus, weights, extra):
    engine = ROOT / 'model' / task
    python = sys.executable
    distributed = [python, '-m', 'torch.distributed.run', '--standalone',
                   '--nnodes=1', f'--nproc_per_node={gpus}']
    if task == 'seg':
        if action == 'train':
            model = read_config(config).get('MODEL', {})
            if model.get('NAME') not in {'RGBBackbone', 'LiFRV1', 'LiFRV2'}:
                raise ValueError('Semantic training requires a backbone, lifr_v1 or lifr_v2 configuration')
        if action in {'train', 'distill'}:
            entry = 'train.py' if action == 'train' else 'train_event_distill.py'
            command = distributed + [f'tools/{entry}', '--cfg', str(config)]
        else:
            command = [python, 'tools/val.py', '--cfg', str(config), '--model_path', weights]
    elif task == 'depth':
        if action in {'train', 'distill'}:
            prefix = [python]
            if gpus > 1:
                cfg = read_config(config)
                train = cfg.get('train', {})
                precision = train.get('precision', 'bf16' if train.get('mixed_precision_dtype') == 'bfloat16' else 'no')
                precision = {'fp32': 'no', 'amp': 'fp16'}.get(precision, precision)
                prefix = [python, '-m', 'accelerate.commands.launch', '--multi_gpu',
                          '--num_machines=1', f'--num_processes={gpus}',
                          f'--mixed_precision={precision}', '--dynamo_backend=no']
            entry = 'train.py' if action == 'train' else 'train_event_distill.py'
            command = prefix + [f'tools/{entry}', '--config', str(config)]
        elif action == 'eval':
            command = distributed + ['tools/val.py', '--config', str(config), '--pretrained', weights]
        else:
            command = [python, 'tools/infer.py', '--pretrained', weights]
    else:
        if action == 'distill':
            command = distributed + ['tools/train_event_distill.py', '--config', str(config)]
        else:
            entry = 'train.py' if action == 'train' else 'val.py'
            command = distributed + [entry, '--config_exp', str(config),
                                     '--run_mode', 'train' if action == 'train' else 'infer']
            if weights: command += ['--trained_model', weights]
    return engine, command + extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True,
                        choices=['seg', 'depth', 'multi_task'],
                        help='Model task: seg, depth, or multi_task.')
    parser.add_argument('--action', required=True, choices=['train', 'distill', 'eval', 'infer'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--weights', default='')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--data-root')
    parser.add_argument('--weights-root')
    parser.add_argument('--output-root')
    parser.add_argument('--dry-run', action='store_true', help='Render configs and print the command without starting a job.')
    args, extra = parser.parse_known_args()
    task = args.task
    if extra[:1] == ['--']: extra = extra[1:]
    if args.gpus < 1: parser.error('--gpus must be at least 1')
    if not args.config and not (task == 'depth' and args.action == 'infer'):
        parser.error('--config is required for this action')
    if args.action in {'eval', 'infer'} and not args.weights:
        parser.error('--weights is required for evaluation and inference')
    config = args.config.resolve() if args.config else None
    if config:
        try: relative = config.relative_to(ROOT / 'configs')
        except ValueError: parser.error('Put the config under this repository\'s configs/ directory.')
        if not config.is_file(): parser.error(f'Config does not exist: {config}')
        if relative.parts[0] != task:
            parser.error(f'Use a configuration from configs/{task}/ for --task {task}.')
    variables = roots(args.data_root, args.weights_root, args.output_root)
    runtime = Path(variables['OUTPUT_ROOT']) / '.configs' / f'{time.time_ns()}_{os.getpid()}'
    render_tree(runtime, variables)
    config = runtime / relative if config else None
    weights = str(Path(args.weights).expanduser().resolve()) if args.weights else ''
    if weights and not args.dry_run and not Path(weights).is_file():
        parser.error(f'Checkpoint does not exist: {weights}')
    engine, command = build_command(task, args.action, config, args.gpus, weights, extra)
    env = os.environ.copy()
    # Isolate the selected engine's Python packages.
    paths = [str(engine), str(ROOT)]
    if task == 'multi_task': paths.append(str(ROOT / 'model/seg'))
    env['PYTHONPATH'] = os.pathsep.join(paths)
    env['LIFR_DATA_ROOT'] = variables['DATA_ROOT']
    env['LIFR_WEIGHTS_ROOT'] = variables['WEIGHTS_ROOT']
    env['LIFR_OUTPUT_ROOT'] = variables['OUTPUT_ROOT']
    env['MTMAMBA_DATA_ROOT'] = variables['DATA_ROOT']
    print('Working directory:', engine, flush=True)
    print('Command:', shlex.join(command), flush=True)
    (runtime / 'launch.json').write_text(json.dumps({'task': task, 'action': args.action,
        'config': str(config), 'command': command, 'cwd': str(engine)}, indent=2)+'\n')
    if not args.dry_run:
        raise SystemExit(subprocess.call(command, cwd=engine, env=env))


if __name__ == '__main__': main()
