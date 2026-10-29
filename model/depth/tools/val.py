"""Evaluate a depth checkpoint on the full configured validation split."""
import argparse
import os
from pathlib import Path
import sys

for root in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[3]):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--pretrained')
    parser.add_argument('--output', '-o')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--num-tokens', '--num_tokens', type=int)
    parser.add_argument('--resolution-level', '--resolution_level', type=int)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist
    from moge.config import default_checkpoint, expand_eval_config, read_config
    from moge.model import load_model
    from moge.evaluation.runner import evaluate

    raw = read_config(args.config)
    paths, options = raw.get('paths', {}), raw.get('eval', {})
    checkpoint = args.pretrained or paths.get('checkpoint')
    if checkpoint is None and raw.get('preset') == 'backbone':
        checkpoint = default_checkpoint(raw['backbone'])
    output = args.output or paths.get('output')
    if not checkpoint or not output:
        parser.error('Provide checkpoint and output paths in config or --pretrained/--output')
    device = torch.device(args.device)
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if distributed:
        if device.type == 'cuda':
            device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
            torch.cuda.set_device(device)
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo')
    try:
        model = load_model(checkpoint).to(device).eval()
        evaluate(model, expand_eval_config(raw), output,
                 batch_size=int(options.get('batch_size', 1)),
                 num_tokens=args.num_tokens if args.num_tokens is not None else options.get('num_tokens'),
                 resolution_level=args.resolution_level if args.resolution_level is not None else options.get('resolution_level', 9),
                 use_fp16=args.fp16 or options.get('precision') == 'fp16')
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
