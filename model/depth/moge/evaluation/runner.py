"""Full validation with per-image metric depth scores on one or more devices."""
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

from .dataloader import EvalDataLoaderPipeline
from .metrics import compute_metrics
from moge.utils.temporal import event_keys_for_warps
from moge.utils.tools import flatten_nested_dict, key_average, set_nested_dict


def evaluate(model, benchmarks, output, *, batch_size=1, num_tokens=None,
             resolution_level=9, use_fp16=False):
    if batch_size < 1:
        raise ValueError('Evaluation batch size must be positive')
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    device = model.device
    results = {}
    for name, config in benchmarks.items():
        # Count each metric separately: a range may be empty in some images.
        totals = {}
        with EvalDataLoaderPipeline(**config, rank=rank, world_size=world_size) as loader:
            for start in tqdm(range(0, len(loader), batch_size), desc=name, disable=rank != 0):
                size = min(batch_size, len(loader) - start)
                samples = [loader.get() for _ in range(size)]
                samples = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in sample.items()} for sample in samples]
                padded = samples + [samples[-1]] * (batch_size - size)
                image = torch.stack([sample['image'] for sample in padded])
                kwargs = {}
                if model.anytime_enabled:
                    kwargs['event_sequence'] = [
                        torch.stack([sample[key] for sample in padded])
                        for key in event_keys_for_warps(model.num_warps)
                    ]
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                start_time = time.perf_counter()
                with torch.inference_mode():
                    prediction = model.infer(
                        image, apply_mask=False, num_tokens=num_tokens,
                        resolution_level=resolution_level, use_fp16=use_fp16, **kwargs,
                    )
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                elapsed = (time.perf_counter() - start_time) / batch_size
                for index, sample in enumerate(samples):
                    scores = compute_metrics({'depth_metric': prediction['depth'][index]}, sample)
                    scores['inference_time'] = elapsed
                    for key, value in flatten_nested_dict(scores).items():
                        if not math.isfinite(value):
                            raise ValueError(f'Non-finite validation metric {key}: {sample["filename"]}')
                        total, count = totals.get(key, (0., 0))
                        totals[key] = (total + value, count + 1)
        gathered = [totals]
        if distributed:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, totals)
        if rank == 0:
            combined = {}
            for part in gathered:
                for key, (value, count) in part.items():
                    old_sum, old_count = combined.get(key, (0., 0))
                    combined[key] = (old_sum + value, old_count + count)
            result = {}
            for key, (value, count) in combined.items():
                set_nested_dict(result, key, value / count)
            if 'depth_metric' not in result:
                raise ValueError(f'No valid depth ground truth in {name}')
            results[name] = result
    if rank == 0:
        results['mean'] = key_average(list(results.values()))
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    return results
