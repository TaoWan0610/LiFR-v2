"""Endpoint validation, aggregate segmentation metrics and optional visualizations."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate

from semseg.augmentations import get_val_augmentation
from semseg.datasets import get_dataset_class
from semseg.metrics import Metrics
from semseg.models import build_model, prepare_model_config
from .checkpoints import _extract_model_state, _strip_module_prefix


def _is_primary():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _save_predictions(save_dir, palette, names, indices, images, predictions, targets):
    colors = np.zeros((256, 3), dtype=np.uint8)
    for label, color in palette.items():
        colors[int(label)] = color
    mean = images.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    rgb = ((images * std + mean).clamp(0, 1) * 255).byte().cpu().numpy()
    predictions = predictions.argmax(dim=1).byte().cpu().numpy()
    targets = targets.byte().cpu().numpy()
    for name, index, image, prediction, target in zip(names, indices, rgb, predictions, targets):
        folder = Path(save_dir) / str(name)
        folder.mkdir(parents=True, exist_ok=True)
        stem = Path(str(index)).stem
        Image.fromarray(prediction).save(folder / f'{stem}_labelTrainIds11.png')
        panels = [Image.fromarray(image.transpose(1, 2, 0)),
                  Image.fromarray(colors[target]), Image.fromarray(colors[prediction])]
        canvas = Image.new('RGB', (sum(panel.width for panel in panels), max(panel.height for panel in panels)))
        offset = 0
        for panel in panels:
            canvas.paste(panel, (offset, 0))
            offset += panel.width
        canvas.save(folder / f'{stem}_color.png')


@torch.no_grad()
def evaluate(model, dataloader, device, save_dir=None, palette=None):
    """Return class/mean accuracy, F1 and IoU over one GT endpoint per sample."""
    if save_dir is not None and palette is None:
        raise ValueError('Saving predictions requires a label palette')
    model.eval()
    dataset = dataloader.dataset
    metrics = Metrics(dataset.n_classes, dataset.ignore_label, device)
    for names, indices, images, labels in tqdm(dataloader, disable=not _is_primary(), desc='Evaluating'):
        if len(labels) != 1:
            raise ValueError('Validation requires exactly one GT at t0 + duration')
        images = [image.to(device, non_blocking=True) for image in images]
        target = labels[0].to(device, non_blocking=True)
        outputs = model(images)
        prediction = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
        metrics.update(prediction, target)
        if save_dir is not None:
            _save_predictions(save_dir, palette, names, indices, images[0], prediction, target)
    metrics.synchronize()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou


def _validation_dataset(cfg, duration):
    """Use the configured validation split, including its published sampling rules."""
    data = cfg['DATASET']
    options = {}
    if data['NAME'] in {'DSEC', 'SHFDSEC', 'SHFEmerge'}:
        options.update(
            frame_interval_ms=int(data.get('FRAME_INTERVAL_MS', 50)),
            frame_index_stride=int(data.get('FRAME_INDEX_STRIDE', 1)),
            anchor_modulo=int(data.get('ANCHOR_MODULO', 0)),
            anchor_remainders=data.get('ANCHOR_REMAINDERS'),
            sequence_names=data.get('VAL_SEQUENCES'),
            exclude_sequence_names=data.get('VAL_EXCLUDE_SEQUENCES'),
        )
    dataset = get_dataset_class(data['NAME'])(
        data['ROOT'].replace('${DURATION}', str(duration)), 'val', 11,
        get_val_augmentation(cfg['EVAL']['IMAGE_SIZE']), data['MODALS'],
        duration=duration, dataset_type=data['TYPE'],
        sample_gap=int(data.get('VAL_SAMPLE_GAP', 0)), **options,
    )
    if len(dataset) == 0:
        raise RuntimeError('Validation dataset is empty; check paths and split settings')
    return dataset


def run_validation(cfg, model_path, duration, logger):
    cfg = prepare_model_config(cfg)
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    device = torch.device(cfg['DEVICE'])
    dataset = _validation_dataset(cfg, duration)
    model = build_model(cfg, num_classes=dataset.n_classes, duration_ms=int(duration),
                        event_interval_ms=dataset.frame_interval_ms)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(_strip_module_prefix(_extract_model_state(checkpoint)), strict=True)
    model.to(device)
    options = cfg['EVAL']
    loader = DataLoader(dataset, batch_size=options['BATCH_SIZE'],
                        num_workers=int(options.get('NUM_WORKERS', 0)),
                        pin_memory=device.type == 'cuda')
    scene = str(cfg['DATASET'].get('SCENE', 'day'))
    tag = str(options.get('OUTPUT_TAG', '')).strip().replace(' ', '_')
    name = '_'.join(filter(None, (scene, 'eval', tag, time.strftime('%Y%m%d_%H%M%S'))))
    output = model_path.parent / name
    output.mkdir(parents=True, exist_ok=True)
    taxonomy = dataset.SEGMENTATION_CONFIGS[dataset.n_classes]
    visualization_dir = output if options.get('SAVE_VISUALIZATIONS', False) else None
    acc, macc, f1, mf1, ious, miou = evaluate(
        model, loader, device, visualization_dir, palette=taxonomy['PALETTE'],
    )
    class_names = list(taxonomy['CLASSES'])
    table = tabulate({'Class': class_names + ['Mean'], 'IoU': ious + [miou],
                      'F1': f1 + [mf1], 'Acc': acc + [macc]}, headers='keys')
    payload = {
        'checkpoint': str(model_path.resolve()), 'dataset': cfg['DATASET']['NAME'],
        'dataset_root': str(Path(dataset.root).resolve()), 'scene': scene,
        'duration_ms': int(duration), 'num_samples': len(dataset),
        'miou': float(miou), 'mean_f1': float(mf1), 'mean_pixel_accuracy': float(macc),
        'per_class': {
            name: {'iou': float(iou), 'f1': float(score), 'pixel_accuracy': float(accuracy)}
            for name, iou, score, accuracy in zip(class_names, ious, f1, acc)
        },
    }
    (output / 'metrics.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    output.with_suffix('.txt').write_text(f'{model_path}\nSamples: {len(dataset)}\n{table}\n', encoding='utf-8')
    logger.info('\n%s', table)
    print(f'mIoU: {miou:.2f}; results saved in {output}')
