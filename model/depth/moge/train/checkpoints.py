"""Atomic training-state checkpoints."""
from pathlib import Path
import torch


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, workspace):
    if path is None:
        return None
    path = Path(workspace, 'checkpoint', 'latest.pt') if path == 'latest' else Path(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if 'model' not in checkpoint:
        raise ValueError(f'Checkpoint has no model state: {path}')
    if 'step' not in checkpoint and path.stem.isdigit():
        checkpoint['step'] = int(path.stem) - 1
    if 'optimizer' not in checkpoint:
        checkpoint['_model_only_resume'] = True
    return checkpoint
