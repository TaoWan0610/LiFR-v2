from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from torchvision.transforms import functional as TF

def _event_files(root: Path) -> list[Path]:
    """Read both DSEC .npy event voxels and compressed CARLA .npz event voxels."""
    return sorted((*root.rglob('*.npy'), *root.rglob('*.npz')))


def _load_event(path: Path) -> Tensor:
    if path.suffix == '.npy':
        array = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as archive:
            array = archive['event']
    return torch.from_numpy(array).float()[:, :440]


class EventDistillationDataset(Dataset):
    """Return a future event voxel and the RGB image at its endpoint."""

    def __init__(
        self,
        root: str,
        split: str,
        dataset_type: str = 'dsec',
        frame_index_stride: int = 2,
        image_size: tuple[int, int] = (440, 640),
        event_scale: float = 255.0,
        horizontal_flip: bool = False,
        exclude_sequences: list[str] | None = None,
        include_semantic: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.dataset_type = str(dataset_type).lower()
        self.frame_index_stride = int(frame_index_stride)
        self.image_size = tuple(image_size)
        self.event_scale = float(event_scale)
        self.horizontal_flip = bool(horizontal_flip)
        self.exclude_sequences = set(exclude_sequences or [])
        self.include_semantic = bool(include_semantic)
        if self.dataset_type == 'm3ed':
            if self.frame_index_stride != 1:
                raise ValueError('M3ED 40 ms distillation requires event[N] -> RGB[N+1]')
            event_root = self.root / 'event_t0_t1' / 'event_20' / split
            if not event_root.is_dir():
                event_root = self.root / 'event_t0' / split
            image_roots = [self.root / 'image' / split]
        elif self.dataset_type in {'carla', 'dsec'}:
            event_root = self.root / 'event_t0_t1' / 'event_20' / split
            image_roots = [self.root / 'images' / split]
            if split == 'val':
                image_roots.insert(0, self.root / 'images' / 'val' / 'val_semantic')
        else:
            raise ValueError(f'Unsupported distillation dataset type: {self.dataset_type}')

        self.samples = []
        semantic_root = self.root / 'semantic' / split
        for event_path in _event_files(event_root):
            sequence = event_path.parent.name
            if sequence in self.exclude_sequences:
                continue
            event_index = int(event_path.stem)
            target_index = event_index + self.frame_index_stride
            image_name = f'{target_index:06d}.png'
            image_path = next(
                (root / sequence / image_name for root in image_roots if (root / sequence / image_name).is_file()),
                None,
            )
            label_path = semantic_root / sequence / image_name
            if image_path is not None and (not self.include_semantic or label_path.is_file()):
                self.samples.append((event_path, image_path, sequence, target_index, label_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        event_path, image_path, sequence, target_index, label_path = self.samples[index]
        event = _load_event(event_path)
        image = io.read_image(str(image_path))[:3, :440].float()
        event = TF.resize(event, self.image_size, TF.InterpolationMode.BILINEAR, antialias=True)
        image = TF.resize(image, self.image_size, TF.InterpolationMode.BILINEAR, antialias=True)
        sample = {'event': event, 'image': image}
        if self.include_semantic:
            label = io.read_image(str(label_path))[0, :440].long()
            sample['label'] = TF.resize(label.unsqueeze(0), self.image_size, TF.InterpolationMode.NEAREST).squeeze(0)
        if self.horizontal_flip and random.random() < 0.5:
            event = TF.hflip(event)
            image = TF.hflip(image)
            if self.include_semantic:
                sample['label'] = TF.hflip(sample['label'])
        event = event / self.event_scale
        image = TF.normalize(image / 255.0, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        sample.update(event=event, image=image, sequence=sequence, target_index=f'{target_index:06d}')
        return sample

