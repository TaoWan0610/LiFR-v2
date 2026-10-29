import os
from pathlib import Path

import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
DEPTH_EXTENSIONS = {'.png', '.npy'}


def numeric_stem_sets(root, extensions):
    stems = {}
    with os.scandir(root) as sequences:
        for sequence in sequences:
            if not sequence.is_dir(follow_symlinks=True):
                continue
            values = set()
            with os.scandir(sequence.path) as files:
                for entry in files:
                    path = Path(entry.name)
                    if (
                        entry.is_file(follow_symlinks=True)
                        and path.suffix.lower() in extensions
                        and path.stem.isdigit()
                    ):
                        values.add(int(path.stem))
            stems[sequence.name] = values
    return stems


def numeric_file_map(directory, extensions):
    files = {}
    for path in sorted(directory.glob('*/*')):
        if not path.is_file() or path.suffix.lower() not in extensions or not path.stem.isdigit():
            continue
        files[f'{path.parent.name}/{int(path.stem):06d}'] = path
    return files


def manifest_samples(root, split, do_semseg, do_depth, semantic_root=None):
    """Read the canonical MoGe-style M3ED sample manifest.

    Each manifest row points to one same-time sample directory containing
    ``image.jpg`` and ``depth.png`` symlinks plus ``meta.json``.  Using the
    manifest keeps MTMamba on exactly the same train/validation split as MoGe
    without rescanning or inventing a second sample index.
    """
    manifest = root / f'.{split}.txt'
    if not manifest.is_file():
        return None

    samples = []
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding='utf-8').splitlines(), start=1
    ):
        relative = raw_line.strip().replace('\\', '/')
        if not relative or relative.startswith('#'):
            continue
        relative_path = Path(relative)
        if relative_path.is_absolute() or '..' in relative_path.parts:
            raise ValueError(
                f'Unsafe M3ED manifest entry at {manifest}:{line_number}: {relative!r}'
            )
        if not relative_path.parts or relative_path.parts[0] != split:
            raise ValueError(
                f'M3ED manifest entry must start with {split!r} at '
                f'{manifest}:{line_number}: {relative!r}'
            )

        sample_dir = root / relative_path
        image_path = sample_dir / 'image.jpg'
        depth_path = sample_dir / 'depth.png'
        semseg_path = (
            (semantic_root / relative_path).with_suffix('.png')
            if semantic_root is not None
            else sample_dir / 'semantic.png'
        )

        sample_id = '/'.join(relative_path.parts[1:])
        samples.append({
            'id': sample_id,
            'image': image_path,
            'semseg': semseg_path,
            'depth': depth_path,
            'meta_path': sample_dir / 'meta.json',
        })
    return samples


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.float32)


def load_semseg(path, image_shape):
    with Image.open(path) as image:
        semseg = np.asarray(image, dtype=np.float32)
    if semseg.ndim == 3:
        semseg = semseg[:, :, 0]
    if semseg.shape != image_shape:
        semseg = cv2.resize(semseg, image_shape[::-1], interpolation=cv2.INTER_NEAREST)
    return semseg[:, :, None]


def load_depth(path, image_shape):
    if path.suffix.lower() == '.npy':
        depth = np.load(path, allow_pickle=False).astype(np.float32)
    else:
        with Image.open(path) as image:
            if 'near' not in image.info or 'far' not in image.info:
                raise ValueError(f'M3ED depth PNG is missing near/far metadata: {path}')
            near = float(image.info['near'])
            far = float(image.info['far'])
            encoded = np.asarray(image, dtype=np.uint16)
        normalized = (encoded.astype(np.float32) - 1.0) / 65533.0
        depth = near ** (1.0 - normalized) * far ** normalized
        depth[encoded == 0] = np.nan
        depth[encoded == 65535] = np.inf
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.shape != image_shape:
        depth = cv2.resize(depth, image_shape[::-1], interpolation=cv2.INTER_NEAREST)
    return depth[:, :, None].astype(np.float32, copy=False)


def load_event(path, image_shape, event_bins):
    event = np.load(path, allow_pickle=False).astype(np.float32)
    if event.ndim != 3 or event.shape[0] != event_bins:
        raise ValueError(f'Expected ({event_bins}, H, W) event voxel, got {event.shape}: {path}')
    event = np.moveaxis(event, 0, -1)
    if event.shape[:2] != image_shape:
        event = cv2.resize(event, image_shape[::-1], interpolation=cv2.INTER_LINEAR)
        if event.ndim == 2:
            event = event[:, :, None]
    return np.ascontiguousarray(event, dtype=np.float32)


class M3ED(data.Dataset):
    """M3ED static samples: same-time ``RGB[N] -> depth/semantic[N]``."""

    def __init__(
        self,
        root=None,
        split='train',
        transform=None,
        retname=True,
        do_semseg=True,
        do_depth=True,
        semantic_root=None,
    ):
        if split not in {'train', 'val'}:
            raise ValueError(f'M3ED only supports train/val, got {split!r}')
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.retname = retname
        self.do_semseg = do_semseg
        self.do_depth = do_depth
        self.semantic_root = Path(semantic_root) if semantic_root is not None else None

        manifest_records = manifest_samples(
            self.root,
            split,
            do_semseg=do_semseg,
            do_depth=do_depth,
            semantic_root=self.semantic_root,
        )
        if manifest_records is not None:
            self.samples = manifest_records
            layout = 'moge2-manifest'
        else:
            image_dir = self.root / 'image' / split
            if not image_dir.is_dir():
                image_dir = self.root / 'image' / 'all'
            image_map = numeric_file_map(image_dir, IMAGE_EXTENSIONS)
            semseg_map = numeric_file_map(self.root / 'semantic' / split, IMAGE_EXTENSIONS)
            depth_map = numeric_file_map(self.root / 'depth' / split, DEPTH_EXTENSIONS)
            sample_ids = set(image_map)
            if do_semseg:
                sample_ids &= set(semseg_map)
            if do_depth:
                sample_ids &= set(depth_map)

            self.samples = [
                {
                    'id': sample_id,
                    'image': image_map[sample_id],
                    'semseg': semseg_map.get(sample_id),
                    'depth': depth_map.get(sample_id),
                    'meta_path': None,
                }
                for sample_id in sorted(sample_ids)
            ]
            layout = 'legacy-folders'
        if not self.samples:
            raise RuntimeError(f'No aligned M3ED static samples found under {self.root} ({split})')
        print(
            f'[M3ED] split={split} samples={len(self.samples)} layout={layout} '
            'mapping=RGB[N]->GT[N]'
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]
        image = load_rgb(record['image'])
        sample = {'image': image}
        if self.do_semseg:
            sample['semseg'] = load_semseg(record['semseg'], image.shape[:2])
        if self.do_depth:
            sample['depth'] = load_depth(record['depth'], image.shape[:2])
        if self.retname:
            sample['meta'] = {
                'img_name': record['id'],
                'input_name': record.get('input_id', record['id']),
                'img_size': image.shape[:2],
            }
        if self.transform is not None:
            sample = self.transform(sample)
        sample['index'] = index
        return sample

    def __str__(self):
        return f'M3ED(split={self.split})'
