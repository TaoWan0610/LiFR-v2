import os
from pathlib import Path
import re

import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image


def _resolve_first_existing(root, candidates):
    for name in candidates:
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Could not find any of {candidates} under {root}')


class DSEC(data.Dataset):
    """DSEC RGB semantic segmentation + depth dataset.

    Expected directory structure under ``root``:
    - images/ or image/
    - semantic/ or segmentation/
    - depth/

    Optional split definitions:
    - splits/train.txt and splits/val.txt containing sample stems, or
    - train/val subdirectories inside the image/semantic/depth folders.

    If no explicit split is provided, a deterministic 90/10 split is used.
    """

    IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    LABEL_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    DEPTH_EXTS = ('.npy', '.png', '.tif', '.tiff', '.exr')
    SEQUENCE_PATTERN = re.compile(r'^[A-Za-z]+(?:_[A-Za-z]+)*_[0-9]{2}_[A-Za-z]$')

    def __init__(
        self,
        root=None,
        split='train',
        transform=None,
        retname=True,
        do_semseg=True,
        do_depth=True,
        split_ratio=0.9,
        split_seed=42,
        train_sequences=None,
        val_sequences=None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.retname = retname
        self.do_semseg = do_semseg
        self.do_depth = do_depth
        self.split_ratio = split_ratio
        self.split_seed = split_seed
        self.train_sequences = set(train_sequences) if train_sequences else None
        self.val_sequences = val_sequences or {}
        self.val_sequences_semseg = set(self.val_sequences.get('semseg', []))
        self.val_sequences_depth = set(self.val_sequences.get('depth', []))
        self.known_sequences = set()
        if self.train_sequences:
            self.known_sequences.update(self.train_sequences)
        self.known_sequences.update(self.val_sequences_semseg)
        self.known_sequences.update(self.val_sequences_depth)

        if isinstance(split, str):
            self.split = [split]
        else:
            split = list(split)
            split.sort()
            self.split = split

        self.image_root = _resolve_first_existing(self.root, ('images', 'image'))
        self.semseg_root = _resolve_first_existing(self.root, ('semantic', 'segmentation'))
        self.depth_root = _resolve_first_existing(self.root, ('depth',))

        self.samples = []
        print('Initializing dataloader for DSEC {} set'.format(''.join(self.split)))
        for split_name in self.split:
            self.samples.extend(self._gather_split(split_name))


        self.im_ids = [sample['id'] for sample in self.samples]
        self.images = [sample['image'] for sample in self.samples]
        self.semsegs = [sample['semseg'] for sample in self.samples]
        self.depths = [sample['depth'] for sample in self.samples]

        if self.do_semseg:
            assert len(self.images) == len(self.semsegs)
        if self.do_depth:
            assert len(self.images) == len(self.depths)

        print('Number of dataset images: {:d}'.format(len(self.images)))

    def _list_stem_map(self, directory, allowed_exts, kind):
        stem_map = {}
        for path in sorted(directory.rglob('*')):
            if not path.is_file():
                continue
            if path.suffix.lower() not in allowed_exts:
                continue
            if not self._is_target_file(path, kind):
                continue
            relative = path.relative_to(directory)
            key = self._canonical_key(relative)
            if key is None:
                continue
            stem_map[key] = path
        return stem_map

    def _is_target_file(self, path, kind):
        parent_names = {parent.name for parent in path.parents}
        if kind == 'image':
            # Support both:
            # - images/train/<seq>/left/aligned_rgb_640x440/000000.png
            # - images/train/<seq>/000000.png
            return 'aligned_rgb_640x440' in parent_names or path.stem.isdigit()
        if kind == 'semseg':
            # Support both:
            # - semantic/train/<seq>/11classes/000000.png
            # - semantic/train/<seq>/000000.png
            return '11classes' in parent_names or path.stem.isdigit()
        if kind == 'depth':
            return path.suffix.lower() == '.npy' and 'depth' in parent_names
        raise ValueError(f'Unknown kind: {kind}')

    def _extract_sequence_name(self, parts):
        for part in parts:
            if part in self.known_sequences:
                return part
        for part in parts:
            if self.SEQUENCE_PATTERN.match(part):
                return part
        return None

    def _canonical_key(self, relative_path):
        relative_no_suffix = relative_path.with_suffix('')
        parts = relative_no_suffix.parts
        sequence_name = self._extract_sequence_name(parts)
        if sequence_name is None:
            return None
        sample_stem = parts[-1]
        return f'{sequence_name}/{sample_stem}'

    def _sequence_name(self, sample_key):
        if '/' in sample_key:
            return sample_key.split('/')[0]
        return None

    def _filter_ids_by_sequences(self, ids, sequences):
        if not sequences:
            return ids
        return [sample_id for sample_id in ids if self._sequence_name(sample_id) in sequences]

    def _read_split_file(self, split_name):
        split_file = self.root / 'splits' / f'{split_name}.txt'
        if not split_file.is_file():
            return None
        with split_file.open('r') as f:
            return [line.strip() for line in f if line.strip()]

    def _gather_split(self, split_name):
        split_file_ids = self._read_split_file(split_name)
        image_map = self._list_stem_map(self.image_root, self.IMAGE_EXTS, 'image')
        semseg_map = self._list_stem_map(self.semseg_root, self.LABEL_EXTS, 'semseg')
        depth_map = self._list_stem_map(self.depth_root, self.DEPTH_EXTS, 'depth')

        if split_file_ids is not None:
            ids = split_file_ids
        else:
            ids_all = sorted(image_map.keys())
            if split_name == 'train' and self.train_sequences:
                ids = self._filter_ids_by_sequences(ids_all, self.train_sequences)
            elif split_name == 'val' and (self.val_sequences_semseg or self.val_sequences_depth):
                union_sequences = self.val_sequences_semseg | self.val_sequences_depth
                ids = self._filter_ids_by_sequences(ids_all, union_sequences)
            else:
                ids_all = sorted(set(image_map.keys()) & set(semseg_map.keys()) & set(depth_map.keys()))
                if not ids_all:
                    raise RuntimeError(f'No matched DSEC samples found under {self.root}')
                rng = np.random.default_rng(self.split_seed)
                ids_all = list(ids_all)
                rng.shuffle(ids_all)
                if len(ids_all) > 1:
                    split_idx = min(len(ids_all) - 1, max(1, int(len(ids_all) * self.split_ratio)))
                else:
                    split_idx = 1
                if split_name == 'train':
                    ids = sorted(ids_all[:split_idx])
                elif split_name == 'val':
                    ids = sorted(ids_all[split_idx:])
                else:
                    ids = sorted(ids_all)

        samples = []
        for sample_id in ids:
            if sample_id not in image_map:
                continue
            seq_name = self._sequence_name(sample_id)
            semseg_enabled = self.do_semseg and (
                not self.val_sequences_semseg or split_name != 'val' or seq_name in self.val_sequences_semseg
            )
            depth_enabled = self.do_depth and (
                not self.val_sequences_depth or split_name != 'val' or seq_name in self.val_sequences_depth
            )
            if split_name == 'train':
                semseg_enabled = self.do_semseg
                depth_enabled = self.do_depth

            if semseg_enabled and sample_id not in semseg_map:
                continue
            if depth_enabled and sample_id not in depth_map:
                continue
            samples.append({
                'id': sample_id,
                'image': str(image_map[sample_id]),
                'semseg': str(semseg_map[sample_id]) if semseg_enabled and sample_id in semseg_map else None,
                'depth': str(depth_map[sample_id]) if depth_enabled and sample_id in depth_map else None,
            })
        return samples

    def __getitem__(self, index):
        sample = {}

        image = self._load_img(index)
        sample['image'] = image

        if self.do_semseg:
            semseg = self._load_semseg(index, image.shape[:2])
            if semseg.shape[:2] != image.shape[:2]:
                semseg = cv2.resize(semseg, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
                if semseg.ndim == 2:
                    semseg = np.expand_dims(semseg, axis=2)
            sample['semseg'] = semseg

        if self.do_depth:
            depth = self._load_depth(index, image.shape[:2])
            if depth.shape[:2] != image.shape[:2]:
                depth = cv2.resize(depth, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
                if depth.ndim == 2:
                    depth = np.expand_dims(depth, axis=2)
            sample['depth'] = depth

        if self.retname:
            sample['meta'] = {
                'img_name': str(self.im_ids[index]),
                'input_name': str(self.samples[index].get('input_id', self.im_ids[index])),
                'img_size': (image.shape[0], image.shape[1]),
            }

        if self.transform is not None:
            sample = self.transform(sample)

        sample['index'] = index
        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        return np.asarray(image, dtype=np.float32)

    def _load_semseg(self, index, image_shape):
        if self.semsegs[index] is None:
            semseg = np.full(image_shape, 255, dtype=np.float32)
            return np.expand_dims(semseg, axis=2)
        semseg = Image.open(self.semsegs[index])
        semseg = np.asarray(semseg, dtype=np.float32)
        if semseg.ndim == 3:
            semseg = semseg[:, :, 0]
        return np.expand_dims(semseg, axis=2)

    def _load_depth(self, index, image_shape):
        if self.depths[index] is None:
            depth = np.zeros(image_shape, dtype=np.float32)
            return np.expand_dims(depth, axis=2)
        depth_path = self.depths[index]
        suffix = Path(depth_path).suffix.lower()
        if suffix == '.npy':
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(f'Failed to read depth map: {depth_path}')
            depth = depth.astype(np.float32)
            if depth.ndim == 3:
                depth = depth[:, :, 0]
        if depth.ndim == 2:
            depth = np.expand_dims(depth, axis=2)
        return depth

    def __str__(self):
        return f'DSEC(split={self.split})'
