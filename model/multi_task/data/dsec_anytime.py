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


class DSECAnytime(data.Dataset):
    """DSEC RGB + event anytime dataset for semseg + depth prediction.

    Expected root layout:
    - images/ or image/
    - semantic/ or segmentation/
    - depth/
    - event_t0_t1/event_<bins>/
    - event_t-1_t0/event_<bins>/
    - event_t1_t2/event_<bins>/   (used when time_window_steps == 2)

    Targets are read from the future timestamp and RGB/event inputs are gathered
    from the aligned previous timestamps following the original semseg anytime
    pipeline.
    """

    IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    LABEL_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    DEPTH_EXTS = ('.npy', '.png', '.tif', '.tiff', '.exr')
    EVENT_EXTS = ('.npy', '.npz')
    SEQUENCE_PATTERN = re.compile(r'^[A-Za-z]+(?:_[A-Za-z]+)*_[0-9]{2}_[A-Za-z]$')

    def __init__(
        self,
        root=None,
        split='train',
        transform=None,
        retname=True,
        do_semseg=True,
        do_depth=True,
        train_sequences=None,
        val_sequences=None,
        time_window_steps=1,
        event_bins=20,
    ):
        self.root = Path(root)
        self.transform = transform
        self.retname = retname
        self.do_semseg = do_semseg
        self.do_depth = do_depth
        self.time_window_steps = int(time_window_steps)
        self.event_bins = int(event_bins)
        self.train_sequences = set(train_sequences) if train_sequences else None
        self.val_sequences = val_sequences or {}
        self.val_sequences_semseg = set(self.val_sequences.get('semseg', []))
        self.val_sequences_depth = set(self.val_sequences.get('depth', []))
        self.known_sequences = set()
        if self.train_sequences:
            self.known_sequences.update(self.train_sequences)
        self.known_sequences.update(self.val_sequences_semseg)
        self.known_sequences.update(self.val_sequences_depth)

        if self.time_window_steps not in (0, 1, 2):
            raise NotImplementedError(
                f'DSECAnytime currently supports time_window_steps in {{0, 1, 2}}, got {self.time_window_steps}'
            )

        if isinstance(split, str):
            self.split = [split]
        else:
            split = list(split)
            split.sort()
            self.split = split

        self.image_root = _resolve_first_existing(self.root, ('images', 'image'))
        self.semseg_root = _resolve_first_existing(self.root, ('semantic', 'segmentation'))
        self.depth_root = _resolve_first_existing(self.root, ('depth',))
        event_folder = f'event_{self.event_bins}'
        self.event_root = _resolve_first_existing(self.root / 'event_t0_t1', (event_folder,))
        self.event_before_root = _resolve_first_existing(self.root / 'event_t-1_t0', (event_folder,))
        self.event_after_root = None
        if self.time_window_steps == 2:
            self.event_after_root = _resolve_first_existing(self.root / 'event_t1_t2', (event_folder,))

        self.samples = []
        print('Initializing dataloader for DSECAnytime {} set'.format(''.join(self.split)))
        for split_name in self.split:
            self.samples.extend(self._gather_split(split_name))


        self.im_ids = [sample['id'] for sample in self.samples]
        print('Number of dataset images: {:d}'.format(len(self.im_ids)))

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
            return 'aligned_rgb_640x440' in parent_names or path.stem.isdigit()
        if kind == 'semseg':
            return '11classes' in parent_names or path.stem.isdigit()
        if kind == 'depth':
            return path.suffix.lower() == '.npy' and 'depth' in parent_names
        if kind == 'event':
            return path.suffix.lower() in self.EVENT_EXTS and f'event_{self.event_bins}' in parent_names
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
        if not sample_stem.isdigit():
            return None
        return f'{sequence_name}/{sample_stem}'

    def _sequence_name(self, sample_key):
        if '/' in sample_key:
            return sample_key.split('/')[0]
        return None

    def _filter_ids_by_sequences(self, ids, sequences):
        if not sequences:
            return ids
        return [sample_id for sample_id in ids if self._sequence_name(sample_id) in sequences]

    def _build_target_ids(self, split_name, image_map, semseg_map, depth_map):
        if split_name == 'train':
            if self.do_depth:
                base_ids = sorted(depth_map.keys())
            elif self.do_semseg:
                base_ids = sorted(semseg_map.keys())
            else:
                base_ids = sorted(image_map.keys())
            return self._filter_ids_by_sequences(base_ids, self.train_sequences)

        if split_name == 'val' and (self.val_sequences_semseg or self.val_sequences_depth):
            ids = set()
            if self.do_semseg:
                ids.update(self._filter_ids_by_sequences(sorted(semseg_map.keys()), self.val_sequences_semseg))
            if self.do_depth:
                ids.update(self._filter_ids_by_sequences(sorted(depth_map.keys()), self.val_sequences_depth))
            return sorted(ids)

        ids_all = set(image_map.keys())
        if self.do_semseg:
            ids_all &= set(semseg_map.keys())
        if self.do_depth:
            ids_all &= set(depth_map.keys())
        return sorted(ids_all)

    def _gather_split(self, split_name):
        image_map = self._list_stem_map(self.image_root, self.IMAGE_EXTS, 'image')
        semseg_map = self._list_stem_map(self.semseg_root, self.LABEL_EXTS, 'semseg')
        depth_map = self._list_stem_map(self.depth_root, self.DEPTH_EXTS, 'depth')
        event_map = self._list_stem_map(self.event_root, self.EVENT_EXTS, 'event')
        event_before_map = self._list_stem_map(self.event_before_root, self.EVENT_EXTS, 'event')
        event_after_map = self._list_stem_map(self.event_after_root, self.EVENT_EXTS, 'event') if self.event_after_root else {}

        target_ids = self._build_target_ids(split_name, image_map, semseg_map, depth_map)

        samples = []
        for sample_id in target_ids:
            seq_name, target_stem = sample_id.split('/')
            target_index = int(target_stem)
            input_index = target_index - self.time_window_steps if self.time_window_steps > 0 else target_index
            if input_index < 0:
                continue

            image_id = f'{seq_name}/{input_index:06d}'
            event_id = image_id
            prev_event_id = f'{seq_name}/{input_index - 1:06d}'
            next_event_id = f'{seq_name}/{input_index + 1:06d}'

            if image_id not in image_map:
                continue
            if self.time_window_steps > 0 and event_id not in event_map:
                continue
            if self.time_window_steps > 0 and prev_event_id not in event_before_map and prev_event_id not in event_map:
                continue
            if self.time_window_steps == 2 and next_event_id not in event_after_map and next_event_id not in event_map:
                continue

            semseg_enabled = self.do_semseg and (
                split_name == 'train' or not self.val_sequences_semseg or seq_name in self.val_sequences_semseg
            )
            depth_enabled = self.do_depth and (
                split_name == 'train' or not self.val_sequences_depth or seq_name in self.val_sequences_depth
            )

            if split_name == 'train':
                if semseg_enabled and sample_id not in semseg_map:
                    continue
                if depth_enabled and sample_id not in depth_map:
                    continue

            samples.append({
                'id': sample_id,
                'image': str(image_map[image_id]),
                'semseg': str(semseg_map[sample_id]) if semseg_enabled and sample_id in semseg_map else None,
                'depth': str(depth_map[sample_id]) if depth_enabled and sample_id in depth_map else None,
                'event': str(event_map[event_id]) if self.time_window_steps > 0 else None,
                'event_before': str(event_before_map.get(prev_event_id, event_map.get(prev_event_id))) if self.time_window_steps > 0 else None,
                'event_after': str(event_after_map.get(next_event_id, event_map.get(next_event_id))) if self.time_window_steps == 2 else None,
                'input_id': image_id,
            })
        return samples

    def __getitem__(self, index):
        sample = {}
        record = self.samples[index]

        image = self._load_img(record['image'])
        sample['image'] = image

        if self.do_semseg:
            sample['semseg'] = self._load_semseg(record['semseg'], image.shape[:2])

        if self.do_depth:
            sample['depth'] = self._load_depth(record['depth'], image.shape[:2])

        if self.time_window_steps > 0:
            sample['event'] = self._load_event(record['event'], image.shape[:2])
            sample['event_before'] = self._load_event(record['event_before'], image.shape[:2])
            if self.time_window_steps == 2:
                sample['event_after'] = self._load_event(record['event_after'], image.shape[:2])

        if self.retname:
            sample['meta'] = {
                'img_name': str(record['id']),
                'input_name': str(record['input_id']),
                'img_size': (image.shape[0], image.shape[1]),
            }

        if self.transform is not None:
            sample = self.transform(sample)

        sample['index'] = index
        return sample

    def __len__(self):
        return len(self.samples)

    def _load_img(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return np.asarray(image, dtype=np.float32)

    def _load_semseg(self, semseg_path, image_shape):
        if semseg_path is None:
            semseg = np.full(image_shape, 255, dtype=np.float32)
            return np.expand_dims(semseg, axis=2)
        semseg = Image.open(semseg_path)
        semseg = np.asarray(semseg, dtype=np.float32)
        if semseg.ndim == 3:
            semseg = semseg[:, :, 0]
        if semseg.shape[:2] != image_shape:
            semseg = cv2.resize(semseg, image_shape[::-1], interpolation=cv2.INTER_NEAREST)
        return np.expand_dims(semseg, axis=2)

    def _load_depth(self, depth_path, image_shape):
        if depth_path is None:
            depth = np.zeros(image_shape, dtype=np.float32)
            return np.expand_dims(depth, axis=2)
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
        if depth.shape[:2] != image_shape:
            depth = cv2.resize(depth, image_shape[::-1], interpolation=cv2.INTER_NEAREST)
        if depth.ndim == 2:
            depth = np.expand_dims(depth, axis=2)
        return depth

    def _load_event(self, event_path, image_shape):
        loaded = np.load(event_path)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            key = 'event' if 'event' in loaded.files else loaded.files[0]
            event = loaded[key].astype(np.float32)
            loaded.close()
        else:
            event = loaded.astype(np.float32)
        if event.ndim != 3:
            raise ValueError(f'Unsupported event voxel shape: {event_path} {event.shape}')
        if event.shape[0] <= 64:
            event = np.transpose(event, (1, 2, 0))
        target_h, target_w = image_shape
        if event.shape[1] == target_w and event.shape[0] >= target_h:
            event = event[:target_h, :target_w]
        if event.shape[:2] != image_shape:
            event = cv2.resize(event, image_shape[::-1], interpolation=cv2.INTER_LINEAR)
            if event.ndim == 2:
                event = np.expand_dims(event, axis=2)
        return event

    def __str__(self):
        return 'DSECAnytime(split=' + str(self.split) + ')'
