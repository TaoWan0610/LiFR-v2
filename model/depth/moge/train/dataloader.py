import os
from pathlib import Path
import random
from typing import *
import threading

import numpy as np
from PIL import Image
import torch
import utils3d
import pipeline
from tqdm import tqdm

from ..utils.io import *
from ..utils.data_augmentation import sample_perspective, warp_perspective, image_color_augmentation
from ..utils.temporal import event_keys_for_warps, event_keys_in_mapping, event_offset_from_key


class TrainDataLoaderPipeline:
    def __init__(self, config: dict, batch_size: int, num_load_workers: int = 4, num_process_workers: int = 8, buffer_size: int = 8):
        self.config = config

        self.batch_size = batch_size
        self.clamp_max_depth = config['clamp_max_depth']
        self.fov_range_absolute = config.get('fov_range_absolute', 0.0)
        self.fov_range_relative = config.get('fov_range_relative', 0.0)
        self.center_augmentation = config.get('center_augmentation', 0.0)
        self.image_augmentation = config.get('image_augmentation', [])
        self.depth_interpolation = config.get('depth_interpolation', 'bilinear')
        self._raw_event_h5 = {}

        if 'image_sizes' in config:
            self.image_size_strategy = 'fixed'
            self.image_sizes = config['image_sizes']
        elif 'aspect_ratio_range' in config and 'area_range' in config:
            self.image_size_strategy = 'aspect_area'
            self.aspect_ratio_range = config['aspect_ratio_range']
            self.area_range = config['area_range']
        else:
            raise ValueError('Invalid image size configuration')

        # Load datasets
        self.datasets = {}
        for dataset in tqdm(config['datasets'], desc='Loading datasets'):
            name = dataset['name']
            content = Path(dataset['path'], dataset.get('index', '.index.txt')).joinpath().read_text()
            filenames = content.splitlines()
            filenames = self._filter_temporal_filenames(filenames, dataset)
            if dataset.get('load_raw_event_activity', False):
                raw_root = Path(dataset['raw_event_h5_root'])
                valid_filenames = []
                for filename in filenames:
                    parts = Path(filename).parts
                    if len(parts) >= 2 and (raw_root / parts[0] / f'{parts[1]}.h5').is_file():
                        valid_filenames.append(filename)
                skipped = len(filenames) - len(valid_filenames)
                if skipped:
                    print(f"Skipped {skipped} samples without raw event H5 from {name}")
                filenames = valid_filenames
            self.datasets[name] = {
                **dataset,
                'path': dataset['path'],
                'filenames': filenames,
            }
        self.dataset_names = [dataset['name'] for dataset in config['datasets']]
        self.dataset_weights = [dataset['weight'] for dataset in config['datasets']]

        # Build pipeline
        self.pipeline = pipeline.Sequential([
            self._sample_batch,
            pipeline.Unbatch(),
            pipeline.Parallel([self._load_instance] * num_load_workers),
            pipeline.Parallel([self._process_instance] * num_process_workers),
            pipeline.Batch(self.batch_size),
            self._collate_batch,
            pipeline.Buffer(buffer_size),
        ])

        self.invalid_instance = {
            'intrinsics': np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32),
            'image': np.zeros((256, 256, 3), dtype=np.uint8),
            'depth': np.ones((256, 256), dtype=np.float32),
            'depth_mask': np.ones((256, 256), dtype=bool),
            'depth_mask_inf': np.zeros((256, 256), dtype=bool),
            'label_type': 'invalid',
        }


    @staticmethod
    def _duration_frames(dataset_config: Dict[str, Any]) -> int:
        duration_ms = dataset_config.get('duration_ms', None)
        if duration_ms in (None, 0):
            return 0
        frame_step_ms = int(dataset_config.get('frame_step_ms', 50))
        if int(duration_ms) % frame_step_ms != 0:
            raise ValueError(f"duration_ms={duration_ms} must be divisible by frame_step_ms={frame_step_ms}")
        return int(duration_ms) // frame_step_ms

    @classmethod
    def _event_keys(cls, dataset_config: Dict[str, Any]) -> List[str]:
        configured_keys = dataset_config.get('event_keys')
        if configured_keys is not None:
            return list(configured_keys)
        num_warps = cls._duration_frames(dataset_config) or 2
        return event_keys_for_warps(num_warps)

    @staticmethod
    def _sample_index(filename: str) -> Optional[int]:
        try:
            return int(Path(filename).name)
        except ValueError:
            return None

    @staticmethod
    def _inline_temporal_target(
        meta: Dict[str, Any],
        dataset_config: Dict[str, Any],
    ) -> Optional[Tuple[int, int, int]]:
        """Return (input index, target index, offset ms) for in-folder future GT."""
        duration_ms = int(dataset_config.get('duration_ms', 0) or 0)
        target_indices = meta.get('target_indices')
        if duration_ms <= 0 or not isinstance(target_indices, dict):
            return None
        target_idx = target_indices.get(str(duration_ms), target_indices.get(duration_ms))
        if target_idx is None:
            return None
        input_idx = int(meta.get('input_image_index', meta.get('image_index')))
        return input_idx, int(target_idx), duration_ms

    @staticmethod
    def _resolve_temporal_image_path(instance_path: Path, meta: Dict[str, Any], image_idx: int) -> Path:
        image_dirs = []
        direct = instance_path.parent / f'{image_idx:06d}' / 'image.jpg'
        if direct.is_file():
            return direct

        current_image = instance_path / 'image.jpg'
        if current_image.is_file():
            resolved_image = current_image.resolve()
            image_dirs.append(resolved_image.parent)
            resolved_temporal = resolved_image.with_name(
                f'{image_idx:06d}{resolved_image.suffix}'
            )
            if resolved_temporal.is_file():
                return resolved_temporal

        meta_image_path = meta.get('image_path') or meta.get('source_image')
        if meta_image_path is not None:
            meta_image_path = Path(meta_image_path)
            candidates = [
                meta_image_path.with_name(f'{image_idx:06d}{meta_image_path.suffix}'),
                meta_image_path.parent / f'{image_idx:06d}.png',
                meta_image_path.parent / f'{image_idx:06d}.jpg',
            ]
            if 'images' in meta_image_path.parts:
                moge_root = next(
                    (parent for parent in instance_path.parents if parent.name.startswith('moge2')),
                    None,
                )
                if moge_root is not None:
                    images_index = meta_image_path.parts.index('images')
                    relative_image = Path(*meta_image_path.parts[images_index:])
                    current_image_path = moge_root.parent / relative_image
                    image_dirs.append(current_image_path.parent)
                    candidates.extend([
                        current_image_path.with_name(f'{image_idx:06d}{current_image_path.suffix}'),
                        current_image_path.parent / f'{image_idx:06d}.png',
                        current_image_path.parent / f'{image_idx:06d}.jpg',
                    ])
            # M3ED keeps train/val as views of one continuous ``all`` timeline.
            # The first validation window may therefore begin before the split.
            if meta_image_path.parent.parent.name in {'train', 'val'}:
                current_split = meta_image_path.parent.parent.name
                sibling_split = 'val' if current_split == 'train' else 'train'
                sibling_dir = meta_image_path.parent.parent.parent / sibling_split / meta_image_path.parent.name
                all_dir = meta_image_path.parent.parent.parent / 'all' / meta_image_path.parent.name
                candidates.extend([
                    sibling_dir / f'{image_idx:06d}{meta_image_path.suffix}',
                    sibling_dir / f'{image_idx:06d}.png',
                    sibling_dir / f'{image_idx:06d}.jpg',
                    all_dir / f'{image_idx:06d}{meta_image_path.suffix}',
                    all_dir / f'{image_idx:06d}.png',
                    all_dir / f'{image_idx:06d}.jpg',
                ])
            for candidate in candidates:
                if candidate.is_file():
                    return candidate

        # Sparse CARLA standard sequences keep original frame IDs (1, 10, 19, ...).
        # Use the latest real RGB at or before the nominal causal input index.
        for image_dir in image_dirs:
            if not image_dir.is_dir():
                continue
            previous = [
                candidate for candidate in image_dir.iterdir()
                if candidate.is_file()
                and candidate.stem.isdigit()
                and int(candidate.stem) <= image_idx
                and candidate.suffix.lower() in {'.png', '.jpg', '.jpeg'}
            ]
            if previous:
                return max(previous, key=lambda candidate: int(candidate.stem))

        return direct

    @classmethod
    def _filter_temporal_filenames(cls, filenames: List[str], dataset_config: Dict[str, Any]) -> List[str]:
        if not dataset_config.get('requires_events', False):
            return filenames
        duration_frames = cls._duration_frames(dataset_config)
        if duration_frames == 0:
            return filenames

        filtered = []
        skipped_missing_input = 0
        skipped_missing_target = 0
        skipped_missing_events = 0
        minimum_event_offset = min(
            (event_offset_from_key(key) for key in cls._event_keys(dataset_config)),
            default=0,
        )
        for filename in filenames:
            target_idx = cls._sample_index(filename)
            if target_idx is None:
                filtered.append(filename)
                continue
            instance_path = Path(dataset_config['path'], filename)
            meta_path = instance_path / 'meta.json'
            if meta_path.is_file():
                meta = read_json(meta_path)
                inline_target = cls._inline_temporal_target(meta, dataset_config)
                if inline_target is not None:
                    image_idx, _, duration_ms = inline_target
                    runtime_meta = {**meta, 'temporal_input_index': image_idx}
                    image_path = cls._resolve_temporal_image_path(instance_path, runtime_meta, image_idx)
                    if not image_path.is_file():
                        skipped_missing_input += 1
                        continue
                    depth_path = instance_path / f'depth_{duration_ms:03d}ms.png'
                    semantic_path = instance_path / f'semantic_{duration_ms:03d}ms.png'
                    if not depth_path.is_file() or (
                        dataset_config.get('semantic') is not None and not semantic_path.is_file()
                    ):
                        skipped_missing_target += 1
                        continue
                    event_paths = [
                        cls._resolve_event_path_from_meta(
                            dataset_config['path'], dataset_config, runtime_meta, event_key,
                        )
                        for event_key in cls._event_keys(dataset_config)
                    ]
                    if any(path is None or not path.is_file() for path in event_paths):
                        skipped_missing_events += 1
                        continue
                    filtered.append(filename)
                    continue
                target_idx = int(meta.get(
                    'temporal_target_index',
                    meta.get('depth_index', meta.get('image_index', target_idx)),
                ))
            else:
                meta = {}
            image_idx = target_idx - duration_frames
            if image_idx + minimum_event_offset < 0:
                continue
            if not cls._resolve_temporal_image_path(instance_path, meta, image_idx).is_file():
                skipped_missing_input += 1
                continue
            filtered.append(filename)
        if skipped_missing_input:
            print(
                f"Skipped {skipped_missing_input} temporal samples with missing causal input RGB "
                f"from {dataset_config['name']}"
            )
        if skipped_missing_target:
            print(
                f"Skipped {skipped_missing_target} temporal samples with missing in-folder future GT "
                f"from {dataset_config['name']}"
            )
        if skipped_missing_events:
            print(
                f"Skipped {skipped_missing_events} temporal samples with incomplete event windows "
                f"from {dataset_config['name']}"
            )
        return filtered

    @staticmethod
    def _load_event_voxel(path: Path) -> np.ndarray:
        event = np.load(path, allow_pickle=True).astype(np.float32)
        if event.ndim != 3:
            raise ValueError(f"Expected event voxel to be CxHxW, got {event.shape} from {path}")
        return np.nan_to_num(event, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _align_event_to_image(event: np.ndarray, image: np.ndarray, path: Path) -> np.ndarray:
        image_h, image_w = image.shape[:2]
        _, event_h, event_w = event.shape
        if event_h < image_h or event_w < image_w:
            raise ValueError(
                f"Event voxel is smaller than image for {path}: event={event.shape}, image={image.shape}"
            )
        return event[:, :image_h, :image_w]

    @staticmethod
    def _warp_event_voxel(event: np.ndarray, transform: np.ndarray, tgt_size: Tuple[int, int]) -> np.ndarray:
        event_hwc = np.moveaxis(event, 0, -1)
        warped = warp_perspective(event_hwc, transform, tgt_size=tgt_size, interpolation='bilinear')
        if warped.ndim == 2:
            warped = warped[..., None]
        warped = np.moveaxis(warped, -1, 0).astype(np.float32)
        return np.nan_to_num(warped, nan=0.0, posinf=0.0, neginf=0.0)

    def _load_raw_event_activity(
        self,
        dataset_config: Dict[str, Any],
        meta: Dict[str, Any],
        event_idx: int,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("Raw H5 event activity loading requires h5py") from exc

        root = Path(dataset_config['raw_event_h5_root'])
        path = root / str(meta['split']) / f"{meta['sequence']}.h5"
        cache_key = (os.getpid(), threading.get_ident(), str(path))
        h5 = self._raw_event_h5.get(cache_key)
        if h5 is None:
            h5 = h5py.File(path, 'r')
            self._raw_event_h5[cache_key] = h5

        timestamps = h5['image_timestamps']
        source_step = float(np.median(np.diff(timestamps[:min(len(timestamps), 64)])))
        duration_ms = int(dataset_config['duration_ms'])
        duration_ticks = duration_ms / 1000.0 if abs(source_step) < 1 else duration_ms * 1000.0
        source_frames = int(round(duration_ticks / source_step))
        start_frame = event_idx * source_frames
        end_frame = (event_idx + 1) * source_frames
        actual_ticks = float(timestamps[end_frame] - timestamps[start_frame])
        if not np.isclose(actual_ticks, duration_ticks, rtol=0.0, atol=max(abs(source_step) * 0.01, 1e-6)):
            raise ValueError(
                f"Raw-event interval mismatch for {path.name}/{event_idx:06d}: "
                f"frames={start_frame}->{end_frame}, ticks={actual_ticks}, expected={duration_ticks}"
            )

        event_offsets = h5['image_event_indices']
        begin = int(event_offsets[start_frame])
        end = int(event_offsets[end_frame])
        x = np.asarray(h5['events/x'][begin:end], dtype=np.int64)
        y = np.asarray(h5['events/y'][begin:end], dtype=np.int64)
        height, width = image_shape
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        activity = np.zeros((1, height, width), dtype=np.float32)
        np.add.at(activity[0], (y[valid], x[valid]), 1.0)
        return activity

    @staticmethod
    def _event_source_root(dataset_path: Union[str, Path], dataset_config: Dict[str, Any]) -> Optional[Path]:
        if 'event_root' in dataset_config:
            return Path(dataset_config['event_root'])
        path = Path(dataset_path)
        if path.name.startswith('moge2'):
            return path.parent
        return None

    @staticmethod
    def _resolve_event_path_from_meta(
        dataset_path: Union[str, Path],
        dataset_config: Dict[str, Any],
        meta: Dict[str, Any],
        event_key: str,
    ) -> Optional[Path]:
        source_root = TrainDataLoaderPipeline._event_source_root(dataset_path, dataset_config)
        if source_root is None:
            return None

        split = meta.get('split')
        seq = meta.get('sequence')
        image_index = meta.get('image_index')
        if split is None or seq is None or image_index is None:
            return None

        image_idx = int(meta.get('temporal_input_index', image_index))
        duration_frames = TrainDataLoaderPipeline._duration_frames(dataset_config)
        if 'temporal_input_index' in meta:
            pass
        elif duration_frames > 0:
            target_idx = int(meta.get('temporal_target_index', meta.get('depth_index', image_index)))
            image_idx = target_idx - duration_frames
        elif 'duration_ms' in meta and 'depth_index' in meta:
            duration_frames = int(meta['duration_ms']) // int(dataset_config.get('frame_step_ms', 50))
            image_idx = int(meta['depth_index']) - duration_frames
        event_offset = event_offset_from_key(event_key)
        event_bins = int(dataset_config.get('event_bins', 20))
        event_idx = image_idx + event_offset
        if event_idx < 0:
            return None
        if dataset_config.get('event_layout') == 'continuous':
            candidates = [
                source_root / 'event_t0' / split / seq / f'{event_idx:06d}.npy',
                source_root / 'event_t0' / 'all' / seq / f'{event_idx:06d}.npy',
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
            return candidates[0]
        split_variants = [split]
        source_split = meta.get('source_split')
        if source_split and source_split not in split_variants:
            split_variants.append(str(source_split))
        if dataset_config.get('continuous_cross_split', False) and split in {'train', 'val'}:
            split_variants.append('val' if split == 'train' else 'train')
        if split == 'val':
            split_variants.extend(['val/val_depth', 'val/val_semantic'])
        preferred_folders = {
            -1: 'event_t-1_t0',
            0: 'event_t0_t1',
            1: 'event_t1_t2',
        }
        folders = []
        if event_offset in preferred_folders:
            folders.append(preferred_folders[event_offset])
        # ``event_t0_t1`` is a continuous per-index timeline in the DSEC
        # preprocessing output and is therefore the canonical source for
        # offsets beyond the legacy t1->t2 window.
        if 'event_t0_t1' not in folders:
            folders.append('event_t0_t1')
        candidates = []
        for split_variant in split_variants:
            for folder in folders:
                candidates.extend([
                    source_root / folder / f'event_{event_bins}' / split_variant / seq / f'{event_idx:06d}.npy',
                    source_root / folder / split_variant / seq / f'{event_idx:06d}.npy',
                    source_root / 'events' / folder / f'event_{event_bins}' / split_variant / seq / f'{event_idx:06d}.npy',
                    source_root / 'events' / folder / split_variant / seq / f'{event_idx:06d}.npy',
                ])
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    def _sample_batch(self):
        batch_id = 0
        last_area = None
        while True:
            # Depending on the sample strategy, choose a dataset and a filename
            batch_id += 1
            batch = []

            # Sample instances
            for _ in range(self.batch_size):
                dataset_name = random.choices(self.dataset_names, weights=self.dataset_weights)[0]
                dataset_config = self.datasets[dataset_name]
                filename = random.choice(dataset_config['filenames'])
                path = Path(dataset_config['path'], filename)

                instance = {
                    'batch_id': batch_id,
                    'seed': random.randint(0, 2 ** 32 - 1),
                    'dataset': dataset_name,
                    'filename': filename,
                    'path': path,
                    'label_type': self.datasets[dataset_name]['label_type'],
                }
                batch.append(instance)

            # Decide the image size for this batch
            if self.image_size_strategy == 'fixed':
                width, height = random.choice(self.config['image_sizes'])
            elif self.image_size_strategy == 'aspect_area':
                area = random.uniform(*self.area_range)
                aspect_ratio_ranges = [self.datasets[instance['dataset']].get('aspect_ratio_range', self.aspect_ratio_range) for instance in batch]
                aspect_ratio_range = (min(r[0] for r in aspect_ratio_ranges), max(r[1] for r in aspect_ratio_ranges))
                aspect_ratio = random.uniform(*aspect_ratio_range)
                width, height = int((area * aspect_ratio) ** 0.5), int((area / aspect_ratio) ** 0.5)
            else:
                raise ValueError('Invalid image size strategy')

            for instance in batch:
                instance['width'], instance['height'] = width, height

            yield batch


    def _load_instance(self, instance: dict):
        try:
            dataset_config = self.datasets[instance['dataset']]

            meta = read_json(Path(instance['path'], 'meta.json'))
            duration_frames = self._duration_frames(dataset_config)
            configured_duration_ms = int(dataset_config.get('duration_ms', 0) or 0)
            inline_target = self._inline_temporal_target(meta, dataset_config)
            metadata_duration_matches = (
                'duration_ms' not in meta
                or int(meta['duration_ms']) == configured_duration_ms
            )
            target_idx = int(meta.get(
                'temporal_target_index',
                meta.get('depth_index', meta.get('image_index', Path(instance['path']).name)),
            ))
            image_idx = int(meta.get('image_index', Path(instance['path']).name))
            if inline_target is not None:
                image_idx, target_idx, _ = inline_target
                image_path = self._resolve_temporal_image_path(
                    Path(instance['path']), meta, image_idx,
                )
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Missing in-folder temporal input image for {instance['path']}: {image_path}"
                    )
                meta = {**meta, 'temporal_input_index': image_idx}
            elif duration_frames > 0:
                image_idx = target_idx - duration_frames
                image_path = self._resolve_temporal_image_path(Path(instance['path']), meta, image_idx)
                if image_idx < 0 or not image_path.is_file():
                    raise FileNotFoundError(f"Missing temporal input image for {instance['path']}: {image_path}")
                image_idx = int(image_path.stem) if image_path.stem.isdigit() else image_idx
                meta = {**meta, 'temporal_input_index': image_idx}
            else:
                image_path = Path(instance['path'], 'image.jpg')
            image = read_image(image_path)
            target_image = None
            if dataset_config.get('load_target_image', False):
                target_image_path = self._resolve_temporal_image_path(Path(instance['path']), meta, target_idx)
                if not target_image_path.is_file():
                    raise FileNotFoundError(f"Missing target image for {instance['path']}: {target_image_path}")
                target_image = read_image(target_image_path)
            depth_name = (
                f'depth_{configured_duration_ms:03d}ms.png'
                if inline_target is not None
                else self.datasets[instance['dataset']].get('depth', 'depth.png')
            )
            depth = read_depth(Path(instance['path'], depth_name))
            intrinsics = np.array(meta['intrinsics'], dtype=np.float32)
            data = {
                'image': image,
                'depth': depth,
                'intrinsics': intrinsics
            }
            semantic_name = dataset_config.get('semantic')
            if semantic_name is not None:
                if inline_target is not None:
                    semantic_name = f'semantic_{configured_duration_ms:03d}ms.png'
                semantic_path = Path(instance['path'], semantic_name)
                if not semantic_path.is_file():
                    raise FileNotFoundError(f"Missing semantic GT for {instance['path']}: {semantic_path}")
                semantic = np.asarray(Image.open(semantic_path))
                if semantic.ndim != 2 or semantic.shape != depth.shape:
                    raise ValueError(
                        f"Semantic/depth shape mismatch for {instance['path']}: "
                        f"semantic={semantic.shape}, depth={depth.shape}"
                    )
                data['semantic'] = semantic
            if target_image is not None:
                data['target_image'] = target_image
            if self.datasets[instance['dataset']].get('requires_events', False):
                event_files = self.datasets[instance['dataset']].get('events', {})
                for event_key in self._event_keys(dataset_config):
                    expected_event_idx = image_idx + event_offset_from_key(event_key)
                    meta_event_idx = meta.get(f'{event_key}_index') if metadata_duration_matches else None
                    if meta_event_idx is not None and int(meta_event_idx) != expected_event_idx:
                        raise ValueError(
                            f"{event_key} index mismatch for {instance['path']}: "
                            f"metadata={meta_event_idx}, expected={expected_event_idx:06d} "
                            f"relative to input RGB {image_idx:06d}"
                        )
                    event_path = Path(instance['path'], event_files.get(event_key, f'{event_key}.npy'))
                    meta_path = meta.get(f'{event_key}_path', None) if metadata_duration_matches else None
                    if meta_path is not None:
                        meta_parts = Path(meta_path).parts
                        bin_markers = {
                            part for part in meta_parts
                            if part.startswith('event_') and part[6:].isdigit()
                        }
                        expected_marker = f"event_{int(dataset_config.get('event_bins', 20))}"
                        if bin_markers and expected_marker not in bin_markers:
                            meta_path = None
                    if not metadata_duration_matches:
                        # Do not reuse legacy two-warp symlinks when the same
                        # sample folder is trained with a different horizon.
                        event_path = Path(instance['path'], f'.runtime_{event_key}.npy')
                    if not event_path.is_file() and meta_path is not None:
                        event_path = Path(meta_path)
                    if not event_path.is_file():
                        resolved = self._resolve_event_path_from_meta(
                            dataset_config['path'],
                            dataset_config,
                            meta,
                            event_key,
                        )
                        if resolved is not None:
                            event_path = resolved
                    if not event_path.is_file():
                        raise FileNotFoundError(f"Missing {event_key} for {instance['path']}: {event_path}")
                    if event_path.stem.isdigit() and int(event_path.stem) != expected_event_idx:
                        raise ValueError(
                            f"{event_key} file mismatch for {instance['path']}: "
                            f"resolved={event_path.name}, expected={expected_event_idx:06d}.npy"
                        )
                    event_voxel = self._load_event_voxel(event_path)
                    expected_bins = int(dataset_config.get('event_bins', 20))
                    if event_voxel.shape[0] != expected_bins:
                        raise ValueError(
                            f"{event_key} bin mismatch for {event_path}: "
                            f"loaded={event_voxel.shape[0]}, expected={expected_bins}"
                        )
                    data[event_key] = self._align_event_to_image(
                        event_voxel,
                        image,
                        event_path,
                    )
                if dataset_config.get('load_raw_event_activity', False):
                    data['event_activity'] = self._load_raw_event_activity(
                        dataset_config,
                        meta,
                        image_idx,
                        image.shape[:2],
                    )
            instance.update({
                **data,
            })
        except Exception as e:
            print(f"Failed to load instance {instance['dataset']}/{instance['filename']} because of exception:", e)
            if (
                self.datasets[instance['dataset']].get('requires_events', False)
            ):
                raise
            invalid_instance = dict(self.invalid_instance)
            instance.update(invalid_instance)
        return instance

    def _process_instance(self, instance: Dict[str, Union[np.ndarray, str, float, bool]]):
        raw_image, raw_depth, raw_intrinsics, label_type = instance['image'], instance['depth'], instance['intrinsics'], instance['label_type']
        raw_target_image = instance.get('target_image')
        dataset_config = self.datasets[instance['dataset']]
        # Apply the benchmark-validity contract before geometric
        # augmentation, so all downstream targets share the exact same mask.
        depth_unit = dataset_config.get('depth_unit', None)
        depth_for_limits = raw_depth * depth_unit if depth_unit is not None else raw_depth
        raw_valid_depth = np.isfinite(raw_depth)
        raw_valid_depth &= depth_for_limits > float(dataset_config.get('min_depth', 0.0))
        raw_valid_depth &= depth_for_limits <= float(dataset_config.get('max_depth', np.inf))
        raw_semantic = instance.get('semantic')
        if raw_semantic is not None:
            for semantic_id in dataset_config.get('invalid_semantic_ids', []):
                raw_valid_depth &= raw_semantic != int(semantic_id)
        raw_depth = np.where(raw_valid_depth, raw_depth, np.nan).astype(np.float32)
        raw_height, raw_width = raw_image.shape[:2]
        raw_fov_x, raw_fov_y = utils3d.np.intrinsics_to_fov(raw_intrinsics)
        tgt_width, tgt_height = instance['width'], instance['height']
        tgt_aspect = tgt_width / tgt_height

        rng = np.random.default_rng(instance['seed'])

        # Sample perspective transformation
        tgt_intrinsics, R = sample_perspective(
            raw_intrinsics,
            tgt_aspect=tgt_aspect,
            center_augmentation=self.datasets[instance['dataset']].get('center_augmentation', self.center_augmentation),
            fov_range_absolute=self.datasets[instance['dataset']].get('fov_range_absolute', self.fov_range_absolute),
            fov_range_relative=self.datasets[instance['dataset']].get('fov_range_relative', self.fov_range_relative),
            rng=rng
        )

        # Warp
        transform = tgt_intrinsics @ R @ np.linalg.inv(raw_intrinsics)
        # - Warp image
        tgt_image = warp_perspective(raw_image, transform, tgt_size=(tgt_height, tgt_width), interpolation='lanczos')
        tgt_target_image = (
            warp_perspective(raw_target_image, transform, tgt_size=(tgt_height, tgt_width), interpolation='lanczos')
            if raw_target_image is not None else None
        )
        # - Warp depth
        depth_edge_mask = utils3d.np.depth_map_edge(raw_depth, mask=np.isfinite(raw_depth), kernel_size=5, ltol=0.01)
        depth_bilinear_mask = np.isfinite(raw_depth) & ~depth_edge_mask
        warped_depth_bilinear_mask = warp_perspective(depth_bilinear_mask.astype(np.float32), transform, (tgt_height, tgt_width), interpolation='bilinear')
        warped_depth_nearest = warp_perspective(raw_depth, transform, (tgt_height, tgt_width), interpolation='nearest', sparse_mask=~np.isnan(raw_depth))
        warped_depth_bilinear = 1 / warp_perspective(1 / raw_depth, transform, (tgt_height, tgt_width), interpolation='bilinear')   # NOTE: Bilinear intepolation in disparity space maintains planar surfaces.
        warped_depth = np.where(warped_depth_bilinear_mask == 1., warped_depth_bilinear, warped_depth_nearest)
        tgt_uvhomo = np.concatenate([utils3d.np.uv_map((tgt_height, tgt_width)), np.ones((tgt_height, tgt_width, 1), dtype=np.float32)], axis=-1)
        tgt_depth = warped_depth / np.dot(tgt_uvhomo, np.linalg.inv(transform)[2, :])
        event_keys = self._event_keys(dataset_config)
        if all(k in instance for k in event_keys):
            tgt_events = {
                key: self._warp_event_voxel(instance[key], transform, (tgt_height, tgt_width))
                for key in event_keys
            }
        else:
            tgt_events = {}
        tgt_event_activity = (
            self._warp_event_voxel(instance['event_activity'], transform, (tgt_height, tgt_width))
            if 'event_activity' in instance else None
        )

        # always make sure that mask is not empty
        if np.isfinite(tgt_depth).sum() / tgt_depth.size < 0.001:
            tgt_depth = np.ones_like(tgt_depth)
            instance['label_type'] = 'invalid'

        # Flip augmentation
        if rng.choice([True, False]):
            tgt_image = np.flip(tgt_image, axis=1).copy()
            if tgt_target_image is not None:
                tgt_target_image = np.flip(tgt_target_image, axis=1).copy()
            tgt_depth = np.flip(tgt_depth, axis=1).copy()
            tgt_events = {key: np.flip(value, axis=2).copy() for key, value in tgt_events.items()}
            if tgt_event_activity is not None:
                tgt_event_activity = np.flip(tgt_event_activity, axis=2).copy()
            # NOTE: if cx != 0.5, flip intrinsics accordingly.

        # Color augmentation
        image_augmentation = dataset_config.get('image_augmentation', self.image_augmentation)
        tgt_image = image_color_augmentation(
            tgt_image,
            augmentations=image_augmentation,
            rng=rng,
            depth=tgt_depth,
        )

        # Set metric flag if depth is in metric unit
        if depth_unit is not None:
            tgt_depth *= depth_unit
            instance['is_metric'] = True
        else:
            instance['is_metric'] = False

        # Legacy datasets use a scene-relative clamp.  Datasets with an
        # explicit metric max_depth were already invalidated above and must
        # not be silently restricted to a smaller, image-dependent range.
        if 'max_depth' not in dataset_config:
            max_depth = np.nanquantile(
                np.where(np.isfinite(tgt_depth), tgt_depth, np.nan), 0.01
            ) * self.clamp_max_depth
            tgt_depth = np.where(
                np.isfinite(tgt_depth), np.clip(tgt_depth, 0, max_depth), tgt_depth
            )

        tgt_depth_mask_inf = np.isinf(tgt_depth)
        if dataset_config.get('finite_depth_mask', None) == "only_known":
            tgt_depth_mask_fin = np.isfinite(tgt_depth)
        else:
            tgt_depth_mask_fin = ~tgt_depth_mask_inf

        instance.update({
            'image': torch.from_numpy(tgt_image.astype(np.float32) / 255.0).permute(2, 0, 1),
            'depth': torch.from_numpy(tgt_depth).float(),
            'depth_mask_fin': torch.from_numpy(tgt_depth_mask_fin).bool(),
            'depth_mask_inf': torch.from_numpy(tgt_depth_mask_inf).bool(),
            'intrinsics': torch.from_numpy(tgt_intrinsics).float(),
        })
        if tgt_target_image is not None:
            instance['target_image'] = torch.from_numpy(
                tgt_target_image.astype(np.float32) / 255.0
            ).permute(2, 0, 1)
        for key, value in tgt_events.items():
            instance[key] = torch.from_numpy(value).float()
        if tgt_event_activity is not None:
            instance['event_activity'] = torch.from_numpy(tgt_event_activity).float()
        return instance

    def _collate_batch(self, instances: List[Dict[str, Any]]):
        batch = {k: torch.stack([instance[k] for instance in instances], dim=0) for k in ['image', 'depth', 'depth_mask_fin', 'depth_mask_inf', 'intrinsics']}
        if 'target_image' in instances[0]:
            batch['target_image'] = torch.stack([instance['target_image'] for instance in instances], dim=0)
        for event_key in event_keys_in_mapping(instances[0]):
            if event_key in instances[0]:
                batch[event_key] = torch.stack([instance[event_key] for instance in instances], dim=0)
        if 'event_activity' in instances[0]:
            batch['event_activity'] = torch.stack(
                [instance['event_activity'] for instance in instances], dim=0
            )
        batch = {
            'label_type': [instance['label_type'] for instance in instances],
            'is_metric': [instance['is_metric'] for instance in instances],
            'info': [{'dataset': instance['dataset'], 'filename': instance['filename']} for instance in instances],
            **batch,
        }
        return batch

    def get(self) -> Dict[str, Union[torch.Tensor, str]]:
        return self.pipeline.get()

    def start(self):
        self.pipeline.start()

    def stop(self):
        self.pipeline.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pipeline.stop()
        return False
