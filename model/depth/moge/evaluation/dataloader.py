from typing import *
from pathlib import Path
import math

import numpy as np
import torch
from PIL import Image
import cv2
import utils3d
import pipeline

from ..utils.geometry_numpy import norm3d
from ..utils.io import *
from ..utils.temporal import event_keys_for_warps, event_offset_from_key


class EvalDataLoaderPipeline:

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        split: str = '.index.txt',
        drop_max_depth: float = 1000.,
        num_load_workers: int = 4,
        num_process_workers: int = 8,
        depth_unit: str = None,
        requires_events: bool = False,
        event_bins: int = 20,
        event_root: str = None,
        temporal_image_root: str = None,
        event_layout: str = 'dsec',
        continuous_cross_split: bool = False,
        duration_ms: int = 0,
        frame_step_ms: int = 50,
        min_depth: float = 0.0,
        max_depth: float = None,
        semantic: str = None,
        invalid_semantic_ids: Sequence[int] = (),
        depth_metric_ranges: Sequence[Sequence[float]] = (),
        rank: int = 0,
        world_size: int = 1,
    ):
        filenames = Path(path).joinpath(split).read_text(encoding='utf-8').splitlines()
        self.width = width
        self.height = height
        self.drop_max_depth = drop_max_depth
        self.path = Path(path)
        self.depth_unit = depth_unit
        self.requires_events = requires_events
        self.event_bins = int(event_bins)
        self.event_root = Path(event_root) if event_root is not None else (self.path.parent if self.path.name.startswith('moge2') else None)
        self.temporal_image_root = Path(temporal_image_root) if temporal_image_root is not None else None
        self.event_layout = event_layout
        self.continuous_cross_split = bool(continuous_cross_split)
        self.duration_ms = int(duration_ms or 0)
        self.frame_step_ms = int(frame_step_ms)
        self.duration_frames = self._duration_frames()
        self.min_depth = float(min_depth)
        self.max_depth = None if max_depth is None else float(max_depth)
        self.semantic = semantic
        self.invalid_semantic_ids = tuple(int(value) for value in invalid_semantic_ids)
        self.depth_metric_ranges = tuple(
            (float(lower), float(upper)) for lower, upper in depth_metric_ranges
        )
        if any(lower < 0 or upper <= lower for lower, upper in self.depth_metric_ranges):
            raise ValueError(f"Invalid metric depth ranges: {self.depth_metric_ranges}")
        filenames = self._filter_temporal_filenames(filenames)
        self.filenames = filenames[int(rank)::int(world_size)]

        self.rng = np.random.default_rng(seed=0)

        self.pipeline = pipeline.Sequential([
            self._generator,
            pipeline.Parallel([self._load_instance] * num_load_workers),
            pipeline.Parallel([self._process_instance] * num_process_workers),
            pipeline.Buffer(4)
        ])

    def __len__(self):
        return math.ceil(len(self.filenames))

    def _generator(self):
        for idx in range(len(self)):
            yield idx

    def _duration_frames(self) -> int:
        if self.duration_ms == 0:
            return 0
        if self.duration_ms % self.frame_step_ms != 0:
            raise ValueError(f"duration_ms={self.duration_ms} must be divisible by frame_step_ms={self.frame_step_ms}")
        return self.duration_ms // self.frame_step_ms


    def _event_keys(self) -> List[str]:
        return event_keys_for_warps(self.duration_frames or 2)

    @staticmethod
    def _sample_index(filename: str) -> Optional[int]:
        try:
            return int(Path(filename).name)
        except ValueError:
            return None

    def _inline_temporal_target(
        self,
        meta: Dict[str, Any],
    ) -> Optional[Tuple[int, int, int]]:
        target_indices = meta.get('target_indices')
        if self.duration_ms <= 0 or not isinstance(target_indices, dict):
            return None
        target_idx = target_indices.get(str(self.duration_ms), target_indices.get(self.duration_ms))
        if target_idx is None:
            return None
        input_idx = int(meta.get('input_image_index', meta.get('image_index')))
        return input_idx, int(target_idx), self.duration_ms

    def _resolve_temporal_image_path(self, path: Path, meta: Dict[str, Any], image_idx: int) -> Path:
        image_dirs = []
        direct = path.parent / f'{image_idx:06d}' / 'image.jpg'
        if direct.is_file():
            return direct

        current_image = path / 'image.jpg'
        if current_image.is_file():
            resolved_image = current_image.resolve()
            image_dirs.append(resolved_image.parent)
            resolved_temporal = resolved_image.with_name(
                f'{image_idx:06d}{resolved_image.suffix}'
            )
            if resolved_temporal.is_file():
                return resolved_temporal

        candidates = []
        if self.temporal_image_root is not None:
            split = str(meta.get('split', ''))
            seq = str(meta.get('sequence', path.parent.name))
            roots = [
                self.temporal_image_root / seq,
                self.temporal_image_root / split / seq,
            ]
            if split == 'val':
                roots.extend([
                    self.temporal_image_root / split / 'val_depth' / seq,
                    self.temporal_image_root / split / 'val_depth_full50ms' / seq,
                ])
            for root in roots:
                candidates.extend([
                    root / 'left' / 'aligned_rgb_640x440' / f'{image_idx:06d}.png',
                    root / 'left' / 'aligned_rgb_640x440' / f'{image_idx:06d}.jpg',
                    root / f'{image_idx:06d}.png',
                    root / f'{image_idx:06d}.jpg',
                ])

        meta_image_path = meta.get('image_path') or meta.get('source_image')
        if meta_image_path is not None:
            meta_image_path = Path(meta_image_path)
            candidates.extend([
                meta_image_path.with_name(f'{image_idx:06d}{meta_image_path.suffix}'),
                meta_image_path.parent / f'{image_idx:06d}.png',
                meta_image_path.parent / f'{image_idx:06d}.jpg',
            ])
            if 'images' in meta_image_path.parts:
                moge_root = next(
                    (parent for parent in path.parents if parent.name.startswith('moge2')),
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

    def _filter_temporal_filenames(self, filenames: List[str]) -> List[str]:
        if not self.requires_events or self.duration_frames == 0:
            return filenames
        filtered = []
        for filename in filenames:
            target_idx = self._sample_index(filename)
            if target_idx is None:
                filtered.append(filename)
                continue
            path = self.path / filename
            meta = read_meta(path / 'meta.json')
            inline_target = self._inline_temporal_target(meta)
            if inline_target is not None:
                image_idx, _, duration_ms = inline_target
                runtime_meta = {**meta, 'temporal_input_index': image_idx}
                if not (path / f'depth_{duration_ms:03d}ms.png').is_file():
                    continue
                if self.semantic is not None and not (
                    path / f'semantic_{duration_ms:03d}ms.png'
                ).is_file():
                    continue
                event_paths = [
                    self._resolve_event_path(path, runtime_meta, event_key)
                    for event_key in self._event_keys()
                ]
                if all(event_path is not None and event_path.is_file() for event_path in event_paths):
                    filtered.append(filename)
            else:
                image_idx = target_idx - self.duration_frames
                if image_idx - 1 >= 0:
                    filtered.append(filename)
        return filtered

    def _load_instance(self, idx):
        if idx >= len(self.filenames):
            return None

        path = self.path.joinpath(self.filenames[idx])
        target_path = path

        instance = {
            'filename': self.filenames[idx],
            'width': self.width,
            'height': self.height,
        }
        meta = read_meta(Path(path, 'meta.json'))
        inline_target = self._inline_temporal_target(meta)
        target_meta = meta
        if inline_target is not None:
            image_idx, target_idx, _ = inline_target
            image_path = self._resolve_temporal_image_path(path, meta, image_idx)
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing in-folder temporal input image for {path}: {image_path}")
            meta = {**meta, 'temporal_input_index': image_idx}
        elif self.duration_frames > 0:
            target_idx = int(meta.get(
                'temporal_target_index',
                meta.get('depth_index', meta.get('image_index', path.name)),
            ))
            image_idx = target_idx - self.duration_frames
            image_path = self._resolve_temporal_image_path(path, meta, image_idx)
            if image_idx < 0 or not image_path.is_file():
                raise FileNotFoundError(f"Missing temporal input image for {path}: {image_path}")
            image_idx = int(image_path.stem) if image_path.stem.isdigit() else image_idx
            meta = {**meta, 'temporal_input_index': image_idx}
        else:
            image_path = Path(path, 'image.jpg')
        instance['image'] = read_image(image_path)

        depth_name = (
            f'depth_{self.duration_ms:03d}ms.png'
            if inline_target is not None
            else 'depth.png'
        )
        depth = read_depth(Path(target_path, depth_name))
        instance.update({
            'depth': np.nan_to_num(depth, nan=1, posinf=1, neginf=1),
            'depth_mask': np.isfinite(depth),
            'depth_mask_inf': np.isinf(depth),
        })

        if self.semantic is not None:
            semantic_name = (
                f'semantic_{self.duration_ms:03d}ms.png'
                if inline_target is not None
                else self.semantic
            )
            semantic_path = Path(target_path, semantic_name)
            if not semantic_path.is_file():
                raise FileNotFoundError(f"Missing semantic GT for {target_path}: {semantic_path}")
            semantic_mask = np.asarray(Image.open(semantic_path))
            if semantic_mask.ndim != 2 or semantic_mask.shape != depth.shape:
                raise ValueError(
                    f"Semantic/depth shape mismatch for {target_path}: "
                    f"semantic={semantic_mask.shape}, depth={depth.shape}"
                )
            instance['validity_semantic_mask'] = semantic_mask


        instance['intrinsics'] = np.array(target_meta['intrinsics'], dtype=np.float32)
        instance['meta'] = {
            **target_meta,
            'input_meta': meta,
            'input_index': (
                image_idx
                if self.duration_frames > 0
                else meta.get('image_index', path.name)
            ),
            'target_index': (
                target_idx
                if inline_target is not None
                else target_meta.get(
                    'temporal_target_index', target_meta.get('depth_index', target_path.name)
                )
            ),
            'duration_ms': self.duration_ms,
            'target_offset_us': int(target_meta.get('depth_timestamp_us', 0))
            - int(meta.get('rgb_timestamp_us', 0)),
        }
        if self.requires_events:
            for event_key in self._event_keys():
                event_path = self._resolve_event_path(path, meta, event_key)
                if event_path is None or not event_path.is_file():
                    raise FileNotFoundError(f"Missing {event_key} for {path}: {event_path}")
                event_voxel = np.load(event_path, allow_pickle=True).astype(np.float32)
                if event_voxel.shape[0] != self.event_bins:
                    raise ValueError(
                        f"{event_key} bin mismatch for {event_path}: "
                        f"loaded={event_voxel.shape[0]}, expected={self.event_bins}"
                    )
                instance[event_key] = self._align_event_to_image(
                    event_voxel,
                    instance['image'],
                    event_path,
                )

        return instance

    def _resolve_event_path(self, path: Path, meta: Dict[str, Any], event_key: str) -> Optional[Path]:
        configured_duration_ms = int(self.duration_ms or 0)
        metadata_duration_matches = (
            'duration_ms' not in meta
            or int(meta['duration_ms']) == configured_duration_ms
        )
        if metadata_duration_matches:
            direct = Path(path, f'{event_key}.npy')
            if direct.is_file():
                return direct
            meta_path = meta.get(f'{event_key}_path')
            if meta_path is not None:
                meta_path = Path(meta_path)
                bin_markers = {
                    part for part in meta_path.parts
                    if part.startswith('event_') and part[6:].isdigit()
                }
                expected_marker = f"event_{self.event_bins}"
                if meta_path.is_file() and (not bin_markers or expected_marker in bin_markers):
                    return meta_path
        if self.event_root is None:
            return None

        split = meta.get('split')
        seq = meta.get('sequence')
        image_index = meta.get('image_index')
        if split is None or seq is None or image_index is None:
            return None

        image_idx = int(meta.get('temporal_input_index', image_index))
        if 'temporal_input_index' in meta:
            pass
        elif self.duration_frames > 0:
            image_idx = int(
                meta.get('temporal_target_index', meta.get('depth_index', image_index))
            ) - self.duration_frames
        elif 'duration_ms' in meta and 'depth_index' in meta:
            image_idx = int(meta['depth_index']) - int(meta['duration_ms']) // 50
        event_offset = event_offset_from_key(event_key)
        event_idx = image_idx + event_offset
        if event_idx < 0:
            return None
        if self.event_layout == 'continuous':
            candidates = [
                self.event_root / 'event_t0' / split / seq / f'{event_idx:06d}.npy',
                self.event_root / 'event_t0' / 'all' / seq / f'{event_idx:06d}.npy',
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
            return candidates[0]
        split_variants = [split]
        source_split = meta.get('source_split')
        if source_split and source_split not in split_variants:
            split_variants.append(str(source_split))
        if self.continuous_cross_split and split in {'train', 'val'}:
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
        if 'event_t0_t1' not in folders:
            folders.append('event_t0_t1')
        candidates = []
        for split_variant in split_variants:
            for folder in folders:
                candidates.extend([
                    self.event_root / folder / f'event_{self.event_bins}' / split_variant / seq / f'{event_idx:06d}.npy',
                    self.event_root / folder / split_variant / seq / f'{event_idx:06d}.npy',
                    self.event_root / 'events' / folder / f'event_{self.event_bins}' / split_variant / seq / f'{event_idx:06d}.npy',
                    self.event_root / 'events' / folder / split_variant / seq / f'{event_idx:06d}.npy',
                ])
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    @staticmethod
    def _align_event_to_image(event: np.ndarray, image: np.ndarray, path: Path) -> np.ndarray:
        if event.ndim != 3:
            raise ValueError(f"Expected event voxel to be CxHxW, got {event.shape} from {path}")
        event = np.nan_to_num(event, nan=0.0, posinf=0.0, neginf=0.0)
        image_h, image_w = image.shape[:2]
        _, event_h, event_w = event.shape
        if event_h < image_h or event_w < image_w:
            raise ValueError(
                f"Event voxel is smaller than image for {path}: event={event.shape}, image={image.shape}"
            )
        return event[:, :image_h, :image_w]

    @staticmethod
    def _resize_remap_event(event: np.ndarray, size: Tuple[int, int], pixel_remap: np.ndarray) -> np.ndarray:
        rescaled_h, rescaled_w = size
        resized = np.stack([
            cv2.resize(channel, (rescaled_w, rescaled_h), interpolation=cv2.INTER_LINEAR)
            for channel in event
        ], axis=0)
        remapped = np.stack([
            cv2.remap(channel, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_LINEAR)
            for channel in resized
        ], axis=0)
        remapped = remapped.astype(np.float32)
        return np.nan_to_num(remapped, nan=0.0, posinf=0.0, neginf=0.0)

    def _process_instance(self, instance: dict):
        if instance is None:
            return None

        image, depth, depth_mask, intrinsics = instance['image'], instance['depth'], instance['depth_mask'], instance['intrinsics']
        validity_semantic_mask = instance.get('validity_semantic_mask')

        raw_height, raw_width = image.shape[:2]
        raw_horizontal, raw_vertical = abs(1.0 / intrinsics[0, 0]), abs(1.0 / intrinsics[1, 1])
        raw_pixel_w, raw_pixel_h = raw_horizontal / raw_width, raw_vertical / raw_height
        tgt_width, tgt_height = instance['width'], instance['height']
        tgt_aspect = tgt_width / tgt_height

        # set expected target view field
        tgt_horizontal = min(raw_horizontal, raw_vertical * tgt_aspect)
        tgt_vertical = tgt_horizontal / tgt_aspect

        # set target view direction
        cu, cv = 0.5, 0.5
        direction = utils3d.np.unproject_cv(np.array([[cu, cv]], dtype=np.float32), np.array([1.0], dtype=np.float32), intrinsics=intrinsics)[0]
        R = utils3d.np.rotation_matrix_from_vectors(direction, np.array([0, 0, 1], dtype=np.float32))

        # restrict target view field within the raw view
        corners = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
        corners = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1) @ (np.linalg.inv(intrinsics).T @ R.T)   # corners in viewport's camera plane
        corners = corners[:, :2] / corners[:, 2:3]

        warp_horizontal, warp_vertical = abs(1.0 / intrinsics[0, 0]), abs(1.0 / intrinsics[1, 1])
        for i in range(4):
            intersection, _ = utils3d.np.ray_intersection(
                np.array([0., 0.]), np.array([[tgt_aspect, 1.0], [tgt_aspect, -1.0]]),
                corners[i - 1], corners[i] - corners[i - 1],
            )
            warp_horizontal, warp_vertical = min(warp_horizontal, 2 * np.abs(intersection[:, 0]).min()), min(warp_vertical, 2 * np.abs(intersection[:, 1]).min())
        tgt_horizontal, tgt_vertical = min(tgt_horizontal, warp_horizontal), min(tgt_vertical, warp_vertical)

        # get target view intrinsics
        fx, fy = 1.0 / tgt_horizontal, 1.0 / tgt_vertical
        tgt_intrinsics = utils3d.np.intrinsics_from_focal_center(fx, fy, 0.5, 0.5).astype(np.float32)

        # do homogeneous transformation with the rotation and intrinsics
        # 4.1 The image and depth is resized first to approximately the same pixel size as the target image with PIL's antialiasing resampling
        tgt_pixel_w, tgt_pixel_h = tgt_horizontal / tgt_width, tgt_vertical / tgt_height        # (should be exactly the same for x and y axes)
        rescaled_w, rescaled_h = int(raw_width * raw_pixel_w / tgt_pixel_w), int(raw_height * raw_pixel_h / tgt_pixel_h)
        image = np.array(Image.fromarray(image).resize((rescaled_w, rescaled_h), Image.Resampling.LANCZOS))

        depth, depth_mask = utils3d.np.masked_nearest_resize(depth, mask=depth_mask, size=(rescaled_h, rescaled_w))
        distance = norm3d(utils3d.np.depth_map_to_point_map(depth, intrinsics=intrinsics))
        validity_semantic_mask = cv2.resize(validity_semantic_mask, (rescaled_w, rescaled_h), interpolation=cv2.INTER_NEAREST) if validity_semantic_mask is not None else None

        # 4.2 calculate homography warping
        transform = intrinsics @ np.linalg.inv(R) @ np.linalg.inv(tgt_intrinsics)
        uv_tgt = utils3d.np.uv_map(tgt_height, tgt_width)
        pts = np.concatenate([uv_tgt, np.ones((tgt_height, tgt_width, 1), dtype=np.float32)], axis=-1) @ transform.T
        uv_remap = pts[:, :, :2] / (pts[:, :, 2:3] + 1e-12)
        pixel_remap = utils3d.np.uv_to_pixel(uv_remap, (rescaled_h, rescaled_w)).astype(np.float32)

        tgt_image = cv2.remap(image, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_LINEAR)
        tgt_distance = cv2.remap(distance, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST)
        tgt_events = {}
        if self.requires_events:
            tgt_events = {
                key: self._resize_remap_event(instance[key], (rescaled_h, rescaled_w), pixel_remap)
                for key in self._event_keys()
            }
        tgt_ray_length = utils3d.np.unproject_cv(uv_tgt, np.ones_like(uv_tgt[:, :, 0]), intrinsics=tgt_intrinsics)
        tgt_ray_length = (tgt_ray_length[:, :, 0] ** 2 + tgt_ray_length[:, :, 1] ** 2 + tgt_ray_length[:, :, 2] ** 2) ** 0.5
        tgt_depth = tgt_distance / (tgt_ray_length + 1e-12)
        tgt_depth_mask = cv2.remap(depth_mask.astype(np.uint8), pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) > 0
        tgt_validity_semantic_mask = cv2.remap(validity_semantic_mask, pixel_remap[:, :, 0], pixel_remap[:, :, 1], cv2.INTER_NEAREST) if validity_semantic_mask is not None else None

        # Preserve the upstream scene-relative cutoff for old benchmarks.
        # An explicit metric max_depth takes precedence and is applied below.
        if self.max_depth is None:
            max_depth = np.nanquantile(
                np.where(tgt_depth_mask, tgt_depth, np.nan), 0.01
            ) * self.drop_max_depth
            tgt_depth_mask &= tgt_depth <= max_depth

        if self.depth_unit is not None:
            tgt_depth *= self.depth_unit
        tgt_depth_mask &= np.isfinite(tgt_depth) & (tgt_depth > self.min_depth)
        if self.max_depth is not None:
            tgt_depth_mask &= tgt_depth <= self.max_depth
        if tgt_validity_semantic_mask is not None:
            for semantic_id in self.invalid_semantic_ids:
                tgt_depth_mask &= tgt_validity_semantic_mask != semantic_id
        tgt_depth = np.nan_to_num(tgt_depth, nan=0.0)

        instance.update({
            'image': torch.from_numpy(tgt_image.astype(np.float32) / 255.0).permute(2, 0, 1),
            'depth': torch.from_numpy(tgt_depth).float(),
            'depth_mask': torch.from_numpy(tgt_depth_mask).bool(),
            'intrinsics': torch.from_numpy(tgt_intrinsics).float(),
            'is_metric': self.depth_unit is not None,
            'depth_metric_ranges': self.depth_metric_ranges,
        })
        for key, value in tgt_events.items():
            instance[key] = torch.from_numpy(value).float()

        instance = {k: v for k, v in instance.items() if v is not None}

        return instance

    def start(self):
        self.pipeline.start()

    def stop(self):
        self.pipeline.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def get(self):
        return self.pipeline.get()
