"""Time-aligned DSEC and SHF-DSEC RGB/event/semantic endpoint samples."""

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io


def _is_rank0() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def _event_paths(root: Path, frame_idx: int) -> Tuple[Path, Path]:
    stem = f"{frame_idx:06d}"
    return root / f"{stem}.npy", root / f"{stem}.npz"


def _load_event_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    with np.load(path, allow_pickle=False) as archive:
        return archive["event"]


class DSEC(Dataset):
    """Load temporally aligned RGB images, event voxels, and semantic labels."""

    # ------------------------------------------------------------------
    # Semantic label definitions
    # ------------------------------------------------------------------
    SEGMENTATION_CONFIGS = {
        11: {
            "CLASSES": [
                "background", "building", "fence", "person", "pole", "road",
                "sidewalk", "vegetation", "car", "wall", "traffic sign",
            ],
            "PALETTE": {
                0: [0, 0, 0], 1: [70, 70, 70], 2: [190, 153, 153],
                3: [220, 20, 60], 4: [153, 153, 153], 5: [128, 64, 128],
                6: [244, 35, 232], 7: [107, 142, 35], 8: [0, 0, 142],
                9: [102, 102, 156], 10: [220, 220, 0],
            },
        },
    }
    CARLA_SEGMENTATION_CONFIGS = {
        11: {
            "CLASSES": [
                "background",
                "building",
                "fence",
                "person",
                "pole",
                "road",
                "sidewalk",
                "vegetation",
                "car",
                "road lines",
                "traffic sign",
            ],
            "PALETTE": {
                0: [0, 0, 0],
                1: [70, 70, 70],
                2: [190, 153, 153],
                3: [220, 20, 60],
                4: [153, 153, 153],
                5: [128, 64, 128],
                6: [244, 35, 232],
                7: [107, 142, 35],
                8: [0, 0, 142],
                9: [157, 234, 50],
                10: [220, 220, 0],
            },
        }
    }

    def __init__(
        self, root: str = 'data/dsec', split: str = 'train', n_classes: int = 11,
        transform=None, modals=('img',), case=None,
        duration: int = None, dataset_type: str = None, sample_gap: int = 0,
        frame_interval_ms: int = 50, frame_index_stride: int = 1,
        sequence_names=None, exclude_sequence_names=None,
        anchor_modulo: int = 0, anchor_remainders=None,
    ) -> None:
        super().__init__()
        self.root = root
        self.root_path = Path(root)
        context_layout = self.root_path / 'event_t-1_t0' / 'context_indexing.txt'
        self.aligned_context_at_start = (
            context_layout.is_file()
            and context_layout.read_text(encoding='utf-8').strip() == 'aligned_t0'
        )
        self.split = split
        assert split in ['train', 'val', 'test']
        self.transform = transform
        self.n_classes = n_classes
        self.ignore_label = 255
        self.modals = tuple(modals)
        self.case = case
        self.duration = int(duration or 0)
        self.frame_interval_ms = int(frame_interval_ms)
        if self.frame_interval_ms <= 0:
            raise ValueError(f'frame_interval_ms must be positive, got {self.frame_interval_ms}')
        self.frame_index_stride = int(frame_index_stride)
        if self.frame_index_stride <= 0:
            raise ValueError(f'frame_index_stride must be positive, got {self.frame_index_stride}')
        if self.duration < 0 or self.duration % self.frame_interval_ms != 0:
            raise ValueError(f'duration must be a non-negative multiple of {self.frame_interval_ms} ms, got {self.duration} ms')
        self.time_window = self.duration // self.frame_interval_ms
        self.index_window = self.time_window * self.frame_index_stride
        self.dataset_type = dataset_type
        if dataset_type not in ('dsec', 'carla', None):
            raise ValueError(f'Unsupported dataset_type for DSEC: {dataset_type!r}')
        if dataset_type == 'carla':
            self.SEGMENTATION_CONFIGS = self.CARLA_SEGMENTATION_CONFIGS
        self.sample_gap = int(sample_gap or 0)
        self.anchor_modulo = int(anchor_modulo or 0)
        self.anchor_remainders = (
            frozenset(int(value) for value in anchor_remainders)
            if anchor_remainders is not None else frozenset()
        )
        if self.anchor_modulo < 0:
            raise ValueError('anchor_modulo cannot be negative')
        if self.anchor_remainders and self.anchor_modulo <= 0:
            raise ValueError('anchor_remainders requires a positive anchor_modulo')
        if any((value < 0 or value >= self.anchor_modulo for value in self.anchor_remainders)):
            raise ValueError('anchor_remainders must be in [0, anchor_modulo)')
        if sequence_names is None:
            self.sequence_names = None
        elif isinstance(sequence_names, str):
            self.sequence_names = (sequence_names,)
        else:
            self.sequence_names = tuple((str(name) for name in sequence_names))
        if exclude_sequence_names is None:
            self.exclude_sequence_names = ()
        elif isinstance(exclude_sequence_names, str):
            self.exclude_sequence_names = (exclude_sequence_names,)
        else:
            self.exclude_sequence_names = tuple((str(name) for name in exclude_sequence_names))
        overlap = set(self.sequence_names or ()) & set(self.exclude_sequence_names)
        if overlap:
            raise ValueError(f'Sequences cannot be both included and excluded: {sorted(overlap)}')
        self.event_interval_ms = self.frame_interval_ms
        self.supervision_offsets_ms = [self.duration]
        self.bin = 20
        if _is_rank0():
            print('Root: ', self.root)
            print('Loading semantic segmentation ground truth.')
            context_mapping = f'S-{self.frame_index_stride}'
            if self.aligned_context_at_start:
                context_mapping = f'aligned_t0 when S=0, otherwise S-{self.frame_index_stride}'
            if self.time_window > 0:
                print(
                    f'[final_mapping] RGB=S, context_event={context_mapping}, '
                    f'forward_events=S..S+{(self.time_window - 1) * self.frame_index_stride}, '
                    f'GT=S+{self.index_window} ({self.duration}ms)'
                )
        semantic_split_root = self.root_path / 'semantic' / split
        files = []
        candidate_count = 0
        complete_count = 0
        seq_dirs = sorted((path for path in semantic_split_root.iterdir() if path.is_dir()))
        available_sequences = {seq_dir.name for seq_dir in seq_dirs}
        if self.sequence_names is not None:
            missing = sorted(set(self.sequence_names) - available_sequences)
            if missing:
                raise FileNotFoundError(f'Unknown {split} sequence(s) under {semantic_split_root}: {missing}')
            selected = set(self.sequence_names)
            seq_dirs = [seq_dir for seq_dir in seq_dirs if seq_dir.name in selected]
            if _is_rank0():
                print(f'[sequence_filter] split={split} sequences={sorted(selected)}')
        if self.exclude_sequence_names:
            missing = sorted(set(self.exclude_sequence_names) - available_sequences)
            if missing:
                raise FileNotFoundError(f'Unknown excluded {split} sequence(s) under {semantic_split_root}: {missing}')
            excluded = set(self.exclude_sequence_names)
            seq_dirs = [seq_dir for seq_dir in seq_dirs if seq_dir.name not in excluded]
            if _is_rank0():
                print(f'[sequence_filter] split={split} excluded={sorted(excluded)}')
        for seq_dir in seq_dirs:
            candidates = sorted(seq_dir.glob('*.png'))
            candidate_count += len(candidates)
            complete = [path for path in candidates if self._has_complete_inputs(path)]
            complete_count += len(complete)
            files.extend((str(path) for path in complete))
        self.files = files
        if self.anchor_remainders:
            before_anchor_filter = len(self.files)
            self.files = [path for path in self.files if self._sample_anchor_index(Path(path)) % self.anchor_modulo in self.anchor_remainders]
            if _is_rank0():
                print(f'[anchor_filter] split={split} modulo={self.anchor_modulo} remainders={sorted(self.anchor_remainders)} kept={len(self.files)}/{before_anchor_filter}')
        if _is_rank0():
            print(f'Found {len(self.files)} {split} images.')
            incomplete_count = candidate_count - complete_count
            if incomplete_count > 0:
                print(f'[causal_filter] split={split} duration={self.duration}ms excluded={incomplete_count} incomplete samples')
        if self.sample_gap > 0:
            self.files = self.files[::self.sample_gap + 1]
            if _is_rank0():
                print(f'[sample_gap] split={split} sample_gap={self.sample_gap} kept={len(self.files)}')

    def __len__(self) -> int:
        return len(self.files)

    # ----------------------------------------------------------------------
    # Dataset directory layout and path resolution
    # ----------------------------------------------------------------------
    def _semantic_event_split_roots(self, folder_name: str) -> Tuple[Path, ...]:
        split_root = self.root_path / folder_name / f"event_{self.bin}" / self.split
        if self.split == "val":
            return split_root, split_root / "val_semantic"
        return (split_root,)

    def _semantic_rgb_path(self, seq_name: str, frame_idx: int) -> Path:
        split_root = self.root_path / "images" / self.split
        if self.split == "val":
            legacy_path = split_root / "val_semantic" / seq_name / f"{frame_idx:06d}.png"
            if legacy_path.is_file():
                return legacy_path
        return split_root / seq_name / f"{frame_idx:06d}.png"



    def _context_event_path(self, seq_name: str, t0_idx: int):
        if t0_idx == 0 and self.aligned_context_at_start:
            return self._semantic_event_path(seq_name, 0, ("event_t-1_t0",))
        previous_idx = t0_idx - self.frame_index_stride
        if previous_idx < 0:
            return None
        return self._semantic_event_path(
            seq_name,
            previous_idx,
            ("event_t-1_t0", "event_t0_t1"),
        )

    def _semantic_event_path(
        self,
        seq_name: str,
        frame_idx: int,
        folder_names: Tuple[str, ...],
    ):
        for folder_name in folder_names:
            for split_root in self._semantic_event_split_roots(folder_name):
                for path in _event_paths(split_root / seq_name, frame_idx):
                    if path.is_file():
                        return path
        return None



    def _event_exists(self, seq_name: str, frame_idx: int, folder_names: Tuple[str, ...]) -> bool:
        return self._semantic_event_path(seq_name, frame_idx, folder_names) is not None

    # ----------------------------------------------------------------------
    # Check sample completeness for each mode.
    # ----------------------------------------------------------------------
    def _has_complete_inputs(self, label_path: Path) -> bool:
        label_idx = int(label_path.stem)
        seq_name = label_path.parent.name
        if self.time_window == 0:
            return self._semantic_rgb_path(seq_name, label_idx).is_file()
        t0_idx = label_idx - self.index_window
        if t0_idx < 0 or not self._semantic_rgb_path(seq_name, t0_idx).is_file():
            return False
        if self._context_event_path(seq_name, t0_idx) is None:
            return False
        event_requirements = [(t0_idx, ('event_t0_t1',))]
        event_requirements.extend(((t0_idx + step * self.frame_index_stride, ('event_t0_t1', 'event_t1_t2')) for step in range(1, self.time_window)))
        if not all((self._event_exists(seq_name, frame_idx, folders) for (frame_idx, folders) in event_requirements)):
            return False
        return True

    def _sample_anchor_index(self, label_path: Path) -> int:
        """Map a candidate semantic endpoint to the RGB/event input anchor S."""
        label_index = int(label_path.stem)
        return label_index - self.index_window if self.time_window else label_index

    def _load_semantic_event_voxel(self, seq_name: str, frame_idx: int, folder_names: Tuple[str, ...]):
        path = self._semantic_event_path(seq_name, frame_idx, folder_names)
        if path is None:
            raise FileNotFoundError(
                f"Missing event voxel: split={self.split} sequence={seq_name} index={frame_idx:06d}"
            )
        return torch.from_numpy(_load_event_array(path)[:, :440])

    def _load_context_event_voxel(self, seq_name: str, t0_idx: int):
        path = self._context_event_path(seq_name, t0_idx)
        if path is None:
            raise FileNotFoundError(
                f"Missing context event: split={self.split} "
                f"sequence={seq_name} t0={t0_idx:06d}"
            )
        return torch.from_numpy(_load_event_array(path)[:, :440])


    # ----------------------------------------------------------------------
    # Construct samples for each mode.
    # ----------------------------------------------------------------------
    def __getitem__(self, index: int):
        lbl_path = Path(self.files[index])
        label_idx = int(lbl_path.stem)
        t0_idx = label_idx if self.time_window == 0 else label_idx - self.index_window
        seq_name = lbl_path.parent.name
        sample = {}
        rgb_path = self._semantic_rgb_path(seq_name, t0_idx)
        img = io.read_image(str(rgb_path))[:3, ...]
        future_event_keys = []
        if self.time_window:
            sample['mask'] = io.read_image(str(lbl_path))[0].unsqueeze(0)[:, :440]
            sample['event'] = self._load_semantic_event_voxel(seq_name, t0_idx, ('event_t0_t1',))
            sample['event_before'] = self._load_context_event_voxel(seq_name, t0_idx)
            for step in range(1, self.time_window):
                key = f'event_future_{step}'
                sample[key] = self._load_semantic_event_voxel(seq_name, t0_idx + step * self.frame_index_stride, ('event_t0_t1', 'event_t1_t2'))
                future_event_keys.append(key)
            (target_h, target_w) = sample['event'].shape[-2:]
        else:
            sample['mask'] = io.read_image(str(lbl_path))[0].unsqueeze(0)
            (target_h, target_w) = sample['mask'].shape[-2:]
        if img.shape[-2:] != (target_h, target_w):
            (h_img, w_img) = img.shape[-2:]
            scale = max(target_h / h_img, target_w / w_img)
            new_h = int(math.ceil(h_img * scale))
            new_w = int(math.ceil(w_img * scale))
            img = TF.resize(img, [new_h, new_w], TF.InterpolationMode.BILINEAR)
            img = img[:, :target_h, :target_w]
        sample['img'] = img
        if self.transform:
            sample = self.transform(sample)
        labels = [self.encode(sample.pop('mask').squeeze().numpy()).long()]
        sample_list = [sample[key] for key in self.modals]
        if self.time_window:
            sample_list.extend([sample['event'], sample['event_before']])
            sample_list.extend((sample[key] for key in future_event_keys))
        seq_idx = f'{t0_idx:06d}' if self.time_window else f'{label_idx:06d}'
        return (seq_name, seq_idx, sample_list, labels)

    def encode(self, label: np.ndarray) -> Tensor:
        return torch.from_numpy(label)


