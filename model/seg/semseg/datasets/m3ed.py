from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import io

from .dsec import DSEC, _load_event_array


class M3ED(Dataset):
    """M3ED Falcon/Spot: same-time RGB training or one-step 40 ms event propagation."""

    SEGMENTATION_CONFIGS = DSEC.SEGMENTATION_CONFIGS

    def __init__(
        self,
        root: str,
        split: str = "train",
        n_classes: int = 11,
        transform=None,
        modals=("img",),
        case=None,
        duration: int = 40,
        dataset_type: str = "m3ed",
        sample_gap: int = 0,
    ) -> None:
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError(f"M3ED only supports train/val, got {split!r}")
        if int(duration) not in {0, 40} or dataset_type != "m3ed":
            raise ValueError("M3ED requires duration=0 or 40, TYPE=m3ed")

        self.root = Path(root)
        self.split = split
        self.n_classes = int(n_classes)
        self.ignore_label = 255
        self.transform = transform
        self.modals = tuple(modals)
        self.case = case
        self.duration = int(duration)
        self.frame_interval_ms = 40
        self.event_interval_ms = 40
        self.time_window = self.duration // 40
        self.index_window = self.time_window
        self.supervision_offsets_ms = [self.duration]

        event_root = self.root / "event_t0_t1" / "event_20" / split
        context_root = self.root / "event_t-1_t0" / "event_20" / split
        if not event_root.is_dir():
            event_root = self.root / "event_t0" / split
            context_root = event_root
        self.samples = []
        if self.duration == 0:
            for rgb_path in sorted((self.root / 'image' / split).glob('*/*.png')):
                sequence, t0 = rgb_path.parent.name, int(rgb_path.stem)
                label_path = self.root / 'semantic' / split / sequence / rgb_path.name
                if label_path.is_file():
                    self.samples.append((sequence, t0, rgb_path, None, None, label_path))
        else:
            for event_path in sorted(event_root.glob("*/*.npy")):
                sequence = event_path.parent.name
                t0 = int(event_path.stem)
                context_path = context_root / sequence / f"{t0 - 1:06d}.npy"
                rgb_path = self.root / "image" / split / sequence / f"{t0:06d}.png"
                label_path = self.root / "semantic" / split / sequence / f"{t0 + 1:06d}.png"
                if context_path.is_file() and rgb_path.is_file() and label_path.is_file():
                    self.samples.append((sequence, t0, rgb_path, event_path, context_path, label_path))

        step = int(sample_gap) + 1
        if step > 1:
            first_index = {}
            for sequence, t0, *_ in self.samples:
                first_index.setdefault(sequence, t0)
            self.samples = [
                sample for sample in self.samples
                if (sample[1] - first_index[sample[0]]) % step == 0
            ]
        self.files = [str(sample[-1]) for sample in self.samples]
        print(
            f"[M3ED] split={split} samples={len(self.samples)} "
            f"horizon={self.duration}ms"
        )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_event(path: Path) -> torch.Tensor:
        event = _load_event_array(path)
        if event.ndim != 3 or event.shape[0] != 20:
            raise ValueError(f"Expected a 20-bin event voxel, got {event.shape} from {path}")
        return torch.from_numpy(event)

    def __getitem__(self, index: int):
        sequence, t0, rgb_path, event_path, context_path, label_path = self.samples[index]
        sample = {
            "img": io.read_image(str(rgb_path))[:3],
            "mask": io.read_image(str(label_path))[0].unsqueeze(0),
        }
        if self.duration:
            sample.update(event=self._load_event(event_path), event_before=self._load_event(context_path))
        if self.transform:
            sample = self.transform(sample)
        label = sample.pop("mask").squeeze(0).long()
        inputs = [sample[key] for key in self.modals]
        if self.duration:
            inputs.extend([sample["event"], sample["event_before"]])
        return sequence, f"{t0:06d}", inputs, [label]
