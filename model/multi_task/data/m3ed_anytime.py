from pathlib import Path

import torch.utils.data as data

from .m3ed import (
    load_depth,
    load_event,
    load_rgb,
    load_semseg,
    manifest_samples,
    numeric_stem_sets,
)


class M3EDAnytime(data.Dataset):
    """M3ED temporal samples shared by query and future event-refine models.

    One step represents 40 ms:
    RGB[N] + event[N-1] + event[N] -> semantic/depth[N+1].
    """

    frame_interval_ms = 40

    def __init__(
        self,
        root=None,
        split='train',
        transform=None,
        retname=True,
        do_semseg=True,
        do_depth=True,
        time_window_steps=1,
        event_bins=20,
        semantic_root=None,
    ):
        if split not in {'train', 'val'}:
            raise ValueError(f'M3EDAnytime only supports train/val, got {split!r}')
        if int(time_window_steps) not in {1, 2}:
            raise ValueError(f'M3EDAnytime supports one or two 40 ms steps, got {time_window_steps}')
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.retname = retname
        self.do_semseg = do_semseg
        self.do_depth = do_depth
        self.time_window_steps = int(time_window_steps)
        self.event_bins = int(event_bins)

        records = manifest_samples(
            self.root,
            split,
            do_semseg=do_semseg,
            do_depth=do_depth,
            semantic_root=Path(semantic_root) if semantic_root else None,
        )
        if records is None:
            raise FileNotFoundError(f'M3ED temporal loading requires {self.root / f".{split}.txt"}')
        dataset_root = self.root.parent
        event_root = dataset_root / 'event_t0_t1' / f'event_{self.event_bins}' / split
        if not event_root.is_dir():
            event_root = dataset_root / 'event_t0' / 'all'
        image_root = dataset_root / 'image' / 'all'
        image_ids = numeric_stem_sets(image_root, {'.png', '.jpg', '.jpeg'})
        event_ids = numeric_stem_sets(event_root, {'.npy'})
        self.samples = []
        for record in records:
            target_id = record['id']
            sequence, target_stem = target_id.split('/')
            target_index = int(target_stem)
            input_index = target_index - self.time_window_steps
            if input_index < 1:
                continue
            input_id = f'{sequence}/{input_index:06d}'
            input_image = image_root / sequence / f'{input_index:06d}.png'
            event_paths = [
                event_root / sequence / f'{event_index:06d}.npy'
                for event_index in range(input_index - 1, target_index)
            ]
            if input_index not in image_ids.get(sequence, set()):
                continue
            if any(index not in event_ids.get(sequence, set()) for index in range(input_index - 1, target_index)):
                continue
            self.samples.append({
                'id': target_id,
                'input_id': input_id,
                'image': input_image,
                'semseg': record['semseg'],
                'depth': record['depth'],
                'events': event_paths,
            })

        if not self.samples:
            raise RuntimeError(f'No aligned M3ED temporal samples found under {self.root} ({split})')
        duration = self.frame_interval_ms * self.time_window_steps
        print(
            f'[{type(self).__name__}] split={split} samples={len(self.samples)} duration={duration}ms '
            f'mapping=RGB[N-{self.time_window_steps}]+adjacent-events->GT[N]'
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]
        image = load_rgb(record['image'])
        events = [load_event(path, image.shape[:2], self.event_bins) for path in record['events']]
        sample = {
            'image': image,
            'event_before': events[0],
            'event': events[1],
        }
        if self.time_window_steps == 2:
            sample['event_after'] = events[2]
        if self.do_semseg:
            sample['semseg'] = load_semseg(record['semseg'], image.shape[:2])
        if self.do_depth:
            sample['depth'] = load_depth(record['depth'], image.shape[:2])
        if self.retname:
            sample['meta'] = {
                'img_name': record['id'],
                'input_name': record['input_id'],
                'img_size': image.shape[:2],
            }
        if self.transform is not None:
            sample = self.transform(sample)
        sample['index'] = index
        return sample

    def __str__(self):
        return f'M3EDAnytime(split={self.split}, steps={self.time_window_steps})'
