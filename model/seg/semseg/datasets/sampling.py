"""Repeat training samples to pad the dataset length to a full batch."""
from torch.utils.data import Dataset


class RepeatDataset(Dataset):
    """Repeat the source dataset cyclically to reach the requested training length."""

    def __init__(self, original_dataset: Dataset, target_length: int):
        self.original_dataset = original_dataset
        self.original_length = len(original_dataset)
        self.target_length = int(target_length)
        self.frame_interval_ms = original_dataset.frame_interval_ms
        self.supervision_offsets_ms = original_dataset.supervision_offsets_ms
        self.SEGMENTATION_CONFIGS = original_dataset.SEGMENTATION_CONFIGS
        self.n_classes = original_dataset.n_classes
        self.ignore_label = original_dataset.ignore_label

    def __len__(self):
        return self.target_length

    def __getitem__(self, index):
        return self.original_dataset[index % self.original_length]
