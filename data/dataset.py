"""Dataset class for Qaida Whisper training."""

import json
from torch.utils.data import Dataset


class QaidaWhisperDataset(Dataset):
    """Lazy-loading dataset that returns entries for the collator to process."""

    def __init__(self, entries):
        """
        Args:
            entries: list of dicts with audio_path, arabic_text, etc.
        """
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]


def load_dataset_splits(dataset_path, split_path):
    """Load unified dataset and split into train/val."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_entries = json.load(f)

    with open(split_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    train_entries = [all_entries[i] for i in split_info["train_indices"]]
    val_entries = [all_entries[i] for i in split_info["val_indices"]]

    return (
        QaidaWhisperDataset(train_entries),
        QaidaWhisperDataset(val_entries),
    )
