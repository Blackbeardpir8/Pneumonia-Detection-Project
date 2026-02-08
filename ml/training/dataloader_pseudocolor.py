# ml/training/dataloader_pseudocolor.py

from torch.utils.data import DataLoader
from training.dataset_pseudocolor import ChestXrayDatasetPseudoColor


def get_dataloader_pseudocolor(data_dir, batch_size=16, shuffle=True):
    dataset = ChestXrayDatasetPseudoColor(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Windows-safe
    )

