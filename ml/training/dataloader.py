# ml/training/dataloader.py

from torch.utils.data import DataLoader
from training.dataset import ChestXrayDataset


def get_dataloader(data_dir, batch_size=16, shuffle=True):
    dataset = ChestXrayDataset(data_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Windows-safe
    )

    return dataloader
