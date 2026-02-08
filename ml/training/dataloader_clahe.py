# ml/training/dataloader_clahe.py

from torch.utils.data import DataLoader
from training.dataset_clahe import ChestXrayDatasetCLAHE


def get_dataloader_clahe(data_dir, batch_size=16, shuffle=True):
    dataset = ChestXrayDatasetCLAHE(data_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Windows safe
    )
