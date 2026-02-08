# ml/training/dataset_clahe.py

import os
import torch
from torch.utils.data import Dataset

from preprocessing.preprocess import preprocess_clahe


class ChestXrayDatasetCLAHE(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

        classes = {
            "NORMAL": 0,
            "PNEUMONIA": 1
        }

        for class_name, label in classes.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_tensor = preprocess_clahe(self.image_paths[idx])
        image_tensor = image_tensor.squeeze(0)  # remove batch dim
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image_tensor, label
