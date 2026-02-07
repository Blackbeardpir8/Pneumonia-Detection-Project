# ml/training/dataset.py

import os
import torch
from torch.utils.data import Dataset

from preprocessing.preprocess import preprocess


class ChestXrayDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir example:
        data/raw/chest_xray/train
        """
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

        # Folder names must match dataset
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
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image_tensor = preprocess(image_path)

        # REMOVE batch dimension (first dim)
        image_tensor = image_tensor.squeeze(0)

        return image_tensor, torch.tensor(label, dtype=torch.long)

# to run this file:
# (venv) D:\pneumonia-detection-project\ml> python dataset.py
