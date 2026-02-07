# ml/training/test_dataset.py

from training.dataset import ChestXrayDataset

dataset_path = "data/raw/kaggle/train"

dataset = ChestXrayDataset(dataset_path)

print("Total samples:", len(dataset))

image, label = dataset[0]

print("Image shape:", image.shape)
print("Label:", label)

# to run this file:
# (venv) D:\pneumonia-detection-project\ml> python -m training.test_dataset


