# ml/training/test_dataloader.py

from training.dataloader import get_dataloader

loader = get_dataloader("data/raw/kaggle/train", batch_size=8)

images, labels = next(iter(loader))

print("Batch image shape:", images.shape)
print("Batch labels:", labels)

# to run this file:
# (venv) D:\pneumonia-detection-project\ml> python -m training.test_dataloader

