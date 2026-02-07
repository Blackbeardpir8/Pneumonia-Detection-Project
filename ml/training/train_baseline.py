# ml/training/train_baseline.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from training.dataloader import get_dataloader
from models.baseline_cnn import BaselineCNN


# ---------------- CONFIG ----------------
TRAIN_DIR = "data/raw/kaggle/train"
VAL_DIR = "data/raw/kaggle/val"  # if not present, we will fix later

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return running_loss / len(loader), accuracy


def main():
    print("Using device:", DEVICE)

    train_loader = get_dataloader(TRAIN_DIR, batch_size=BATCH_SIZE, shuffle=True)

    # OPTIONAL validation
    try:
        val_loader = get_dataloader(VAL_DIR, batch_size=BATCH_SIZE, shuffle=False)
    except:
        val_loader = None
        print("Validation folder not found. Skipping validation.")

    model = BaselineCNN().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Train Loss: {train_loss:.4f}")

        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion)
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), "checkpoints/baseline_grayscale.pth")
    print("\nModel saved to checkpoints/baseline_grayscale.pth")


if __name__ == "__main__":
    main()


# to run this 
# (venv) D:\pneumonia-detection-project\ml> python -m training.train_baseline
