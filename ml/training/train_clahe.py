# ml/training/train_clahe.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from training.dataloader_clahe import get_dataloader_clahe
from models.baseline_cnn import BaselineCNN


TRAIN_DIR = "data/raw/kaggle/train"
VAL_DIR = "data/raw/kaggle/val"

BATCH_SIZE = 16
EPOCHS = 3
LR = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return total_loss / len(loader), acc


def main():
    print("Using device:", DEVICE)

    train_loader = get_dataloader_clahe(TRAIN_DIR, BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader_clahe(VAL_DIR, BATCH_SIZE, shuffle=False)

    model = BaselineCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "checkpoints/clahe_model.pth")
    print("\nCLAHE model saved to checkpoints/clahe_model.pth")


if __name__ == "__main__":
    main()

