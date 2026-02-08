# ml/training/train_pseudocolor.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from training.dataloader_pseudocolor import get_dataloader_pseudocolor
from models.baseline_cnn_rgb import BaselineCNN_RGB


TRAIN_DIR = "data/raw/kaggle/train"
VAL_DIR = "data/raw/kaggle/val"

BATCH_SIZE = 16
EPOCHS = 3
LR = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(DEVICE)
        y = y.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.float().unsqueeze(1).to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total


def main():
    print("Using device:", DEVICE)

    train_loader = get_dataloader_pseudocolor(TRAIN_DIR, BATCH_SIZE, True)
    val_loader = get_dataloader_pseudocolor(VAL_DIR, BATCH_SIZE, False)

    model = BaselineCNN_RGB().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for e in range(EPOCHS):
        print(f"\nEpoch {e+1}/{EPOCHS}")
        tl = train_one_epoch(model, train_loader, criterion, optimizer)
        vl, va = validate(model, val_loader, criterion)
        print(f"Train Loss: {tl:.4f}")
        print(f"Val Loss: {vl:.4f} | Val Acc: {va:.4f}")

    torch.save(model.state_dict(), "checkpoints/pseudocolor_model.pth")
    print("\nPseudo-color model saved to checkpoints/pseudocolor_model.pth")


if __name__ == "__main__":
    main()


# to run this file:
# (venv) D:\pneumonia-detection-project\ml> python -m training.train_pseudocolor
