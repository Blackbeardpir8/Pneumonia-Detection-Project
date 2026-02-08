# ml/evaluation/evaluate_baseline.py

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from training.dataloader import get_dataloader
from models.baseline_cnn import BaselineCNN


# ---------------- CONFIG ----------------
TEST_DIR = "data/raw/kaggle/val"
MODEL_PATH = "checkpoints/baseline_grayscale.pth"
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------


def main():
    print("Using device:", DEVICE)

    # Load test data
    test_loader = get_dataloader(TEST_DIR, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = BaselineCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long().squeeze(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n--- BASELINE MODEL EVALUATION ---")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()

# to run this file:
# (venv) D:\pneumonia-detection-project\ml> python -m evaluation.evaluate_baseline

