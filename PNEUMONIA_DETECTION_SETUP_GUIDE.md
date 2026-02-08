# 🫁 Pneumonia Detection Project
## ML Setup & Run Guide

This document provides comprehensive instructions for training and evaluating all models (Grayscale, CLAHE, and Pseudo-color) in the Pneumonia Detection project.

> **Important:** Follow these steps sequentially. Do not skip any steps.

---

## 📋 Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Project Setup](#2-project-setup)
3. [Virtual Environment](#3-virtual-environment)
4. [Dependencies Installation](#4-dependencies-installation)
5. [Dataset Configuration](#5-dataset-configuration)
6. [Preprocessing Verification](#6-preprocessing-verification)
7. [Image Enhancement Visualization](#7-image-enhancement-visualization)
8. [Component Testing](#8-component-testing)
9. [Model Training](#9-model-training)
10. [Model Evaluation](#10-model-evaluation)
11. [Expected Outputs](#11-expected-outputs)

---

## 1️⃣ System Requirements

| Component | Requirement |
|-----------|-------------|
| **Operating System** | Windows / Linux / macOS |
| **Python Version** | 3.9 or 3.10 |
| **RAM** | 8GB minimum (CPU training supported) |
| **Disk Space** | ~2-3 GB free |

---

## 2️⃣ Project Setup

Place the project in your desired location, for example:
```
D:\pneumonia-detection-project
```

### Expected Project Structure

```
pneumonia-detection-project/
├── ml/
├── backend/
├── frontend/
└── requirements.txt
```

---

## 3️⃣ Virtual Environment

### Create Virtual Environment

Navigate to the project root directory and run:

```bash
python -m venv venv
```

### Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### Verify Installation

```bash
python --version
```

---

## 4️⃣ Dependencies Installation

### Install from Requirements File

```bash
pip install -r requirements.txt
```

### Manual Installation (if requirements.txt is missing)

```bash
pip install torch torchvision torchaudio
pip install opencv-python numpy pillow matplotlib scikit-learn tqdm
```

---

## 5️⃣ Dataset Configuration

### Download Dataset

Download the [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

### Required Directory Structure

> ⚠️ **Critical:** The dataset must be placed exactly as shown below.

```
ml/data/raw/kaggle/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

> ⚠️ **Note:** The `val/` directory must NOT be empty. Use ~10-15% of training images if the validation set is not provided.

---

## 6️⃣ Preprocessing Verification

### Navigate to ML Directory

```bash
cd ml
```

### Test Grayscale Preprocessing

```bash
python inference_test.py
```

**Expected Output:**
```
Tensor shape: [1, 1, 224, 224]
```

---

## 7️⃣ Image Enhancement Visualization

### CLAHE Visualization (Optional but Recommended)

```bash
python visualize_clahe.py
```

### Pseudo-Color Visualization

```bash
python visualize_pseudocolor.py
```

You should see side-by-side comparison images demonstrating the enhancement techniques.

---

## 8️⃣ Component Testing

### Test Dataset Loader

```bash
python -m training.test_dataset
```

**Expected Output:**
```
Total samples: > 0
Image shape: [1, 224, 224]
```

### Test DataLoader

```bash
python -m training.test_dataloader
```

**Expected Output:**
```
Batch image shape: [8, 1, 224, 224]
```

### Test CNN Forward Pass

```bash
python -m models.test_model
```

**Expected Output:**
```
Output shape: [8, 1]
```

---

## 9️⃣ Model Training

> **Important:** Execute these training commands in the order presented.

### Step 1: Train Baseline (Grayscale) Model

```bash
python -m training.train_baseline
```

**Output Location:** `checkpoints/baseline_grayscale.pth`

### Step 2: Train CLAHE Model

```bash
python -m training.train_clahe
```

**Output Location:** `checkpoints/clahe_model.pth`

### Step 3: Train Pseudo-Color Model

```bash
python -m training.train_pseudocolor
```

**Output Location:** `checkpoints/pseudocolor_model.pth`

---

## 🔟 Model Evaluation

### Evaluate Baseline Model

```bash
python -m evaluation.evaluate_baseline
```

> 💡 **Tip:** Save the printed metrics for comparison.

### Evaluate CLAHE Model

```bash
python -m evaluation.evaluate_clahe
```

### Evaluate Pseudo-Color Model

```bash
python -m evaluation.evaluate_pseudocolor
```

---

## 1️⃣1️⃣ Expected Outputs

Upon successful completion, your checkpoint directory should contain:

```
ml/checkpoints/
├── baseline_grayscale.pth
├── clahe_model.pth
└── pseudocolor_model.pth
```

---

## 🎯 Summary

You have successfully:

✅ Set up the development environment  
✅ Configured the dataset  
✅ Verified all preprocessing pipelines  
✅ Trained three distinct models (Grayscale, CLAHE, Pseudo-color)  
✅ Evaluated model performance  

---

## 📞 Support

For issues or questions, please refer to the project documentation or submit an issue in the project repository.

---

**Last Updated:** February 2026  
**Project:** Pneumonia Detection using Deep Learning
