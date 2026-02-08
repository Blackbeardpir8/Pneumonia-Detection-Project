# ml/preprocessing/preprocess.py

import cv2
import numpy as np
import torch
from preprocessing.clahe import apply_clahe
from preprocessing.pseudocolor import apply_pseudocolor

IMG_SIZE = 224

def load_image(image_path):
    """
    Load chest X-ray image in grayscale
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or invalid image path")
    return image


def resize_image(image):
    """
    Resize image to 224x224
    """
    return cv2.resize(image, (IMG_SIZE, IMG_SIZE))


def normalize_image(image):
    """
    Normalize pixel values to range [0, 1]
    """
    image = image.astype(np.float32)
    image /= 255.0
    return image


def to_tensor(image):
    """
    Convert numpy image to PyTorch tensor
    Shape: (1, 1, 224, 224)
    """
    tensor = torch.from_numpy(image)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor


def preprocess(image_path):
    """
    Complete preprocessing pipeline
    """
    image = load_image(image_path)
    image = resize_image(image)
    image = normalize_image(image)
    tensor = to_tensor(image)
    return tensor
def preprocess_clahe(image_path):
    """
    Preprocessing pipeline with CLAHE
    """
    image = load_image(image_path)       # grayscale
    image = resize_image(image)
    image = apply_clahe(image)            # 🔥 CLAHE
    image = normalize_image(image)
    tensor = to_tensor(image)
    return tensor


def preprocess_pseudocolor(image_path):
    """
    Preprocessing pipeline with pseudo-color mapping
    """
    image = load_image(image_path)        # grayscale
    image = resize_image(image)
    image = apply_pseudocolor(image)      # pseudo-color (3-channel)
    image = normalize_image(image)        # normalize to [0,1]
    tensor = to_tensor_rgb(image)          # will produce (1, 3, 224, 224)
    return tensor

def to_tensor_rgb(image):
    """
    Convert HWC RGB image to CHW tensor
    Input: (224, 224, 3)
    Output: (1, 3, 224, 224)
    """
    image = image.astype("float32")

    # HWC -> CHW
    image = image.transpose(2, 0, 1)

    tensor = torch.from_numpy(image)
    tensor = tensor.unsqueeze(0)
    return tensor


# to use this file:
# from preprocessing.preprocess import preprocess_clahe
# tensor = preprocess_clahe("path/to/image.jpg")
