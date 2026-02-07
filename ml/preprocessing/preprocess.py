# ml/preprocessing/preprocess.py

import cv2
import numpy as np
import torch

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
