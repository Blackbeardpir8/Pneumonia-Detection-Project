# ml/preprocessing/clahe.py

import cv2
import numpy as np

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to a grayscale image
    """
    if len(image.shape) != 2:
        raise ValueError("CLAHE expects a grayscale image")

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    enhanced_image = clahe.apply(image)
    return enhanced_image
