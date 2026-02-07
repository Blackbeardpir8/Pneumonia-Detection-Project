# ml/preprocessing/pseudocolor.py

import cv2
import numpy as np

def apply_pseudocolor(image, colormap=cv2.COLORMAP_JET):
    """
    Apply pseudo-color mapping to a grayscale image.
    Returns a 3-channel color image.
    """
    if len(image.shape) != 2:
        raise ValueError("Pseudo-color expects a grayscale image")

    # OpenCV expects uint8 for colormaps
    if image.dtype != np.uint8:
        image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    else:
        image_uint8 = image

    color_image = cv2.applyColorMap(image_uint8, colormap)
    return color_image

# to run this file:
# (venv) D:\pneumonia-detection-project\ml> python pseudocolor.py
