# ml/visualize_pseudocolor.py

import os
import cv2
import matplotlib.pyplot as plt

from preprocessing.preprocess import load_image
from preprocessing.pseudocolor import apply_pseudocolor

# CHANGE THIS PATH TO A VALID IMAGE (same one you used for CLAHE)
image_path = "data/raw/kaggle/train/PNEUMONIA/person9_bacteria_38.jpeg"

print("File exists:", os.path.exists(image_path))

# Load grayscale image
gray = load_image(image_path)

# Apply pseudo-color
pseudo = apply_pseudocolor(gray, colormap=cv2.COLORMAP_JET)

# Convert BGR (OpenCV) to RGB for matplotlib
pseudo_rgb = cv2.cvtColor(pseudo, cv2.COLOR_BGR2RGB)

# Plot side-by-side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Grayscale")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Pseudo-Color (JET)")
plt.imshow(pseudo_rgb)
plt.axis("off")

plt.tight_layout()
plt.show()


# to run this file:
# (venv) D:\pneumonia-detection-project\ml> python visualize_pseudocolor.py

