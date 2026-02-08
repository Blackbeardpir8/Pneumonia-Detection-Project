# ml/visualize_clahe.py

import cv2
import matplotlib.pyplot as plt

from preprocessing.clahe import apply_clahe
from preprocessing.preprocess import load_image

# CHANGE PATH IF NEEDED
image_path = "data/raw/kaggle/train/PNEUMONIA/person1_virus_6.jpeg"

# Load original grayscale image
original = load_image(image_path)

# Apply CLAHE
clahe_image = apply_clahe(original)

# Plot side-by-side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Grayscale")
plt.imshow(original, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("CLAHE Enhanced")
plt.imshow(clahe_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()


# to run this file:
# (venv) D:\pneumonia-detection-project\ml> python visualize_clahe.py