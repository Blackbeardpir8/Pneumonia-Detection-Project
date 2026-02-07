# ml/inference_test.py

from preprocessing.preprocess import preprocess

image_path = "data/raw/kaggle/train/PNEUMONIA/person9_bacteria_38.jpeg"

tensor = preprocess(image_path)

print("Tensor shape:", tensor.shape)
print("Min value:", tensor.min().item())
print("Max value:", tensor.max().item())

# to run this file:
# (venv) D:\pneumonia-detection-project\ml> python inference_test.py
