# ml/inference_test.py

from preprocessing.preprocess import preprocess

image_path = "data/raw/kaggle/train/PNEUMONIA/person1_virus_6.jpeg"

tensor = preprocess(image_path)

print("Tensor shape:", tensor.shape)
print("Min value:", tensor.min().item())
print("Max value:", tensor.max().item())
