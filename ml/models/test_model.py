# ml/models/test_model.py

import torch
from models.baseline_cnn import BaselineCNN

model = BaselineCNN()

dummy_input = torch.randn(8, 1, 224, 224)

output = model(dummy_input)

print("Output shape:", output.shape)

# to run this file:
# (venv) D:\pneumonia-detection-project\ml> python -m models.test_model

