```bash
pneumonia-detection-project/
|-- backend
|-- clean_tree.md
|-- frontend
|-- generate_clean_tree.py
|-- ml
|   |-- augmentation
|   |-- checkpoints
|   |   `-- baseline_grayscale.pth
|   |-- data
|   |   |-- processed
|   |   `-- raw
|   |       `-- kaggle
|   |           |-- test
|   |           |-- train
|   |           |   |-- NORMAL
|   |           |   `-- PNEUMONIA
|   |           `-- val
|   |-- evaluation
|   |-- explainability
|   |-- inference_test.py
|   |-- models
|   |   |-- baseline_cnn.py
|   |   `-- test_model.py
|   |-- preprocessing
|   |   |-- clahe.py
|   |   |-- preprocess.py
|   |   `-- pseudocolor.py
|   |-- training
|   |   |-- __init__.py
|   |   |-- dataloader.py
|   |   |-- dataset.py
|   |   |-- test_dataloader.py
|   |   |-- test_dataset.py
|   |   `-- train_baseline.py
|   |-- visualize_clahe.py
|   `-- visualize_pseudocolor.py
`-- requirements.txt
```
