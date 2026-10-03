# fashion-ann-pipeline

A fully-connected ANN (Flatten -> Dense/ReLU -> Dropout -> Dense/Softmax) that
classifies Fashion-MNIST images, built as a reproducible DVC pipeline with a
Google Drive remote.

## Pipeline

| Stage      | Script               | Output                                   |
|------------|----------------------|------------------------------------------|
| prepare    | `src/prepare.py`     | `data/raw/`                              |
| preprocess | `src/preprocess.py`  | `data/processed/`                        |
| train      | `src/train.py`       | `models/model.h5`, `models/history.csv`  |
| evaluate   | `src/evaluate.py`    | `metrics.json`, `reports/confusion_matrix.png` |

Hyperparameters live in `params.yaml`.

## Setup

```
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

## Reproduce

```
dvc pull     # fetch data/model from the Google Drive remote
dvc repro    # run the full pipeline
```
