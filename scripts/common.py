"""Shared helpers used by every q*.py script — keeps train/test split and
feature columns identical across all 7 questions, since they all reuse the
same BSDS500-derived dataset (data/bsds_features.csv)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "bsds_features.csv"
FIG_DIR = ROOT / "results" / "figures"
METRIC_DIR = ROOT / "results" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ["R", "G", "B", "gray", "grad_mag", "grad_dir",
                "laplacian", "local_std", "x_norm", "y_norm"]
LABEL_COL = "is_edge"
RANDOM_STATE = 42


def load_split(test_size=0.2, scale=True):
    df = pd.read_csv(DATA_CSV)
    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df[LABEL_COL].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    if scale:
        mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
        sigma[sigma == 0] = 1.0
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma

    return X_train, X_test, y_train, y_test


def save_metrics(name, d):
    path = METRIC_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=float)
    print(f"saved metrics -> {path}")
