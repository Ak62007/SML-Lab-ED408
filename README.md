# SML Lab Work — ED408

Machine learning lab exercises applied to the **BSDS500** (Berkeley
Segmentation Data Set) image dataset (`archive/`).

## Dataset adaptation

BSDS500 ships as JPEG images with hand-annotated segmentation/boundary
ground truth (`.mat` files, 6 annotators per image). It is an image
segmentation dataset, not a tabular classification dataset — so every
question below is answered on a single **derived tabular dataset** built
from it (`data/bsds_features.csv`, `scripts/build_dataset.py`):

- One row = one pixel, sampled (balanced) from all 200 BSDS500 training
  images.
- **Features**: `R, G, B, gray` (colour), `grad_mag, grad_dir, laplacian,
  local_std` (gradient/texture), `x_norm, y_norm` (position).
- **Label**: `is_edge` — 1 if a majority of the 6 human annotators marked
  the pixel as a segment boundary, else 0.
- 24,000 rows, perfectly balanced (12,000 / 12,000).

This turns BSDS500 into a genuine binary classification problem (edge vs.
non-edge pixel) and the *same* dataset is reused, unchanged, across every
question — including the ones whose original prompt named a different
dataset (Iris, spam/ham, MNIST) — per lab instructions.

## Structure

```
scripts/build_dataset.py   builds data/bsds_features.csv from archive/
notebooks/00_EDA.ipynb     full exploratory data analysis
scripts/q1_knn_scratch.py           Q1 - k-NN from scratch
scripts/q2_decision_tree.py         Q2 - Decision Tree (sklearn) + viz
scripts/q3_naive_bayes.py           Q3 - Naive Bayes (spam/ham-style)
scripts/q4_logistic_regression.py   Q4 - Logistic Regression from scratch
scripts/q5_pca_tsne.py              Q5 - PCA & t-SNE to 2D
scripts/q6_outliers.py              Q6 - outlier detection (z-score, IQR)
scripts/q7_compare_models.py        Q7 - LR vs SVM metric comparison
results/figures/           all generated plots
results/metrics/           numeric results (json/txt) per question
docs/                      lab documentation (.docx) per question
```

## Running

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib seaborn scipy jupyter python-docx
python3 scripts/build_dataset.py
python3 scripts/q1_knn_scratch.py   # etc.
```
