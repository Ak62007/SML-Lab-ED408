# SML Lab Work — ED408

Machine learning lab exercises for ED408. Questions 1–7 use a derived
**BSDS500** edge-pixel dataset; Questions 11–12 use real macroeconomic data
from the **World Bank World Development Indicators**.

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
non-edge pixel) for Questions 1–7.

## World Bank economics dataset

`data/world_bank_macro_2010_2023.csv` is a local snapshot of country-year
macroeconomic indicators downloaded from the official World Bank Indicators
API. Aggregate regions are excluded. Its metadata JSON records indicator
codes, source URLs, the date range, and download time.

- **Question 11:** predicts log real GDP per capita with Linear Regression,
  Lasso, Ridge, and Elastic Net using country-grouped validation and testing.
- **Question 12:** compares ordinary PCA and RBF Kernel PCA for separating
  high-income from non-high-income economies. GDP per capita is excluded from
  the predictors.

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
scripts/build_world_bank_macro.py   downloads the Q11-Q12 economics snapshot
scripts/create_q11_q12_notebooks.py regenerates the Q11-Q12 notebooks
notebooks/Q11_L1_L2_Elastic_Net_Regularization.ipynb
notebooks/Q12_Kernel_PCA_Nonlinear_Separation.ipynb
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

# Optional: refresh the World Bank snapshot and regenerate Q11-Q12.
python3 scripts/build_world_bank_macro.py
python3 scripts/create_q11_q12_notebooks.py
jupyter nbconvert --to notebook --execute --inplace notebooks/Q11_L1_L2_Elastic_Net_Regularization.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/Q12_Kernel_PCA_Nonlinear_Separation.ipynb
```
