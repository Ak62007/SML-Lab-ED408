"""
Q6 — Detect and treat outliers in the BSDS500-derived edge-pixel dataset
using the Z-score method and the IQR method.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import DATA_CSV, FIG_DIR, save_metrics

# grad_dir is circular (radians, -pi..pi) and x_norm/y_norm are positional —
# outlier analysis is only meaningful for the continuous magnitude features.
OUTLIER_COLS = ["R", "G", "B", "gray", "grad_mag", "laplacian", "local_std"]
Z_THRESH = 3.0
IQR_K = 1.5


def zscore_outliers(series):
    z = (series - series.mean()) / series.std()
    return z.abs() > Z_THRESH


def iqr_outliers(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - IQR_K * iqr, q3 + IQR_K * iqr
    return (series < lower) | (series > upper), lower, upper


def main():
    df = pd.read_csv(DATA_CSV)
    summary = {}
    df_treated = df.copy()

    for col in OUTLIER_COLS:
        z_mask = zscore_outliers(df[col])
        iqr_mask, lower, upper = iqr_outliers(df[col])

        # Treatment: cap (winsorize) IQR-flagged values at the fences.
        df_treated[col] = df[col].clip(lower=lower, upper=upper)

        summary[col] = {
            "n_outliers_zscore": int(z_mask.sum()),
            "pct_outliers_zscore": float(100 * z_mask.mean()),
            "n_outliers_iqr": int(iqr_mask.sum()),
            "pct_outliers_iqr": float(100 * iqr_mask.mean()),
            "iqr_lower_fence": float(lower),
            "iqr_upper_fence": float(upper),
        }
        print(f"{col:10s} zscore={z_mask.sum():4d} ({100*z_mask.mean():.2f}%)  "
              f"iqr={iqr_mask.sum():4d} ({100*iqr_mask.mean():.2f}%)  "
              f"fences=[{lower:.3f}, {upper:.3f}]")

    # ---------- before/after boxplots ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df[OUTLIER_COLS].boxplot(ax=axes[0], rot=45)
    axes[0].set_title("Before treatment")
    df_treated[OUTLIER_COLS].boxplot(ax=axes[1], rot=45)
    axes[1].set_title("After IQR-capping (winsorized)")
    plt.suptitle("Outlier treatment: box plots before vs after")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q6_outlier_boxplots.png", bbox_inches="tight")
    plt.close()

    # bar chart comparing method counts
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(OUTLIER_COLS))
    width = 0.35
    z_counts = [summary[c]["n_outliers_zscore"] for c in OUTLIER_COLS]
    iqr_counts = [summary[c]["n_outliers_iqr"] for c in OUTLIER_COLS]
    ax.bar(x - width/2, z_counts, width, label="Z-score (|z|>3)")
    ax.bar(x + width/2, iqr_counts, width, label="IQR (1.5x)")
    ax.set_xticks(x); ax.set_xticklabels(OUTLIER_COLS, rotation=45, ha="right")
    ax.set_ylabel("# outliers detected")
    ax.set_title(f"Outlier counts by method (n={len(df)})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q6_outlier_counts.png", bbox_inches="tight")
    plt.close()

    treated_path = DATA_CSV.parent / "bsds_features_outliers_treated.csv"
    df_treated.to_csv(treated_path, index=False)
    print(f"saved treated dataset -> {treated_path}")

    save_metrics("q6_outliers", {"per_feature": summary, "total_rows": len(df)})


if __name__ == "__main__":
    main()
