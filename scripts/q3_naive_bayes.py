"""
Q3 — Naive Bayes classifier ("spam vs ham" style binary classification),
applied to the BSDS500-derived edge-pixel dataset.

Adaptation note: the lab asks for spam/ham email classification, but per
lab instructions every question must use the BSDS500-derived dataset only.
We treat it as the same kind of problem Naive Bayes is classically used
for: binary classification from a feature vector under a conditional
independence assumption -- here "edge" plays the role of "spam" and
"non-edge" the role of "ham", with pixel colour/gradient/texture features
in place of word/token features.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, RocCurveDisplay)

from common import load_split, save_metrics, FIG_DIR


def check_normality(X, y, feature_names=None, max_n=5000, seed=0):
    """
    Check normality of each feature, separately for each class.
    (GaussianNB assumes each feature is normal *within a class*,
    so checking the pooled data would be the wrong test.)

    Prints a table, saves Q-Q plots, and returns a dict of results.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    results = {}
    print("\n--- Normality check (training set, per class) ---")
    print(f"{'class':<6}{'feature':<14}{'skew':>8}{'ex.kurt':>9}"
          f"{'shapiro_p':>12}{'dagostino_p':>13}")
    for cls in np.unique(y):
        Xc = X[y == cls]
        for j, name in enumerate(feature_names):
            col = Xc[:, j]
            if np.ptp(col) == 0:          # constant feature, tests are meaningless
                continue
            # Shapiro is only reliable up to ~5000 points, so subsample
            sample = col if len(col) <= max_n else rng.choice(col, max_n, replace=False)
            _, sh_p = stats.shapiro(sample)
            _, dag_p = stats.normaltest(col)   # fine for large n
            sk = stats.skew(col)
            ku = stats.kurtosis(col)           # 0 means normal-like tails
            results[f"class{cls}_{name}"] = dict(
                skew=float(sk), excess_kurtosis=float(ku),
                shapiro_p=float(sh_p), dagostino_p=float(dag_p))
            print(f"{cls:<6}{name:<14}{sk:>8.2f}{ku:>9.2f}{sh_p:>12.2e}{dag_p:>13.2e}")

    # Q-Q plots: points on the line = normal-like
    n_feat = len(feature_names)
    ncols = min(4, n_feat)
    nrows = int(np.ceil(n_feat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3 * nrows),
                             squeeze=False)
    for j, name in enumerate(feature_names):
        ax = axes[j // ncols][j % ncols]
        for cls, color in zip(np.unique(y), ["tab:blue", "tab:red"]):
            col = X[y == cls][:, j]
            if np.ptp(col) == 0:
                continue
            if len(col) > max_n:
                col = rng.choice(col, max_n, replace=False)
            (osm, osr), (slope, intercept, _) = stats.probplot(col, dist="norm")
            ax.plot(osm, osr, ".", ms=2, alpha=0.5, color=color, label=f"class {cls}")
            ax.plot(osm, slope * osm + intercept, "-", lw=1, color=color)
        ax.set_title(name, fontsize=9)
    for k in range(n_feat, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    axes[0][0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q3_normality_qq.png", bbox_inches="tight")
    plt.close()
    return results


def main():
    # GaussianNB is not scale-sensitive, so no scaling needed here.
    X_train, X_test, y_train, y_test = load_split(scale=False)
    print(f"train={X_train.shape}, test={X_test.shape}")

    # ---------- Normality check (training set only, no test-set leakage) ----------
    # Pass feature_names=[...] if you have them, for readable labels.
    normality = check_normality(X_train, y_train)

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=["ham (non-edge)", "spam (edge)"])
    print(f"Test accuracy: {acc:.4f}")
    print(report)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["ham", "spam"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["ham", "spam"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Naive Bayes Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q3_nb_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, probs, ax=ax, name="GaussianNB")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("Naive Bayes ROC Curve")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q3_nb_roc.png", bbox_inches="tight")
    plt.close()

    # ---------- Parameter Modification: sensitivity to var_smoothing ----------
    smoothing_values = [1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0]
    smoothing_accs = []
    for vs in smoothing_values:
        c = GaussianNB(var_smoothing=vs)
        c.fit(X_train, y_train)
        a = accuracy_score(y_test, c.predict(X_test))
        smoothing_accs.append(a)
        print(f"var_smoothing={vs:<9g} accuracy={a:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([str(v) for v in smoothing_values], smoothing_accs, marker="o")
    ax.set_xlabel("var_smoothing")
    ax.set_ylabel("test accuracy")
    ax.set_title("Naive Bayes: accuracy vs var_smoothing")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q3_smoothing_sensitivity.png", bbox_inches="tight")
    plt.close()

    save_metrics("q3_naive_bayes", {
        "test_accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "class_priors": clf.class_prior_.tolist(),
        "smoothing_sweep": {"var_smoothing": smoothing_values, "accuracies": smoothing_accs},
        "normality_check": normality,
    })


if __name__ == "__main__":
    main()
