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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, RocCurveDisplay)

from common import load_split, save_metrics, FIG_DIR


def main():
    # GaussianNB is not scale-sensitive, but scaling keeps things consistent
    # with the other questions and doesn't change its predictions.
    X_train, X_test, y_train, y_test = load_split(scale=False)
    print(f"train={X_train.shape}, test={X_test.shape}")

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
    })


if __name__ == "__main__":
    main()
