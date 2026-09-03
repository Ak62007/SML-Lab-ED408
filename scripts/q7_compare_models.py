"""
Q7 — Compare Accuracy, Precision, Recall, F1-score and ROC-AUC for
Logistic Regression vs SVM on the BSDS500-derived edge-pixel dataset.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve)

from common import load_split, save_metrics, FIG_DIR


def evaluate(name, clf, X_test, y_test, use_decision_function=False):
    preds = clf.predict(X_test)
    if use_decision_function:
        scores = clf.decision_function(X_test)
    else:
        scores = clf.predict_proba(X_test)[:, 1]

    return {
        "name": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, scores),
        "scores": scores,
        "preds": preds,
    }


def main():
    X_train, X_test, y_train, y_test = load_split(scale=True)
    print(f"train={X_train.shape}, test={X_test.shape}")

    log_reg = LogisticRegression(max_iter=2000, random_state=42)
    log_reg.fit(X_train, y_train)
    res_lr = evaluate("Logistic Regression", log_reg, X_test, y_test)

    # RBF-kernel SVM; probability=True enables predict_proba for ROC-AUC
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    res_svm = evaluate("SVM (RBF)", svm, X_test, y_test)

    results = [res_lr, res_svm]
    for r in results:
        print(f"\n{r['name']}")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            print(f"  {k:10s} {r[k]:.4f}")

    # ---------- comparison bar chart ----------
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, [res_lr[m] for m in metrics], width, label="Logistic Regression", color="#4C72B0")
    ax.bar(x + width/2, [res_svm[m] for m in metrics], width, label="SVM (RBF)", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Logistic Regression vs SVM — metric comparison")
    ax.legend()
    for i, m in enumerate(metrics):
        ax.text(i - width/2, res_lr[m] + 0.01, f"{res_lr[m]:.3f}", ha="center", fontsize=8)
        ax.text(i + width/2, res_svm[m] + 0.01, f"{res_svm[m]:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q7_metric_comparison.png", bbox_inches="tight")
    plt.close()

    # ---------- ROC curves ----------
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for r, color in zip(results, ["#4C72B0", "#DD8452"]):
        fpr, tpr, _ = roc_curve(y_test, r["scores"])
        ax.plot(fpr, tpr, label=f"{r['name']} (AUC={r['roc_auc']:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Logistic Regression vs SVM")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q7_roc_curves.png", bbox_inches="tight")
    plt.close()

    table = pd.DataFrame([{k: r[k] for k in ["name"] + metrics} for r in results])
    print("\n", table.to_string(index=False))

    # ---------- Parameter Modification: sensitivity to SVM's C ----------
    C_values = [0.01, 0.1, 1, 10, 100]
    C_accs, C_f1s = [], []
    for C in C_values:
        m = SVC(kernel="rbf", C=C, gamma="scale", random_state=42)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        a, f = accuracy_score(y_test, preds), f1_score(y_test, preds)
        C_accs.append(a)
        C_f1s.append(f)
        print(f"C={C:<6g} accuracy={a:.4f}  f1={f:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([str(c) for c in C_values], C_accs, marker="o", label="accuracy")
    ax.plot([str(c) for c in C_values], C_f1s, marker="s", label="f1")
    ax.set_xlabel("SVM regularization C")
    ax.set_ylabel("score")
    ax.set_title("SVM: accuracy / F1 vs regularization C")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q7_C_sensitivity.png", bbox_inches="tight")
    plt.close()

    save_metrics("q7_compare_models", {
        **{r["name"]: {k: float(r[k]) for k in metrics} for r in results},
        "svm_C_sweep": {"C_values": C_values, "accuracies": C_accs, "f1_scores": C_f1s},
    })


if __name__ == "__main__":
    main()
