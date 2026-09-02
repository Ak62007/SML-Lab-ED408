"""
Q1 — k-Nearest Neighbours classifier implemented from scratch (numpy only),
tested on the BSDS500-derived edge-pixel dataset.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from common import load_split, save_metrics, FIG_DIR


class KNNFromScratch:
    """Brute-force k-NN classifier, batched for memory efficiency."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def _batch_predict(self, X, k, batch_size=500):
        preds = np.empty(len(X), dtype=int)
        train_sq = (self.X_train ** 2).sum(axis=1)  # (n_train,)
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            batch = X[start:end]
            # squared Euclidean distance via ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b
            # (keeps memory at batch_size x n_train instead of batch_size x n_train x dim)
            batch_sq = (batch ** 2).sum(axis=1, keepdims=True)  # (batch,1)
            dists = batch_sq + train_sq[None, :] - 2 * batch @ self.X_train.T
            nn_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
            nn_labels = self.y_train[nn_idx]
            # majority vote
            votes = nn_labels.sum(axis=1)
            preds[start:end] = (votes > k / 2).astype(int)
        return preds

    def predict(self, X, k=None):
        return self._batch_predict(X, k or self.k)


def main():
    X_train, X_test, y_train, y_test = load_split()
    print(f"train={X_train.shape}, test={X_test.shape}")

    model = KNNFromScratch().fit(X_train, y_train)

    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    accs = []
    for k in k_values:
        preds = model.predict(X_test, k=k)
        acc = accuracy_score(y_test, preds)
        accs.append(acc)
        print(f"k={k:3d}  accuracy={acc:.4f}")

    best_k = k_values[int(np.argmax(accs))]
    print(f"\nBest k = {best_k}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, accs, marker="o")
    ax.axvline(best_k, color="red", ls="--", alpha=0.5, label=f"best k={best_k}")
    ax.set_xlabel("k")
    ax.set_ylabel("test accuracy")
    ax.set_title("k-NN (from scratch): accuracy vs k")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q1_knn_k_selection.png", bbox_inches="tight")
    plt.close()

    final_preds = model.predict(X_test, k=best_k)
    acc = accuracy_score(y_test, final_preds)
    cm = confusion_matrix(y_test, final_preds)
    report = classification_report(y_test, final_preds, target_names=["non-edge", "edge"])
    print(report)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["non-edge", "edge"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["non-edge", "edge"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"k-NN Confusion Matrix (k={best_k})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q1_knn_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    save_metrics("q1_knn", {
        "k_values": k_values,
        "accuracies": accs,
        "best_k": best_k,
        "test_accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    })


if __name__ == "__main__":
    main()
