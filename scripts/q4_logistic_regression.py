"""
Q4 — Logistic Regression implemented from scratch using batch gradient
descent (numpy only), on the BSDS500-derived edge-pixel dataset.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from common import load_split, save_metrics, FIG_DIR, FEATURE_COLS


class LogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iters=2000, l2=0.0):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n, d = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])  # bias term
        self.w = np.zeros(d + 1)

        for it in range(self.n_iters):
            z = Xb @ self.w
            p = self._sigmoid(z)
            grad = Xb.T @ (p - y) / n
            grad[1:] += self.l2 * self.w[1:] / n  # don't regularize bias
            self.w -= self.lr * grad

            if it % 20 == 0:
                eps = 1e-12
                loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
                self.loss_history.append(loss)
        return self

    def predict_proba(self, X):
        Xb = np.hstack([np.ones((len(X), 1)), X])
        return self._sigmoid(Xb @ self.w)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def main():
    X_train, X_test, y_train, y_test = load_split(scale=True)
    print(f"train={X_train.shape}, test={X_test.shape}")

    model = LogisticRegressionScratch(lr=0.5, n_iters=3000, l2=0.0)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=["non-edge", "edge"])
    print(f"Test accuracy: {acc:.4f}")
    print(report)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(model.loss_history)) * 20, model.loss_history)
    ax.set_xlabel("iteration")
    ax.set_ylabel("binary cross-entropy loss")
    ax.set_title("Logistic Regression (from scratch): gradient descent convergence")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q4_logreg_loss_curve.png", bbox_inches="tight")
    plt.close()

    coefs = sorted(zip(FEATURE_COLS, model.w[1:]), key=lambda t: -abs(t[1]))
    print("Learned weights (standardized features):")
    for name, w in coefs:
        print(f"  {name:12s} {w:+.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    names, ws = zip(*coefs)
    colors = ["#C44E52" if w < 0 else "#4C72B0" for w in ws]
    ax.barh(names, ws, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("weight (standardized features)")
    ax.set_title("Logistic Regression coefficients")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q4_logreg_coefficients.png", bbox_inches="tight")
    plt.close()

    # ---------- Parameter Modification: sensitivity to learning rate ----------
    lr_values = [0.001, 0.01, 0.1, 0.5, 1.0, 3.0]
    lr_accs, lr_final_losses = [], []
    for lr in lr_values:
        m = LogisticRegressionScratch(lr=lr, n_iters=3000)
        m.fit(X_train, y_train)
        a = accuracy_score(y_test, m.predict(X_test))
        lr_accs.append(a)
        lr_final_losses.append(m.loss_history[-1])
        print(f"lr={lr:<6g} final_loss={m.loss_history[-1]:.4f}  accuracy={a:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([str(v) for v in lr_values], lr_accs, marker="o")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("test accuracy")
    ax.set_title("Logistic Regression: accuracy vs learning rate")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q4_lr_sensitivity.png", bbox_inches="tight")
    plt.close()

    save_metrics("q4_logistic_regression", {
        "test_accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "final_loss": model.loss_history[-1],
        "weights": {n: float(w) for n, w in zip(FEATURE_COLS, model.w[1:])},
        "bias": float(model.w[0]),
        "lr_sweep": {"learning_rates": lr_values, "accuracies": lr_accs,
                     "final_losses": lr_final_losses},
    })


if __name__ == "__main__":
    main()
