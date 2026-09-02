"""
Q5 — PCA and t-SNE to reduce the BSDS500-derived edge-pixel dataset to 2D,
and visualize the resulting clusters (edge vs non-edge).

Adaptation note: the lab prompt names MNIST, but per lab instructions every
question must use the BSDS500-derived dataset only. PCA/t-SNE serve the
same purpose here -- projecting a multi-dimensional feature space (colour +
gradient + texture + position) down to 2D to see whether the two classes
form separable clusters.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from common import DATA_CSV, FEATURE_COLS, LABEL_COL, FIG_DIR, METRIC_DIR, save_metrics
import pandas as pd

RANDOM_STATE = 42
TSNE_SAMPLE_SIZE = 4000  # subsample for tractable t-SNE runtime


def main():
    df = pd.read_csv(DATA_CSV)
    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df[LABEL_COL].values.astype(int)

    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma

    # ---------- PCA (full dataset) ----------
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(Xs)
    explained = pca.explained_variance_ratio_
    print(f"PCA explained variance ratio: {explained} (sum={explained.sum():.3f})")

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, name, color in [(0, "non-edge", "#4C72B0"), (1, "edge", "#DD8452")]:
        mask = y == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=4, alpha=0.35, c=color, label=name)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    ax.set_title("PCA projection to 2D (full dataset, n=24000)")
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q5_pca_2d.png", bbox_inches="tight")
    plt.close()

    # PCA loadings (which original features drive PC1/PC2)
    fig, ax = plt.subplots(figsize=(6, 4))
    comp = pca.components_
    x = np.arange(len(FEATURE_COLS))
    width = 0.35
    ax.bar(x - width/2, comp[0], width, label="PC1")
    ax.bar(x + width/2, comp[1], width, label="PC2")
    ax.set_xticks(x); ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right")
    ax.set_ylabel("loading")
    ax.set_title("PCA component loadings")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q5_pca_loadings.png", bbox_inches="tight")
    plt.close()

    # ---------- t-SNE (subsample for speed) ----------
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(Xs), size=min(TSNE_SAMPLE_SIZE, len(Xs)), replace=False)
    X_sub, y_sub = Xs[idx], y[idx]

    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE, init="pca")
    X_tsne = tsne.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, name, color in [(0, "non-edge", "#4C72B0"), (1, "edge", "#DD8452")]:
        mask = y_sub == label
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=6, alpha=0.5, c=color, label=name)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE projection to 2D (subsample n={len(idx)})")
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q5_tsne_2d.png", bbox_inches="tight")
    plt.close()

    save_metrics("q5_pca_tsne", {
        "pca_explained_variance_ratio": explained.tolist(),
        "pca_total_variance_explained_2d": float(explained.sum()),
        "tsne_sample_size": int(len(idx)),
        "tsne_perplexity": 30,
    })


if __name__ == "__main__":
    main()
