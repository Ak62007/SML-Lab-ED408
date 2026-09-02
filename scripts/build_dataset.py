"""
Build a tabular pixel-level feature dataset from the BSDS500 image
segmentation dataset (archive/images, archive/ground_truth).

Task: binary "edge pixel vs non-edge pixel" classification.
For every sampled pixel we compute simple colour / gradient / texture /
position features, and label it 1 (edge) if a majority of the human
annotators marked it as a segment boundary, else 0.

Output: data/bsds_features.csv  -- used by every question in this lab.
"""
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import scipy.io as sio
from pathlib import Path

RNG = np.random.default_rng(42)

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "archive" / "images" / "train"
GT_DIR = ROOT / "archive" / "ground_truth" / "train"
OUT_CSV = ROOT / "data" / "bsds_features.csv"

SAMPLES_PER_CLASS_PER_IMAGE = 60   # -> up to 120 rows / image
BOUNDARY_VOTE_THRESHOLD = 0.5      # fraction of annotators that must agree


def consensus_boundary(mat_path):
    d = sio.loadmat(mat_path)
    gt = d["groundTruth"]
    n = gt.shape[1]
    acc = None
    for i in range(n):
        b = gt[0, i][0, 0]["Boundaries"].astype(np.float32)
        acc = b if acc is None else acc + b
    return (acc / n) >= BOUNDARY_VOTE_THRESHOLD


def extract_features(gray, rgb):
    sobel_x = ndimage.sobel(gray, axis=1)
    sobel_y = ndimage.sobel(gray, axis=0)
    grad_mag = np.hypot(sobel_x, sobel_y)
    grad_mag = grad_mag / (grad_mag.max() + 1e-8)
    grad_dir = np.arctan2(sobel_y, sobel_x)

    laplacian = np.abs(ndimage.laplace(gray))

    local_mean = ndimage.uniform_filter(gray, size=5)
    local_sqmean = ndimage.uniform_filter(gray ** 2, size=5)
    local_std = np.sqrt(np.clip(local_sqmean - local_mean ** 2, 0, None))

    return {
        "R": rgb[..., 0],
        "G": rgb[..., 1],
        "B": rgb[..., 2],
        "gray": gray,
        "grad_mag": grad_mag,
        "grad_dir": grad_dir,
        "laplacian": laplacian,
        "local_std": local_std,
    }


def main():
    mat_files = sorted(GT_DIR.glob("*.mat"))
    rows = []

    for k, mat_path in enumerate(mat_files):
        img_path = IMG_DIR / (mat_path.stem + ".jpg")
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        rgb = np.asarray(img, dtype=np.float32) / 255.0
        gray = rgb.mean(axis=2)

        edge_map = consensus_boundary(mat_path)
        # BSDS images can be landscape or portrait; ground truth matches orientation.
        if edge_map.shape != gray.shape:
            continue

        feats = extract_features(gray, rgb)
        h, w = gray.shape

        yy, xx = np.mgrid[0:h, 0:w]
        x_norm = xx / (w - 1)
        y_norm = yy / (h - 1)

        # avoid a 4px border so the filters above are well defined
        border = 4
        valid = np.zeros_like(edge_map, dtype=bool)
        valid[border:-border, border:-border] = True

        edge_idx = np.argwhere(edge_map & valid)
        nonedge_idx = np.argwhere((~edge_map) & valid)

        n_edge = min(SAMPLES_PER_CLASS_PER_IMAGE, len(edge_idx))
        if n_edge == 0:
            continue
        n_nonedge = min(SAMPLES_PER_CLASS_PER_IMAGE, len(nonedge_idx))

        pick_edge = edge_idx[RNG.choice(len(edge_idx), n_edge, replace=False)]
        pick_nonedge = nonedge_idx[RNG.choice(len(nonedge_idx), n_nonedge, replace=False)]
        picks = np.vstack([pick_edge, pick_nonedge])
        labels = np.array([1] * n_edge + [0] * n_nonedge)

        for (r, c), lab in zip(picks, labels):
            rows.append({
                "image_id": mat_path.stem,
                "R": feats["R"][r, c],
                "G": feats["G"][r, c],
                "B": feats["B"][r, c],
                "gray": feats["gray"][r, c],
                "grad_mag": feats["grad_mag"][r, c],
                "grad_dir": feats["grad_dir"][r, c],
                "laplacian": feats["laplacian"][r, c],
                "local_std": feats["local_std"][r, c],
                "x_norm": x_norm[r, c],
                "y_norm": y_norm[r, c],
                "is_edge": lab,
            })

        if (k + 1) % 40 == 0:
            print(f"  processed {k + 1}/{len(mat_files)} images, rows so far={len(rows)}")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} cols -> {OUT_CSV}")
    print(df["is_edge"].value_counts())


if __name__ == "__main__":
    main()
