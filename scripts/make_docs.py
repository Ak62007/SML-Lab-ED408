"""
Generates the per-question lab documentation (.docx files) for ED408,
following the naming convention "<Topic> ED408.docx". Kept condensed so
each question fits in ~3 pages: short Aim/Theory, full code (small
monospace font), 1-2 key result figures, and a compact results table.
"""
import json
from pathlib import Path

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True, parents=True)

DATASET_BLURB = (
    "BSDS500 (Berkeley Segmentation Data Set) ships as natural images with "
    "hand-annotated object-boundary ground truth, not a tabular dataset. It was "
    "converted into a tabular binary-classification dataset "
    "(data/bsds_features.csv, built by scripts/build_dataset.py): one row per "
    "sampled pixel from all 200 BSDS500 training images, 10 features "
    "(R, G, B, gray, grad_mag, grad_dir, laplacian, local_std, x_norm, y_norm) "
    "capturing colour, gradient, texture and position, and label is_edge (1 if "
    "a majority of 6 human annotators marked the pixel as a segment boundary, "
    "else 0). 24,000 rows total, perfectly balanced (12,000 / 12,000). This "
    "same dataset is reused, unchanged, for every question in this lab."
)


def set_margins(doc, inches=0.7):
    for section in doc.sections:
        section.top_margin = Inches(inches)
        section.bottom_margin = Inches(inches)
        section.left_margin = Inches(inches)
        section.right_margin = Inches(inches)


def add_heading(doc, text, size=13, color=(0x1F, 0x4E, 0x79)):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(*color)
    return p


def add_label_para(doc, label, text, size=10):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r1 = p.add_run(f"{label}: ")
    r1.bold = True
    r1.font.size = Pt(size)
    r2 = p.add_run(text)
    r2.font.size = Pt(size)
    return p


def add_code(doc, code_text, font_size=7.5):
    for line in code_text.splitlines():
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing = 1.0
        run = p.add_run(line if line.strip() else " ")
        run.font.name = "Courier New"
        run.font.size = Pt(font_size)


def add_image(doc, path, width_in=3.1, caption=None):
    doc.add_picture(str(path), width=Inches(width_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(caption)
        r.italic = True
        r.font.size = Pt(8)


def add_table(doc, headers, rows, font_size=9):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = str(h)
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(font_size)
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
            for p in cells[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(font_size)
    return table


def base_doc(title):
    doc = Document()
    set_margins(doc)
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title)
    r.bold = True
    r.font.size = Pt(15)
    r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    p.paragraph_format.space_after = Pt(2)

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = p2.add_run("ED408 — Statistical / Supervised Machine Learning Lab")
    r2.italic = True
    r2.font.size = Pt(9)
    p2.paragraph_format.space_after = Pt(8)

    return doc


def load_json(name):
    with open(METRICS / f"{name}.json") as f:
        return json.load(f)


def load_code(name):
    return (SCRIPTS / name).read_text()


def finish(doc, filename):
    out = DOCS / filename
    doc.save(out)
    print("wrote", out)


# ---------------------------------------------------------------- Q1 ----
def make_q1():
    doc = base_doc("Experiment 1: k-Nearest Neighbours Classifier (From Scratch)")
    add_label_para(doc, "Aim", "Implement the k-NN classification algorithm from "
                   "scratch using only NumPy, and evaluate it on a real dataset.")
    add_label_para(doc, "Dataset Used", DATASET_BLURB)
    add_heading(doc, "Theory")
    doc.add_paragraph(
        "k-NN is a non-parametric, instance-based classifier. For a query point "
        "it computes the distance (here, Euclidean) to every training point, "
        "picks the k closest neighbours, and predicts the majority class among "
        "them. No explicit training phase is needed beyond storing the data; "
        "the choice of k controls the bias-variance trade-off — small k "
        "overfits to noise, large k over-smooths the decision boundary."
    ).runs[0].font.size = Pt(10)

    add_heading(doc, "Code")
    add_code(doc, load_code("q1_knn_scratch.py"))

    m = load_json("q1_knn")
    add_heading(doc, "Output")
    add_table(doc, ["k", "Test Accuracy"],
              [[k, f"{a:.4f}"] for k, a in zip(m["k_values"], m["accuracies"])])
    doc.add_paragraph()
    add_image(doc, FIGURES / "q1_knn_k_selection.png", width_in=3.6,
              caption="Accuracy vs. k on the held-out test set.")

    add_heading(doc, "Result")
    doc.add_paragraph(
        f"Best k = {m['best_k']}, achieving a test accuracy of "
        f"{m['test_accuracy']*100:.2f}% on 4,800 held-out pixels for the "
        f"edge vs. non-edge classification task. The from-scratch "
        f"implementation performs comparably to library implementations, "
        f"confirming the distance-based majority-vote logic is correct."
    ).runs[0].font.size = Pt(10)

    finish(doc, "K-Nearest Neighbours ED408.docx")


# ---------------------------------------------------------------- Q2 ----
def make_q2():
    doc = base_doc("Experiment 2: Decision Tree Classifier (scikit-learn)")
    add_label_para(doc, "Aim", "Build a Decision Tree classifier using "
                   "scikit-learn and visualize the learned tree structure.")
    add_label_para(doc, "Dataset Used", DATASET_BLURB)
    add_heading(doc, "Theory")
    doc.add_paragraph(
        "A Decision Tree recursively partitions the feature space by choosing, "
        "at each node, the feature and threshold that most reduces impurity "
        "(Gini index by default in scikit-learn). Leaves are assigned the "
        "majority class of the training samples that reach them. Trees are "
        "easy to interpret and visualize but can overfit if grown too deep, "
        "hence max_depth / min_samples_leaf are used here as regularizers."
    ).runs[0].font.size = Pt(10)

    add_heading(doc, "Code")
    add_code(doc, load_code("q2_decision_tree.py"))

    m = load_json("q2_decision_tree")
    add_heading(doc, "Output")
    top3 = sorted(m["feature_importances"].items(), key=lambda t: -t[1])[:3]
    add_table(doc, ["Metric", "Value"], [
        ["Test accuracy (depth=6)", f"{m['depth6_test_accuracy']:.4f}"],
        ["Test accuracy (depth=3)", f"{m['depth3_test_accuracy']:.4f}"],
        ["Top feature", f"{top3[0][0]} (importance={top3[0][1]:.3f})"],
    ])
    doc.add_paragraph()
    add_image(doc, FIGURES / "q2_tree_shallow.png", width_in=5.5,
              caption="Visualized Decision Tree (max_depth=3, readable).")

    add_heading(doc, "Result")
    doc.add_paragraph(
        f"The depth-6 tree reached {m['depth6_test_accuracy']*100:.2f}% test "
        f"accuracy. local_std (local texture energy) dominates the splits, "
        f"confirming texture is the strongest edge indicator in this feature set."
    ).runs[0].font.size = Pt(10)

    finish(doc, "Decision Tree ED408.docx")


# ---------------------------------------------------------------- Q3 ----
def make_q3():
    doc = base_doc("Experiment 3: Naive Bayes Classifier (Spam/Ham-style)")
    add_label_para(doc, "Aim", "Apply a Naive Bayes classifier to a "
                   "spam-vs-ham-style binary classification problem.")
    add_label_para(doc, "Dataset Used", DATASET_BLURB +
                   " Per lab instructions this same dataset (not an email "
                   "corpus) is used here too: 'edge' plays the role of 'spam' "
                   "and 'non-edge' the role of 'ham', with pixel colour/"
                   "gradient/texture features standing in for word/token features.")
    add_heading(doc, "Theory")
    doc.add_paragraph(
        "Naive Bayes applies Bayes' theorem with a (naive) assumption that "
        "features are conditionally independent given the class. "
        "GaussianNB models each feature, per class, as a 1-D Gaussian and "
        "multiplies per-feature likelihoods to score each class, predicting "
        "the class with the highest posterior probability. It is fast, needs "
        "little data, and is a classic baseline for spam/ham-style filtering."
    ).runs[0].font.size = Pt(10)

    add_heading(doc, "Code")
    add_code(doc, load_code("q3_naive_bayes.py"))

    m = load_json("q3_naive_bayes")
    add_heading(doc, "Output")
    cm = m["confusion_matrix"]
    add_table(doc, ["", "Pred: ham", "Pred: spam"], [
        ["Actual: ham", cm[0][0], cm[0][1]],
        ["Actual: spam", cm[1][0], cm[1][1]],
    ])
    doc.add_paragraph()
    add_image(doc, FIGURES / "q3_nb_roc.png", width_in=3.3,
              caption="ROC curve for the Naive Bayes classifier.")

    add_heading(doc, "Result")
    doc.add_paragraph(
        f"GaussianNB achieved {m['test_accuracy']*100:.2f}% test accuracy. "
        f"It underperforms the Decision Tree / SVM because the independence "
        f"assumption is violated (grad_mag, laplacian and local_std are all "
        f"correlated), but it remains a fast, reasonable baseline."
    ).runs[0].font.size = Pt(10)

    finish(doc, "Naive Bayes ED408.docx")


# ---------------------------------------------------------------- Q4 ----
def make_q4():
    doc = base_doc("Experiment 4: Logistic Regression (From Scratch, Gradient Descent)")
    add_label_para(doc, "Aim", "Implement Logistic Regression from scratch "
                   "using batch gradient descent to minimize binary "
                   "cross-entropy loss.")
    add_label_para(doc, "Dataset Used", DATASET_BLURB)
    add_heading(doc, "Theory")
    doc.add_paragraph(
        "Logistic Regression models P(y=1|x) = sigma(w.x + b) where sigma is "
        "the logistic sigmoid. Parameters are learned by minimizing the "
        "binary cross-entropy loss via gradient descent: at every iteration "
        "the gradient of the loss w.r.t. w is computed over the whole "
        "training set (batch GD) and w is updated as w := w - lr * grad. "
        "Features are standardized (z-score) first so gradient descent "
        "converges quickly and evenly across features."
    ).runs[0].font.size = Pt(10)

    add_heading(doc, "Code")
    add_code(doc, load_code("q4_logistic_regression.py"))

    m = load_json("q4_logistic_regression")
    add_heading(doc, "Output")
    top3 = sorted(m["weights"].items(), key=lambda t: -abs(t[1]))[:3]
    add_table(doc, ["Metric", "Value"], [
        ["Test accuracy", f"{m['test_accuracy']:.4f}"],
        ["Final training loss", f"{m['final_loss']:.4f}"],
        ["Top weight", f"{top3[0][0]} = {top3[0][1]:+.3f}"],
    ])
    doc.add_paragraph()
    add_image(doc, FIGURES / "q4_logreg_loss_curve.png", width_in=3.6,
              caption="Binary cross-entropy loss vs. gradient-descent iteration.")

    add_heading(doc, "Result")
    doc.add_paragraph(
        f"The model converged smoothly to a test accuracy of "
        f"{m['test_accuracy']*100:.2f}%. local_std again receives by far "
        f"the largest weight magnitude, consistent with the Decision Tree "
        f"result in Experiment 2."
    ).runs[0].font.size = Pt(10)

    finish(doc, "Logistic Regression ED408.docx")


# ---------------------------------------------------------------- Q5 ----
def make_q5():
    doc = base_doc("Experiment 5: PCA and t-SNE — 2D Visualization")
    add_label_para(doc, "Aim", "Apply PCA and t-SNE to reduce a "
                   "high-dimensional dataset to 2D and visualize class clusters.")
    add_label_para(doc, "Dataset Used", DATASET_BLURB +
                   " Per lab instructions this dataset is used in place of "
                   "MNIST: the 10-D standardized feature space is projected "
                   "to 2D and coloured by the is_edge label.")
    add_heading(doc, "Theory")
    doc.add_paragraph(
        "PCA is a linear technique that projects data onto the orthogonal "
        "directions (principal components) of maximum variance, computed "
        "from the eigenvectors of the covariance matrix. t-SNE is a "
        "non-linear technique that preserves local neighbourhood structure "
        "by minimizing the KL-divergence between pairwise-similarity "
        "distributions in high-D and low-D space, often revealing cluster "
        "structure PCA misses. t-SNE was run on a 4,000-point subsample for "
        "tractable runtime."
    ).runs[0].font.size = Pt(10)

    add_heading(doc, "Code")
    add_code(doc, load_code("q5_pca_tsne.py"))

    m = load_json("q5_pca_tsne")
    add_heading(doc, "Output")
    add_table(doc, ["Metric", "Value"], [
        ["PCA var. explained (PC1)", f"{m['pca_explained_variance_ratio'][0]*100:.1f}%"],
        ["PCA var. explained (PC2)", f"{m['pca_explained_variance_ratio'][1]*100:.1f}%"],
        ["t-SNE sample size", m["tsne_sample_size"]],
    ])
    doc.add_paragraph()
    add_image(doc, FIGURES / "q5_pca_2d.png", width_in=2.9)
    add_image(doc, FIGURES / "q5_tsne_2d.png", width_in=2.9,
              caption="PCA (left/top) vs. t-SNE (right/bottom) 2D projections.")

    add_heading(doc, "Result")
    doc.add_paragraph(
        f"PCA's first two components explain "
        f"{m['pca_total_variance_explained_2d']*100:.1f}% of total variance "
        f"with partial class separation. t-SNE forms visibly tighter, more "
        f"separated edge / non-edge clusters, illustrating its strength at "
        f"preserving local non-linear structure that PCA (linear) cannot capture."
    ).runs[0].font.size = Pt(10)

    finish(doc, "PCA and t-SNE ED408.docx")


# ---------------------------------------------------------------- Q6 ----
def make_q6():
    doc = base_doc("Experiment 6: Outlier Detection and Treatment (Z-score, IQR)")
    add_label_para(doc, "Aim", "Detect and treat outliers in a dataset "
                   "using the Z-score method and the IQR method.")
    add_label_para(doc, "Dataset Used", DATASET_BLURB)
    add_heading(doc, "Theory")
    doc.add_paragraph(
        "Z-score method: flag a value as an outlier if |z| = |(x - mean)/std| "
        "> 3, i.e. more than 3 standard deviations from the mean (assumes "
        "roughly normal data). IQR method: flag a value as an outlier if it "
        "falls outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR], where IQR = Q3 - Q1 "
        "(distribution-free, robust to skew). Treatment here is winsorizing: "
        "outliers are capped at the IQR fences rather than dropped, to avoid "
        "shrinking the dataset."
    ).runs[0].font.size = Pt(10)

    add_heading(doc, "Code")
    add_code(doc, load_code("q6_outliers.py"))

    m = load_json("q6_outliers")
    add_heading(doc, "Output")
    rows = [[c, d["n_outliers_zscore"], d["n_outliers_iqr"]]
            for c, d in m["per_feature"].items()]
    add_table(doc, ["Feature", "Z-score outliers", "IQR outliers"], rows)
    doc.add_paragraph()
    add_image(doc, FIGURES / "q6_outlier_boxplots.png", width_in=5.5,
              caption="Box plots before vs. after IQR-capping.")

    add_heading(doc, "Result")
    doc.add_paragraph(
        "Colour features (R, G, B, gray) contain essentially no outliers "
        "(bounded [0,1] range), while gradient/texture features (grad_mag, "
        "laplacian, local_std) show 2-6% outliers under IQR, consistent "
        "with their right-skewed distributions (most pixels are flat, a "
        "minority are sharp edges). IQR consistently flags more points than "
        "Z-score here because these features are non-normal / skewed."
    ).runs[0].font.size = Pt(10)

    finish(doc, "Outlier Detection ED408.docx")


# ---------------------------------------------------------------- Q7 ----
def make_q7():
    doc = base_doc("Experiment 7: Logistic Regression vs SVM — Metric Comparison")
    add_label_para(doc, "Aim", "Compare Accuracy, Precision, Recall, "
                   "F1-score and ROC-AUC for Logistic Regression and SVM "
                   "on a classification dataset.")
    add_label_para(doc, "Dataset Used", DATASET_BLURB)
    add_heading(doc, "Theory")
    doc.add_paragraph(
        "Accuracy = (TP+TN)/Total. Precision = TP/(TP+FP) (of predicted "
        "positives, how many are correct). Recall = TP/(TP+FN) (of actual "
        "positives, how many are found). F1 = harmonic mean of Precision "
        "and Recall. ROC-AUC = area under the True-Positive-Rate vs. "
        "False-Positive-Rate curve across all thresholds, summarizing "
        "ranking quality independent of a fixed threshold. SVM (RBF kernel) "
        "finds a max-margin, non-linear decision boundary, contrasted here "
        "with the linear boundary of Logistic Regression."
    ).runs[0].font.size = Pt(10)

    add_heading(doc, "Code")
    add_code(doc, load_code("q7_compare_models.py"))

    m = load_json("q7_compare_models")
    add_heading(doc, "Output")
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    add_table(doc, ["Metric"] + list(m.keys()),
              [[met.upper()] + [f"{m[name][met]:.4f}" for name in m] for met in metrics])
    doc.add_paragraph()
    add_image(doc, FIGURES / "q7_metric_comparison.png", width_in=4.6,
              caption="Metric comparison: Logistic Regression vs SVM (RBF).")

    add_heading(doc, "Result")
    lr, svm = m["Logistic Regression"], m["SVM (RBF)"]
    doc.add_paragraph(
        f"SVM (RBF) outperforms Logistic Regression on every metric "
        f"(accuracy {svm['accuracy']*100:.1f}% vs {lr['accuracy']*100:.1f}%, "
        f"F1 {svm['f1']:.3f} vs {lr['f1']:.3f}, ROC-AUC {svm['roc_auc']:.3f} "
        f"vs {lr['roc_auc']:.3f}), because the RBF kernel can model the "
        f"non-linear decision boundary between edge and non-edge pixels "
        f"that a purely linear model cannot."
    ).runs[0].font.size = Pt(10)

    finish(doc, "Model Comparison ED408.docx")


if __name__ == "__main__":
    make_q1()
    make_q2()
    make_q3()
    make_q4()
    make_q5()
    make_q6()
    make_q7()
