"""
Q2 — Decision Tree classifier using scikit-learn, with tree visualization,
on the BSDS500-derived edge-pixel dataset.
"""
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from common import load_split, save_metrics, FIG_DIR, FEATURE_COLS


def main():
    # Decision trees don't need feature scaling; keep raw feature values.
    X_train, X_test, y_train, y_test = load_split(scale=False)
    print(f"train={X_train.shape}, test={X_test.shape}")

    clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=["non-edge", "edge"])
    print(f"Test accuracy: {acc:.4f}")
    print(report)

    # feature importances
    importances = sorted(zip(FEATURE_COLS, clf.feature_importances_),
                          key=lambda t: -t[1])
    print("Feature importances:")
    for name, imp in importances:
        print(f"  {name:12s} {imp:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    names, vals = zip(*importances)
    ax.barh(names, vals, color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("importance")
    ax.set_title("Decision Tree feature importances")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q2_tree_feature_importance.png", bbox_inches="tight")
    plt.close()

    # full-tree visualization
    fig, ax = plt.subplots(figsize=(26, 12))
    plot_tree(clf, feature_names=FEATURE_COLS, class_names=["non-edge", "edge"],
              filled=True, rounded=True, fontsize=7, ax=ax)
    ax.set_title(f"Decision Tree (max_depth=6)  test accuracy={acc:.3f}")
    plt.savefig(FIG_DIR / "q2_tree_full.png", bbox_inches="tight", dpi=150)
    plt.close()

    # shallow tree for a readable diagram
    clf_shallow = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42)
    clf_shallow.fit(X_train, y_train)
    preds_shallow = clf_shallow.predict(X_test)
    acc_shallow = accuracy_score(y_test, preds_shallow)

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(clf_shallow, feature_names=FEATURE_COLS, class_names=["non-edge", "edge"],
              filled=True, rounded=True, fontsize=10, ax=ax)
    ax.set_title(f"Decision Tree (max_depth=3, readable)  test accuracy={acc_shallow:.3f}")
    plt.savefig(FIG_DIR / "q2_tree_shallow.png", bbox_inches="tight", dpi=150)
    plt.close()

    save_metrics("q2_decision_tree", {
        "depth6_test_accuracy": acc,
        "depth3_test_accuracy": acc_shallow,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importances": {n: float(v) for n, v in importances},
    })


if __name__ == "__main__":
    main()
