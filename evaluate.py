"""
Infera – Component 4 (Evaluation):
Compares the Bayesian Network against a Traditional Rule-Based System
across Accuracy, Precision, Recall, F1-Score and False Positive Rate.
Saves evaluation charts to data/plots/.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from pgmpy.inference import VariableElimination

sns.set_theme(style="whitegrid", palette="muted")
PLOTS_DIR = os.path.join("data", "plots")


# ── Rule-Based Classifier ────────────────────────────────────────────────────
def rule_based_predict(df: pd.DataFrame) -> np.ndarray:
    """
    Traditional rule-based logic:
      intrusion  → door=1 AND (motion=1 OR glass=1)
      authorized → door=1 AND motion=1 AND glass=0
      no_entry   → otherwise
    """
    preds = []
    for _, row in df.iterrows():
        d, m, g = row["door_sensor"], row["motion_sensor"], row["glass_sensor"]
        if d == 1 and g == 1:
            preds.append("intrusion")
        elif d == 1 and m == 1:
            preds.append("authorized")
        else:
            preds.append("no_entry")
    return np.array(preds)


# ── Bayesian Network Classifier ───────────────────────────────────────────────
def bn_predict(model, df: pd.DataFrame) -> np.ndarray:
    ve = VariableElimination(model)
    preds = []
    classes = ["no_entry", "authorized", "intrusion"]
    for _, row in df.iterrows():
        evidence = {
            "door_sensor":   int(row["door_sensor"]),
            "motion_sensor": int(row["motion_sensor"]),
            "glass_sensor":  int(row["glass_sensor"]),
        }
        result = ve.query(["entry_status"], evidence=evidence, show_progress=False)
        states = result.state_names["entry_status"]
        values = result.values
        best = states[int(np.argmax(values))]
        preds.append(best)
    return np.array(preds)


# ── Metrics Helper ───────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, label: str) -> dict:
    classes = ["no_entry", "authorized", "intrusion"]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # False Positive Rate (macro-average over classes)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fprs = []
    for i in range(len(classes)):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    fpr = float(np.mean(fprs))

    return {
        "Model": label,
        "Accuracy":  round(acc * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall":    round(rec * 100, 2),
        "F1-Score":  round(f1 * 100, 2),
        "FPR (%)":   round(fpr * 100, 2),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_comparison(metrics_df: pd.DataFrame):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "FPR (%)"]

    fig, axes = plt.subplots(1, len(metrics_cols), figsize=(18, 5))
    fig.suptitle("Infera – Bayesian Network vs Rule-Based System", fontsize=14, fontweight="bold")

    colors = ["#4C72B0", "#DD8452"]
    for ax, metric in zip(axes, metrics_cols):
        bars = ax.bar(metrics_df["Model"], metrics_df[metric], color=colors, edgecolor="white", width=0.5)
        ax.set_title(metric, fontsize=12)
        ax.set_ylim(0, 115)
        ax.set_ylabel("(%)")
        for bar, val in zip(bars, metrics_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "comparison_metrics.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved   : {out}")


def plot_confusion_matrices(y_true, y_rb, y_bn):
    classes = ["no_entry", "authorized", "intrusion"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    titles = ["Rule-Based System", "Bayesian Network (Infera)"]
    preds  = [y_rb, y_bn]

    for ax, title, y_pred in zip(axes, titles, preds):
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices", fontsize=13)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved   : {out}")


def plot_class_distribution(df: pd.DataFrame):
    counts = df["entry_status"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=["#4878d0", "#ee854a", "#6acc65"], edgecolor="white")
    ax.set_title("Dataset – Class Distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Records")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved   : {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 55)
    print("  Infera – Model Evaluation")
    print("=" * 55)

    # Load dataset
    df = pd.read_csv(os.path.join("data", "sensor_data.csv"))
    df_model = df[["entry_status", "door_sensor", "motion_sensor", "glass_sensor"]]

    # Train / test split (80 / 20)
    train_df, test_df = train_test_split(df_model, test_size=0.20,
                                         random_state=42, stratify=df_model["entry_status"])
    y_true = test_df["entry_status"].values
    print(f"  Test-set size: {len(test_df):,} records")

    # Load BN model
    with open(os.path.join("data", "bayesian_model.pkl"), "rb") as f:
        model = pickle.load(f)

    # Predictions
    print("\n  Running Rule-Based predictions …")
    y_rb = rule_based_predict(test_df)

    print("  Running Bayesian Network predictions …")
    y_bn = bn_predict(model, test_df)

    # Metrics
    rb_metrics = compute_metrics(y_true, y_rb,  "Rule-Based")
    bn_metrics = compute_metrics(y_true, y_bn,  "Bayesian Network")
    metrics_df = pd.DataFrame([rb_metrics, bn_metrics])

    print("\n" + "=" * 55)
    print("  Results (weighted averages on test set)")
    print("=" * 55)
    print(metrics_df.to_string(index=False))

    # Detailed classification reports
    classes = ["no_entry", "authorized", "intrusion"]
    print("\n  Rule-Based – Classification Report:")
    print(classification_report(y_true, y_rb, target_names=classes, zero_division=0))
    print("  Bayesian Network – Classification Report:")
    print(classification_report(y_true, y_bn, target_names=classes, zero_division=0))

    # Save metrics CSV
    metrics_df.to_csv(os.path.join("data", "evaluation_metrics.csv"), index=False)
    print(f"  Metrics CSV  : {os.path.join('data','evaluation_metrics.csv')}")

    # Plots
    print("\n  Generating plots …")
    plot_comparison(metrics_df)
    plot_confusion_matrices(y_true, y_rb, y_bn)
    plot_class_distribution(df)

    print("\n  Evaluation complete. ✓")


if __name__ == "__main__":
    main()
