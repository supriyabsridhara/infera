"""
Infera – Flask Web Application
Run with:  python app_flask.py
Then open:  http://localhost:5000
"""

import os
import io
import base64
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from flask import Flask, render_template, request, jsonify
from train_model import infer, load_model

app = Flask(__name__)

# ── Load model once at startup ────────────────────────────────────────────────
MODEL_PATH = os.path.join("data", "bayesian_model.pkl")
model = load_model(MODEL_PATH)

# ── Helper: generate a base64-encoded PNG chart ────────────────────────────────
def prob_bar_chart(probs: dict) -> str:
    labels = ["No Entry", "Authorized", "Intrusion"]
    keys   = ["no_entry", "authorized", "intrusion"]
    values = [probs.get(k, 0) * 100 for k in keys]
    colors = ["#4878d0", "#6acc65", "#ee854a"]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.5)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Probability (%)", fontsize=10)
    ax.set_title("Entry Status Probabilities", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    # Load evaluation metrics if available
    metrics = []
    metrics_path = os.path.join("data", "evaluation_metrics.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        metrics = df.to_dict(orient="records")

    # Dataset stats
    dataset_stats = {}
    data_path = os.path.join("data", "sensor_data.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        counts = df["entry_status"].value_counts().to_dict()
        total  = len(df)
        dataset_stats = {
            "total": f"{total:,}",
            "no_entry":   {"count": f"{counts.get('no_entry',0):,}",
                           "pct":   f"{counts.get('no_entry',0)/total*100:.1f}"},
            "authorized": {"count": f"{counts.get('authorized',0):,}",
                           "pct":   f"{counts.get('authorized',0)/total*100:.1f}"},
            "intrusion":  {"count": f"{counts.get('intrusion',0):,}",
                           "pct":   f"{counts.get('intrusion',0)/total*100:.1f}"},
        }

    # Encode plot images as base64
    plots = {}
    for name in ["comparison_metrics", "confusion_matrices", "class_distribution"]:
        path = os.path.join("data", "plots", f"{name}.png")
        if os.path.exists(path):
            with open(path, "rb") as f:
                plots[name] = base64.b64encode(f.read()).decode("utf-8")

    # CPT summary
    cpt_data = []
    for cpd in model.get_cpds():
        cpt_data.append({"variable": cpd.variable, "table": str(cpd)})

    return render_template("index.html",
                           metrics=metrics,
                           dataset_stats=dataset_stats,
                           plots=plots,
                           cpt_data=cpt_data)


@app.route("/predict", methods=["POST"])
def predict():
    data   = request.get_json()
    door   = int(data.get("door",   0))
    motion = int(data.get("motion", 0))
    glass  = int(data.get("glass",  0))

    result = infer(model, door=door, motion=motion, glass=glass)

    chart = prob_bar_chart(result)

    return jsonify({
        "no_entry":          round(result.get("no_entry",   0) * 100, 1),
        "authorized":        round(result.get("authorized", 0) * 100, 1),
        "intrusion":         round(result.get("intrusion",  0) * 100, 1),
        "risk_level":        result["risk_level"],
        "recommended_action": result["recommended_action"],
        "chart_b64":         chart,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
