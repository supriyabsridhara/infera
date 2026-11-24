"""
Infera – Smart Security Monitoring Dashboard
Run with:  streamlit run app.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from train_model import infer, load_model

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Infera – Smart Security Monitoring",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.4rem; font-weight:800; color:#1a1a2e; }
    .sub-title   { font-size:1.1rem; color:#555; margin-bottom:1rem; }
    .metric-card { background:#f0f4ff; border-radius:12px; padding:1rem 1.5rem;
                   text-align:center; box-shadow:0 2px 8px rgba(0,0,0,.08); }
    .metric-label{ font-size:.85rem; color:#666; }
    .metric-value{ font-size:2rem; font-weight:700; }
    .risk-low    { color:#27ae60; }
    .risk-moderate{ color:#e67e22; }
    .risk-high   { color:#e74c3c; }
    .action-box  { border-radius:10px; padding:1rem 1.4rem;
                   font-size:1.05rem; font-weight:600; margin-top:1rem; }
    .section-hdr { font-size:1.3rem; font-weight:700; color:#1a1a2e;
                   margin-top:2rem; border-bottom:2px solid #4C72B0;
                   padding-bottom:.3rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load Model (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Bayesian Network model …")
def get_model():
    model_path = os.path.join("data", "bayesian_model.pkl")
    if not os.path.exists(model_path):
        st.error("Model file not found. Please run `train_model.py` first.")
        st.stop()
    return load_model(model_path)

model = get_model()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">Infera</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Intelligent Intrusion Detection &amp; Risk Prediction '
    'using Bayesian Networks</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – About
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## About Infera")
    st.info(
        "Infera uses a **Bayesian Network** to probabilistically estimate the "
        "likelihood of unauthorized building entry based on sensor readings.\n\n"
        "**Sensors monitored:**\n"
        "- Door Sensor\n"
        "- Motion Sensor\n"
        "- Glass-Break Sensor"
    )
    st.markdown("### Risk Thresholds")
    st.markdown("""
| Level    | Intrusion Probability |
|----------|-----------------------|
| Low      | < 30 %                |
| Moderate | 30 – 69.9 %           |
| High     | ≥ 70 %                |
""")
    st.markdown("---")
    st.caption("Built with Python · pgmpy · Streamlit")

# ─────────────────────────────────────────────────────────────────────────────
# Main layout: left = inputs, right = results
# ─────────────────────────────────────────────────────────────────────────────
col_in, col_out = st.columns([1, 2], gap="large")

with col_in:
    st.markdown('<p class="section-hdr">Sensor Observations</p>', unsafe_allow_html=True)

    door_val   = st.radio("Door Sensor",        ["Triggered", "Not Triggered"], horizontal=True)
    motion_val = st.radio("Motion Sensor",      ["Triggered", "Not Triggered"], horizontal=True)
    glass_val  = st.radio("Glass-Break Sensor", ["Triggered", "Not Triggered"], horizontal=True)

    door   = 1 if door_val   == "Triggered" else 0
    motion = 1 if motion_val == "Triggered" else 0
    glass  = 1 if glass_val  == "Triggered" else 0

    st.markdown("**Current sensor combination:**")
    st.code(
        f"Door: {'Triggered' if door else 'Not Triggered'} · "
        f"Motion: {'Triggered' if motion else 'Not Triggered'} · "
        f"Glass: {'Triggered' if glass else 'Not Triggered'}"
    )

    predict_btn = st.button("Predict Intrusion Risk", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
with col_out:
    st.markdown('<p class="section-hdr">Risk Assessment</p>', unsafe_allow_html=True)

    if predict_btn:
        result = infer(model, door=door, motion=motion, glass=glass)

        p_no_entry   = result.get("no_entry",   0.0)
        p_authorized = result.get("authorized", 0.0)
        p_intrusion  = result.get("intrusion",  0.0)
        risk         = result["risk_level"]
        action       = result["recommended_action"]

        # ── Probability Metrics ──────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("No Entry",          f"{p_no_entry*100:.1f}%")
        with c2:
            st.metric("Authorized",        f"{p_authorized*100:.1f}%")
        with c3:
            st.metric("Intrusion",         f"{p_intrusion*100:.1f}%")

        # ── Risk Level ───────────────────────────────────────────────────────
        risk_class = {"Low": "risk-low", "Moderate": "risk-moderate", "High": "risk-high"}[risk]
        risk_bg    = {"Low": "#eafaf1", "Moderate": "#fef9e7", "High": "#fdf2f8"}[risk]
        risk_border= {"Low": "#27ae60", "Moderate": "#e67e22", "High": "#e74c3c"}[risk]

        st.markdown(
            f'<div class="action-box" style="background:{risk_bg};'
            f'border-left:5px solid {risk_border};">'
            f'<span class="{risk_class}">Risk Level: {risk}</span><br>'
            f'<span style="font-weight:400;font-size:.95rem">{action}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Probability Bar Chart ────────────────────────────────────────────
        st.markdown("#### Probability Distribution")
        fig, ax = plt.subplots(figsize=(7, 2.8))
        labels = ["No Entry", "Authorized", "Intrusion"]
        values = [p_no_entry, p_authorized, p_intrusion]
        colors = ["#4878d0", "#6acc65", "#ee854a"]
        bars = ax.barh(labels, [v * 100 for v in values],
                       color=colors, edgecolor="white", height=0.5)
        ax.set_xlim(0, 110)
        ax.set_xlabel("Probability (%)", fontsize=10)
        ax.set_title("Entry Status Probabilities", fontsize=11, fontweight="bold")
        for bar, val in zip(bars, values):
            ax.text(val * 100 + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val*100:.1f}%", va="center", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        st.info("Select sensor states and click **Predict Intrusion Risk** to see results.")

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Results Section
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-hdr">Model Evaluation Results</p>', unsafe_allow_html=True)

metrics_path = os.path.join("data", "evaluation_metrics.csv")
plots_dir    = os.path.join("data", "plots")

if os.path.exists(metrics_path):
    mdf = pd.read_csv(metrics_path)

    t1, t2, t3 = st.tabs(["Metrics Table", "Comparison Chart", "Confusion Matrices"])

    with t1:
        st.dataframe(
            mdf.style.highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1-Score"],
                color="#d5f5e3",
            ).highlight_min(
                subset=["FPR (%)"],
                color="#d5f5e3",
            ),
            use_container_width=True,
        )
        st.caption("Green highlights = better score for each metric.")

    with t2:
        cmp_path = os.path.join(plots_dir, "comparison_metrics.png")
        if os.path.exists(cmp_path):
            st.image(cmp_path, use_container_width=True)
        else:
            st.warning("Run `evaluate.py` to generate charts.")

    with t3:
        cm_path = os.path.join(plots_dir, "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.warning("Run `evaluate.py` to generate charts.")
else:
    st.warning("No evaluation results found. Run `evaluate.py` first.")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Overview Section
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-hdr">Training Dataset Overview</p>', unsafe_allow_html=True)

data_path = os.path.join("data", "sensor_data.csv")
if os.path.exists(data_path):
    df = pd.read_csv(data_path)

    da1, da2 = st.columns([1, 2])

    with da1:
        dist_path = os.path.join(plots_dir, "class_distribution.png")
        if os.path.exists(dist_path):
            st.image(dist_path, caption="Class distribution in training data")

    with da2:
        st.markdown(f"**Total records:** {len(df):,}")
        counts = df["entry_status"].value_counts()
        for cls, cnt in counts.items():
            pct = cnt / len(df) * 100
            st.progress(pct / 100, text=f"{cls.replace('_',' ').title()}: {cnt:,} ({pct:.1f}%)")
        st.markdown("**Sample records:**")
        st.dataframe(
            df[["entry_status","door_sensor","motion_sensor","glass_sensor"]].head(10),
            use_container_width=True,
        )
else:
    st.warning("Dataset not found. Run `generate_dataset.py` first.")

# ─────────────────────────────────────────────────────────────────────────────
# Bayesian Network CPTs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-hdr">Bayesian Network – CPT Viewer</p>', unsafe_allow_html=True)

with st.expander("View Conditional Probability Tables"):
    for cpd in model.cpds:
        st.markdown(f"**Node: `{cpd.variable}`**")
        st.text(str(cpd))
        st.markdown("---")

st.markdown(
    '<div style="text-align:center;color:#aaa;font-size:.85rem;margin-top:2rem;">'
    'Infera © 2026 · Intelligent Intrusion Detection using Bayesian Networks'
    '</div>',
    unsafe_allow_html=True,
)
