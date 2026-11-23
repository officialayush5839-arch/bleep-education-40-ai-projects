"""
app.py
Interactive Streamlit Application for DL-03: Chest X-Ray Pneumonia Screening.
"""

import streamlit as st
import torch
import json
from pathlib import Path

from src.data_loader import CLASS_NAMES, load_dataset
from src.model import DeepClassifier
from src.preprocessor import create_dataloaders
from src.evaluate import evaluate_model

st.set_page_config(page_title="Chest X-Ray Pneumonia Screening", page_icon="🧠", layout="wide")

st.title("🧠 DL-03: Chest X-Ray Pneumonia Screening")
st.caption("Domain: Healthcare | Bleep Education 40 AI Projects")
st.info("""Strictly restricted to educational research. Must NOT be presented or used as a clinical diagnostic tool.""")

@st.cache_resource
def load_trained_model():
    train_ds, test_ds = load_dataset()
    train_loader, test_loader = create_dataloaders(train_ds, test_ds)
    model = DeepClassifier()
    model.eval()
    return test_ds, model

test_ds, model = load_trained_model()

tabs = st.tabs(["🚀 Model Inference", "📊 Performance Metrics", "ℹ️ Architecture"])

with tabs[0]:
    st.subheader("Live Neural Inference")
    sample_idx = st.slider("Select Sample Index", 0, len(test_ds)-1, 0)
    x_sample, y_true = test_ds[sample_idx]
    
    with torch.no_grad():
        out = model(x_sample.unsqueeze(0))
        if "vision_2d" == "sequence_lstm":
            pred_val = round(float(out[0][0]), 4)
            st.metric("Predicted Output", pred_val)
            st.metric("Ground Truth Value", round(float(y_true[0]), 4))
        else:
            pred_idx = int(out.argmax(dim=1)[0])
            pred_label = CLASS_NAMES[pred_idx]
            true_label = CLASS_NAMES[int(y_true)]
            col1, col2 = st.columns(2)
            col1.metric("Predicted Class", pred_label)
            col2.metric("Ground Truth", true_label)

with tabs[1]:
    st.subheader("Test Evaluation Metrics")
    metrics_file = Path("results/metrics.json")
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        cols = st.columns(len(metrics))
        for i, (k, v) in enumerate(metrics.items()):
            cols[i].metric(k.replace("_", " ").title(), str(v))
    else:
        st.warning("No metrics.json found.")

with tabs[2]:
    st.subheader("Model Architecture")
    st.code(str(model), language="text")
