"""
app.py
Interactive Streamlit Application for ML-02: Hospital Patient Readmission Risk Prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

from src.data_loader import load_data
from src.preprocessor import preprocess_data
from src.model import MLModel
from src.evaluate import evaluate_predictions

st.set_page_config(page_title="Hospital Patient Readmission Risk Prediction", page_icon="📈", layout="wide")

st.title("📈 ML-02: Hospital Patient Readmission Risk Prediction")
st.caption("Domain: Healthcare | Bleep Education 40 AI Projects")
st.info("""Strictly restricted to educational and care-coordination planning. Not a clinical diagnostic decision tool.""")

@st.cache_data
def get_trained_pipeline():
    df = load_data()
    prep = preprocess_data(df)
    model = MLModel()
    model.train(prep["X_train"], prep["y_train"])
    return prep, model

prep, model = get_trained_pipeline()

tabs = st.tabs(["🚀 Interactive Inference", "📊 Quantitative Evaluation", "📄 Dataset Inspector"])

with tabs[0]:
    st.subheader("Run Model Inference")
    st.markdown("Adjust feature values to generate instant predictions:")
    sample_idx = st.slider("Select Sample Profile Index", min_value=0, max_value=len(prep["X_test"])-1, value=5)
    sample_feat = prep["X_test"][sample_idx:sample_idx+1]
    
    pred = model.predict(sample_feat)[0]
    actual = prep["y_test"][sample_idx]

    col1, col2 = st.columns(2)
    col1.metric("Predicted Value", f"{pred}")
    col2.metric("Ground Truth Actual", f"{actual}")

with tabs[1]:
    st.subheader("Evaluation Performance Metrics")
    metrics_path = Path("results/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        cols = st.columns(len(metrics))
        for i, (k, v) in enumerate(metrics.items()):
            cols[i].metric(k.replace("_", " ").title(), str(v))
    else:
        st.warning("No metrics.json found.")

with tabs[2]:
    st.subheader("Raw Dataset Preview")
    st.dataframe(prep["raw_df"].head(100))
