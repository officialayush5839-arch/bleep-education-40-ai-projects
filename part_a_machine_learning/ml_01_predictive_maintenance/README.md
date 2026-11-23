# ML-01 — Predictive Maintenance & Equipment Failure

## 1. Overview
- **Domain**: Manufacturing
- **Spec**: Bleep Education Part A (Machine Learning)

### Problem Statement
Predict industrial equipment failure using sensor telemetry (temperature, rotational speed, torque, tool wear) to optimize maintenance intervals and prevent catastrophic downtime.

> **Disclaimer**: Industrial predictive model for operational decision support. Regular scheduled physical inspections remain mandatory.

## 2. Setup & Execution
```bash
pip install -r requirements.txt
pytest tests/ -v
streamlit run app.py
```
