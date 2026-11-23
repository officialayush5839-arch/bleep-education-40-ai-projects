# ML-08 — Solar Power Generation Forecasting

## 1. Overview
- **Domain**: Renewable Energy
- **Spec**: Bleep Education Part A (Machine Learning)

### Problem Statement
Forecast short-term photovoltaic power generation using ambient weather sensor telemetry, panel module temperatures, and solar irradiance.

> **Disclaimer**: Grid dispatch forecast system. Weather forecasting variance may affect peak hourly estimations.

## 2. Setup & Execution
```bash
pip install -r requirements.txt
pytest tests/ -v
streamlit run app.py
```
