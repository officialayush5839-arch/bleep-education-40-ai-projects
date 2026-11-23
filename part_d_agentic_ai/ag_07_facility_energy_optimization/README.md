# AG-07 — Commercial Facility Energy Optimization

## 1. Overview
- **Domain**: Energy Efficiency
- **Spec**: Bleep Education Part D (Agentic AI)

### Problem Statement
Monitors building power consumption, detects energy consumption anomalies, and drafts HVAC schedule optimizations subject to facility manager approval.

> **Disclaimer**: Building energy advisory tool. HVAC setpoint overrides require on-site building engineer sign-off.

## 2. Multi-Agent Roles
- TelemetryIngestionAgent
- AnomalyDetectorAgent
- HVACOptimizationAgent

## 3. Setup & Execution
```bash
pip install -r requirements.txt
pytest tests/ -v
streamlit run app.py
```
