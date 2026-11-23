# AG-09 — Inventory Reordering Multi-Agent

## 1. Overview
- **Domain**: Supply Chain
- **Spec**: Bleep Education Part D (Agentic AI)

### Problem Statement
Monitors warehouse inventory, forecasts depletion timelines, and drafts purchase orders subject to procurement officer approval.

> **Disclaimer**: Supply chain optimization tool. Supplier purchase orders require human procurement approval.

## 2. Multi-Agent Roles
- StockLevelMonitorAgent
- DemandForecasterAgent
- PurchaseDrafterAgent

## 3. Setup & Execution
```bash
pip install -r requirements.txt
pytest tests/ -v
streamlit run app.py
```
