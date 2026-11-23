# AG-08 — Security Incident Response Multi-Agent

## 1. Overview
- **Domain**: Cyber Defense
- **Spec**: Bleep Education Part D (Agentic AI)

### Problem Statement
Analyzes security telemetry, maps IOCs against MITRE ATT&CK, and drafts incident containment playbooks without performing disruptive network actions.

> **Disclaimer**: Cyber defense triage tool. Network containment actions strictly require SOC Lead approval.

## 2. Multi-Agent Roles
- SIEMLogAnalystAgent
- ThreatIntelMapper
- ResponsePlaybookDrafter

## 3. Setup & Execution
```bash
pip install -r requirements.txt
pytest tests/ -v
streamlit run app.py
```
