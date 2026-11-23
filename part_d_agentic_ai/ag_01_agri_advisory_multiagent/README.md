# AG-01 — Agricultural Advisory Multi-Agent System

## 1. Overview
- **Domain**: Agriculture
- **Spec**: Bleep Education Part D (Agentic AI)

### Problem Statement
Coordinates soil nutrient testing and local micro-climate forecast tools to generate certified agronomic guidance.

> **Disclaimer**: Advisory workflow system. Large-scale fertilizer or pesticide deployment requires agronomy officer sign-off.

## 2. Multi-Agent Roles
- SoilAnalystAgent
- WeatherForecastAgent
- AgronomyAdvisoryAgent

## 3. Setup & Execution
```bash
pip install -r requirements.txt
pytest tests/ -v
streamlit run app.py
```
