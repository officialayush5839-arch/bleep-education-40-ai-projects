# AG-04 — Recruitment Administration & Screening

## 1. Overview
- **Domain**: Human Resources
- **Spec**: Bleep Education Part D (Agentic AI)

### Problem Statement
Extracts candidate skills and drafts customized interview questions without ranking or automatically rejecting candidates.

> **Disclaimer**: HR screening assistant. Does not make autonomous candidate rejection or ranking decisions.

## 2. Multi-Agent Roles
- ResumeParserAgent
- JDMatcherAgent
- InterviewQuestionDrafter

## 3. Setup & Execution
```bash
pip install -r requirements.txt
pytest tests/ -v
streamlit run app.py
```
