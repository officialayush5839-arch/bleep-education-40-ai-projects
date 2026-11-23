# AG-10 — Educational Content Development Agent

## 1. Overview
- **Domain**: EdTech
- **Spec**: Bleep Education Part D (Agentic AI)

### Problem Statement
Generates structured lesson outlines, instructional reading content, and formative assessments through coordinated pedagogical stages.

> **Disclaimer**: Instructional design assistant. Content publication requires educator pedagogical review.

## 2. Multi-Agent Roles
- CurriculumArchitectAgent
- ContentWriterAgent
- PedagogicalReviewerAgent

## 3. Setup & Execution
```bash
pip install -r requirements.txt
pytest tests/ -v
streamlit run app.py
```
