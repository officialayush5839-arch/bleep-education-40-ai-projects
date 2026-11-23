# AG-02 — Customer Support Ticket Triage & Resolution

## 1. Overview
- **Domain**: Customer Service
- **Spec**: Bleep Education Part D (Agentic AI)

### Problem Statement
Classifies incoming customer service tickets, retrieves policy documentation, drafts responses, and enforces approval before financial actions.

> **Disclaimer**: Customer support assistant. Financial credits and account cancellations require supervisor approval.

## 2. Multi-Agent Roles
- TicketClassifierAgent
- PolicyRetrievalAgent
- ResponseDrafterAgent

## 3. Setup & Execution
```bash
pip install -r requirements.txt
pytest tests/ -v
streamlit run app.py
```
