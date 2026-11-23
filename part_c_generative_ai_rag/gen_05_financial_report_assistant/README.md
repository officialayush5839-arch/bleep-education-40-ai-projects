# GEN-05 — Financial Report Analysis Assistant

## 1. Project Overview
- **Domain**: Finance
- **Task**: Retrieval-Augmented Generation (RAG) with Evidence Citations & Deterministic Abstention
- **Specification**: Bleep Education Internship Assignment — Part C

### Problem Statement
Answers questions about financial results, risks, and management commentary using traceable citations.

> **Disclaimer**: Informational research tool. Does not provide investment, tax, or financial advisory recommendations.

---

## 2. System Architecture & Workflow
```
[User Query] ──> [Query Normalization]
                     │
                     ▼
          [TF-IDF Vector Search] ──> Cosine Similarity Matching
                     │
         [Score Threshold Check (tau=0.18)]
          ├── Below Threshold ──> [Deterministic Abstention Message]
          └── Above Threshold ──> [Context Assembler]
                                         │
                                         ▼
                             [Grounded Extractive Synthesis]
                                         │
                                         ▼
                            [Evidence Citations + Disclaimer]
```

---

## 3. Directory Layout
```
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py
├── data/
│   ├── README.md
│   └── raw_documents/
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── chunker.py
│   ├── vectorstore.py
│   ├── rag_pipeline.py
│   ├── evaluate.py
│   └── utils.py
├── tests/
│   ├── test_data.py
│   ├── test_rag.py
│   └── test_pipeline.py
├── results/
│   └── metrics.json
└── reports/
    ├── project_report.md
    └── project_report.pdf
```

---

## 4. Setup & Execution Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Run automated tests
pytest tests/ -v

# Launch Streamlit web application
streamlit run app.py
```

---

## 5. Evaluation Results
- **Benchmark Queries**: 15 (10 In-Scope Answerable + 5 Out-of-Scope Unanswerable)
- **Citation Precision**: 100.0%
- **Abstention Accuracy**: 100.0%
- **Overall Semantic Accuracy**: 100.0%
- **Average Latency**: < 20 ms
