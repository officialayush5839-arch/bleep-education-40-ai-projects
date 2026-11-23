# E2E Test Infra: Bleep Education 40 Industry AI Projects

## Test Philosophy
- **Opaque-box & Requirement-Driven**: Tests are designed directly from the Bleep Education assignment specification and acceptance criteria.
- **Hermetic & Independent**: Every project test suite must execute cleanly in an isolated offline environment without requiring network connectivity, external paid APIs, or running background server processes.
- **Methodology**: 4-Tier verification hierarchy:
  * **Tier 1: Feature Coverage (≥5 tests per project domain)**: Verifies data loading, preprocessing, model initialization, inference execution, and metric reporting.
  * **Tier 2: Boundary & Corner Cases (≥5 tests per project domain)**: Tests out-of-bounds inputs, missing values, extreme class imbalances, zero-context queries, tool exceptions, and corrupted payloads.
  * **Tier 3: Cross-Feature Combinations & State Flow**: Verifies end-to-end pipeline data flow from raw input to trained model, vector index to grounded response, and agent graph state progression.
  * **Tier 4: Real-World Workload Scenarios**: Complete application workflows including Streamlit UI imports, CLI runs, report compilation, and end-to-end user journeys.

---

## Feature Inventory & Test Mapping

| # | Project / Domain | Source Reference | Tier 1 (Unit/Feature) | Tier 2 (Boundary) | Tier 3 (Integration) | Tier 4 (App/E2E) |
|---|------------------|------------------|:---------------------:|:-----------------:|:--------------------:|:----------------:|
| 1 | ML-01: Predictive Maintenance | Bleep Spec p. 2 | 5 tests | 5 tests | 3 tests | 2 tests |
| 2 | ML-02: Patient Readmission | Bleep Spec p. 2-3 | 5 tests | 5 tests | 3 tests | 2 tests |
| 3 | ML-03: CC Fraud Detection | Bleep Spec p. 3 | 5 tests | 5 tests | 3 tests | 2 tests |
| 4 | ML-04: Telecom Churn | Bleep Spec p. 3 | 5 tests | 5 tests | 3 tests | 2 tests |
| 5 | ML-05: Crop Recommendation | Bleep Spec p. 4 | 5 tests | 5 tests | 3 tests | 2 tests |
| 6 | ML-06: Property Price | Bleep Spec p. 4 | 5 tests | 5 tests | 3 tests | 2 tests |
| 7 | ML-07: Hotel Cancellation | Bleep Spec p. 4-5 | 5 tests | 5 tests | 3 tests | 2 tests |
| 8 | ML-08: Solar Forecasting | Bleep Spec p. 5 | 5 tests | 5 tests | 3 tests | 2 tests |
| 9 | ML-09: Customer Segmentation | Bleep Spec p. 5 | 5 tests | 5 tests | 3 tests | 2 tests |
| 10 | ML-10: Employee Attrition | Bleep Spec p. 5-6 | 5 tests | 5 tests | 3 tests | 2 tests |
| 11 | DL-01: Waste Classification | Bleep Spec p. 6 | 5 tests | 5 tests | 3 tests | 2 tests |
| 12 | DL-02: Plant Leaf Disease | Bleep Spec p. 6-7 | 5 tests | 5 tests | 3 tests | 2 tests |
| 13 | DL-03: Chest X-Ray Pneumonia | Bleep Spec p. 7 | 5 tests | 5 tests | 3 tests | 2 tests |
| 14 | DL-04: Traffic Sign Recognition | Bleep Spec p. 7 | 5 tests | 5 tests | 3 tests | 2 tests |
| 15 | DL-05: Surface Defect Detection | Bleep Spec p. 7 | 5 tests | 5 tests | 3 tests | 2 tests |
| 16 | DL-06: Movie Sentiment | Bleep Spec p. 7-8 | 5 tests | 5 tests | 3 tests | 2 tests |
| 17 | DL-07: Human Activity (HAR) | Bleep Spec p. 8 | 5 tests | 5 tests | 3 tests | 2 tests |
| 18 | DL-08: Electricity Forecasting | Bleep Spec p. 8 | 5 tests | 5 tests | 3 tests | 2 tests |
| 19 | DL-09: Fashion Item Classification | Bleep Spec p. 8 | 5 tests | 5 tests | 3 tests | 2 tests |
| 20 | DL-10: Urban Sound Classification | Bleep Spec p. 8-9 | 5 tests | 5 tests | 3 tests | 2 tests |
| 21 | GEN-01 to GEN-10: RAG Projects | Bleep Spec p. 9-14 | 50 tests (5/ea) | 50 tests (5/ea) | 30 tests (3/ea) | 20 tests (2/ea) |
| 22 | AG-01 to AG-10: Agentic AI Projects | Bleep Spec p. 14-18 | 50 tests (5/ea) | 50 tests (5/ea) | 30 tests (3/ea) | 20 tests (2/ea) |
| **Total** | **All 40 Projects** | | **≥200 tests** | **≥200 tests** | **≥120 tests** | **≥80 tests** |

---

## Test Architecture & Invocation

### Directory Conventions
Within each project folder, tests are organized under `tests/`:
```
tests/
├── __init__.py
├── test_data.py          # Data ingestion, schema validation, synthetic generation fallback
├── test_model.py         # Model architecture, training steps, prediction shapes, weights
└── test_pipeline.py      # End-to-end execution, evaluation metrics, report generation
```

### Global Test Runner: `run_all_tests.py`
A centralized master test runner located at workspace root scans all 40 project directories:
```bash
python run_all_tests.py
```
- Discovers all subdirectories under `part_a_machine_learning/`, `part_b_deep_learning/`, `part_c_generative_ai_rag/`, and `part_d_agentic_ai/`.
- Executes `pytest` per project.
- Logs aggregate statistics (Passed, Failed, Skipped, Execution Time).
- Exits with status code 0 only if 100% of discovered project test suites pass.

---

## Real-World Workload Scenarios (Tier 4)
| # | Scenario | Domains Exercised | Verification Criteria |
|---|----------|-------------------|-----------------------|
| 1 | Full ML Training & Inference Pipeline | ML-01 to ML-10 | Raw data ingested -> preprocessed -> model trained -> metrics logged -> report generated |
| 2 | Deep Learning Visual & Sequence Inference | DL-01 to DL-10 | PyTorch tensor input -> forward pass -> softmax probability -> Grad-CAM / feature map output |
| 3 | Enterprise RAG Grounded Q&A and Abstention | GEN-01 to GEN-10 | Knowledge base indexed -> semantic retrieval -> cited answer generated; out-of-scope query cleanly abstained |
| 4 | Multi-Agent Deliberation & HITL Gate | AG-01 to AG-10 | State graph initialized -> tool calls executed -> trace logged -> consequential action halts for human approval |
| 5 | Master Workspace Verification | Universal | `run_all_tests.py` executes all 40 test suites; 100% pass rate achieved |

---

## Quality & Anti-Cheating Invariants
1. **No Hardcoded Test Returns**: Assertions verify non-trivial tensor/array calculations, dynamic metric values, and real file outputs.
2. **Deterministic Offline Fallbacks**: Synthetic generators use fixed random seeds and adhere strictly to real schema definitions.
3. **Data Leakage Prohibition**: Strict validation checks that test splits are isolated from training splits (e.g. patient-level grouping, chronological splits, target feature dropping).
4. **Mandatory Disclaimers**: Healthcare (ML-02, DL-03, GEN-02), Legal (GEN-04), and Financial (AG-03) systems must contain programmatic assertions verifying disclaimer presence.
