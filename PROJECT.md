# Project: Bleep Education 40 Industry AI Projects (Comprehensive Master Execution Plan)

## Executive Summary & Architecture
This project implements all 40 industry-grade AI internship projects from the official Bleep Education assignment specification across four specialized domains:
1. **Part A: Machine Learning (ML-01 to ML-10)** — Tabular classification, regression, imbalanced learning, time series forecasting, customer clustering, and explainable AI (SHAP/fairness).
2. **Part B: Deep Learning (DL-01 to DL-10)** — Computer vision (CNNs, MobileNetV2, ResNet, defect detection, medical chest X-ray screening), natural language processing (LSTM sentiment analysis), sensor time-series (1D CNN HAR), sequence forecasting (GRU electricity load), and audio classification (Mel-Spectrogram 2D CNN).
3. **Part C: Generative AI / RAG (GEN-01 to GEN-10)** — Retrieval-Augmented Generation across 10 verticals (HR, Medical, Legal, Finance, Agriculture, Cybersecurity, Developer Docs, Climate Science, Technical Education), with semantic chunking, vector indexing (Chroma/FAISS), evidence citation `[Doc, Page, Section]`, deterministic abstention, and offline evaluation harnesses.
4. **Part D: Agentic AI (AG-01 to AG-10)** — Multi-agent stateful graph architectures (LangGraph/state machine) featuring >=2 specialized agents, >=2 functional tools, execution trace logging (`results/agent_trace.json`), Human-in-the-Loop (HITL) approval gates for consequential actions, and strict safety guardrails.

---

## Universal Repository Layout (Per Project Standard)
Every one of the 40 project folders adheres strictly to the 10 standard deliverables defined in Part E of the Bleep specification:

```
<project_dir>/
├── README.md                      # Comprehensive documentation (Overview, Architecture, Setup, Usage, Metrics, Citations)
├── requirements.txt               # Exact pinned dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Standard Python/data ignore rules
├── app.py                         # Interactive Streamlit application interface
├── data/
│   ├── README.md                  # Dataset schema, source, acquisition instructions, and data dictionary
│   └── raw/                       # Cached public or synthetic dataset
├── notebooks/
│   └── exploratory_analysis.ipynb # Interactive EDA, feature distributions, baseline experiments
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Acquisition, validation, synthetic fallback generator
│   ├── preprocessor.py           # Feature engineering, scaling, tokenization, chunking
│   ├── model.py                  # Model architecture, training pipeline, agent state graph
│   ├── evaluate.py               # Quantitative evaluation metrics, confusion matrix, plots
│   └── utils.py                  # Common utilities, logger, config loaders
├── tests/
│   ├── __init__.py
│   ├── test_data.py              # Schema validation, missingness checks, split integrity
│   ├── test_model.py             # Inference checks, shape consistency, reproducibility
│   └── test_pipeline.py          # End-to-end integration test (offline-compatible)
├── results/
│   ├── metrics.json              # Key evaluation metrics (accuracy, F1, RMSE, RAGAS, trace stats)
│   └── evaluation_plots/         # Confusion matrix, ROC/PR curves, training loss curves
└── reports/
    ├── project_report.md         # Full 12-section technical project report
    └── project_report.pdf        # Programmatically compiled PDF report (ReportLab)
```

---

## Feature Inventory & Project Assignment

| # | Project ID | Project Title | Domain | Milestone | Source Reference |
|---|------------|---------------|--------|-----------|------------------|
| 1 | ML-01 | Predictive Maintenance & Equipment Failure | Manufacturing | Milestone 2 | Bleep Spec p. 2, UCI AI4I 2020 |
| 2 | ML-02 | Hospital Patient Readmission Risk Prediction | Healthcare | Milestone 2 | Bleep Spec p. 2-3, UCI Diabetes 130 |
| 3 | ML-03 | Credit Card Fraud Detection | FinTech | Milestone 2 | Bleep Spec p. 3, Kaggle CC Fraud |
| 4 | ML-04 | Telecom Customer Churn Prediction | Telecommunications | Milestone 2 | Bleep Spec p. 3, IBM Telco Churn |
| 5 | ML-05 | Crop Recommendation System | Agriculture | Milestone 2 | Bleep Spec p. 4, Kaggle Crop Rec |
| 6 | ML-06 | Real Estate Property Price Prediction | Real Estate | Milestone 2 | Bleep Spec p. 4, Kaggle Ames Housing |
| 7 | ML-07 | Hotel Booking Cancellation Prediction | Hospitality | Milestone 2 | Bleep Spec p. 4-5, Kaggle Hotel Booking |
| 8 | ML-08 | Solar Power Generation Forecasting | Energy / Renewables | Milestone 2 | Bleep Spec p. 5, Kaggle Solar Power |
| 9 | ML-09 | Retail Customer Segmentation | E-Commerce / Retail | Milestone 2 | Bleep Spec p. 5, UCI Online Retail |
| 10 | ML-10 | Employee Attrition Analytics | Human Resources | Milestone 2 | Bleep Spec p. 5-6, IBM HR Analytics |
| 11 | DL-01 | Automated Waste Classification | Environmental Tech | Milestone 3 | Bleep Spec p. 6, Kaggle Garbage Class. |
| 12 | DL-02 | Plant Leaf Disease Classification | AgriTech | Milestone 3 | Bleep Spec p. 6-7, Kaggle PlantVillage |
| 13 | DL-03 | Chest X-Ray Pneumonia Classification | Healthcare | Milestone 3 | Bleep Spec p. 7, Kaggle Chest X-Ray |
| 14 | DL-04 | Road Traffic Sign Recognition | Autonomous Systems | Milestone 3 | Bleep Spec p. 7, GTSRB Benchmark |
| 15 | DL-05 | Industrial Surface Defect Classification | Manufacturing | Milestone 3 | Bleep Spec p. 7, NEU Surface Defect |
| 16 | DL-06 | Movie Review Sentiment Classification | Natural Language | Milestone 3 | Bleep Spec p. 7-8, Stanford IMDB |
| 17 | DL-07 | Smartphone Human Activity Recognition | IoT / Mobile Sensing | Milestone 3 | Bleep Spec p. 8, UCI HAR Dataset |
| 18 | DL-08 | Household Electricity Consumption Forecasting | Energy Management | Milestone 3 | Bleep Spec p. 8, UCI Household Power |
| 19 | DL-09 | Fashion Item Classification | E-Commerce / Retail | Milestone 3 | Bleep Spec p. 8, Fashion-MNIST |
| 20 | DL-10 | Urban Sound Classification | Smart Cities / Audio | Milestone 3 | Bleep Spec p. 8-9, UrbanSound8K |
| 21 | GEN-01 | Enterprise HR Policy Assistant | Human Resources | Milestone 4 | Bleep Spec p. 9-10 |
| 22 | GEN-02 | Medical Research Literature Assistant | Life Sciences | Milestone 4 | Bleep Spec p. 10 |
| 23 | GEN-03 | University Academic Information Assistant | Higher Education | Milestone 4 | Bleep Spec p. 10 |
| 24 | GEN-04 | Legal Document Research Assistant | Legal Tech | Milestone 4 | Bleep Spec p. 10-11 |
| 25 | GEN-05 | Financial Report Analysis Assistant | Corporate Finance | Milestone 4 | Bleep Spec p. 11 |
| 26 | GEN-06 | Agricultural Extension Knowledge Assistant | AgriTech | Milestone 4 | Bleep Spec p. 11-12 |
| 27 | GEN-07 | Cybersecurity Awareness Assistant | Information Security | Milestone 4 | Bleep Spec p. 12 |
| 28 | GEN-08 | Software Developer Documentation Assistant | Developer Tools | Milestone 4 | Bleep Spec p. 12 |
| 29 | GEN-09 | Climate & Sustainability Research Assistant | Environmental Science | Milestone 4 | Bleep Spec p. 12-13 |
| 30 | GEN-10 | Technical Training & Revision Assistant | EdTech | Milestone 4 | Bleep Spec p. 13 |
| 31 | AG-01 | Multi-Agent Agricultural Advisory System | Agriculture | Milestone 5 | Bleep Spec p. 14-15 |
| 32 | AG-02 | Customer Support Ticket Triage & Resolution | Customer Service | Milestone 5 | Bleep Spec p. 15 |
| 33 | AG-03 | Financial Research & Market Analysis | FinTech | Milestone 5 | Bleep Spec p. 15 |
| 34 | AG-04 | Recruitment Administration & Screening | HR Tech | Milestone 5 | Bleep Spec p. 15 |
| 35 | AG-05 | Travel Itinerary & Budget Planning | Travel & Tourism | Milestone 5 | Bleep Spec p. 15-16 |
| 36 | AG-06 | Research Paper Analysis & Matrix Workflow | Academia / Research | Milestone 5 | Bleep Spec p. 16 |
| 37 | AG-07 | Commercial Facility Energy Optimization | Energy Efficiency | Milestone 5 | Bleep Spec p. 16 |
| 38 | AG-08 | Security Incident Triage & Playbook Draft | Cyber Defense | Milestone 5 | Bleep Spec p. 16-17 |
| 39 | AG-09 | Inventory Monitoring & Reordering Assistant | Supply Chain | Milestone 5 | Bleep Spec p. 16-17 |
| 40 | AG-10 | Educational Content Development Multi-Agent | EdTech | Milestone 5 | Bleep Spec p. 17 |
| 41 | E2E-INFRA | Master Test Infrastructure & Quality Harness | Universal | Milestone 1 | TEST_INFRA.md, run_all_tests.py |
| 42 | E2E-VERIFY | Master Verification, 100% Test Pass & Audit | Universal | Milestone 6 | TEST_READY.md, Forensic Audit |

---

## Master Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|--------------|--------|
| M1 | Infrastructure & E2E Testing Harness | Global test runner (`run_all_tests.py`), universal project scaffolder, synthetic data engines, `TEST_INFRA.md` | None | DONE |
| M2 | Part A: Machine Learning (ML-01 to ML-10) | 10 end-to-end ML projects complete with models, preprocessing, app UIs, tests, metrics, and PDF reports | M1 | IN_PROGRESS |
| M3 | Part B: Deep Learning (DL-01 to DL-10) | 10 end-to-end DL projects complete with PyTorch neural nets, train/val loops, app UIs, tests, metrics, and PDF reports | M1 | IN_PROGRESS |
| M4 | Part C: Generative AI / RAG (GEN-01 to GEN-10) | 10 end-to-end RAG projects with vector stores, citations, abstention, app UIs, 15-query test suites, and PDF reports | M1 | IN_PROGRESS |
| M5 | Part D: Agentic AI (AG-01 to AG-10) | 10 end-to-end Agentic projects with >=2 agents, >=2 tools, state machine, HITL gates, app UIs, 10-scenario tests, and PDF reports | M1 | IN_PROGRESS |
| M6 | Final Verification, Hardening & Audit | Run all 40 automated test suites, verify 100% pass, compile final deliverables index, and pass forensic integrity audit | M2, M3, M4, M5 | PLANNED |

---

## Detailed Technical Specifications (All 40 Projects)

### Part A: Machine Learning (ML-01 to ML-10)

#### ML-01: Predictive Maintenance & Equipment Failure
- **Directory**: `part_a_machine_learning/ml_01_predictive_maintenance/`
- **Dataset**: UCI AI4I 2020 Predictive Maintenance Dataset (10,000 records). Features: Air temp [K], Process temp [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Type.
- **Target**: `Machine failure` (binary 0/1, 3.39% positive rate).
- **Leakage Prevention**: Strictly drop `UDI`, `Product ID`, and failure mode breakdown columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) during training.
- **Models**: Logistic Regression baseline, Random Forest, XGBoost/LightGBM. Class weighting & SMOTE.
- **Metrics**: Recall, F1-Score, ROC-AUC, Precision-Recall Curve.
- **Deliverables**: Modular `src/`, Streamlit UI (`app.py`), pytest suite (`tests/`), `results/metrics.json`, `reports/project_report.pdf`.

#### ML-02: Hospital Patient Readmission Risk Prediction
- **Directory**: `part_a_machine_learning/ml_02_patient_readmission/`
- **Dataset**: UCI Diabetes 130-US Hospitals (101,766 encounters, 50 features). Features: demographics, admission type, time in hospital, lab procedures, medication changes, ICD-9 diagnoses.
- **Target**: Early readmission within 30 days (`<30` vs remainder).
- **Leakage Prevention**: Group encounters by `patient_nbr` so same patient never appears across train and test sets. Exclude deceased/hospice discharges.
- **Mandatory Constraint**: Educational disclaimer — strictly no clinical diagnostic advice.
- **Models**: Random Forest, Gradient Boosting, Logistic Regression with high-cardinality ICD-9 category grouping.
- **Metrics**: Precision, Recall (minimizing false negatives), F1-Score, Confusion Matrix.

#### ML-03: Credit Card Fraud Detection
- **Directory**: `part_a_machine_learning/ml_03_credit_card_fraud/`
- **Dataset**: Kaggle Credit Card Fraud (284,807 transactions, 492 frauds, 0.172% imbalance). Features: `Time`, `Amount`, PCA components `V1` to `V28`.
- **Target**: `Class` (0 = Normal, 1 = Fraud).
- **Strategy**: Standard accuracy is invalid. Optimize Precision-Recall AUC (AUPRC), cost-sensitive loss, anomaly detection / LightGBM.
- **Metrics**: AUPRC, Recall at 95% Precision, F1-Score, Cost-Matrix Savings.
- **UI**: Interactive transaction risk scoring simulator with threshold slider.

#### ML-04: Telecom Customer Churn Prediction
- **Directory**: `part_a_machine_learning/ml_04_telecom_churn/`
- **Dataset**: IBM Telco Customer Churn (7,043 customers, 21 features). Features: tenure, contract type, internet services, payment methods, monthly charges.
- **Target**: `Churn` (Yes/No).
- **Key Preprocessing**: Clean whitespace/empty strings in `TotalCharges` when `tenure=0`. One-hot encoding of categorical features.
- **Explainability**: SHAP summary values computing top churn drivers per customer.
- **UI**: Customer profile inspector displaying churn probability and specific retention actions.

#### ML-05: Crop Recommendation System
- **Directory**: `part_a_machine_learning/ml_05_crop_recommendation/`
- **Dataset**: Kaggle Crop Recommendation (2,200 samples across 22 crops). Features: N, P, K (soil nutrients), Temperature, Humidity, pH, Rainfall.
- **Target**: Recommended Crop (22 multi-class categories).
- **Validation**: Bounds validation (pH 0-14, positive N/P/K).
- **Models**: Random Forest Classifier, Gaussian Naive Bayes, LightGBM.
- **Metrics**: Multiclass Accuracy, Macro F1-Score, Top-3 Accuracy.
- **UI**: Agronomy portal with nutrient sliders and top-3 probabilistic crop recommendations.

#### ML-06: Real Estate Property Price Prediction
- **Directory**: `part_a_machine_learning/ml_06_property_price_prediction/`
- **Dataset**: Kaggle Ames Housing Dataset (1,460 samples, 79 features).
- **Target**: `SalePrice` (continuous $).
- **Key Preprocessing**: Log-transform target (`log1p(SalePrice)`) to correct right skew. Missingness semantics (NA means absence of amenity, e.g. Pool, Fireplace, Garage).
- **Models**: Ridge Regression, ElasticNet, Gradient Boosting, XGBoost Regressor.
- **Metrics**: RMSE, MAE, R-squared ($R^2 \ge 0.88$).

#### ML-07: Hotel Booking Cancellation Prediction
- **Directory**: `part_a_machine_learning/ml_07_hotel_cancellation/`
- **Dataset**: Hotel Booking Demand (119,390 bookings). Features: lead time, arrival dates, guests, meals, market segment, ADR, previous cancellations.
- **Target**: `is_canceled` (0/1).
- **Leakage Prevention**: Strictly drop `reservation_status` and `reservation_status_date`.
- **Business Logic**: ADR > 0 validation; overbooking risk calculation.
- **Metrics**: Accuracy, F1-Score, ROC-AUC, Booking Lead Time Impact Curve.

#### ML-08: Solar Power Generation Forecasting
- **Directory**: `part_a_machine_learning/ml_08_solar_power_forecasting/`
- **Dataset**: Kaggle Solar Power Generation & Weather Sensor Data (merged 15-minute time series).
- **Target**: AC Power (kW) and DC Power (kW).
- **Strict Constraint**: Chronological time-series split (no random shuffle). Night-time zero solar radiation handled explicitly.
- **Models**: Time-lagged feature engineering, Ridge, Random Forest Regressor, LightGBM Regressor.
- **Metrics**: RMSE, MAE, MAPE, Day-ahead load forecasting visualizer.

#### ML-09: Retail Customer Segmentation
- **Directory**: `part_a_machine_learning/ml_09_customer_segmentation/`
- **Dataset**: UCI Online Retail Transaction Data (541,909 transactions).
- **Methodology**: RFM (Recency, Frequency, Monetary) feature extraction. Filter out cancellations (`InvoiceNo` starting with 'C') and missing `CustomerID`.
- **Models**: Log1p transform of RFM, StandardScaler, K-Means clustering with Elbow Method and Silhouette Score analysis.
- **Output**: Actionable customer clusters (Champions, Loyal, At-Risk, Lost) and interactive 3D scatter visualization.

#### ML-10: Employee Attrition Analytics
- **Directory**: `part_a_machine_learning/ml_10_employee_attrition/`
- **Dataset**: IBM HR Analytics Employee Attrition & Performance (1,470 employees, 35 features).
- **Target**: `Attrition` (Yes/No).
- **Fairness & Ethics**: Disallow individual hiring/firing usage; demographic parity and disparate impact analysis across age and gender.
- **Models**: Logistic Regression, Random Forest, Balanced Random Forest.
- **Explainability**: Permutation importance and partial dependence plots for policy insights.

---

### Part B: Deep Learning (DL-01 to DL-10)

#### DL-01: Automated Waste Classification
- **Directory**: `part_b_deep_learning/dl_01_waste_classification/`
- **Dataset**: Garbage Classification Dataset (2,527 images across 6 classes: Cardboard, Glass, Metal, Paper, Plastic, Trash).
- **Architecture**: Lightweight CNN / MobileNetV2 transfer learning backbone.
- **Training**: Data augmentation (rotation, horizontal flip, zoom). Early stopping on validation loss.
- **Metrics**: Categorical Accuracy, Per-class F1-Score, Confusion Matrix.
- **UI**: Image upload widget with real-time class classification and recycling instructions.

#### DL-02: Plant Leaf Disease Classification
- **Directory**: `part_b_deep_learning/dl_02_plant_disease_classification/`
- **Dataset**: PlantVillage Dataset (38 crop-disease categories, tomato, potato, apple, corn, etc.).
- **Architecture**: ResNet18 / MobileNetV2 with transfer learning and dropout regularizer.
- **Robustness**: Color jitter and random affine transformations to mitigate laboratory background bias.
- **Metrics**: Top-1 and Top-3 Accuracy, Multiclass Precision-Recall, Confusion Matrix.

#### DL-03: Chest X-Ray Pneumonia Classification
- **Directory**: `part_b_deep_learning/dl_03_chest_xray_pneumonia/`
- **Dataset**: Chest X-Ray Images (Pneumonia) (5,863 anterior-posterior chest radiographs: Normal vs Pneumonia).
- **Objective**: High-sensitivity medical screening prioritizing Recall for Pneumonia to eliminate false negatives.
- **Explainability**: Grad-CAM visual heatmaps overlaying lung field activation zones.
- **Mandatory Constraint**: Prominent educational use disclaimer stating no clinical diagnostic validity.

#### DL-04: Road Traffic Sign Recognition
- **Directory**: `part_b_deep_learning/dl_04_traffic_sign_recognition/`
- **Dataset**: German Traffic Sign Recognition Benchmark (GTSRB, 43 classes, variable resolutions).
- **Preprocessing**: Contrast Limited Adaptive Histogram Equalization (CLAHE), standardization to 32x32x3.
- **Architecture**: Multi-scale convolutional neural network (Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout).
- **Metrics**: Top-1 Accuracy (>95%), Macro F1-Score, speed-sign vs warning-sign confusion analysis.

#### DL-05: Industrial Surface Defect Classification
- **Directory**: `part_b_deep_learning/dl_05_surface_defect_detection/`
- **Dataset**: NEU Surface Defect Database (1,800 grayscale images, 200x200, 6 defect types: Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches).
- **Architecture**: Specialized defect detection CNN with residual connections for low-contrast textures.
- **Metrics**: Mean Average Precision, Multiclass Classification Report, Per-defect accuracy.

#### DL-06: Movie Review Sentiment Classification
- **Directory**: `part_b_deep_learning/dl_06_sentiment_analysis/`
- **Dataset**: Stanford IMDB Large Movie Review Dataset (50,000 reviews: 25k train, 25k test).
- **Architecture**: PyTorch BiLSTM / GRU network with learned Word Embeddings and sequence padding.
- **Evaluation**: Binary Accuracy (>87%), AUC-ROC, length invariance evaluation, and sample review inference.

#### DL-07: Smartphone Human Activity Recognition (HAR)
- **Directory**: `part_b_deep_learning/dl_07_human_activity_recognition/`
- **Dataset**: UCI HAR Dataset (30 subjects, triaxial accelerometer and gyroscope, 6 activities).
- **Split Rule**: Strict subject-independent split (70% subjects train, 30% subjects test) to prevent cross-person identity leakage.
- **Architecture**: 1D Convolutional Neural Network (1D-CNN) operating on raw temporal sensor windows (128 time steps x 9 channels).
- **Metrics**: Classification Accuracy, Walking vs Stairs confusion analysis.

#### DL-08: Household Electricity Consumption Forecasting
- **Directory**: `part_b_deep_learning/dl_08_electricity_forecasting/`
- **Dataset**: UCI Individual Household Electric Power Consumption (2,075,259 measurements).
- **Objective**: Multi-step sequence forecasting of Global Active Power.
- **Architecture**: Recurrent Neural Network (GRU / LSTM sequence-to-sequence).
- **Preprocessing**: Chronological time-series sequencing, forward imputation of missing values, min-max scaling.
- **Metrics**: RMSE, MAE, R2-score, 24-hour horizon forecast plot.

#### DL-09: Fashion Item Classification
- **Directory**: `part_b_deep_learning/dl_09_fashion_mnist/`
- **Dataset**: Fashion-MNIST (70,000 28x28 grayscale images across 10 apparel classes).
- **Architecture**: Modern CNN with Batch Normalization, Dropout (0.25, 0.5), and Adam optimizer.
- **Analysis**: Error matrix focusing on subtle differences between Shirt, T-Shirt, Coat, and Pullover.
- **Metrics**: Test Accuracy (>90%), Classification Report, Top-K Misclassifications display.

#### DL-10: Urban Sound Classification
- **Directory**: `part_b_deep_learning/dl_10_urban_sound_classification/`
- **Dataset**: UrbanSound8K (8,732 sound clips <=4s across 10 urban sound categories).
- **Feature Extraction**: Mel-Spectrogram audio representation (128 Mel bands x time frames).
- **Validation**: Strict adherence to pre-defined 10-fold cross-validation folds to prevent audio slice leakage.
- **Architecture**: 2D Audio Spectrogram CNN.
- **Metrics**: Cross-validated Accuracy, F1-Score, Soundwave & Spectrogram viewer in UI.

---

### Part C: Generative AI / RAG (GEN-01 to GEN-10)

Universal RAG Architecture across all 10 projects:
1. **Document Ingestion**: Multi-format parser (PDF, Markdown, HTML, TXT) with document metadata preservation.
2. **Chunking & Indexing**: Recursive character chunking (500-1000 chars, 150 overlap). Embeddings using sentence-transformers. Local vector index using FAISS or Chroma.
3. **Retrieval**: Top-$k$ ($k=3$) semantic similarity with relevance distance thresholding.
4. **Context Grounding**: Prompt template enforcing zero-hallucination and exact evidence citations: `[Doc: <name>, Page: <p>, Section: <s>]`.
5. **Deterministic Abstention**: When relevance score < threshold or context is empty, the system returns standard abstention: *"I cannot answer this question based on the provided documentation."*
6. **Evaluation Harness**: Programmatic test suite evaluating at least 15 domain questions (including 3+ unanswerable/out-of-scope queries) testing answer faithfulness, citation accuracy, and correct abstention.
7. **Interactive UI**: Streamlit application allowing document upload, search parameter adjustments ($k$, threshold), grounded Q&A, and citation viewing.

#### Projects in Part C:
- **GEN-01**: Enterprise HR Policy Assistant (handbooks, leaves, benefits, grievance policies).
- **GEN-02**: Medical Research Literature Assistant (PubMed Central papers; comparative synthesis; non-clinical disclaimer).
- **GEN-03**: University Academic Information Assistant (academic regulations, syllabi, course prerequisites).
- **GEN-04**: Legal Document Research Assistant (statutes, public acts, regulatory guidelines; non-legal-advice disclaimer).
- **GEN-05**: Financial Report Analysis Assistant (10-K filings, earnings calls, MD&A disclosures, audited vs unaudited tables).
- **GEN-06**: Agricultural Extension Knowledge Assistant (ICAR/USDA extension manuals, pest control, soil treatments).
- **GEN-07**: Cybersecurity Awareness Assistant (NIST SP 800, CISA guidelines, phishing, zero dual-use exploit generation).
- **GEN-08**: Software Developer Documentation Assistant (API references, code snippet generation, deprecation warnings).
- **GEN-09**: Climate & Sustainability Research Assistant (IPCC AR6 Working Group reports, uncertainty terminology preservation).
- **GEN-10**: Technical Training & Revision Assistant (course syllabi, lecture notes, quiz generation with Bloom's taxonomy).

---

### Part D: Agentic AI (AG-01 to AG-10)

Universal Agentic Architecture across all 10 projects:
1. **Multi-Agent State Graph**: Stateful workflow orchestrating at least 2 specialized agents with distinct personas and responsibilities.
2. **Tool Ecosystem**: At least 2 functional tools with strict input schemas, runtime exception handling, and structured outputs.
3. **Trace & State Logging**: Every execution cycle writes full step-by-step agent deliberations, tool calls, inputs, outputs, and timestamps to `results/agent_trace.json`.
4. **Human-in-the-Loop (HITL) Gate**: Explicit pause-and-confirm safety barrier before any consequential action (e.g. order dispatch, escalation, network change).
5. **Ethical & Safety Guardrails**: Hardcoded domain-specific veto filters:
   - No automated trading in AG-03.
   - No automated candidate ranking/rejection in AG-04.
   - No automated network containment/shutdown in AG-08.
   - Explicit confirmation for customer refunds in AG-02 and purchase orders in AG-09.
6. **Scenario Evaluation**: Programmatic test suite evaluating at least 10 realistic multi-step user scenarios (happy path, tool failure recovery, HITL approval, and HITL rejection).
7. **Interactive UI**: Streamlit/CLI interface showcasing dynamic agent conversation graph, live tool execution logs, and HITL approval dialogs.

#### Projects in Part D:
- **AG-01**: Multi-Agent Agricultural Advisory System (Soil Analyst Agent + Weather Forecast Agent + Agronomy Synthesizer). Tools: `SoilAnalyzerTool`, `WeatherLookupTool`.
- **AG-02**: Customer Support Ticket Triage & Resolution (Ticket Classifier Agent + Policy Retrieval Agent + Response Drafter Agent). Tools: `TicketClassifierTool`, `PolicySearchTool`. HITL: Refund/Escalation approval.
- **AG-03**: Financial Research & Market Analysis (Fundamental Analyst Agent + News Sentiment Agent + Compliance Auditor). Tools: `FinancialRatioTool`, `NewsSentimentTool`. Hard constraint: No live trading.
- **AG-04**: Recruitment Administration & Screening (Resume Parser Agent + JD Matcher Agent + Interview Question Drafter). Tools: `ResumeParserTool`, `SkillGapTool`. Hard constraint: No candidate ranking/rejection.
- **AG-05**: Travel Itinerary & Budget Planning (Destination Research Agent + Route Planner Agent + Budget Auditor). Tools: `TransitEstimatorTool`, `AccommodationBudgetTool`. Cost estimation disclaimer.
- **AG-06**: Research Paper Analysis & Matrix Workflow (Paper Discovery Agent + Methodology Extractor + Synthesis Reviewer). Tools: `PaperMetadataTool`, `MethodologyMatrixTool`.
- **AG-07**: Commercial Facility Energy Optimization (Telemetry Ingestion Agent + Anomaly Detector + HVAC Optimization Agent). Tools: `LoadAnomalyTool`, `ConservationSavingsTool`.
- **AG-08**: Security Incident Triage & Playbook Draft (SIEM Log Analyst Agent + Threat Intel Mapper + Response Playbook Drafter). Tools: `LogParserTool`, `MitreAttackLookupTool`. Hard constraint: Read-only simulation.
- **AG-09**: Inventory Monitoring & Reordering Assistant (Stock Level Monitor Agent + Demand Forecaster + Purchase Drafter). Tools: `StockLevelCheckTool`, `ReorderQuantityTool`. HITL: Purchase Order approval.
- **AG-10**: Educational Content Development Multi-Agent (Curriculum Architect Agent + Content Writer Agent + Pedagogical Reviewer Agent). Tools: `SyllabusAlignerTool`, `QuizGeneratorTool`. Two-stage author-critic loop.

---

## Master Verification Strategy (Milestone 1 & Milestone 6)
- **Universal Test Runner**: `run_all_tests.py` at workspace root discovers and executes tests across all 40 project folders using `pytest`.
- **Dual Mode Testing**:
  * **Offline/Hermetic Mode**: Tests use local synthetic fixtures and deterministic mock engines to guarantee 100% test pass without requiring internet or running background daemons.
  * **Interactive Mode**: Applications seamlessly connect to real public datasets and local Ollama (`qwen2.5:latest`) endpoints.
- **Automated PDF Report Compilation**: Every project includes a ReportLab script generating the complete 12-section PDF project report in `reports/project_report.pdf` adhering to the Bleep 100-mark rubric.
- **Zero-Tolerance Forensic Audit**: Every deliverable is verified by `teamwork_auditor` for code authenticity, non-fabricated metrics, genuine algorithms, and absence of hardcoded cheat facades.
