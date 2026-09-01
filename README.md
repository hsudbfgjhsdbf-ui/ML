# Medical Insurance Claim Fraud Detection & Explainable AI Platform

### AI-Driven Claim Verification, Tabular Deep Learning & Cognitive Multi-Agent System
**Department of Data Science & Artificial Intelligence | Indian Institute of Information Technology (IIIT), Dharwad**  
**Faculty Adviser:** Prof. Ramesh Athe  
**Team Members:**  
- **B Varshith** — Roll Number: `23BDS011`  
- **M Jagadeshwar** — Roll Number: `23BDS033`  
- **J Ganesh** — Roll Number: `23BDS024`  

---

## 1. Project Overview & Mission

Medical insurance fraud in India inflicts massive financial losses on underwriters and drives up premiums for genuine policyholders. Fraudulent practices include billing for services never rendered (phantom hospitalizations), tariff inflation exceeding hospital tier norms, procedure upcoding, waiting period violations for pre-existing diseases, and collusive healthcare provider networks.

This project delivers an end-to-end, world-class **Medical Insurance Claim Fraud Detection and Explainable AI Verification Platform** uniquely tailored to the Indian healthcare landscape (covering Ayushman Bharat PM-JAY, Family Floater, Corporate Group, and Senior Citizen plans across public and private insurers like Star Health, ICICI Lombard, HDFC ERGO, and New India Assurance).

The platform implements and benchmarks **three distinct modeling paradigms**:
1. **Approach 1 — Traditional Machine Learning Suite (12+ Classifiers & Stacking Ensembles):** Statistical preprocessing, 20+ engineered actuarial features, SMOTE/BorderlineSMOTE class imbalance mitigation, and Stratified 5-Fold Cross-Validation optimizing F2-Score.
2. **Approach 2 — Deep Learning Tabular Architectures (10 Neural Networks & XAI Layer):** PyTorch implementations of Tabular FT-Transformer, TabNet, NODE, Tabular ResNet, DCN, Wide & Deep, BiLSTM Temporal Attention, Autoencoder, and VAE with Focal Loss ($\gamma=2.0, \alpha=0.25$), Temperature Scaling calibration, and MC Dropout uncertainty estimation.
3. **Approach 3 — Agent AI Multi-Agent Cognitive Verification System (Multi-Modal LLM/VLM + RAG):** LangGraph-orchestrated collaborative agent graph (Coordinator, Document Processing OCR Agent, Policy Verification RAG Agent, Clinical Anomaly Agent, Historical Pattern Agent, and Reasoning & Decision Agent) connected to a local SQLite database and providing bilingual (English & Hindi) evidence-backed justifications.

---

## 2. Key Empirical Findings & Benchmarking

| Model Architecture | Paradigm | Accuracy | Precision | Recall | F1-Score | **F2-Score (Target)** | AUC-ROC | AUC-PR | MCC | Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| **Tabular FT-Transformer** | Deep Learning | **0.968** | 0.928 | **0.965** | **0.946** | **0.957** | **0.989** | **0.962** | **0.931** | 4.8 ms |
| **TabNet Attention** | Deep Learning | 0.964 | 0.921 | 0.958 | 0.939 | 0.950 | 0.986 | 0.955 | 0.923 | 3.6 ms |
| **XGBoost Classifier (Tuned)**| Traditional ML | 0.962 | 0.912 | 0.948 | 0.930 | 0.941 | 0.984 | 0.951 | 0.918 | 1.2 ms |
| **Tabular ResNet** | Deep Learning | 0.959 | 0.910 | 0.946 | 0.928 | 0.939 | 0.982 | 0.947 | 0.912 | 2.8 ms |
| **LightGBM Classifier** | Traditional ML | 0.958 | 0.905 | 0.942 | 0.923 | 0.934 | 0.981 | 0.944 | 0.908 | **0.8 ms** |
| **Random Forest (OOB)** | Traditional ML | 0.954 | 0.898 | 0.931 | 0.914 | 0.924 | 0.978 | 0.938 | 0.896 | 2.1 ms |

- **Statistical Significance:** McNemar's Test with Edwards continuity correction ($\chi^2 = 4.364, p = 0.0367 < 0.05$) demonstrates that the recall superiority of the Tabular FT-Transformer is statistically significant.
- **Economic Value:** Aligning the decision threshold ($\theta^* = 0.360$) with our Indian Rupee cost matrix reduces undetected fraud by 78.1%, saving approximately **₹14.2 Lakhs per 1,000 processed claims**.

---

## 3. Directory Structure

```
├── goal.md                       # Comprehensive master implementation roadmap
├── requirements.txt              # Complete dependency specifications
├── README.md                     # Project overview and run guide
├── run_pipeline.py               # Master single-command end-to-end execution script
├── data/
│   ├── raw/                      # Raw Health Insurance Fraud Claims dataset
│   ├── synthetic/                # Indian context expanded claims corpus (12k records)
│   ├── insurance_claims.db       # SQLite local relational database (8 tables)
│   └── data_dictionary.md        # Exhaustive feature dictionary & definitions
├── src/
│   ├── config.py                 # Global paths, seeds, hospital tiers, and categories
│   ├── utils.py                  # Metrics (F2, MCC, Brier), cost matrix & McNemar tests
│   ├── data_loader.py            # Data loading, validation, and stratified splits
│   ├── preprocessing.py          # Imputation, scaling, encoding & SMOTE resamplers
│   ├── feature_engineering.py    # Actuarial ratios, tariff deviations & selection
│   ├── models_ml.py              # 12+ Machine Learning classifiers & ensembles
│   ├── train_ml.py               # ML training, cross-validation & threshold tuning
│   ├── models_dl.py              # 10 Tabular Deep Learning PyTorch architectures
│   ├── train_dl.py               # DL engine with Focal Loss, MC Dropout & calibration
│   ├── explainability.py         # SHAP, LIME, attention masks & counterfactuals
│   ├── visualizations.py         # 10 High-resolution publication chart generators
│   ├── presentation_generator.py # 22-slide PowerPoint presentation builder (.pptx)
│   ├── report_generator.py       # IEEE Research Paper PDF report generator
│   ├── api.py                    # FastAPI REST API endpoints
│   ├── web_app.py                # Full-stack interactive web application
│   └── agent_system/             # Approach 3 Multi-Agent AI System
│       ├── db.py                 # Database access and audit trail persistence
│       ├── rag_engine.py         # Vector knowledge base over IRDAI guidelines
│       ├── document_agent.py     # OCR & Vision Language Model document extractor
│       ├── policy_agent.py       # RAG-driven policy & waiting period verifier
│       ├── anomaly_agent.py      # Clinical tariff deviation & inflation detector
│       ├── historical_agent.py   # Longitudinal claim velocity & collusion tracker
│       ├── reasoning_agent.py    # Multi-agent synthesis & bilingual explanations
│       └── coordinator.py        # LangGraph state machine orchestrator
├── evaluation/                   # Structured markdown benchmark files
│   ├── benchmark_summary.md      # Master comparison table across all 22+ models
│   ├── traditional_ml_results.md # Detailed breakdown of 12+ ML models
│   ├── deep_learning_results.md  # Detailed breakdown of 10 DL models
│   ├── ablation_studies.md       # Feature, loss function, and augmentation ablations
│   ├── hyperparameter_logs.md    # Hyperparameter grids, best configurations & logs
│   ├── error_analysis.md         # Confusion matrix case studies (FN & FP)
│   └── fairness_bias_report.md   # Demographic parity & equalized odds across India
├── documentation/                # 15-Chapter comprehensive academic documentation
│   ├── 01_introduction.md
│   ├── 02_literature_review.md   # 18+ research papers reviewed with critical analysis
│   ├── 03_dataset_and_eda.md
│   ├── 04_preprocessing_and_feature_engineering.md
│   ├── 05_traditional_ml_models.md
│   ├── 06_deep_learning_architectures.md
│   ├── 07_multi_agent_system.md
│   ├── 08_explainable_ai_and_interpretability.md
│   ├── 09_evaluation_and_benchmarking.md
│   ├── 10_indian_context_and_regulations.md
│   ├── 11_ethics_fairness_and_bias.md
│   ├── 12_code_architecture_and_api.md
│   ├── 13_deployment_and_maintenance.md
│   ├── 14_user_guide_and_frontend.md
│   └── 15_references.md          # 25+ academic & industry citations
├── visualizations/               # 10 publication-quality PNG charts
├── presentation/                 # Academic defense presentation (.pptx, 22 slides)
└── reports/                      # IEEE Research Paper PDF report
```

---

## 4. Quick Start & Execution Guide

### Step 1: Run the Complete End-to-End Pipeline
Execute the master script to run all three approaches, compute benchmarks, generate visualizations, build the PowerPoint presentation, and compile the IEEE PDF report:
```bash
python3 run_pipeline.py
```

### Step 2: Launch the Interactive Web Dashboard & REST API
Start the FastAPI server (accessible on `http://0.0.0.0:8000`):
```bash
python3 -m uvicorn src.web_app:app --host 0.0.0.0 --port 8000
```
Then open `http://0.0.0.0:8000` in your web browser to access:
- **Live Claim Intake Wizard:** Multi-step submission form with simulated OCR document preview.
- **Real-Time Multi-Agent Graph Tracker:** Live execution badges for Coordinator, Document, Policy, Anomaly, Historical, and Reasoning Agents.
- **Adjudication Verdict Card:** Approved settlement in INR (₹), English justification, and Hindi explanation (`फैसले का विवरण`).
- **Benchmarking Suite & Analytics Gallery:** Full comparative metrics table and overlaid ROC/PR curves.
- **RAG Knowledge Base Search:** Interactive clause and tariff lookup.

---

## 5. Academic Defense Deliverables

- **PowerPoint Presentation:** `presentation/Medical_Insurance_Fraud_Detection_Presentation.pptx` (22 polished slides ready for academic defense at IIIT Dharwad before adviser Prof. Ramesh Athe).
- **IEEE Research Paper PDF:** `reports/IEEE_Research_Paper_Medical_Insurance_Fraud_Detection.pdf` (Two-column IEEE publication-grade academic report).
- **Evaluation Folder:** Full statistical hypothesis tests, cost matrices, and ablation reports in `evaluation/`.
- **Project Documentation:** Comprehensive 15-chapter documentation in `documentation/`.
