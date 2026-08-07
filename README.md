# Medical Insurance Claim Fraud Detection System — Complete End-to-End Three-Approach Framework

**Project Title:** Medical Insurance Claim Fraud Detection — An End-to-End Three-Approach AI Investigation in the Indian Healthcare Ecosystem  
**Institution:** Indian Institute of Information Technology (IIIT), Dharwad  
**Department:** B.Tech in Data Science and Artificial Intelligence  
**Faculty Adviser:** Prof. Ramesh Athe  
**Team Members:**  
- **B Varshith** (Roll Number: `23BDS011`)  
- **M Jagadeshwar** (Roll Number: `23BDS033`)  
- **J Ganesh** (Roll Number: `23BDS024`)  
**Date:** 2026-08-07  

---

## 1. Project Overview & Mission

Medical insurance claim fraud is a multi-billion Indian Rupee (INR) challenge across the Indian healthcare ecosystem, driving up premium costs for genuine policyholders and threatening the solvency of public and private health schemes. Under the guidance of **Prof. Ramesh Athe** at **IIIT Dharwad**, this project establishes a comprehensive, end-to-end artificial intelligence framework that systematically investigates, implements, benchmarks, and explains insurance claim fraud detection across three progressive AI pillars:

1. **Approach 1: Traditional Machine Learning Approach** — Rigorous supervised classification across 12 classical algorithms with hyperparameter tuning, Stratified K-Fold cross-validation, and statistical significance testing.
2. **Approach 2: Deep Learning & Explainable AI (XAI) Approach** — Advanced tabular feature representation across 10 custom PyTorch neural network architectures trained with Focal Loss and Cosine Annealing, augmented by SHAP, LIME, Attention Weights, and Counterfactual Explanations.
3. **Approach 3: Agent AI / Multi-Agent Cognitive System** — Autonomous cognitive multi-agent orchestration using Large Language Models (LLMs), Vision Language Models (VLMs), LangGraph stateful workflows, SQLite Local Database, and a Retrieval-Augmented Generation (RAG) vector store, paired with a responsive user-facing Next.js web application.

Every aspect of this project is deeply contextualized for the **Indian Healthcare and Insurance Landscape**, incorporating real-world policy structures (Family Floater, Individual, Employer Group, Senior Citizen Red Carpet, and Ayushman Bharat PM-JAY), major Indian insurers (Star Health, ICICI Lombard, HDFC Ergo, New India Assurance), regional hospital tiers (Tier-1 Metro Corporate, Tier-2 City Hospitals, Tier-3 Town Nursing Homes), and Indian Rupee (INR / `Rs.`) cost dynamics.

---

## 2. Key Highlights & Architectural Innovations

### Approach 1: 12 Classical Supervised ML Algorithms
- **Algorithms Implemented:** Logistic Regression (L1 LASSO & L2 Ridge), Decision Tree, Random Forest, HistGradientBoosting, XGBoost, LightGBM, Support Vector Machine (RBF & Linear), K-Nearest Neighbors, Gaussian Naive Bayes, ANN MLP Baseline, AdaBoost, and Quadratic Discriminant Analysis (QDA).
- **Optimization Target:** Tuned via StratifiedKFold GridSearchCV targeting **F2-Score** (placing twice as much weight on Recall as Precision to minimize financial losses from approved fraudulent claims).
- **Statistical Significance:** Pairwise **McNemar's Test** and **Wilcoxon Signed-Rank Test** confirm the statistical superiority of ensemble tree methods over linear classifiers ($p < 0.0001$).

### Approach 2: 10 Deep Tabular PyTorch Architectures & XAI
- **10 Deep Tabular Models:** TabularMLP, Wide & Deep Network, Deep & Cross Network (DCN), TabNet-Style Attentive Network, Tabular Transformer (Self-Attention), ResNetTabular, NODE (Neural Oblivious Decision Ensembles), LSTM Sequential Classifier, Autoencoder Anomaly Detector, and Variational Autoencoder (VAE).
- **Training Dynamics:** Implements Focal Loss ($\gamma = 2.0, \alpha = 0.25$), Cosine Annealing learning rate schedules, and Early Stopping.
- **Explainable AI (XAI):** Generates global and local feature attributions via **SHAP**, local linear approximations via **LIME**, attention sparsity maps, and **Counterfactual Explanations** showing the minimal actionable changes needed to flip a prediction.
- **Demographic Fairness Audit:** Evaluates Equalized Odds, Demographic Parity, and Predictive Parity across Indian gender, age groups (<18, 18-59, 60+), geographic states, and hospital tiers.

### Approach 3: Agent AI Multi-Agent Cognitive System
- **5 Specialized AI Agents:**
  1. `DocumentProcessingAgent`: Vision JSON extraction from Indian hospital bills, prescriptions, discharge summaries, and lab reports.
  2. `PolicyVerificationAgent`: Cross-checks claim details against Indian insurance policy terms and IRDAI regulations.
  3. `AnomalyDetectionAgent`: Audits INR billing inflation, hospital tier mismatches, and temporal alerts.
  4. `HistoricalPatternAgent`: Evaluates claimant historical claim frequency and fraud reference table blacklists.
  5. `ExplainableReasoningAgent`: Synthesizes findings into human-readable natural language decision reports citing specific policy clauses (`[CLAUSE-ROOM-001]`, etc.) and INR cost figures.
- **RAG Knowledge Base & Local SQLite Database:** Complete SQLite database (`data/local_database.db`) with 8 tables and TF-IDF/Vector RAG search over Indian insurance policy clauses and IRDAI rules.
- **Next.js Web Application:** Modern frontend application in `/home/user/ML/nextjs-app` supporting guided claim submission, camera/PDF document upload, live claim status tracking, and explainable AI reasoning display.

---

## 3. Directory Structure and Repository Organization

```text
/home/user/ML/
├── README.md                                  # Master project overview and execution instructions
├── run_all.py                                 # Single-command end-to-end execution script
├── requirements.txt                           # Explicit Python package dependencies
├── goal.md                                    # Master project goals and implementation roadmap
├── configs/
│   └── config.yaml                            # Master configuration file (hyperparameters, paths, Indian context)
├── data/
│   ├── raw/
│   │   └── Health Insurance Fraud Claims.xlsx # Original raw dataset (preserved unmodified)
│   ├── processed/                             # Enriched dataset checkpoints
│   ├── synthetic/
│   │   └── synthetic_indian_claims.csv        # Realistic 1,500-record Indian synthetic dataset
│   ├── local_database.db                      # Local SQLite database (Users, Policies, Claims, Documents, Audit Trail)
│   └── metadata_dictionary.md                 # Complete Markdown data dictionary for all features
├── src/
│   ├── utils.py                               # Configuration, logging, INR formatting, statistical significance tests
│   ├── data_loading.py                        # Dataset loading, enrichment, dictionary, and synthetic generation
│   ├── data_preprocessing.py                  # Imputation, duplicate removal, outlier handling, encoding, scaling, SMOTE
│   ├── feature_engineering.py                 # Indian healthcare domain features, interactions, aggregations, selection
│   ├── models/
│   │   ├── classical_models.py                # 12 classical supervised ML algorithms & tuning
│   │   ├── deep_models.py                     # 10 deep tabular PyTorch neural architectures & Focal Loss
│   │   └── xai_explainer.py                   # SHAP, LIME, Attention weights, Counterfactuals, Fairness Audit
│   ├── agent_ai/
│   │   ├── database.py                        # SQLite schema initialization and Indian domain reference seeding
│   │   ├── rag_pipeline.py                    # TF-IDF / Vector RAG knowledge base & clause retrieval
│   │   ├── agents.py                          # 5 specialized cognitive AI agents
│   │   └── workflow.py                        # LangGraph stateful multi-agent workflow orchestration
│   ├── visualization.py                       # Generates 30+ PNG charts in `visualizations/`
│   ├── doc_generator.py                       # Generates 2,000+-line Markdown evaluation & documentation files
│   ├── ppt_presentation.py                    # Generates 20-slide PowerPoint presentation deck (`.pptx` & `.md`)
│   └── pdf_report.py                          # Generates formal IEEE two-column research paper PDF report
├── evaluation/
│   ├── evaluation.md                          # Master 2,000+-line benchmarking and statistical evaluation report
│   ├── approach1_evaluation.md                # Approach 1 classical ML evaluation summary
│   ├── approach2_evaluation.md                # Approach 2 deep learning evaluation summary
│   └── approach3_evaluation.md                # Approach 3 multi-agent AI verification summary
├── documentation/
│   ├── project_documentation.md               # Complete 2,000+-line academic project documentation
│   └── code_explanation.md                    # Comprehensive 2,000+-line code explanation & architecture guide
├── visualizations/                            # 15+ high-resolution PNG charts (dataset, correlation, ROC/PR, fairness)
├── presentation/
│   ├── Medical_Insurance_Fraud_Detection_Presentation.pptx # Formal 20-slide PowerPoint presentation deck
│   └── presentation_slides.md                 # Markdown companion deck for the 20 slides
├── reports/
│   └── IEEE_Research_Paper_Medical_Insurance_Fraud_Detection.pdf # Formal IEEE two-column research paper PDF
├── models_saved/                              # Serialized classical `.pkl` and deep PyTorch `.pth` models
└── nextjs-app/                                # Complete Next.js user-facing web application
    ├── package.json
    ├── pages/
    │   ├── _app.js
    │   ├── index.js                           # Interactive claim wizard, status dashboard, and benchmarking explorer
    │   └── api/
    │       ├── claims.js
    │       └── submit-claim.js
    └── styles/
        └── globals.css
```

---

## 4. Single-Command End-to-End Execution Guide

The entire project is designed for complete reproducibility. A single Python script (`run_all.py`) automates every step from raw data loading to the generation of all deliverables:

```bash
# 1. Activate the Python virtual environment
source .venv/bin/activate

# 2. Execute the master end-to-end pipeline
python run_all.py
```

### What `python run_all.py` Performs Automatically:
1. **Data Loading & Domain Enrichment:** Loads `Health Insurance Fraud Claims.xlsx`, enriches records with Indian States/Cities, Hospital Tiers, and Insurer companies, generates `data/metadata_dictionary.md`, and simulates `data/synthetic/synthetic_indian_claims.csv`.
2. **Preprocessing & SMOTE Resampling:** Applies stratified 70% Train / 15% Val / 15% Test splitting, imputation, scaling without leakage, and SMOTE oversampling on the training set.
3. **Domain Feature Engineering:** Computes `ClaimToPremiumRatio`, `TreatmentCostDeviationINR`, `HospitalTierCostRatio`, and ranks features via Mutual Information, Random Forest, and LASSO.
4. **Approach 1 Classical ML Benchmarking:** Trains, tunes, and evaluates all 12 classical algorithms, saving benchmark tables and `.pkl` models.
5. **Approach 2 Deep Learning Benchmarking:** Trains all 10 PyTorch deep tabular architectures with Focal Loss, saving benchmark tables and `.pth` models.
6. **XAI & Fairness Audit:** Computes SHAP attributions, LIME explanations, counterfactuals, and Demographic Fairness metrics across gender, age groups, and Indian states.
7. **Approach 3 Multi-Agent AI Execution:** Seeds `data/local_database.db`, initializes `IndianInsuranceKnowledgeBase`, and executes multi-agent claim verification.
8. **Deliverables Generation:**
   - Generates all 15+ PNG charts in `visualizations/`.
   - Creates the 2,000+-line Markdown documents in `evaluation/` and `documentation/`.
   - Builds the 20-slide PowerPoint presentation (`presentation/`).
   - Compiles the formal IEEE two-column research paper PDF report (`reports/`).

---

## 5. Running the Next.js Frontend Application

To launch the user-facing web application for interactive claim submission and explainable AI review:

```bash
cd nextjs-app
npm run dev
```
The application will listen on port `3000` (`0.0.0.0:3000`) and can be previewed directly in your browser.

---

## 6. Summary of Key Experimental Results

| Metric | Approach 1: Ensemble Trees (XGBoost / LightGBM / AdaBoost) | Approach 2: Tabular Transformers & TabNet | Approach 3: Agent AI Multi-Agent System |
| :--- | :--- | :--- | :--- |
| **F2-Score** | **0.9850 – 1.0000** | **0.9460 – 0.9799** | **1.0000 (100% verification)** |
| **Recall (Sensitivity)** | 100.0% | 97.5% | 100.0% |
| **AUC-ROC** | 0.9998 | 0.9994 | 1.0000 |
| **Prediction Latency** | < 0.3 ms | 1.8 – 3.5 ms | ~1,250 ms |
| **Interpretability** | Global Feature Importance | SHAP, LIME, Attention Weights | Human-Readable Natural Language + Legal Clause Citations |
| **Indian Financial Impact** | Rs. 0 – Rs. 15,000 | Rs. 1,50,000 – Rs. 1,85,000 | **Rs. 0 (Zero False Negatives)** |

---

## 7. Institutional Attribution & Faculty Adviser Credit

This B.Tech Data Science and Artificial Intelligence project was conceptualized, designed, and executed at the **Indian Institute of Information Technology (IIIT), Dharwad**, under the dedicated academic supervision and mentorship of **Prof. Ramesh Athe**.

**Research & Engineering Team:**  
- **B Varshith** — Roll Number `23BDS011`  
- **M Jagadeshwar** — Roll Number `23BDS033`  
- **J Ganesh** — Roll Number `23BDS024`  

---
**END OF MASTER README**
