# Medical Insurance Claim Fraud Detection — End-to-End Master Project Plan (goal.md)

**Institution:** Indian Institute of Information Technology (IIIT), Dharwad  
**Department:** B.Tech in Data Science & Artificial Intelligence  
**Faculty Adviser:** Ramesh Athe  
**Project Team:**  
- B Varshith — Roll Number: 23BDS011  
- M Jagadeshwar — Roll Number: 23BDS033  
- J Ganesh — Roll Number: 23BDS024  
**Academic Year:** 2024–2025 / 2025–2026  

---

## 1. Executive Summary & Project Mission

Medical insurance fraud in India imposes massive financial burdens on insurers and increases premiums for genuine policyholders. Fraudulent practices include billing for unrendered services, inflated medical procedures, falsified diagnostic tests, phantom hospitalizations, identity misrepresentation, and organized collusion between providers and claimants. 

This project builds a state-of-the-art, multi-paradigm **Medical Insurance Claim Fraud Detection and Explainable AI Verification Platform** tailored specifically to the Indian healthcare ecosystem (covering Ayushman Bharat, ECHS, family floater, group, and senior citizen policies across public and private insurers like Star Health, ICICI Lombard, HDFC Ergo, and New India Assurance).

The project implements and benchmarks **three distinct approaches**:
1. **Approach 1 — Traditional Machine Learning Framework (12+ Classifiers & Stacking Ensembles)**: Statistical preprocessing, domain feature engineering, stratified cross-validation, and extensive benchmarking with F2-score optimization.
2. **Approach 2 — Deep Learning & Tabular Neural Architectures (10 Deep Learning Models & XAI Layer)**: Tabular deep learning architectures (MLP, Wide & Deep, DCN, TabNet, Tabular Transformer, Tabular ResNet, NODE, LSTM Sequence, Autoencoder, VAE) with Focal Loss, MC Dropout uncertainty, temperature calibration, and adversarial robustness.
3. **Approach 3 — Agent AI Multi-Agent Cognitive Verification System (Multi-Modal LLM/VLM + RAG)**: Multi-agent orchestration (Coordinator, Document Processing OCR/VLM Agent, Policy Verification RAG Agent, Anomaly Detection Agent, Historical Pattern Agent, Explainable Reasoning Agent) with SQLite local database, knowledge vector retrieval, and bilingual (English/Hindi) evidence-backed natural language justifications.

---

## 2. End-to-End Task Breakdown & Deliverables Checklist

### Pillar 1: Data Acquisition, Synthesis & Indian Context Grounding
- [x] Ingest and audit `Health Insurance Fraud Claims.xlsx` (4,500 claims across 19 attributes).
- [x] Create an expanded Indian context synthetic dataset (10,000+ records) reflecting realistic Indian healthcare patterns (INR currency, tiered hospitals, Indian state/metro distributions, Ayushman Bharat & family floater plans, ICD-10 diagnostic categories).
- [x] Construct a comprehensive `data/data_dictionary.md` detailing all raw and engineered variables, ranges, units, and fraud relevance.
- [x] Implement stratified 70-15-15 train/validation/test splits ensuring identical class distributions.

### Pillar 2: Preprocessing & Advanced Feature Engineering
- [x] Build reproducible imputation pipelines (median for skewed numeric, mean for normal, mode/constant for categoricals).
- [x] Implement outlier detection (IQR, Z-score, domain-bounded clipping preserving true fraud signals).
- [x] Support multi-paradigm categorical encodings: Label Encoding, One-Hot Encoding, Target Encoding, and Ordinal Encoding.
- [x] Engineer 20+ domain-specific features:
  - Claim-to-Premium Ratio
  - Treatment Cost Deviation normalized by Hospital Tier and Geographic Region
  - Days Since Policy Inception & Waiting Period Delta
  - Claim Frequency (rolling 12 months) and Inter-Claim Interval
  - Interaction features (Age × Treatment, Location × Hospital Tier, Amount / Hospitalization Days)
  - Historical policyholder aggregations (mean claim, standard deviation, max claim, current-to-historical ratio)
  - Provider risk statistics (rejection rate, total claims volume, unique patient ratio)
  - 2nd-degree polynomial features with automated feature selection (Correlation filter, Chi-Square, Mutual Information, RFECV, LASSO).
- [x] Class imbalance mitigation: SMOTE, BorderlineSMOTE, ADASYN, RandomUnderSampler, SMOTEENN, and cost-sensitive class weights.

### Pillar 3: Approach 1 — Traditional Machine Learning (12+ Classifiers)
- [x] Implement, tune, and evaluate 12+ classification models:
  1. Logistic Regression (L1 LASSO & L2 Ridge tuned)
  2. Decision Tree Classifier (cost-complexity pruning, max depth)
  3. Random Forest Classifier (OOB error estimation, feature importances)
  4. Gradient Boosting Classifier (Histogram-based)
  5. XGBoost Classifier (scale_pos_weight tuning)
  6. LightGBM Classifier (leaf-wise growth, bagging/feature fraction)
  7. Support Vector Machine (Linear & RBF kernels with probability calibration)
  8. K-Nearest Neighbors (KNN with distance weighting)
  9. Gaussian Naive Bayes & Multinomial Naive Bayes
  10. Artificial Neural Network (MLP baseline classifier)
  11. AdaBoost Classifier (decision stump weak learners)
  12. Quadratic Discriminant Analysis (QDA) / Extra Trees Classifier
  13. Voting Ensemble & Stacking Classifier
- [x] Stratified 5-Fold and 10-Fold Cross Validation.
- [x] Hyperparameter tuning via GridSearchCV, RandomizedSearchCV, and Optuna optimizing F2-score.
- [x] Primary evaluation on test set: Accuracy, Precision, Recall, F1-Score, F2-Score, AUC-ROC, AUC-PR, Matthews Correlation Coefficient (MCC), Brier Score, Training Time (s), Latency (ms), Model Size (KB).
- [x] Confusion Matrix & Financial Cost Matrix in Indian Rupees (₹).
- [x] Threshold optimization curve (0.05 to 0.95) maximizing F2-score.
- [x] Statistical hypothesis testing: McNemar's Test for pairwise model comparisons and Wilcoxon Signed-Rank Test across CV folds (p < 0.05).

### Pillar 4: Approach 2 — Deep Learning & Tabular Neural Architectures (10 Models & XAI)
- [x] PyTorch modular implementation of 10 architectures:
  1. **MLP**: Deep dense network with BatchNorm, Dropout (0.3), and residual skip connections.
  2. **Wide & Deep**: Wide linear model with cross-product memorization combined with deep non-linear layers.
  3. **Deep & Cross Network (DCN)**: Explicit bounded-degree cross layers learning feature interactions automatically.
  4. **TabNet**: Sequential multi-step attention transformer with ghost batch normalization and sparse feature selection.
  5. **Tabular Transformer (FT-Transformer)**: Learned categorical entity embeddings, linear numerical projections, Multi-Head Attention, and CLS classification token.
  6. **Tabular ResNet**: Stacked residual blocks with pre-activation BatchNorm and ReLU activations.
  7. **NODE (Neural Oblivious Decision Ensembles)**: Differentiable oblivious decision trees with temperature-controlled soft splits.
  8. **LSTM / BiLSTM with Attention**: Sequential claim modeling capturing policyholder temporal history and claim spikes.
  9. **Autoencoder Anomaly Detector**: Unsupervised reconstruction error model trained exclusively on legitimate claims.
  10. **Variational Autoencoder (VAE)**: Probabilistic latent space with reparameterization trick and KL-divergence loss for anomaly detection and synthetic generation.
- [x] Training pipeline: Focal Loss $(\gamma=2.0, \alpha=0.25)$, Weighted BCE, AdamW optimizer, Cosine Annealing learning rate scheduler with warmup, Gradient Clipping (1.0), and Early Stopping (patience=20).
- [x] Uncertainty estimation: Monte Carlo (MC) Dropout (50 stochastic forward passes).
- [x] Model calibration: Reliability diagrams, Expected Calibration Error (ECE), and Temperature Scaling.
- [x] Robustness testing: Gaussian noise perturbation, feature dropout testing, FGSM and PGD adversarial attack evaluation, and adversarial training.

### Pillar 5: Approach 3 — Agent AI Multi-Agent Cognitive Verification System
- [x] Design and implement multi-agent architecture:
  1. **Coordinator / Supervisor Agent**: Workflow orchestration, state tracking, conditional routing, retries, and error recovery.
  2. **Document Processing & OCR/VLM Agent**: Multi-modal document understanding for hospital bills, prescriptions, discharge summaries, lab reports, and Aadhaar cards with structured JSON output and field confidence scores.
  3. **Policy Verification & RAG Agent**: Grounded rule verification against Indian policy clauses (sum insured, waiting periods, sub-limits, exclusions, network hospital validation).
  4. **Anomaly Detection Agent**: Multi-dimensional fraud pattern matching (billing inflation against hospital tier, diagnosis-treatment discordance, temporal anomalies, blacklisted provider graph matching).
  5. **Historical Pattern Agent**: Time-series claim velocity, cumulative payout analysis, and repeat claim frequency tracking.
  6. **Reasoning & Explainable Decision Agent**: Comprehensive evidence synthesis, conflict resolution, and bilingual natural language explanations (English & Hindi) with exact clause citations and financial breakdown.
  7. **Human-in-the-Loop Checkpoint**: Flagged claim workflow pausing for manual claim adjuster review.
- [x] SQLite Local Database (`data/insurance_claims.db`) with 8 tables: `users`, `policies`, `claims`, `documents`, `agent_results`, `fraud_rules`, `hospital_reference`, `medical_pricing_benchmarks`.
- [x] Vector Store & RAG Engine (`src/agent_system/rag_engine.py`) indexing policy rulebooks, IRDAI guidelines, fraud topologies, and Indian standard medical rates.

### Pillar 6: Model Interpretability & Explainable AI (XAI)
- [x] SHAP (SHapley Additive exPlanations): TreeExplainer and DeepExplainer computing global and local feature importance.
- [x] LIME (Local Interpretable Model-agnostic Explanations) for instance-level fraud explanations.
- [x] Attention weight extraction and visualization for TabNet and Tabular Transformer.
- [x] Counterfactual Explanation Generator: Minimal feature perturbation needed to flip a fraudulent verdict to legitimate.

### Pillar 7: Comprehensive Documentation, Presentation & Reports
- [x] `documentation/` folder containing 15 detailed chapters:
  - `01_introduction.md`: Problem statement, Indian healthcare insurance context, objectives, team info.
  - `02_literature_review.md`: 18+ research papers critically reviewed with comparative matrix and identified research gaps.
  - `03_dataset_and_eda.md`: Data exploration, statistical profiles, distribution plots, class imbalance analysis.
  - `04_preprocessing_and_feature_engineering.md`: Step-by-step mathematical formulations of 20+ engineered features.
  - `05_traditional_ml_models.md`: Theoretical formulation, hyperparameters, and implementation details for 12+ ML models.
  - `06_deep_learning_architectures.md`: Deep learning architectures, layer dimensions, focal loss, and mathematical formulations.
  - `07_multi_agent_system.md`: LangGraph state machine, agent specifications, prompts, and communication protocols.
  - `08_explainable_ai_and_interpretability.md`: SHAP, LIME, Attention Maps, and Counterfactual reasoning.
  - `09_evaluation_and_benchmarking.md`: Comprehensive evaluation metrics, cross-validation, and benchmarking comparisons.
  - `10_indian_context_and_regulations.md`: IRDAI guidelines, Ayushman Bharat (PM-JAY), hospital tiers, and Indian fraud typologies.
  - `11_ethics_fairness_and_bias.md`: Demographic parity, equalized odds across gender, age groups, and income brackets.
  - `12_code_architecture_and_api.md`: Module breakdown, design patterns, and REST API documentation.
  - `13_deployment_and_maintenance.md`: Production deployment, batch/real-time pipelines, monitoring, and drift detection.
  - `14_user_guide_and_frontend.md`: Full user guide for claimants, auditors, and administrators.
  - `15_references.md`: 30+ peer-reviewed academic references and industry standards in IEEE format.
- [x] `evaluation/` folder with complete benchmark markdown files:
  - `benchmark_summary.md` (Master comparison table of all 22+ models across ML & DL)
  - `traditional_ml_results.md`
  - `deep_learning_results.md`
  - `ablation_studies.md`
  - `hyperparameter_logs.md`
  - `error_analysis.md`
  - `fairness_bias_report.md`
- [x] `visualizations/` folder with high-resolution PNG charts.
- [x] `presentation/Medical_Insurance_Fraud_Detection_Presentation.pptx`: 22-slide professional academic presentation ready for IIIT Dharwad defense before adviser Ramesh Athe.
- [x] `reports/IEEE_Research_Paper_Medical_Insurance_Fraud_Detection.pdf`: Full academic research paper formatted in IEEE two-column style with abstract, formulas, 5+ tables, 5+ figures, and 25+ references.

### Pillar 8: Interactive Web Application & REST API
- [x] Production FastAPI backend serving real-time claim inference, document OCR analysis, multi-agent reasoning, database inspection, and benchmark reporting.
- [x] Interactive Web Dashboard bound to `0.0.0.0:8000` with live preview, multi-step claim submission wizard, real-time OCR document extraction preview, visual agent workflow tracker, explainable decision report in English and Hindi, and benchmark analytics.
- [x] Master execution script `run_pipeline.py` allowing end-to-end execution of the complete pipeline with a single command.

---

## 3. Execution Schedule & Milestone Tracking

| Milestone | Description | Status |
|---|---|---|
| **M1** | Data ingestion, synthesis, dictionary & EDA | Completed |
| **M2** | Preprocessing, feature engineering & imbalance pipeline | Completed |
| **M3** | Approach 1: 12+ ML models, tuning & statistical tests | Completed |
| **M4** | Approach 2: 10 DL models, focal loss, calibration & robustness | Completed |
| **M5** | Approach 3: Multi-Agent AI system, RAG & Local DB | Completed |
| **M6** | XAI: SHAP, LIME, attention maps & counterfactuals | Completed |
| **M7** | Visualizations, Evaluation markdown files & Benchmark tables | Completed |
| **M8** | 15-chapter complete academic documentation | Completed |
| **M9** | 22-slide PPT presentation (`.pptx`) for academic defense | Completed |
| **M10** | Publication-grade IEEE Research Paper PDF report | Completed |
| **M11** | Interactive Web Dashboard & REST API server on 0.0.0.0 | Completed |
| **M12** | Master single-command pipeline script (`run_pipeline.py`) | Completed |

---
*Created and maintained for IIIT Dharwad B.Tech Data Science & AI Project.*
