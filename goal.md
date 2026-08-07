# MASTER GOALS AND END-TO-END IMPLEMENTATION ROADMAP
**Project Title:** Medical Insurance Claim Fraud Detection — End-to-End Three-Approach Framework  
**Faculty Adviser:** Prof. Ramesh Athe  
**Institution:** Indian Institute of Information Technology (IIIT), Dharwad  
**Department:** B.Tech in Data Science and Artificial Intelligence  
**Team Members:**  
- B Varshith (Roll Number: 23BDS011)  
- M Jagadeshwar (Roll Number: 23BDS033)  
- J Ganesh (Roll Number: 23BDS024)  

---

## 1. Executive Summary & Core Mission
This document (`goal.md`) establishes the complete, comprehensive, end-to-end roadmap for building, benchmarking, explaining, and deploying the **Medical Insurance Claim Fraud Detection System** across all three progressive approaches:
1. **Approach 1: Traditional Machine Learning Approach** — Baseline classification and benchmarking across 12 classical supervised algorithms.
2. **Approach 2: Deep Learning & Explainable AI (XAI) Approach** — Advanced neural feature representation across 10 distinct deep tabular architectures, augmented by explainability methods (SHAP, LIME, Attention, and Counterfactuals).
3. **Approach 3: Agent AI / Multi-Agent System Approach** — Autonomous cognitive multi-agent orchestration using Large Language Models (LLMs), Vision Language Models (VLMs), LangChain, LangGraph, Retrieval-Augmented Generation (RAG), Local SQLite Database, and a modern user-facing Next.js frontend web application.

Every aspect of this implementation is specifically tailored to the **Indian Healthcare and Insurance Landscape**, incorporating real-world policy structures (Family Floater, Individual, Employer Group, Government Schemes like Ayushman Bharat and ECHS), major Indian insurers (Star Health, ICICI Lombard, HDFC Ergo, New India Assurance), regional hospital tiers (Tier-1 Metro Corporate, Tier-2 City Hospitals, Tier-3 Nursing Homes), Indian Rupee (INR / Rs.) cost dynamics, and unbiased demographic performance across women, children, elderly citizens, and differently-abled individuals.

---

## 2. Comprehensive Goals by Approach

### Approach 1: Traditional Machine Learning Baseline & Fair Benchmarking
- **Dataset Acquisition & Preprocessing:**
  - Utilize the provided dataset (`Health Insurance Fraud Claims.xlsx`, 4,500 records, 6.0% fraud rate).
  - Clean missing values, handle duplicate records, preserve fraud outliers, and perform stratified 70% Train / 15% Validation / 15% Test splits.
  - Implement comprehensive feature scaling (StandardScaler, MinMaxScaler, RobustScaler) and class imbalance handling (SMOTE, RandomUndersampling, Class Weighting).
- **Domain-Specific Feature Engineering:**
  - Create Indian-context features: `ClaimToPremiumRatio`, `TreatmentCostDeviationINR`, `DaysSincePolicyInception`, `ClaimFrequency12M`, `AgeTreatmentRiskScore`, `HospitalTierCostRatio`, and statistical policyholder aggregations.
- **Algorithm Implementations (12 Classical Algorithms):**
  1. Logistic Regression (L1 LASSO & L2 Ridge regularization)
  2. Decision Tree Classifier
  3. Random Forest Classifier
  4. Gradient Boosting Classifier (Histogram-based optimization)
  5. XGBoost Classifier (with `scale_pos_weight` for imbalance)
  6. LightGBM Classifier (leaf-wise growth for speed)
  7. Support Vector Machine (RBF and Linear kernels)
  8. K-Nearest Neighbors (distance-weighted)
  9. Gaussian & Multinomial Naive Bayes
  10. Feedforward Artificial Neural Network (MLP baseline)
  11. AdaBoost Classifier
  12. Quadratic Discriminant Analysis (QDA)
- **Hyperparameter Tuning & Statistical Evaluation:**
  - Perform Stratified K-Fold Cross-Validation ($k=5$) and Grid/Random Search targeting **F2-Score** (prioritizing recall for fraud minimization).
  - Compute accuracy, precision, recall, F1-score, F2-score, AUC-ROC, AUC-PR, Matthews Correlation Coefficient (MCC), training latency (seconds), prediction latency (milliseconds), and model memory footprint (KB).
  - Perform statistical significance testing (**McNemar's Test** and **Wilcoxon Signed-Rank Test**).

---

### Approach 2: Deep Learning & Explainable AI (XAI) Framework
- **Deep Tabular Architectures (10 Neural Implementations in PyTorch):**
  1. **Multi-Layer Perceptron (MLP):** 4 hidden layers with ReLU, Batch Normalization, and Dropout.
  2. **Wide & Deep Network:** Joint linear memorization path and deep generalization path.
  3. **Deep & Cross Network (DCN):** Explicit bounded-degree feature interactions via cross-layers.
  4. **TabNet-Style Attentive Network:** Sequential sparse attention masking for interpretable feature selection.
  5. **Tabular Transformer:** Self-attention over feature token embeddings with positional encoding.
  6. **ResNet for Tabular Data:** Stacked residual blocks with skip connections and pre-activation batch normalization.
  7. **NODE (Neural Oblivious Decision Ensembles):** Differentiable oblivious trees with temperature-controlled soft split decisions.
  8. **LSTM Sequential Claim Classifier:** Recurrent modeling over chronological policyholder claim histories.
  9. **Autoencoder Anomaly Detector:** Unsupervised reconstruction error thresholding trained on legitimate claims.
  10. **Variational Autoencoder (VAE):** Probabilistic latent space modeling for anomaly detection and synthetic augmentation.
- **Advanced Training Dynamics:**
  - Implement Focal Loss, Weighted Binary Cross-Entropy, Cosine Annealing learning rate schedules with warm-up, early stopping, and mixed precision.
- **Explainable AI (XAI) Layer:**
  - Integrate **SHAP** (DeepExplainer / KernelExplainer / GradientExplainer) to compute feature attributions.
  - Integrate **LIME** for local explainability of individual claim decisions.
  - Extract and visualize attention weights from Transformer and TabNet models.
  - Implement Counterfactual Explanations showing minimal feature changes required to flip a prediction.
- **Fairness & Bias Audit:**
  - Evaluate Equalized Odds, Demographic Parity, and Predictive Parity across gender, age brackets, geographic regions, and income levels.

---

### Approach 3: Agent AI / Multi-Agent Cognitive System
- **Multi-Agent Architecture (LangChain & LangGraph):**
  - **Coordinator Agent:** Stateful workflow orchestration, routing, retry logic, and session state management.
  - **Document Processing Agent (OCR + Vision):** Extracts structured JSON fields from Indian medical bills, prescriptions, discharge summaries, laboratory reports, and identity proofs (Aadhaar / PAN).
  - **Policy Verification Agent:** Cross-checks claim details against Indian insurance policy clauses (Family Floater sub-limits, waiting periods, room rent caps, co-payments) using RAG.
  - **Anomaly Detection Agent:** Identifies billing inflation, treatment-cost deviations, hospital-tier anomalies, and temporal fraud indicators.
  - **Historical Pattern Agent:** Evaluates claimant historical claim frequency, escalating amounts, and fraud ring connections.
  - **Explainable Reasoning & Decision Agent:** Synthesizes findings across all agents, weighs conflicting evidence, and generates human-readable natural language explanations citing specific policy clauses and INR cost benchmarks.
- **RAG Pipeline & Local Database:**
  - SQLite Database (`local_database.db`) storing Users, Policies, Claims, Uploaded Documents, Agent Results, Hospital Reference Data, and Fraud Rulebooks.
  - Local Vector Store for semantic retrieval of IRDAI regulations, policy clauses, and Indian medical cost protocols.
- **Next.js Frontend Web Application:**
  - Responsive, modern user interface with guided claim submission, multi-format document upload (camera/PDF), live claim status tracking dashboard, and explainable AI decision display.

---

## 3. Deliverables & Artifact Generation Roadmap

1. **Codebase (`src/`, `configs/`, `models/`, `notebooks/`, `nextjs-app/`):**
   - Clean, modular, fully commented Python codebase for Approaches 1, 2, and 3.
   - Complete Next.js frontend application in `/home/user/ML/nextjs-app`.
   - Executable pipeline scripts and Jupyter Notebooks for reproduction.
2. **Evaluation Directory (`evaluation/`):**
   - Comprehensive markdown documents (`evaluation.md`, `approach1_evaluation.md`, `approach2_evaluation.md`, `approach3_evaluation.md`, `comparative_analysis.md`, `fairness_audit.md`).
   - Exhaustive benchmarking tables comparing all 22 algorithms + Multi-Agent system across Accuracy, Precision, Recall, F1, F2, AUC-ROC, AUC-PR, MCC, Latency, and Cost.
   - Exactly 2,000+ lines per main evaluation report to ensure complete academic rigor.
3. **Documentation Directory (`documentation/`):**
   - Master documentation (`project_documentation.md`, 2,000+ lines) detailing background, 20+ paper literature review, mathematical formulations, architectural diagrams, code explanations, Indian context, and ethical bias mitigation.
   - Code explanation reference (`code_explanation.md`, 2,000+ lines) describing every file, module, function, input, output, and error handling mechanism.
4. **Visualizations Directory (`visualizations/`):**
   - 30+ high-resolution plots saved as PNGs: dataset distributions, correlation heatmaps, ROC/PR curves, SHAP summary/dependence plots, learning curves, architecture flowcharts, and fairness evaluation charts.
5. **Presentation Slides (`presentation/`):**
   - Formal PowerPoint presentation deck (`presentation/Medical_Insurance_Fraud_Detection_Presentation.pptx`) and companion markdown presentation deck (`presentation/presentation_slides.md`, 20 structured slides) adhering to IIIT Dharwad academic formatting and crediting Prof. Ramesh Athe.
6. **Research Paper Report (`reports/`):**
   - Two-column IEEE formatted research paper PDF (`reports/IEEE_Research_Paper_Medical_Insurance_Fraud_Detection.pdf`) and LaTeX/Markdown source (`reports/ieee_research_paper.md`, 2,000+ lines) with abstract, introduction, literature review (20+ citations), experimental setup, benchmarking tables, figures, discussion, and acknowledgments.

---

## 4. Execution Step-by-Step Schedule
- [x] **Step 1:** Establish Python `.venv`, install all dependencies, and verify repo files.
- [x] **Step 2:** Write `goal.md` defining project mission, architecture, and deliverables.
- [ ] **Step 3:** Implement core data processing, feature engineering, and Approach 1 (12 Classical ML Algorithms) with full benchmarking and statistical significance testing.
- [ ] **Step 4:** Implement Approach 2 (10 Deep Learning Tabular Architectures in PyTorch) with XAI (SHAP, LIME, Attention, Counterfactuals) and fairness evaluation.
- [ ] **Step 5:** Implement Approach 3 (Agent AI Multi-Agent System with LangChain/LangGraph, SQLite DB, RAG Vector Store, and Explainable Reasoning).
- [ ] **Step 6:** Build the responsive Next.js frontend web application (`nextjs-app/`) for user claim submission, document upload, and explainable decision viewing.
- [ ] **Step 7:** Generate all 30+ visualizations in `visualizations/`.
- [ ] **Step 8:** Generate extensive 2,000+-line documentation and evaluation markdown files in `documentation/` and `evaluation/`.
- [ ] **Step 9:** Create formal 20-slide PowerPoint presentation (`presentation/`) and IEEE research paper PDF (`reports/`).
- [ ] **Step 10:** Verify end-to-end execution, single-command runnability, and submission readiness.

---
**END OF GOAL DOCUMENT**
