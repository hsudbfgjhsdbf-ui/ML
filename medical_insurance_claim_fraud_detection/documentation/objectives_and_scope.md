# Objectives and Scope

## Objectives
1. **Data Collection & Understanding**
   - Load public medical insurance fraud dataset (Health Insurance Fraud Claims.xlsx)
   - Validate schema, target, missing, imbalance, outliers, leakage
   - Document data card, dictionary, manifest, lineage

2. **Fraud Detection Models**
   - Traditional ML: benchmark Dummy, Logistic, LinearSVM, Calibrated LinearSVM, KNN, NB, DecisionTree, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, AdaBoost, RBF SVM, XGBoost/LightGBM/CatBoost optional, Voting/Stacking
   - Deep Learning: MLP with embeddings, class-weighted loss, early stopping, CPU-compatible, fallback sklearn MLP
   - Anomaly Detection: IsolationForest, LOF, OneClassSVM, Robust Covariance, Autoencoder optional, Ensemble

3. **Document Intelligence**
   - OCR integration (Tesseract, EasyOCR, PaddleOCR) with fallback
   - VLM API interface optional, env-controlled, no hard-coded keys, no real PHI transmission by default
   - Document type identification, field extraction (dates, amounts, provider, patient ID redacted, diagnosis, procedure, policy, claim numbers)
   - Validation: duplicate detection, bill total vs claimed, date consistency, provider consistency, missing docs
   - Structured JSON output

4. **Agentic RAG Reasoning**
   - 7 logical agents: Document Verification, Policy Rule Matching, Claim Consistency, Historical Pattern/Anomaly, Evidence Retrieval, Decision Synthesis, Explanation Generation
   - RAG over policy docs, coverage rules, exclusion clauses, fraud indicators, guidelines, historical summaries
   - Chunking, embedding (sentence-transformers or TFIDF fallback), vector store (local JSON), similarity search, evidence citations
   - Deterministic fallback when no LLM API
   - Grounded explanations: observed evidence, applied rule, risk signal, model result, recommended action, source reference. No hidden chain-of-thought.

5. **Hybrid End-to-End**
   - Combine best ML, DL, anomaly, document, policy, RAG, explainability, human-review rules
   - Output: claim_id, model_version, fraud_prob, prediction, anomaly_score, doc status, policy status, risk category, decision, key risks, positive evidence, missing/inconsistent, explanation, evidence refs, timestamp, disclaimer
   - Operational outcomes: APPROVE, FLAG_FOR_MANUAL_REVIEW, REJECT_OR_ESCALATE with threshold documented

6. **Evaluation & Explainability**
   - Metrics: Accuracy, Precision, Recall, F1, F2, ROC-AUC, PR-AUC, Balanced Accuracy, MCC, Specificity, Sensitivity, Confusion, Brier, Calibration curve, Precision@k, Recall@k, FPR, FNR, Cost-sensitive
   - Primary metric PR-AUC or fraud recall at acceptable precision, not accuracy alone
   - Same protocol for comparable approaches, label non-comparable (unsupervised vs supervised)
   - Feature importance, SHAP, human-readable explanations

7. **Documentation & Presentation**
   - Full documentation, visualizations, data relationships diagrams, model cards, ethics/privacy
   - 20-slide presentation with actual evaluation outputs (or Pending if not executed)
   - Detailed report PDF

## Scope
- Academic prototype, not production system
- Claim-level prediction (4500 rows)
- Public dataset (synthetic but realistic)
- CPU execution
- Optional external APIs, default offline
- No real PII, synthetic fixtures for docs

## Out of Scope
- Fully autonomous rejection without human review
- Real medical record processing without anonymization
- Legal/insurance final determination
- Real-time FHIR integration (future work)
- Graph-based collusion detection (future)

## Assumptions
- Provided Excel is ground truth for research
- ClaimID unique
- Fraud rate 6% representative of filtered population (real fraud lower)
- Synthetic documents sufficient for OCR demo
- Policy rules in knowledge_base are illustrative, not legal

## Deliverables
- Six approach files in approaches/
- Common utilities
- Data folder with card, dictionary, manifest
- Evaluation folder with metrics, plots, confusion matrices
- Images folder
- Relations folder
- Documentation folder
- Presentation pptx + source md + generator script
- Report pdf + source md + generator script
- Tests, config, requirements, run scripts, API contract
