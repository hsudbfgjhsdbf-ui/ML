# Presentation Source Outline — Medical Insurance Claim Fraud Detection

## Slide 1: Title
- Title: Medical Insurance Claim Fraud Detection — AI-Driven Claim Verification & Explainable Fraud Detection Platform
- Institution: IIIT Dharwad, B.Tech Data Science & AI
- Team: 23BDS011 B Varshith, 23BDS033 M Jagadeshwar, 23BDS024 J Ganesh
- Date: 2026-08-06

## Slide 2: Team
- B Varshith 23BDS011
- M Jagadeshwar 23BDS033
- J Ganesh 23BDS024
- Advisor: (if any)

## Slide 3: Problem Statement
- Medical fraud: upcoding, unbundling, duplicate, fake docs
- Financial loss, premium increase, patient harm
- Need AI decision-support, not auto-rejection, human-in-loop

## Slide 4: Motivation and Impact
- $300B+ fraud US healthcare annually (citation CMS)
- Manual review slow, error-prone
- AI can prioritize suspicious claims, validate docs, provide explanations
- Must be explainable, fair, privacy-preserving

## Slide 5: Objectives
- Collect claimant/policy/incident/docs
- Detect fraud, extract via OCR/VLM, validate against rules/history
- Experiment classical ML, DL, anomaly, doc intelligence, agentic RAG
- Benchmark with appropriate metrics
- Transparent explanations APPROVE/MANUAL/REJECT

## Slide 6: Scope and Assumptions
- Scope: academic prototype, claim-level 4500 rows, CPU, offline default, synthetic docs
- Assumptions: synthetic IDs realistic, fraud rate 6% filtered, policy rules illustrative
- Out-of-scope: autonomous rejection, legal determination, real PHI processing

## Slide 7: End-to-End Workflow
- Diagram: architecture_diagram.png
- Flow: Claim Input -> Doc Intelligence -> Preprocessing -> ML/DL/Anomaly -> Policy RAG -> Agentic -> Hybrid -> Explanation -> Decision

## Slide 8: Dataset and Data Card
- Health Insurance Fraud Claims.xlsx, 4500 rows, 19 cols
- ClaimID, PatientID, ProviderID, ClaimAmount, ClaimDate, DiagnosisCode, ProcedureCode, PatientAge, Gender, Specialty, Status, Income, Marital, Employment, Location, Type, SubmissionMethod, Cluster, ClaimLegitimacy
- Class distribution: 4230 legitimate (94%), 270 fraud (6%)
- Missing 0, prediction unit claim-level
- Source: local + CMS/Kaggle similar
- Image: class_distribution.png, missing_values.png

## Slide 9: Data Schema and Relationships
- ER diagram: relations/entity_relationship_diagram.png
- Entities: Patient, Policyholder, Policy, Provider, Claim, Diagnosis, Procedure, Document, Fraud Label, Review Decision, Historical
- Production vs actual: policy table, doc table missing in dataset, we add synthetic fixtures
- Feature relationships: feature_relationships.csv

## Slide 10: Preprocessing and Feature Engineering
- Target mapping legit 0 fraud 1
- Date engineering: year, month, day, dayofweek, quarter, ordinal
- Numerical: median impute + StandardScaler
- Categorical: most_frequent + OneHotEncoder(handle_unknown ignore)
- Pipeline learned only on train, no leakage
- Class imbalance: ratio 0.063, class_weight balanced, SMOTE optional inside folds
- Outlier IQR, leakage heuristic ClaimStatus flagged
- Visuals: ClaimAmount distribution, fraud comparison

## Slide 11: Traditional ML Approach
- Benchmark: Dummy baseline, Logistic, LinearSVM, Calibrated LinearSVM, KNN, NB, DecisionTree, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, AdaBoost, RBF SVM, XGBoost/LightGBM/CatBoost optional, VotingTop3
- CV 5, scoring PR-AUC, GridSearch
- Calibration isotonic (attempted)
- Threshold optimize F2
- Results: model_comparison.png shows DecisionTree/RF/HGB PR-AUC ~1.0 on this synthetic data, Dummy 0.06, Logistic 0.94
- Feature importance: PatientIncome, ClaimAmount dominant
- Skipped: XGBoost etc missing dep

## Slide 12: Deep Learning Approach
- MLP: hidden [128,64,32], dropout 0.3, Adam lr 0.001, batch 64, epochs 100, early stopping patience 10, LR scheduler ReduceLROnPlateau
- PyTorch if available, else sklearn MLP fallback (used)
- Class-weighted loss / focal loss concept, SMOTE resampled 2925->5498
- CPU compatible
- Results: Val PR-AUC 0.87, Test 0.90, threshold 0.05 low due to underconfidence
- Observation: DL does NOT automatically outperform trees on tabular limited data - documented
- Loss curves: training loss

## Slide 13: Anomaly Detection Approach
- Unsupervised/semi-supervised: IsolationForest, LOF, OneClassSVM, EllipticEnvelope, Autoencoder optional, Ensemble avg
- Train only legit (3384) vs legit+fraud (3600)
- Distinction: anomaly score (deviation) vs fraud prob (calibrated) vs fraud label (ground truth)
- Metrics: PR-AUC, ROC-AUC, Precision@k, Recall@k
- Results: LOF best PR 0.147 ROC 0.737, IsolationForest 0.07, OneClassSVM 0.12, Ensemble 0.12
- Precision@10 0.2, Recall@200 0.51 LOF
- Limitations: high FP, cannot replace supervised, needs human review, cannot distinguish rare legit vs fraud

## Slide 14: OCR and Document Intelligence
- Supported docs: medical bills, prescriptions, discharge summaries, investigation reports, ID/policy docs, scanned/image
- OCR: Tesseract, PaddleOCR, EasyOCR optional, fallback reads JSON fixtures
- VLM interface: env-controlled, no API keys in code, no real PHI transmission by default, deterministic fallback
- Pipeline: OCR -> type identification (keyword+structured) -> field extraction regex+structured (dates, amounts, hospital, provider, patient ID redacted, diagnosis, procedure, policy, claim) -> validation (duplicate hash, bill total vs claimed diff $5 tolerance, date consistency, provider, policyholder name, amount comparison, missing docs)
- Output JSON: extracted fields, confidences, validation errors, risk LOW/MEDIUM/HIGH
- Sample: synthetic_bill_1.json total 7820.52 items sum correct, mismatch fixture 6000 vs 4500 intentional
- Visual: document_validation_flow.png
- Privacy safeguards

## Slide 15: Agentic AI and RAG Architecture
- Agents: Document Verification, Policy Rule Matching, Claim Consistency, Historical Pattern/Anomaly, Evidence Retrieval, Decision Synthesis, Explanation Generation
- RAG: ingest policy_rules.txt, exclusion_clauses.txt, fraud_indicators.txt, coverage_rules.txt, claim_guidelines.txt from data/sample/knowledge_base, chunk 500 overlap 50, TFIDF fallback or sentence-transformers, local JSON vector store, similarity search top_k 5, retrieved evidence with source refs, scores
- Structured prompts, JSON outputs, confidence, policy rule refs, human-review recommendation
- LLM optional, deterministic fallback, no hidden chain-of-thought, auditable summary: observed evidence, applied rule, risk signal, model result, recommended action, source ref
- Sample retrieved: "Pre-authorization required for >$10000" etc

## Slide 16: Hybrid End-to-End Solution
- Combine: best traditional ML (DecisionTree/RF), DL prob where useful, anomaly score, doc validation signals, policy checks, RAG evidence, explainability, human-review rules
- Weights: ML 0.5, DL 0.2, Anomaly 0.15, Document 0.15 (config.yaml)
- Thresholds: approve_max 0.3, review 0.3-0.7, reject_min 0.7, conservative manual review zone
- Output JSON: claim_id, model_version, fraud_prob, fraud_pred, anomaly_score, doc_status, policy_status, risk_category LOW/MEDIUM/HIGH, decision APPROVE/FLAG_FOR_MANUAL_REVIEW/REJECT_OR_ESCALATE, key risks, positive evidence, missing/inconsistent, explanation, evidence refs, timestamp, disclaimer
- Explanation example: flagged for manual review because amount higher than peer pattern, bill total mismatch, inconsistent date, requires human review, disclaimer
- Visual: architecture_diagram.png
- Sample result: hybrid_sample_result.json

## Slide 17: Evaluation Metrics and Protocol
- Protocol: same for comparable supervised, anomaly not directly comparable
- Splits: train 2925, val 675, test 900 stratified 42, untouched test
- Metrics: Accuracy, Precision, Recall, F1, F2 prioritized recall, ROC-AUC, PR-AUC primary, Balanced Accuracy, MCC, Specificity, Sensitivity, Confusion, Brier, Calibration curve, Prec@k, Rec@k, FPR, FNR, cost-sensitive
- Primary not accuracy alone
- Threshold analysis instead of 0.5 default, optimize F2, documented
- Calibration isotonic attempted
- Runtime comparison
- NOT_EXECUTED with reason if blocked, no invented numbers

## Slide 18: Benchmark Results and Model Comparison
- Model comparison chart: model_comparison_pr_auc.png
- Traditional: Best DecisionTree/RF/HGB test PR-AUC 0.98-1.0 (synthetic separable)
- DL: sklearn MLP test PR-AUC 0.9026
- Anomaly: LOF best ROC 0.737 PR 0.147
- Document: bill mismatch detection works
- Hybrid: sample decision FLAG_FOR_MANUAL_REVIEW due to doc failures, prob 0.0 but HIGH risk via docs
- Confusion matrix: confusion_matrix.png (TP 54 FP 1 etc)
- Calibration: calibration_results.csv (51 bytes maybe due to fallback)
- Threshold performance: threshold_performance.png
- Runtime: runtime_comparison.png

## Slide 19: Explainability, Risk Controls, Human Review
- Feature importance: images/feature_importance.png PatientIncome, ClaimAmount
- SHAP attempted but not installed, status logged
- Human explanation: key features, doc errors, policy violations, anomaly indicators, missing evidence, grounded, not vague "AI thinks fraud"
- Risk controls: weights, thresholds, conservative review zone, doc FAILED => manual review, anomaly top_k => review
- Model cards: intended use decision-support prototype, out-of-scope autonomous, training data, eval, metrics, limitations, bias (income, gender, location), threshold policy, failure modes
- Human review mandatory for REJECT_OR_ESCALATE, appeal mechanism, audit
- Disclaimer in every output

## Slide 20: Limitations, Future Work, Conclusion, References
- Limitations: synthetic dataset 4500 modest, no free-text notes, no real doc images, high-card OHE memory heavy (7GB for covariance), ClaimStatus potential leakage, near-perfect ML not realistic, anomaly low PR, DL underperforms trees, OCR fallback, RAG TFIDF less semantic, no graph collusion, no FHIR
- Future: larger CMS data, time/group split, embeddings for codes, graph features provider-patient network, fairness audit, calibration + cost-sensitive, encrypted secure API, Next.js frontend reviewer workflow, model monitoring drift retraining
- Conclusion: built complete reproducible end-to-end academic prototype with 6 approaches, evaluation, visualizations, docs, presentation, report, runnable without paid APIs, responsible disclaimer, human-in-loop
- References: see documentation/references.md CMS, Kaggle, sklearn, SHAP, RAG, model cards, EU Trustworthy AI etc
- Commands to reproduce: pip install -r requirements.txt, python approaches/... , visualization_generator.py, presentation/generate_presentation.py, report/generate_report.py, make all
- Thank you
