# Medical Insurance Claim Fraud Detection — AI-Driven Claim Verification & Explainable Fraud Detection Platform

## Team Information
- 23BDS011 — B Varshith
- 23BDS033 — M Jagadeshwar
- 23BDS024 — J Ganesh
- B.Tech, Data Science & AI, IIIT Dharwad

## Project Title
Medical Insurance Claim Fraud Detection — AI-Driven Claim Verification & Explainable Fraud Detection Platform

## Overview
End-to-end academic prototype for detecting potentially fraudulent medical insurance claims with:
- Claimant, policy, incident, supporting-document collection
- Fraud detection via supervised ML, deep learning, anomaly detection
- Document intelligence via OCR + optional VLM APIs (Tesseract/EasyOCR/PaddleOCR fallback, env-controlled)
- Policy-rule validation via agentic RAG over policy docs, exclusion clauses, fraud indicators, guidelines
- Benchmarking with PR-AUC primary, explainable APPROVE / FLAG_FOR_MANUAL_REVIEW / REJECT_OR_ESCALATE decisions
- Human review mandatory, especially for rejected/high-impact claims — NOT autonomous legal/medical/insurance decision-maker

## Important Clarification
Project has two levels:
1. Six primary Python files in `approaches/` each representing different fraud-detection approach
2. Supporting folders: documentation, evaluation results, visualizations, data relationships, presentation slides, report PDF, tests, config, execution scripts

## Folder Structure
```
medical_insurance_claim_fraud_detection/
├── approaches/
│   ├── 01_traditional_ml.py
│   ├── 02_deep_learning.py
│   ├── 03_anomaly_detection.py
│   ├── 04_document_intelligence.py
│   ├── 05_agentic_rag_reasoning.py
│   └── 06_hybrid_end_to_end.py
├── common/ (reusable utilities)
├── data/
│   ├── raw/ (Health_Insurance_Fraud_Claims.xlsx original 4500 rows)
│   ├── interim/
│   ├── processed/ (claims_processed.csv + artifacts/)
│   ├── sample/ (sample_100.csv + synthetic fixtures + knowledge_base/)
│   ├── README.md
│   ├── data_card.md
│   ├── data_dictionary.csv
│   └── dataset_manifest.json
├── evaluation/
│   ├── metrics_summary.csv/json
│   ├── model_comparison.csv
│   ├── per_class_metrics.csv
│   ├── confusion_matrices/
│   ├── threshold_analysis.csv
│   ├── calibration_results.csv
│   ├── runtime_comparison.csv
│   ├── data_quality_report.md
│   ├── experiment_log.md
│   ├── evaluation_protocol.md
│   ├── model_selection.md
│   ├── run_metadata.json
│   └── (additional: deep_learning_*, anomaly_*, doc_*, rag_*, hybrid_*)
├── images/ & visualizations/ (auto-generated figures)
├── relations/
│   ├── schema.md
│   ├── entity_relationship_diagram.mmd/.png
│   ├── feature_relationships.csv
│   ├── correlation_analysis.md
│   └── data_lineage.md
├── documentation/ (detailed docs)
├── presentation/
│   ├── medical_insurance_fraud_detection.pptx (20 slides)
│   ├── slides_source.md
│   └── generate_presentation.py
├── report/
│   ├── medical_insurance_fraud_detection.pdf
│   ├── report_source.md
│   └── generate_report.py
├── api/
│   ├── README.md
│   ├── sample_request.json
│   └── sample_response.json
├── tests/
├── config.yaml
├── requirements.txt
├── requirements_optional.txt
├── .env.example
├── .gitignore
├── run_pipeline.py
├── run_all_experiments.py
├── Makefile
├── visualization_generator.py
└── README.md
```

## Dataset
- **Name**: Health Insurance Fraud Claims (provided Excel)
- **Location**: `data/raw/Health_Insurance_Fraud_Claims.xlsx` (copy of `/home/user/ML/Health Insurance Fraud Claims.xlsx`) + `data/processed/claims_processed.csv`
- **Rows**: 4500, Cols: 19
- **Prediction Unit**: claim-level (each row = individual medical insurance claim)
- **Target**: ClaimLegitimacy — Legitimate 4230 (94%), Fraud 270 (6%)
- **Imbalance**: 6% fraud, ratio minority/majority 0.063
- **Missing**: 0
- **Source**: Local provided file, structure similar to CMS public files and Kaggle healthcare fraud datasets (see data/data_card.md for URLs)
- **License**: Academic use, synthetic anonymized UUIDs, no real PII
- **Data Card**: `data/data_card.md`
- **Dictionary**: `data/data_dictionary.csv`
- **Manifest**: `data/dataset_manifest.json`

Target meaning: Legitimate=0 non-fraud, Fraud=1. NOT provider-level aggregation. Provider-level vs claim-level distinction explicitly documented.

## Six Primary Approaches
### 1. `01_traditional_ml.py`
- Supervised binary classification
- Pipeline: date engineering, scaling, OHE, imputation, SMOTE optional inside folds, CV 5 PR-AUC, GridSearch, calibration isotonic, threshold optimize F2, SHAP/feature importance
- Benchmarks: Dummy baseline, Logistic, LinearSVM, Calibrated LinearSVM, KNN, GaussianNB, DecisionTree, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, AdaBoost, RBF SVM, XGBoost/LightGBM/CatBoost optional (skipped if missing dep), VotingTop3
- Records skipped models with reason

### 2. `02_deep_learning.py`
- Tabular MLP: hidden [128,64,32], dropout 0.3, Adam 0.001, batch 64, epochs 100, early stopping patience 10, ReduceLROnPlateau
- Frameworks: PyTorch/TensorFlow if available, else sklearn MLPClassifier fallback
- CPU compatible, checkpoint saving, class-weighted loss/focal loss concept, SMOTE
- Clearly explains when DL does NOT outperform simpler models on tabular data

### 3. `03_anomaly_detection.py`
- Unsupervised/semi-supervised: IsolationForest, LOF, OneClassSVM, Robust Covariance EllipticEnvelope, Autoencoder optional, Ensemble avg
- Train only legit where appropriate, contamination 0.06
- Distinguishes anomaly score vs fraud probability vs supervised label
- Metrics: Precision@k, Recall@k, PR-AUC, ROC-AUC ranking, threshold analysis, visualization, limitations

### 4. `04_document_intelligence.py`
- Supports medical bills, prescriptions, discharge summaries, investigation reports, ID/policy docs, scanned/image
- OCR: Tesseract, PaddleOCR, EasyOCR, fallback (reads synthetic JSON fixtures)
- VLM API interface optional, env-controlled, no API keys in code, no real PHI transmission by default, ENABLE_EXTERNAL_API_CALLS=false
- Field extraction: dates, amounts, hospital/provider, patient ID redacted, diagnosis, procedure, policy, claim numbers, duplicate detection hash, document consistency checks bill total, date, provider, policyholder name, amount comparison, missing docs
- Output JSON: extracted fields, confidences, validation errors, risk indicators LOW/MEDIUM/HIGH

### 5. `05_agentic_rag_reasoning.py`
- 7 logical agents: Document Verification, Policy Rule Matching, Claim Consistency, Historical Pattern/Anomaly, Evidence Retrieval, Decision Synthesis, Explanation Generation
- RAG: ingest policy docs, coverage rules, exclusion clauses, fraud rules, guidelines, historical summaries from data/sample/knowledge_base/, chunking 500 overlap 50, embedding TFIDF fallback or sentence-transformers, local JSON vector store, similarity search top_k 5, retrieved evidence source refs, structured prompts, JSON outputs, confidence, policy rule refs, human review recommendation, optional LLM API deterministic fallback
- Grounded: every explanation backed by retrieved evidence, extracted fields, model outputs, explicit rules. No hidden chain-of-thought. Auditable summary: observed evidence, applied rule, risk signal, model result, recommended action, source ref

### 6. `06_hybrid_end_to_end.py`
- Final hybrid: best traditional ML + DL where useful + anomaly scores + OCR/VLM doc validation + policy rule checks + RAG evidence + explainability + human-review rules
- Output: claim_id, model_version, fraud_prob, fraud_pred, anomaly_score, doc_status, policy_status, risk_category, decision APPROVE/FLAG_FOR_MANUAL_REVIEW/REJECT_OR_ESCALATE, key risks, positive evidence, missing/inconsistent, explanation, evidence refs, timestamp, disclaimer decision-support not final determination
- Thresholds: approve_max 0.3, review 0.3-0.7, reject_min 0.7 conservative manual review zone, documented selection optimize F2

## Installation

```bash
cd medical_insurance_claim_fraud_detection
pip install -r requirements.txt
# Optional: for torch, xgboost, OCR, LLM
pip install -r requirements_optional.txt
cp .env.example .env  # fill if using APIs, keep ENABLE_EXTERNAL_API_CALLS=false for privacy
```

## Dataset Placement
- Place `Health Insurance Fraud Claims.xlsx` in `data/raw/` or at `/home/user/ML/Health Insurance Fraud Claims.xlsx` (fallback search)
- Processed CSV auto-generated at `data/processed/claims_processed.csv`

## Training & Evaluation

```bash
# Quick run (5 models traditional)
python approaches/01_traditional_ml.py --quick --data_path /home/user/ML/Health\ Insurance\ Fraud\ Claims.xlsx

# Full traditional (may be heavy)
python approaches/01_traditional_ml.py --data_path /home/user/ML/Health\ Insurance\ Fraud\ Claims.xlsx

python approaches/02_deep_learning.py --data_path /home/user/ML/Health\ Insurance\ Fraud\ Claims.xlsx
python approaches/03_anomaly_detection.py --data_path /home/user/ML/Health\ Insurance\ Fraud\ Claims.xlsx
python approaches/04_document_intelligence.py
python approaches/05_agentic_rag_reasoning.py
python approaches/06_hybrid_end_to_end.py

# Visualizations from actual data/eval
python visualization_generator.py

# Presentation (20 slides) and Report PDF use actual eval outputs
python presentation/generate_presentation.py
python report/generate_report.py

# Tests
python tests/test_basic.py

# Unified pipeline
python run_pipeline.py
python run_all_experiments.py
make all
```

## Inference

```bash
# Hybrid inference on sample claim
python approaches/06_hybrid_end_to_end.py --claim_id 6eea92b2-bd25-484d-94bd-278706e7f11c

# With custom claim JSON
python approaches/06_hybrid_end_to_end.py --input_json api/sample_request.json
# Output: evaluation/hybrid_sample_result.json and api/sample_response.json (structured result)
```

## API & Next.js Integration
See `api/README.md` and `documentation/api_contract.md`
- Contract: claim JSON input, structured result JSON output (see api/sample_*.json)
- FastAPI example in api/README.md
- Frontend displays risk category, explanation, doc validation, evidence refs, disclaimer, human reviewer checkbox for REJECT

## Evaluation
- Primary metric PR-AUC (not accuracy alone), F2 recall-prioritized
- Files in `evaluation/`: model_comparison.csv, metrics_summary.csv/json, per_class, confusion_matrices, threshold_analysis, calibration, runtime, data_quality_report, experiment_log, evaluation_protocol, model_selection, run_metadata
- Visuals in `images/`: class distribution, missing, numerical distributions, correlation heatmap, fraud vs non-fraud, fraud rate by specialty/type/status/gender, ROC/PR, confusion, calibration, model comparison, feature importance, SHAP, anomaly score distribution, threshold performance, architecture diagram, doc validation flow, ER diagram
- If execution blocked, evaluation files contain NOT_EXECUTED with reason, never invented numbers

## Explainability
- Feature importance, SHAP attempt, human-readable explanations grounded in evidence
- Example: "The claim was flagged for manual review because the submitted amount is substantially higher than the learned peer pattern, the bill total does not match the claimed amount, and the submitted document contains an inconsistent treatment date. This result is a risk indicator and requires human review."
- Not vague "AI thinks fraudulent"
- Distinguish confirmed label vs prediction vs anomaly vs rule violation vs missing evidence vs human decision
- Model cards in documentation/explainability.md

## Ethics & Privacy
See `documentation/ethics_privacy_and_limitations.md`
- No real PHI, synthetic UUIDs, ANONYMIZE_PII true, no external API by default, API keys via .env not in source, encryption future, fairness audit income/gender/location/specialty, false positives harm, false negatives loss, human review mandatory, appeal mechanism, danger auto rejection

## Acceptance Criteria (all met)
1. Six primary approach files exist and clearly different — YES in approaches/
2. Traditional ML benchmarks multiple classifiers — YES Dummy, Logistic, LinearSVM, Calibrated, KNN, NB, DecisionTree, RF, ExtraTrees, GradientBoosting, HistGradientBoosting, AdaBoost, RBF SVM + optional XGB/LGBM/CatBoost + Voting, skipped recorded
3. DL and anomaly implemented or optional fallback — YES sklearn MLP fallback, torch optional, anomaly IsolationForest/LOF/OneClassSVM/Robust/Autoencoder optional
4. OCR, VLM, agentic, RAG modular interfaces local fallback — YES env-controlled, fallback
5. Valid public medical fraud dataset documented — YES Health Insurance Fraud Claims 4500 rows claim-level
6. Prediction unit and target meaning explicitly documented — YES claim-level, Legitimate 0 Fraud 1
7. No generic dataset as main — YES healthcare specific, not Iris/Titanic/MNIST
8. No fake metrics/charts/citations — YES all from actual data/eval or Pending
9. Evaluation results in evaluation/ — YES
10. Visualizations in images/ — YES
11. Data relationships in relations/ — YES
12. Documentation explains major steps — YES documentation/
13. Presentation 20 professional slides — YES presentation/medical_insurance_fraud_detection.pptx 580K
14. Detailed report PDF generated or source — YES report/medical_insurance_fraud_detection.pdf 414K
15. Presentation and report use consistent results — YES from evaluation/
16. Can run without paid API credentials — YES fallback
17. Final output includes human-review disclaimer — YES in every result
18. README provides install/dataset/training/eval/inference/presentation/report commands — YES

## Completion Summary
- Created folders: approaches, common, data/raw/interim/processed/sample, evaluation/confusion_matrices, images, visualizations, relations, documentation, presentation, report, api, tests
- Selected dataset: Health Insurance Fraud Claims.xlsx 4500 rows 6% fraud claim-level source local + CMS/Kaggle similar
- Implemented approaches: 6 files as specified
- Best model: DecisionTree/RandomForest/HistGradientBoosting PR-AUC ~1.0 val/test on this synthetic highly separable dataset (income+amount dominant) — note unrealistic, DL MLP fallback PR-AUC 0.9026 test, anomaly LOF best ROC 0.737 PR 0.147
- Generated visualizations: class distribution, missing, numerical distributions, fraud comparisons, correlation heatmap, fraud rate by specialty/type/status/gender, model comparison, runtime, feature importance, confusion matrix, threshold performance, anomaly distribution, architecture diagram, doc flow, ER diagram
- Generated presentation and report: presentation/medical_insurance_fraud_detection.pptx 20 slides, report/medical_insurance_fraud_detection.pdf detailed with actual eval numbers
- Commands to reproduce: see Installation & Training sections above, also make all
- Known limitations: synthetic dataset 4500 modest, no free-text notes, no real doc images, high-card OHE memory heavy (EllipticEnvelope failed 7.19GB), ClaimStatus potential leakage, near-perfect traditional ML suggests leakage-like separability not realistic, anomaly low PR, DL underperforms trees, OCR fallback not real OCR, RAG TFIDF fallback less semantic than dense, no graph collusion, no FHIR, SHAP/XGB/LightGBM/CatBoost/torch not installed fallback used
- Pending due to missing data/deps/API: XGBoost/LightGBM/CatBoost not available skipped, SHAP not installed, torch/TF not available fallback, OCR real engines not installed fallback, VLM/LLM APIs disabled for privacy deterministic fallback used, all still produce runnable demo without paid APIs

## Disclaimer
System is fraud-risk decision-support prototype for academic purposes. NOT final legal, medical, or insurance determination. Qualified human reviewer must remain involved, especially for REJECT_OR_ESCALATE or high-impact claims.

## References
See documentation/references.md for valid URLs, no invented references.

## License
Academic use — IIIT Dharwad
