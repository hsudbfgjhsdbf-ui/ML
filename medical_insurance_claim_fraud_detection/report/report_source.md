# Project Report Source — Medical Insurance Claim Fraud Detection

## Cover page
Title: Medical Insurance Claim Fraud Detection — AI-Driven Claim Verification & Explainable Fraud Detection Platform
Team: 23BDS011 B Varshith, 23BDS033 M Jagadeshwar, 23BDS024 J Ganesh, IIIT Dharwad B.Tech Data Science & AI
Date: 2026-08-06

## Abstract
Medical insurance fraud causes significant financial loss. This project builds an end-to-end academic prototype for AI-driven claim verification and explainable fraud detection. Uses public-like claim dataset (4500 rows, 4500 claims, 6% fraud), benchmarks traditional ML, deep learning MLP, unsupervised anomaly detection, OCR-based document intelligence, agentic RAG over policy rules, and hybrid decision synthesis. Evaluates with PR-AUC primary, F1/F2, ROC-AUC, calibration, precision@k. Produces transparent explanations, risk categories, and operational decisions APPROVE/FLAG_FOR_MANUAL_REVIEW/REJECT_OR_ESCALATE with human-review mandatory disclaimer. System runs without paid APIs via fallbacks. Demonstrates that tree models outperform DL on tabular modest data, anomaly detection has high FP, and document validation is crucial.

Keywords: insurance fraud, claim verification, explainable AI, anomaly detection, document intelligence, RAG, hybrid AI

## Introduction
Background on fraud types, need for AI assistance, responsible AI.

## Problem Statement
See documentation/problem_statement.md

## Motivation
See documentation/project_overview.md

## Objectives
See documentation/objectives_and_scope.md

## Scope
In scope/out-of-scope, assumptions.

## Related Work and References
- CMS Fraud Prevention System
- Kaggle healthcare provider fraud datasets
- SHAP explainability
- RAG Lewis et al.
- Model Cards
See documentation/references.md for full citations with URLs.

## Dataset Description
See data/data_card.md
Rows 4500, cols 19, claim-level, target ClaimLegitimacy 94% legit 6% fraud, missing 0, IDs synthetic, no real PHI.

## Data Dictionary
See data_dictionary.csv

## Data Quality Analysis
Missing 0, outlier IQR 0 for amount/age/income synthetic, imbalance ratio 0.063, leakage heuristic ClaimStatus flagged, correlation heatmap, fraud rate by specialty/type/status/gender visualizations.

## Data Relationships
ER diagram, feature relationships csv, correlation analysis, lineage. Production vs actual schema distinction.

## Feature Engineering
Date engineering year/month/day/dayofweek/quarter/ordinal, numerical median+scaler, categorical most_frequent+OHE 8521 features, IDs dropped, pipeline learned only on train, SMOTE optional inside folds.

## Traditional ML Methodology
Benchmark 14+ classifiers, CV 5 PR-AUC, GridSearch, calibration isotonic attempt, threshold optimize F2, feature importance, SHAP attempted, save best model.

## Deep Learning Methodology
MLP hidden [128,64,32], dropout 0.3, Adam 0.001, batch 64, epochs 100, early stopping patience 10, ReduceLROnPlateau, class-weighted loss pos_weight, CPU, PyTorch/TF optional else sklearn MLP fallback with SMOTE, loss curves, threshold tuning, note DL not automatically superior.

## Anomaly Detection Methodology
IsolationForest, LOF, OneClassSVM, EllipticEnvelope, Autoencoder optional, Ensemble avg, train only legit 3384, contamination 0.06, distinction anomaly score vs fraud prob vs label, precision@k recall@k ranking, threshold analysis, visualization anomaly score distribution, limitations high FP.

## OCR/VLM Document Intelligence Methodology
Supported docs medical bills, prescriptions, discharge summaries, investigation reports, ID/policy, scanned/image. OCR Tesseract/EasyOCR/PaddleOCR optional fallback JSON fixtures. VLM interface env-controlled no hard-coded keys no real PHI transmission by default. Type identification keyword+structured, field extraction regex+structured dates amounts provider patient redacted diagnosis procedure policy claim numbers, validation duplicate hash bill total vs claimed $5 tol date consistency provider policyholder name amount comparison missing docs, output JSON extracted fields confidences validation errors risk indicators. Privacy safeguards.

## Agentic AI and RAG Methodology
7 agents: Document Verification, Policy Rule Matching, Claim Consistency, Historical Pattern/Anomaly, Evidence Retrieval, Decision Synthesis, Explanation Generation. RAG: ingest policy_rules.txt, exclusion_clauses.txt, fraud_indicators.txt, coverage_rules.txt, claim_guidelines.txt, chunk 500 overlap 50, embedding TFIDF fallback or sentence-transformers, local JSON vector store, similarity search top_k 5, retrieved evidence source refs scores, structured prompts JSON outputs confidence evidence citations policy rule refs human review, optional LLM API deterministic fallback, grounded explanations, auditable summary observed evidence applied rule risk signal model result recommended action source ref, no hidden chain-of-thought.

## Hybrid System Methodology
Combine best ML (DecisionTree/RF), DL prob, anomaly scores, doc validation, policy checks, RAG evidence, explainability, human-review rules. Weights ML 0.5 DL 0.2 Anomaly 0.15 Document 0.15 config.yaml. Thresholds approve_max 0.3 review 0.3-0.7 reject_min 0.7 conservative manual review zone. Final system accepts claim JSON returns structured result claim_id model_version fraud_prob fraud_pred anomaly_score doc_status policy_status risk_category decision key risks positive evidence missing/inconsistent explanation evidence refs timestamp disclaimer. Outcomes APPROVE FLAG_FOR_MANUAL_REVIEW REJECT_OR_ESCALATE not arbitrary threshold, documented selection via optimize F2, conservative.

## System Architecture
Diagram architecture_diagram.png, document_validation_flow.png, ER diagram. Data layer, preprocessing, ML layer, policy/RAG, agentic, hybrid synthesis, explainability, output. Next.js integration via JSON contract.

## Implementation Details
Python 3.10+, common utilities reusable, modular, type hints, docstrings, configurable paths, no hard-coded absolute, no hard-coded API keys, logging not uncontrolled print, reproducible seeds, graceful handling missing files deps, clear error messages, unit tests.

## Code Explanation
Walkthrough common/ and approaches/ 6 files, evaluation/, images generation, etc. See documentation/code_walkthrough.md

## Training Procedure
Seed 42, stratified split 2925/675/900, pipeline fit train only, CV 5, GridSearch, SMOTE only inside training, calibration attempt, threshold tuning.

## Evaluation Protocol
Same protocol comparable supervised, anomaly not directly comparable, metrics accuracy precision recall F1 F2 ROC-AUC PR-AUC balanced accuracy MCC specificity sensitivity confusion Brier calibration precision@k recall@k FPR FNR cost-sensitive, primary PR-AUC not accuracy alone, threshold analysis, calibration, runtime, NOT_EXECUTED with reason if blocked.

## Model Benchmark
Model comparison table model_comparison.csv, metrics_summary.json, per_class_metrics.csv, confusion matrices, threshold_analysis.csv, calibration_results.csv, runtime_comparison.csv, visualizations model_comparison_pr_auc.png, confusion_matrix.png, threshold_performance.png etc. Use actual numbers: traditional best DecisionTree/RF/HGB PR-AUC ~1.0 test (synthetic separable), DL MLP fallback test PR-AUC 0.9026, anomaly LOF best PR 0.147 ROC 0.737, document bill mismatch detection works, hybrid sample FLAG.

## Error Analysis
False positives: legit flagged due to doc FAILED or high amount vs peer; false negatives: fraud missed due to low anomaly? Analyze confusion matrix TP 54 FP1 FN0 TN845 @ thr0.95 - near perfect suggests synthetic leak Income feature. Need fairness audit.

## Explainability Analysis
Feature importance PatientIncome, ClaimAmount dominant, SHAP not installed, human-readable explanations with risk signals, evidence citations, auditable summary.

## Security and Privacy
No real PHI, synthetic UUIDs, ANONYMIZE_PII true, no external API by default, API keys via env, encryption future, access control future, audit logs.

## Fairness and Ethical Considerations
Bias income gender location specialty, need subgroup metrics, false positives harm patients, false negatives financial loss, human review mandatory, appeal mechanism, danger auto rejection, disclaimer in every output.

## Limitations
Synthetic dataset 4500 modest, no free-text, no real doc images, high-card OHE memory heavy, ClaimStatus potential leakage, near-perfect ML unrealistic, anomaly low PR, DL underperforms, OCR fallback, RAG TFIDF less semantic, no graph collusion, no FHIR.

## Future Work
Larger CMS data, time/group split, embeddings for codes, graph features, fairness audit, calibration cost-sensitive, encrypted secure API, Next.js frontend reviewer workflow, monitoring drift retraining.

## Conclusion
Complete reproducible end-to-end academic prototype built, 6 approaches, evaluation, visuals, docs, presentation, report, runnable without paid APIs, responsible.

## References
See documentation/references.md with valid URLs

## Appendix
Commands config sample outputs
- pip install -r requirements.txt
- python approaches/01_traditional_ml.py --data_path /path/to/xlsx
- python approaches/02_deep_learning.py
- python approaches/03_anomaly_detection.py
- python approaches/04_document_intelligence.py
- python approaches/05_agentic_rag_reasoning.py
- python approaches/06_hybrid_end_to_end.py
- python visualization_generator.py
- python presentation/generate_presentation.py
- python report/generate_report.py
- python run_all_experiments.py
- make all

Config.yaml excerpt, sample request/response JSON.

