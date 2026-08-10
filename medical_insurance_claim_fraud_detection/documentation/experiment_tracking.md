# Experiment Tracking

## Overview
Logs of experiments run, including parameters, metrics, runtime, notes.

## Log Format
Each run writes to `evaluation/`:
- `run_metadata.json` — timestamp, config, data path, approach, best model
- `metrics_summary.json` — detailed metrics
- `model_comparison.csv` — per model PR-AUC, ROC-AUC, fit time

## Runs (actual)

### 01_traditional_ml quick run (2026-08-06)
- Data: 4500 rows, 19 cols, fraud 6%
- Split: train 2925, val 675, test 900
- Models: Dummy, Logistic, DecisionTree, RandomForest, HistGradientBoosting
- Best: DecisionTree PR-AUC 1.0 val, threshold 0.95 via optimize_f2
- Test metrics @ 0.95: accuracy 0.9989, precision 0.9818, recall 1.0, F1 0.9908, PR-AUC 0.9818, ROC-AUC 0.9994
- HistGradientBoosting also PR-AUC 1.0 test
- Note: near-perfect suggests synthetic separability, possibly PatientIncome + ClaimAmount leakage-like
- Artifacts: best_traditional_ml_model.joblib saved

### 02_deep_learning (2026-08-06)
- Framework: sklearn_mlp_fallback (torch not available)
- Hidden [128,64,32], lr 0.001, batch 64, epochs 100, early stopping patience 10
- Preprocessing: same as traditional, OHE 8521 features
- SMOTE resampled train 2925 -> 5498
- Val PR-AUC 0.8729, Test PR-AUC 0.9026
- Threshold 0.05 via optimize_f2 (low threshold due to class imbalance and model underconfidence)
- Observation: DL does NOT outperform tree models on tabular data with limited rows; documented in doc

### 03_anomaly_detection (2026-08-06)
- Train only legit: 3384 out of 3600
- Contamination 0.06
- Models: IsolationForest PR-AUC 0.0718 ROC 0.487, LOF PR-AUC 0.1468 ROC 0.7374 (best), OneClassSVM PR 0.1228 ROC 0.679, Ensemble PR 0.1293 ROC 0.656
- Precision@10 0.2, Recall@200 up to 0.518 for LOF
- EllipticEnvelope failed due to memory (7.19 GiB for covariance 9822 features)
- Autoencoder not trained (torch missing)
- Note: Anomaly score != fraud prob; high FP, needs supervised calibration

### 04_document_intelligence (2026-08-06)
- OCR engine fallback (Tesseract/EasyOCR/PaddleOCR not installed)
- VLM disabled
- Processed 4 synthetic docs (2 bills, prescription, discharge maybe)
- Bill total mismatch detected as expected (synthetic mismatch)
- Overall validation FAILED due to mismatch, missing prescription/discharge in sample run
- Output saved to evaluation/document_intelligence_sample_output.json

### 05_agentic_rag (2026-08-06)
- KB 5 chunks loaded
- TFIDF fallback (sentence-transformers not available)
- Retrieved policy rules, coverage, guidelines, fraud indicators, exclusion
- Sample claim amount 1703, fraud prob 0.65 moderate
- All agents PASSED originally, synthesis risk MEDIUM? Actually high due to doc issues in hybrid

### 06_hybrid (2026-08-06)
- Loaded best traditional model (DecisionTree)
- Anomaly models: IsolationForest, LOF, OneClassSVM (via joblib)
- Preprocessor anomaly
- Doc pipeline fallback, RAG TFIDF
- Sample claim 6eea92b2... amount 1703 legit? predicted fraud_prob 0.0 (DecisionTree perfect)
- Anomaly score 0.404
- Doc validation FAILED (amount mismatch synthetic bills)
- Policy PASSED
- Risk category HIGH due to doc failure (conservative)
- Decision FLAG_FOR_MANUAL_REVIEW
- Explanation includes missing docs, bill mismatch

## Pending / Not Executed
- XGBoost, LightGBM, CatBoost: missing dependency, recorded skipped
- SHAP: not installed
- Torch / TF: not available, fallback used, noted not superiority claim
- OCR real: Tesseract not installed, fallback used
- VLM/LLM: disabled for privacy, fallback deterministic rules used
- All still produce runnable demo without paid APIs

## Metrics Files
See evaluation/*.csv, *.json

## Visualization
Generated via visualization_generator.py using actual data/eval

## Command to Reproduce
```
pip install -r requirements.txt
python approaches/01_traditional_ml.py
python approaches/02_deep_learning.py
python approaches/03_anomaly_detection.py
python approaches/04_document_intelligence.py
python approaches/05_agentic_rag_reasoning.py
python approaches/06_hybrid_end_to_end.py
python visualization_generator.py
python presentation/generate_presentation.py
python report/generate_report.py
```
Or `make all` or `python run_all_experiments.py`
