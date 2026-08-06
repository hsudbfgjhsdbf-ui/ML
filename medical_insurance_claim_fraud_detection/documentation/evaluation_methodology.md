# Evaluation Methodology

## Protocol
- Same protocol for comparable supervised approaches (traditional ML, deep learning)
- Unsupervised anomaly detection labeled as not directly comparable; evaluated via ranking metrics.

## Splits
- Train 65% (2925), Val 15% (675), Test 20% (900)
- Stratified by fraud label, random_state 42
- Touch test set only once at end (untouched)

## Metrics for Imbalanced Fraud
- Accuracy, Precision, Recall, F1, F2 (recall prioritized), ROC-AUC, PR-AUC (primary), Balanced Accuracy, MCC, Specificity, Sensitivity, Confusion, Brier score, Precision@k, Recall@k, FPR, FNR, Cost-sensitive
- Computed via `common/metrics.py`

## Threshold Analysis
- Bias: default 0.5 not suitable for imbalanced
- Strategies: optimize_f2, optimize_f1, pr_auc_recall_target (recall >=0.8)
- Documented in `threshold_analysis.csv`
- Final decision: conservative manual review zone 0.3-0.7 via config.yaml hybrid.thresholds

## Model Selection
- Primary: PR-AUC on val
- Secondary: recall at acceptable precision, F2, MCC
- Not accuracy alone

## Calibration
- Attempt isotonic/sigmoid calibration via CalibratedClassifierCV
- Calibration curve data in `calibration_results.csv`
- Brier score

## Anomaly Detection
- Train only on legitimate where configured
- Metrics: PR-AUC using fraud as anomaly, ROC-AUC, Precision@k, Recall@k, ranking
- Clearly distinguish anomaly score vs fraud probability

## Document Intelligence
- Validation status PASSED/FAILED/NEEDS_REVIEW
- Risk LOW/MEDIUM/HIGH
- Errors and warnings, field confidences

## RAG
- Retrieved evidence with source refs and scores
- Grounded explanation audit

## Runtime Comparison
- Fit time per model in `runtime_comparison.csv`

## Missing Execution Handling
- If dataset or dependency unavailable, write NOT_EXECUTED with reason, never invented numbers.

## Files
- `evaluation/` folder contains all tables, plots stored in images/
- `data_quality_report.md`, `experiment_log.md`, `evaluation_protocol.md`, `model_selection.md`, `run_metadata.json`
