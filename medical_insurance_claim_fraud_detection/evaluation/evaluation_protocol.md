# Evaluation Protocol

## Overview
Same protocol for comparable supervised approaches; anomaly detection labeled not directly comparable.

## Splits
- Train 65% (2925)
- Val 15% (675)
- Test 20% (900)
- Stratified by fraud label, random_state 42, shuffle
- Test set untouched until final evaluation

## Primary Metric
- PR-AUC (Average Precision) for fraud detection, not accuracy alone
- Secondary: ROC-AUC, F1, F2 (recall prioritized), MCC, Balanced Accuracy, Precision, Recall, Specificity, Sensitivity, Brier, Calibration, Precision@k, Recall@k, FPR, FNR

## Threshold Analysis
- Default 0.5 not suitable for imbalanced 6% fraud
- Strategies: optimize_f1, optimize_f2, pr_auc_recall_target (recall >=0.8)
- Documented in threshold_analysis.csv
- Conservative manual review zone 0.3-0.7 via config.yaml hybrid.thresholds

## Calibration
- Attempt isotonic/sigmoid via CalibratedClassifierCV
- Save calibration_results.csv prob_true vs prob_pred
- Brier score

## Model Selection
- Best model selected on val PR-AUC
- Save model_comparison.csv with val_pr_auc, test_pr_auc, fit_time

## Anomaly Detection
- Train only legit where configured (3384)
- Evaluate PR-AUC, ROC-AUC, Prec@k, Rec@k using fraud labels as reference
- Distinguish anomaly score vs fraud probability
- No calibration unless validated

## Document Intelligence
- Validation status PASSED/FAILED/NEEDS_REVIEW
- Risk LOW/MEDIUM/HIGH
- Field confidences, errors, warnings

## RAG
- Retrieved evidence with source refs, scores
- Grounded explanations

## Missing Execution
- If dataset/dependency unavailable, write NOT_EXECUTED with reason, never invented numbers

## Files
- metrics_summary.csv/json, model_comparison.csv, per_class_metrics.csv, confusion_matrices/, threshold_analysis.csv, calibration_results.csv, runtime_comparison.csv, data_quality_report.md, experiment_log.md, model_selection.md, run_metadata.json

## Reproducibility
- Seed 42, config.yaml, requirements.txt
