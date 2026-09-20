# FORM 3: Traditional ML — Evaluation & Benchmarking Framework
## Medical Insurance Claim Fraud Detection — Approach 1 of 3
### IIIT Dharwad | B.Tech Data Science & AI

---

**Faculty Adviser:** Prof. Ramesh Athe  
**Team Members:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  
**Date:** 01-09-2026 | **Version:** 1.0

---

## 1. Purpose

This document specifies the complete evaluation and benchmarking framework for all models trained in Approach 1. It defines every metric, visualization, statistical test, and reporting format required. All results are stored in the `evaluation/` folder and referenced by documentation, PPT, and PDF reports.

## 2. Evaluation Metrics — Complete Definition

### 2.1 Classification Metrics

| Metric | Formula | Priority | Description |
|--------|---------|----------|-------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Low | Overall correctness (misleading for imbalance) |
| Precision | TP/(TP+FP) | High | Of flagged claims, fraction actually fraud |
| Recall | TP/(TP+FN) | Critical | Of actual fraud, fraction detected |
| F1 Score | 2·(P·R)/(P+R) | High | Harmonic mean of precision and recall |
| F2 Score | 5·(P·R)/(4P+R) | Primary | Recall-weighted (missed fraud costs more) |
| ROC-AUC | Area under ROC curve | High | Ranking quality across all thresholds |
| PR-AUC | Area under PR curve | Primary | Headline metric for imbalanced data |
| MCC | (TP·TN-FP·FN)/√(...) | High | Chance-corrected balanced measure |
| Brier Score | mean((p-y)²) | Medium | Probability calibration quality (lower=better) |
| Log-Loss | -mean(y·log(p)+(1-y)·log(1-p)) | Medium | Penalized confidence measure |

**Note:** Positive class (1) = Fraud in all metrics.

### 2.2 Business Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Detection Cost Factor | (FP·C_FP + FN·C_FN) / (TP·C_TP) | Weighted error cost |
| Savings | TP·avg_fraud_amount - FP·review_cost | Net financial benefit |
| Coverage@K% | Cumulative fraud captured in top K% flagged | Operational triage value |

### 2.3 Cost Assumptions (Indian Context)

| Error Type | Cost (INR) | Rationale |
|-----------|-----------|-----------|
| False Negative (missed fraud) | Average claim amount ~8,734 INR | Insurer pays fraudulent claim |
| False Positive (wrongly flagged) | 500 INR | Manual review cost + customer goodwill |
| True Positive (detected fraud) | +8,734 INR savings | Fraud prevented |
| True Negative (correct legitimate) | 0 | Normal processing |

## 3. Benchmarking Table Structure

### 3.1 Leaderboard CSV Schema

```
model_key,family,status,
val_accuracy,val_precision,val_recall,val_f1,val_f2,
val_roc_auc,val_pr_auc,val_mcc,val_brier,val_logloss,
test_accuracy,test_precision,test_recall,test_f1,test_f2,
test_roc_auc,test_pr_auc,test_mcc,test_brier,test_logloss,
train_seconds,predict_ms_per_1000,artifact_kb,
best_params_json,notes
```

### 3.2 Leaderboard Rendering Rules
- Sort by val_pr_auc descending
- Round metrics to 4 decimal places
- Round times to 2 decimal places
- Status: 'complete', 'skipped', or 'failed'
- Skipped models include reason text
- Include all 19 models (M01-M19)

## 4. Required Artifacts Per Model

### 4.1 Metric Files
- `{model_key}_metrics.json` — Complete metric dictionary
- `{model_key}_confusion_matrix.csv` — 2x2 confusion matrix
- `{model_key}_classification_report.json` — Per-class metrics

### 4.2 Curve Files
- `{model_key}_roc.csv` — ROC curve points (FPR, TPR, threshold)
- `{model_key}_pr.csv` — PR curve points (recall, precision, threshold)
- `{model_key}_threshold_sweep.csv` — 101 threshold values with all metrics

### 4.3 Visualization Files
- `{model_key}_confusion_matrix.png` — Annotated heatmap
- `{model_key}_roc_curve.png` — ROC curve with AUC in legend
- `{model_key}_pr_curve.png` — PR curve with baseline reference
- `{model_key}_probability_histogram.png` — Score distributions by class

### 4.4 Model Card
- `model_cards/{model_key}_card.md` — One-page model summary

## 5. Comparative Visualization Requirements

### 5.1 Model Comparison Plots
| Plot | Description | File |
|------|-------------|------|
| Grouped bar chart | All metrics across all models | `models/all_models_metric_comparison.png` |
| ROC overlay | All model ROC curves on one plot | `models/roc_all_models_overlay.png` |
| PR overlay | All model PR curves on one plot | `models/pr_all_models_overlay.png` |
| Radar chart | Top 5 models on 6 metrics | `models/radar_top5_models.png` |
| Scatter plot | Training time vs PR-AUC | `models/time_vs_performance.png` |
| Heatmap | Prediction correlation matrix | `models/prediction_correlation_heatmap.png` |

### 5.2 Calibration Plots
| Plot | Description | File |
|------|-------------|------|
| Reliability diagram | Before calibration | `calibration/before_calibration.png` |
| Reliability diagram | After calibration | `calibration/after_calibration.png` |
| Brier score comparison | All models | `calibration/brier_comparison.png` |

### 5.3 Learning Curves
| Plot | Description | File |
|------|-------------|------|
| Learning curves | Train/val score vs data size | `models/learning_curves_{model_key}.png` |
| Validation curves | Score vs hyperparameter | `models/validation_curves_{model_key}.png` |

## 6. Statistical Significance Testing

### 6.1 Paired Comparison Tests
- **McNemar's Test:** Pairwise model comparison on test set
- **Paired t-test:** Compare CV fold scores between top models
- **Wilcoxon signed-rank:** Non-parametric alternative to paired t-test

### 6.2 Significance Threshold
- Alpha = 0.05 for all tests
- Results reported with p-values
- "Statistically significant" only when p < 0.05
- Practical significance: PR-AUC difference > 0.005

### 6.3 Bootstrap Confidence Intervals
- 2,000 bootstrap resamples of test set
- 95% percentile confidence intervals for all metrics
- Reported for winner and runner-up

## 7. Threshold Optimization

### 7.1 Method
- Sweep thresholds from 0.01 to 0.99 (101 points)
- Compute precision, recall, F2 at each threshold
- Select threshold maximizing F2 on validation set

### 7.2 Operating Point Bands
| Band | Score Range | Action |
|------|-----------|--------|
| Auto-clear | < 0.30 | Approve claim automatically |
| Review | 0.30 – threshold | Flag for manual review |
| Escalate | > threshold | Priority investigation |

### 7.3 Threshold Memo
Written document justifying the chosen operating threshold:
- Selected threshold value
- Precision and recall at that threshold
- Business cost analysis at that threshold
- Comparison with default 0.5 threshold

## 8. Fairness Analysis

### 8.1 Slice Definitions
| Slice | Groups |
|-------|--------|
| Gender | Female, Male |
| Age Group | 0-17, 18-30, 31-45, 46-60, 61+ |
| Employment | Employed, Student, Unemployed, Retired |
| ClaimType | Inpatient, Outpatient, Emergency, Routine |
| Submission | Paper, Online, Phone |

### 8.2 Per-Slice Metrics
For each slice group: TPR, FPR, Precision, F1, Count, Fraud Rate

### 8.3 Disparity Rules
- Flag if any group's TPR differs by >5 percentage points from majority
- Document explanation for any flagged disparity
- Note: small sample sizes (< 30 positives) → mark as unstable

## 9. Error Analysis

### 9.1 False Negative Analysis
- Examine all FN cases
- Identify patterns: common features, amounts, clusters
- Determine if missed fraud cases share characteristics

### 9.2 False Positive Analysis
- Examine all FP cases
- Identify patterns: legitimate claims incorrectly flagged
- Determine cost of false alarms

### 9.3 Error Pattern Summary
- Table of error types with counts and percentages
- Visualization of error distribution by key features
- Recommendations for feature engineering improvements

## 10. Business Impact Analysis

### 10.1 Financial Summary (Indian Context)

```
Assumptions:
  - Average fraudulent claim: 8,734 INR
  - Manual review cost per claim: 500 INR
  - Total claims in evaluation period: 4,500
  
Expected outcomes (per 100,000 claims):
  - Frauds detected: Recall × 6,000 claims
  - Savings: Detected frauds × 8,734 INR
  - Review costs: (TP + FP) × 500 INR
  - Net savings: Savings - Review costs
```

### 10.2 Investigator Hours Saved
- Estimate: Each manual review takes ~30 minutes
- Model triage reduces reviews by focusing on high-risk claims
- Report estimated hours saved vs. reviewing all claims

## 11. Evaluation Folder Structure

```
evaluation/
├── evaluation.md                    # Human-readable results hub
├── leaderboard.csv                  # Master comparison table
├── selection_memo.md                # Winner selection rationale
├── threshold_memo.md                # Operating point justification
├── test_results.json                # One-time test metrics
├── metrics/
│   ├── {model_key}_metrics.json     # Per-model complete metrics
│   └── ...
├── curves/
│   ├── {model_key}_roc.csv          # ROC curve data
│   ├── {model_key}_pr.csv           # PR curve data
│   └── ...
├── calibration/
│   ├── calibration_table.csv
│   └── calibration_plots.png
├── fairness/
│   ├── slice_metrics.csv
│   └── disparity_analysis.md
├── tuning/
│   ├── {model_key}_trials.csv       # Hyperparameter search log
│   └── ...
└── model_cards/
    ├── {model_key}_card.md          # One-page model summary
    └── ...
```

## 12. Acceptance Criteria

| Criterion | Requirement | Status |
|-----------|-------------|--------|
| All models trained | 19 models minimum | ☐ |
| All metrics computed | 10+ metrics per model | ☐ |
| All curves saved | ROC + PR per model | ☐ |
| Leaderboard complete | All 19 rows filled | ☐ |
| Winner selected | Selection memo written | ☐ |
| Test evaluation done | One-time, documented | ☐ |
| Calibration done | Reliability diagram present | ☐ |
| Fairness checked | Slice table complete | ☐ |
| Business impact | Financial analysis present | ☐ |
| Statistical tests | McNemar + CI reported | ☐ |

## 13. References

1. Saito & Rehmsmeier (2015). The Precision-Recall Plot is More Informative than the ROC Plot. PLoS ONE.
2. Matthews (1975). Comparison of the predicted and observed secondary structure of T4 phage lysozyme.
3. IRDAI Annual Reports: Indian insurance claim statistics
4. scikit-learn evaluation metrics documentation

---

*Last updated: 01-09-2026 | Document version: 1.0*  
*Next: Form 4 — Feature Engineering & Selection Specification*
