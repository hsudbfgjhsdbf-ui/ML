# Exploratory analysis report

**Purpose:** record what was learned before model fitting.  
**Run:** `run_20260807_151423`.  
**Last updated:** 07-08-2026 15:14:23 UTC.

## Executive observations

- The supplied workbook has 4,500 rows and a fraud prevalence of 6.0%; accuracy alone is therefore inadequate.
- Fraud rows in this snapshot have a higher typical claim amount and lower typical reported income; this association is descriptive and may reflect the synthetic construction of the workbook.
- `Cluster` shows strong class concentration in the supplied data. It is retained as a declared source feature, but its importance is audited because a latent cluster can behave like a shortcut.
- IDs and almost-unique diagnosis/procedure/location values are excluded to avoid memorization and poor generalization to unseen entities.
- No missing source cells or exact duplicates were observed in the shipped snapshot; the pipeline still runs imputation as a robustness contract.
- Time is represented by calendar features rather than raw timestamps.
- Figures use teal for legitimate and orange for fraud, with labels and units.
- Fraud-rate charts are descriptive and are not evidence of demographic causation.
- The workbook has fictional-looking location values and no verified Indian state field; Indian-context claims are therefore contextual framing, not population validation.
- EDA is generated before model fitting and saved for independent review.

## Figure index

| Figure | Path | Title | Interpretation / caveat |
| --- | --- | --- | --- |
| Figure 1 | `images/eda/target_distribution_bar.png` | Class distribution | Fraud is the positive class; bars show all validated rows. |
| Figure 2 | `images/eda/target_distribution_pie.png` | Fraud prevalence | The supplied snapshot contains a six-percent fraud prevalence. |
| Figure 3 | `images/eda/missingness_bar.png` | Missing-value audit | No source column is silently dropped; the pipeline records the measured null rate. |
| Figure 4 | `images/eda/numeric_claim_amount_inr_by_class.png` | claim_amount_inr distribution | Training-free distribution view; extreme values are retained for fraud analysis. |
| Figure 5 | `images/eda/numeric_patient_age_years_by_class.png` | patient_age_years distribution | Training-free distribution view; extreme values are retained for fraud analysis. |
| Figure 6 | `images/eda/numeric_patient_income_inr_by_class.png` | patient_income_inr distribution | Training-free distribution view; extreme values are retained for fraud analysis. |
| Figure 7 | `images/eda/numeric_cluster_code_by_class.png` | cluster_code distribution | Training-free distribution view; extreme values are retained for fraud analysis. |
| Figure 8 | `images/eda/numeric_log_claim_amount_by_class.png` | log_claim_amount distribution | Training-free distribution view; extreme values are retained for fraud analysis. |
| Figure 9 | `images/eda/numeric_log_patient_income_by_class.png` | log_patient_income distribution | Training-free distribution view; extreme values are retained for fraud analysis. |
| Figure 10 | `images/eda/numeric_claim_to_income_ratio_by_class.png` | claim_to_income_ratio distribution | Training-free distribution view; extreme values are retained for fraud analysis. |
| Figure 11 | `images/eda/numeric_claim_minus_income_scaled_by_class.png` | claim_minus_income_scaled distribution | Training-free distribution view; extreme values are retained for fraud analysis. |
| Figure 12 | `images/eda/numeric_claim_amount_per_age_by_class.png` | claim_amount_per_age distribution | Training-free distribution view; extreme values are retained for fraud analysis. |
| Figure 13 | `images/eda/numeric_income_age_interaction_by_class.png` | income_age_interaction distribution | Training-free distribution view; extreme values are retained for fraud analysis. |
| Figure 14 | `images/eda/numeric_claim_year_by_class.png` | claim_year distribution | Training-free distribution view; extreme values are retained for fraud analysis. |
| Figure 15 | `images/eda/numeric_claim_month_by_class.png` | claim_month distribution | Training-free distribution view; extreme values are retained for fraud analysis. |
| Figure 16 | `images/eda/numeric_claim_day_of_week_by_class.png` | claim_day_of_week distribution | Training-free distribution view; extreme values are retained for fraud analysis. |
| Figure 17 | `images/eda/numeric_claim_day_of_month_by_class.png` | claim_day_of_month distribution | Training-free distribution view; extreme values are retained for fraud analysis. |
| Figure 18 | `images/eda/numeric_claim_is_weekend_by_class.png` | claim_is_weekend distribution | Training-free distribution view; extreme values are retained for fraud analysis. |
| Figure 19 | `images/eda/numeric_claim_month_sin_by_class.png` | claim_month_sin distribution | Training-free distribution view; extreme values are retained for fraud analysis. |
| Figure 20 | `images/eda/numeric_claim_month_cos_by_class.png` | claim_month_cos distribution | Training-free distribution view; extreme values are retained for fraud analysis. |
| Figure 21 | `images/eda/headline_boxplots_by_class.png` | Headline feature spread | Box plots preserve extreme values rather than treating every outlier as an error. |
| Figure 22 | `images/eda/correlation_heatmap.png` | Correlation heatmap | Correlation supports redundancy review; it is not interpreted as causation. |
| Figure 23 | `images/eda/categorical_age_band_counts.png` | age_band categories | Rare categories are kept for unknown-safe encoding; this plot displays the twelve most common levels. |
| Figure 24 | `images/eda/fraud_rate_age_band.png` | Fraud rate by age_band | Rates are descriptive and should not be treated as causal or sufficient for an adverse decision. |
| Figure 25 | `images/eda/categorical_cluster_category_counts.png` | cluster_category categories | Rare categories are kept for unknown-safe encoding; this plot displays the twelve most common levels. |
| Figure 26 | `images/eda/fraud_rate_cluster_category.png` | Fraud rate by cluster_category | Rates are descriptive and should not be treated as causal or sufficient for an adverse decision. |
| Figure 27 | `images/eda/categorical_patientgender_counts.png` | patientgender categories | Rare categories are kept for unknown-safe encoding; this plot displays the twelve most common levels. |
| Figure 28 | `images/eda/fraud_rate_patientgender.png` | Fraud rate by patientgender | Rates are descriptive and should not be treated as causal or sufficient for an adverse decision. |
| Figure 29 | `images/eda/categorical_providerspecialty_counts.png` | providerspecialty categories | Rare categories are kept for unknown-safe encoding; this plot displays the twelve most common levels. |
| Figure 30 | `images/eda/fraud_rate_providerspecialty.png` | Fraud rate by providerspecialty | Rates are descriptive and should not be treated as causal or sufficient for an adverse decision. |
| Figure 31 | `images/eda/categorical_patientmaritalstatus_counts.png` | patientmaritalstatus categories | Rare categories are kept for unknown-safe encoding; this plot displays the twelve most common levels. |
| Figure 32 | `images/eda/fraud_rate_patientmaritalstatus.png` | Fraud rate by patientmaritalstatus | Rates are descriptive and should not be treated as causal or sufficient for an adverse decision. |
| Figure 33 | `images/eda/categorical_patientemploymentstatus_counts.png` | patientemploymentstatus categories | Rare categories are kept for unknown-safe encoding; this plot displays the twelve most common levels. |
| Figure 34 | `images/eda/fraud_rate_patientemploymentstatus.png` | Fraud rate by patientemploymentstatus | Rates are descriptive and should not be treated as causal or sufficient for an adverse decision. |
| Figure 35 | `images/eda/categorical_claimtype_counts.png` | claimtype categories | Rare categories are kept for unknown-safe encoding; this plot displays the twelve most common levels. |
| Figure 36 | `images/eda/fraud_rate_claimtype.png` | Fraud rate by claimtype | Rates are descriptive and should not be treated as causal or sufficient for an adverse decision. |
| Figure 37 | `images/eda/categorical_claimsubmissionmethod_counts.png` | claimsubmissionmethod categories | Rare categories are kept for unknown-safe encoding; this plot displays the twelve most common levels. |
| Figure 38 | `images/eda/fraud_rate_claimsubmissionmethod.png` | Fraud rate by claimsubmissionmethod | Rates are descriptive and should not be treated as causal or sufficient for an adverse decision. |
| Figure 39 | `images/eda/claim_amount_vs_income.png` | Claim amount relationship | Scatter points are colored by the supplied binary label; overlap is expected in real triage. |
| Figure 40 | `images/eda/claim_amount_vs_age.png` | Claim amount relationship | Scatter points are colored by the supplied binary label; overlap is expected in real triage. |
| Figure 41 | `images/eda/monthly_volume_and_fraud_rate.png` | Monthly claim patterns | Monthly counts and rates are descriptive; the two-year supplied window is not a causal time series. |
| Figure 42 | `images/eda/fraud_rate_heatmap_claim_type_specialty.png` | Fraud-rate heatmap | Cells show descriptive percentages; sparse cells should be interpreted cautiously. |
| Figure 43 | `images/eda/claim_amount_by_cluster_violin.png` | Claim amount by cluster | The cluster is a supplied source field and is shown for shortcut-risk auditing. |
| Figure 44 | `images/eda/monthly_claim_amount_trend.png` | Monthly claim amount | Mean and median amounts are descriptive; the supplied observation window is limited. |
| Figure 45 | `images/eda/claim_type_composition_stacked.png` | Claim-type composition | Counts show volume and class composition without implying a causal claim-type effect. |
| Figure 46 | `images/eda/fraud_rate_by_claim_amount_decile.png` | Fraud rate by amount decile | Decile rates are descriptive and should be checked against policy and provider context. |
| Figure 47 | `images/models/calibration_reliability.png` | Reliability diagram | Calibration is fitted with validation data only and is reported before any locked test evaluation. |
| Figure 48 | `images/models/threshold_sweep_winner.png` | Validation threshold sweep | F2 is maximized on validation data with the selected operating threshold marked. |
| Figure 49 | `images/models/feature_importance_permutation.png` | Permutation importance — Soft voting ensemble | Global importance is descriptive; it does not establish causation. |
| Figure 50 | `images/models/curves/roc_curves_validation.png` | Validation ROC curves | All models use the same validation split; legend values are ROC-AUC. |
| Figure 51 | `images/models/curves/pr_curves_validation.png` | Validation PR curves | PR-AUC is the primary ranking metric under class imbalance. |
| Figure 52 | `images/models/validation_metric_comparison.png` | Validation metric comparison | Models are ordered by validation F2 and show the recall-weighted trade-off. |
| Figure 53 | `images/models/training_time_vs_pr_auc.png` | Training time versus PR-AUC | The highlighted point is the validation-selected winner; efficiency is reported, not used to hide slower models. |
| Figure 54 | `images/models/confusion_matrix_random_forest.png` | Confusion matrix for Random forest | Fraud is the positive class; matrix is evaluated at the validation-selected threshold. |
| Figure 55 | `images/models/confusion_matrix_gradient_boosting.png` | Confusion matrix for Gradient boosting | Fraud is the positive class; matrix is evaluated at the validation-selected threshold. |
| Figure 56 | `images/models/confusion_matrix_hist_gradient_boosting.png` | Confusion matrix for Histogram gradient boosting | Fraud is the positive class; matrix is evaluated at the validation-selected threshold. |
| Figure 57 | `images/models/confusion_matrix_adaboost.png` | Confusion matrix for AdaBoost | Fraud is the positive class; matrix is evaluated at the validation-selected threshold. |
| Figure 58 | `images/models/confusion_matrix_voting.png` | Confusion matrix for Soft voting ensemble | Fraud is the positive class; matrix is evaluated at the validation-selected threshold. |
| Figure 59 | `images/models/confusion_matrix_stacking.png` | Confusion matrix for Stacking ensemble | Fraud is the positive class; matrix is evaluated at the validation-selected threshold. |
| Figure 60 | `images/models/confusion_matrix_svm_rbf.png` | Confusion matrix for Support vector machine (RBF) | Fraud is the positive class; matrix is evaluated at the validation-selected threshold. |
| Figure 61 | `images/models/confusion_matrix_decision_tree.png` | Confusion matrix for Decision tree | Fraud is the positive class; matrix is evaluated at the validation-selected threshold. |

## What was not inferred

The data does not contain policy number, sum insured, premium, hospital tier,
medical documents, diagnosis descriptions, treatment cost reference ranges, or
Indian state/city fields. These missing concepts cannot be reconstructed from
UUIDs or random codes without inventing data. They are documented as future
feature requirements rather than fabricated into the analysis.
