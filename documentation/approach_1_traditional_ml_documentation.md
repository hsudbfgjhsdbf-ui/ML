# Approach 1 — Traditional Machine Learning for Indian Medical Insurance Claim Fraud Screening

**Institution:** IIIT Dharwad, Department of Data Science and AI  
**Faculty Adviser:** Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  
**Generated from the verified run:** 2026-08-07 16:29 UTC  
**Data status:** Transparent synthetic Indian-context educational data; not an insurer dataset and not a real-claim decision system.

> **Safety statement.** The output is a fraud-screening recommendation for human investigation. A high score must never autonomously deny an insurance claim. Coverage, medical necessity, claimant rights and a human reviewer remain decisive.

## Abstract

This study implements a reproducible traditional-machine-learning baseline for medical insurance claim fraud screening in an Indian context. The supplied workbook was retained unchanged but did not meet the study adequacy criteria: it contains 4,500 records, generic rather than Indian locations, and no required policy/waiting-period/history features. We therefore generated a documented synthetic population, preserved realistic Indian product categories and INR price variation, and used stratified 70/15/15 partitions. A train-only pipeline applies duplicate auditing, median/mode imputation, robust scaling, one-hot encoding, smoothed target encoding for high-cardinality fields, limited domain interactions and mutual-information feature selection. The benchmark evaluates 15 classifiers using F2 as the selection metric because missed fraud is costly. The current held-out leader is **XGBoost** (F2=0.988, recall=0.993, PR-AUC=0.999). Results are pedagogical and describe synthetic behavior only; they do not establish expected performance at an Indian insurer. The report includes INR-oriented cost assumptions, threshold selection, statistical comparison, explanation assets and group-level fairness audit.

**Keywords—** health-insurance fraud, India, imbalanced classification, explainable machine learning, fairness audit, F2 score, synthetic data.

## 1. Problem, Scope and Research Questions

Fraudulent bills, duplicated claims, inflated procedures and policy-timing manipulation can impose losses on insurers and ultimately raise premiums for genuine policyholders. India adds material context: family-floater covers, employer-group policies, Ayushman Bharat and ECHS schemes, allopathic and Ayurvedic treatments, cashless and reimbursement workflows, GST components and strong tier/region cost variation. A ₹2 lakh procedure at a metro corporate hospital cannot be evaluated with the same baseline as a small-town nursing-home episode.

This approach answers four bounded questions: (1) which classical model has the strongest recall-prioritised performance on the defined synthetic test set; (2) what threshold best implements the chosen F2 objective; (3) which observed synthetic signals contribute to model screening; and (4) whether error rates differ materially across audited groups. The study does not infer actual national fraud prevalence, clinical validity or insurer-specific policy eligibility.

### Objectives

1. Build an inspectable end-to-end binary fraud-screening pipeline.
2. Preserve and audit the bundled workbook while transparently using a documented fallback population when it is inadequate.
3. Evaluate multiple model families under the same split and F2-led protocol.
4. Report threshold, probability quality, INR assumptions, computational cost, explanation and fairness evidence.
5. Produce reusable assets for the presentation and IEEE-style report.

## 2. Data Provenance, Adequacy and Ethics

### 2.1 Bundled-workbook audit

The supplied `Health Insurance Fraud Claims.xlsx` has SHA-256 `952b6e23c9845b9086994c08bc81323575968c318f845ea69e3b9609614b6a45` and 4,500 data rows. It includes a binary `ClaimLegitimacy` field, but it falls below the 10,000-record criterion and lacks Indian policy/product fields, temporal policy controls and reliable claim-history features. The source was copied unchanged to `data/raw/`; it was not relabelled, currency-converted or represented as an Indian insurer extract.

### 2.2 Synthetic fallback design

The generated population contains claimants across Indian states/cities, INR amounts with a log-normal/right-skewed tail, insurer/policy types, government schemes, hospital tiers, coverage and waiting-period fields, allopathic/Ayurvedic treatment types, historical features and a noisy fraud-generating mechanism. It contains deliberately duplicated records solely to test duplicate removal. Fraud probability is probabilistic and has unobserved noise, so no feature is a deterministic label key. The complete generator and fixed seed are in `src/data_loading.py` and `configs/traditional_ml.yaml`.

### 2.3 Privacy and intended use

All values are synthetic. No Aadhaar number, PAN, real hospital, real patient, medical image or API secret is included. Demographic variables are retained to audit harms, rather than to define an automatic adverse action. Any real deployment would require lawful data governance, insurer-specific validation, privacy impact assessment, security review, clinical/policy expert review, calibration monitoring and meaningful human appeal.

## 3. Literature Review and Identified Gap

The following work motivates the model families and safeguards. It is not treated as evidence that a synthetic benchmark transfers to live claims.

| Ref. | Contribution | Dataset/method | Finding relevant here | Limitation addressed in this project |
|---|---|---|---|---|
| [1] | Broad fraud-mining survey | Multiple fraud domains | Feature engineering and imbalanced evaluation matter | Does not specialise to Indian medical policies |
| [2] | Financial-fraud data-mining review | Classification and clustering | Supervised learning is valuable when labels exist | Domain context and fairness are often underdeveloped |
| [3] | Statistical fraud review | Behavioural anomaly methods | Fraud evolves and distributions matter | This baseline is explicitly bounded to a fixed synthetic period |
| [4] | Fraud-system survey | Rule/ML approaches | Hybrid evidence is useful | Motivates later evidence-grounded Agent-AI track |
| [5] | Medicare fraud detection | Provider-label learning | Provider patterns can be informative | Current-provider proxies are disclosed synthetic variables |
| [6] | Health-care fraud review | Health-data mining | Interpretability and validation are essential | Adds threshold, cost and fairness audit |
| [7] | Class imbalance study | Imbalanced supervised learning | Accuracy can mislead | F2/PR-AUC/MCC are primary reports |
| [8] | SMOTE | Minority synthesis | Sampling can improve minority learning | Sampling comparison is train-only and documented |
| [9] | Imbalanced-learning review | Survey | Cost-sensitive choices require context | INR assumptions are explicit and illustrative |
| [10] | Random forests | Bagged trees | Nonlinear tabular patterns and importances | Importances are not causal explanations |
| [11] | Gradient boosting | Functional gradient descent | Boosting is strong on structured data | Tuned under common split/metric |
| [12] | XGBoost | Regularised tree boosting | Efficient high-performance trees | Optional dependency is recorded, not assumed |
| [13] | LightGBM | Leaf-wise boosting | Speed/accuracy trade-off | Latency/model size are benchmarked |
| [14] | Support-vector networks | Margin classifiers | Kernels model nonlinear boundaries | SVM training cap is disclosed for laptop reproducibility |
| [15] | AdaBoost | Adaptive ensembles | Weak learners can combine effectively | Performance verified on held-out data |
| [17] | SHAP | Additive attributions | Local/global model explanation framework | Attribution is explained as association, not evidence of fraud |
| [18] | PR vs ROC | Evaluation theory | PR better reveals positive-class performance | Both PR-AUC and ROC-AUC are retained |
| [20] | Equality of opportunity | Fair classification | FPR/FNR differences matter | Group outcomes are explicitly audited |
| [21] | Discrimination-aware preprocessing | Fair ML | Mitigation has trade-offs | Guardrails and mitigation route are documented |
| [22] | McNemar test | Paired classifiers | Same-test-set predictions should be compared paired | Exact test is reported descriptively |

**Gap.** Existing general fraud studies do not by themselves yield a transparent, reproducible, Indian-context claim-screening baseline with policy timing, regional costs, sampling comparison, threshold selection, INR-risk assumptions and demographic error audit. This educational synthetic implementation provides that engineering baseline; it does not fill the need for a real governed claims dataset.

## 4. Methodology

![Traditional ML pipeline](visualizations/technical/traditional_ml_pipeline_diagram.png)

### 4.1 Reproducible split and leakage controls

Exact duplicate claims were removed before the stratified split. The resulting partitions are train=8,400, validation=1,800, test=1,800; fraud rates are 14.96%, 14.94% and 14.94%. The preprocessor, target encoder, scaler and mutual-information selector are fitted only on training rows. Validation chooses threshold; test is used only for final model comparison.

### 4.2 Cleaning and outlier policy

Missingness was measured before imputation. Exact duplicates removed: 60. Numeric values use median imputation; low-cardinality categoricals use the training mode; target encoding maps unknown high-cardinality categories to the training global rate. Z-score and IQR candidates are reported but preserved when plausible because a very large claim can be precisely the signal under investigation. Clearly impossible values would be rejected by schema validation; none are silently corrected.

### 4.3 Encoding and scaling

The fitted input has 162 transformed features: numeric fields use robust scaling, low-cardinality nominal fields use one-hot encoding and configured/high-cardinality fields use smoothed training-only target encoding. Robust scaling reduces leverage from INR tails for logistic, SVM, KNN, QDA and neural-network bridge models. Multinomial Naive Bayes additionally receives a fitted train-only min–max transformation because its likelihood requires nonnegative values.

### 4.4 Domain features

The feature engineering code creates only claim-time functions: claim-to-premium ratio, regional treatment-cost deviation, days relative to waiting-period end, claim-frequency intensity, amount per hospital day, current-to-history ratio, provider-distance interaction, age×amount, policy utilisation and two limited degree-two transforms. The regional baseline normalises tier/city cost rather than treating metro pricing as inherently suspicious.

| Engineered feature | Definition | Why it may be useful | Caution |
|---|---|---|---|
| `claim_to_premium_ratio` | Current claim divided by annual premium. | High values can indicate unusual policy utilisation. | Association is not proof; a reviewer must verify evidence. |
| `treatment_cost_deviation_z` | Claim deviation from regional treatment baseline using assumed 36% relative spread. | Large positive deviations may merit review after regional/tier context. | Association is not proof; a reviewer must verify evidence. |
| `days_to_waiting_period_end` | Policy age minus waiting-period duration on claim date. | Near/negative values identify coverage-timing risk. | Association is not proof; a reviewer must verify evidence. |
| `claim_frequency_intensity` | Claims filed in previous 12 months divided by 12. | Frequency spikes can be a fraud indicator but are not proof. | Association is not proof; a reviewer must verify evidence. |
| `amount_per_hospital_day_inr` | Claim amount divided by inpatient days; day-care uses one day. | Flags implausible per-day cost after tier/context controls. | Association is not proof; a reviewer must verify evidence. |
| `current_vs_historical_average` | Current amount divided by policyholder historical mean. | Large departures from personal baseline may warrant review. | Association is not proof; a reviewer must verify evidence. |
| `distance_x_provider_risk` | Travel distance multiplied by historical provider rejection-rate proxy. | Captures distance/provider interaction without a deterministic rule. | Association is not proof; a reviewer must verify evidence. |
| `age_x_claim_amount_lakh` | Age multiplied by amount in lakhs. | Limited interaction used to model age-treatment cost patterns. | Association is not proof; a reviewer must verify evidence. |
| `policy_utilisation_ratio` | Current claim divided by sum insured. | Near-limit amounts may require additional validation. | Association is not proof; a reviewer must verify evidence. |
| `claim_amount_log` | Natural log of one plus claim amount. | Stabilises right-skewed INR claim distributions. | Association is not proof; a reviewer must verify evidence. |
| `claim_amount_squared_lakh` | Squared claim amount in lakh INR units. | Captures limited nonlinear large-claim risk. | Association is not proof; a reviewer must verify evidence. |
| `hospital_stay_squared` | Square of hospitalization duration. | Captures nonlinear duration/cost relationships. | Association is not proof; a reviewer must verify evidence. |

### 4.5 Feature selection

Correlation inspection identifies redundant numerical relationships. Mutual-information filtering selected 60 training-fitted columns, listed in `data/processed/data_quality_report.json`. RFE/LASSO/tree importance remain complementary analytical techniques; this shared filter controls dimensionality fairly across the complete benchmark rather than optimizing separately on the test set.

### 4.6 Imbalance strategies

Five strategies were compared using a train-only logistic reference and validation-only threshold selection: class weighting, random undersampling, Tomek links, SMOTE and SMOTEENN. This does not declare that a single sampler is globally optimal; it documents the recall/precision trade-off prior to the all-model class-weight baseline.

| strategy | train rows | threshold | val F2 | val recall | val precision |
|---|---|---|---|---|---|
| random_under | 2514 | 0.515 | 0.970 | 0.974 | 0.953 |
| tomek | 8379 | 0.320 | 0.970 | 0.970 | 0.967 |
| class_weight | 8400 | 0.660 | 0.964 | 0.963 | 0.966 |
| smote | 14286 | 0.385 | 0.962 | 0.970 | 0.932 |
| smoteenn | 13838 | 0.500 | 0.962 | 0.970 | 0.929 |

### 4.7 Algorithms and tuning protocol

All models use five-fold stratified cross-validation on the training partition and select hyperparameters by mean F2. Small grids are exhaustive; broader tree/neural spaces use bounded random search. The selected threshold is not fixed at 0.50: validation probabilities are evaluated from 0.05 to 0.95 in 0.005 steps and the highest F2 is retained. F2 weights recall four times as strongly as precision in its denominator, reflecting the missed-fraud screening priority while retaining precision, cost and FPR reporting.

#### XGBoost
Search space: `{'subsample': 0.7, 'min_child_weight': 5, 'max_depth': 5, 'colsample_bytree': 0.65}` selected from `see saved model metadata`. Best five-fold CV F2=0.971±0.008; validation threshold=0.295; tuning time=4.79s. The model is serialized with metadata in `models/`.

#### Logistic Regression (L1)
Search space: `{'C': 0.2}` selected from `see saved model metadata`. Best five-fold CV F2=0.972±0.004; validation threshold=0.645; tuning time=1.28s. The model is serialized with metadata in `models/`.

#### LightGBM
Search space: `{'subsample': 0.7, 'num_leaves': 31, 'min_child_samples': 35, 'colsample_bytree': 1.0}` selected from `see saved model metadata`. Best five-fold CV F2=0.967±0.015; validation threshold=0.085; tuning time=1237.41s. The model is serialized with metadata in `models/`.

#### SVM (linear)
Search space: `{'C': 0.1}` selected from `see saved model metadata`. Best five-fold CV F2=0.973±0.006; validation threshold=0.230; tuning time=5.09s. The model is serialized with metadata in `models/`.

#### SVM (RBF)
Search space: `{'C': 1.0, 'gamma': 0.02}` selected from `see saved model metadata`. Best five-fold CV F2=0.964±0.004; validation threshold=0.180; tuning time=35.59s. The model is serialized with metadata in `models/`.

#### Logistic Regression (L2)
Search space: `{'C': 1.0}` selected from `see saved model metadata`. Best five-fold CV F2=0.970±0.009; validation threshold=0.660; tuning time=1.62s. The model is serialized with metadata in `models/`.

#### Histogram Gradient Boosting
Search space: `{'max_leaf_nodes': 15, 'learning_rate': 0.08, 'l2_regularization': 0.2}` selected from `see saved model metadata`. Best five-fold CV F2=0.969±0.012; validation threshold=0.150; tuning time=6.26s. The model is serialized with metadata in `models/`.

#### Random Forest
Search space: `{'min_samples_leaf': 8, 'max_features': 'sqrt', 'max_depth': 8}` selected from `see saved model metadata`. Best five-fold CV F2=0.957±0.009; validation threshold=0.460; tuning time=35.63s. The model is serialized with metadata in `models/`.

#### AdaBoost
Search space: `{'learning_rate': 0.3, 'n_estimators': 160}` selected from `see saved model metadata`. Best five-fold CV F2=0.964±0.008; validation threshold=0.490; tuning time=79.56s. The model is serialized with metadata in `models/`.

#### Shallow Neural Network
Search space: `{'learning_rate_init': 0.001, 'hidden_layer_sizes': (64, 32), 'alpha': 0.0001}` selected from `see saved model metadata`. Best five-fold CV F2=0.995±0.002; validation threshold=0.200; tuning time=19.40s. The model is serialized with metadata in `models/`.

#### Decision Tree
Search space: `{'max_depth': 7, 'min_impurity_decrease': 0.001, 'min_samples_leaf': 10}` selected from `see saved model metadata`. Best five-fold CV F2=0.939±0.008; validation threshold=0.575; tuning time=3.39s. The model is serialized with metadata in `models/`.

#### K-Nearest Neighbors
Search space: `{'n_neighbors': 7, 'p': 2, 'weights': 'distance'}` selected from `see saved model metadata`. Best five-fold CV F2=0.864±0.024; validation threshold=0.145; tuning time=10.74s. The model is serialized with metadata in `models/`.

#### Quadratic Discriminant Analysis
Search space: `{'reg_param': 0.01}` selected from `see saved model metadata`. Best five-fold CV F2=0.878±0.013; validation threshold=0.050; tuning time=0.81s. The model is serialized with metadata in `models/`.

#### Gaussian Naive Bayes
Search space: `{'var_smoothing': 1e-07}` selected from `see saved model metadata`. Best five-fold CV F2=0.877±0.012; validation threshold=0.150; tuning time=0.50s. The model is serialized with metadata in `models/`.

#### Multinomial Naive Bayes
Search space: `{'alpha': 0.05}` selected from `see saved model metadata`. Best five-fold CV F2=0.424±0.027; validation threshold=0.125; tuning time=0.46s. The model is serialized with metadata in `models/`.

## 5. Experimental Results

### 5.1 Held-out benchmark

The following values are on the untouched test partition, at the model-specific threshold chosen from validation F2. The primary rank is F2 and the secondary rank is ROC-AUC. Accuracy is intentionally not the ranking metric because the synthetic class is imbalanced.

| algorithm | accuracy | precision | recall | f1 | f2 | auc_roc | auc_pr | mcc | threshold | train s | latency ms/sample | model_size_kb | # tuned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 0.994 | 0.971 | 0.993 | 0.982 | 0.988 | 1.000 | 0.999 | 0.978 | 0.295 | 4.791 | 0.003 | 317.877 | 4 |
| Logistic Regression (L1) | 0.996 | 0.989 | 0.985 | 0.987 | 0.986 | 1.000 | 0.999 | 0.985 | 0.645 | 1.278 | 0.000 | 3.947 | 1 |
| LightGBM | 0.996 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 0.998 | 0.983 | 0.085 | 1237.405 | 0.007 | 767.413 | 4 |
| SVM (linear) | 0.996 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 0.998 | 0.983 | 0.230 | 5.090 | 0.008 | 148.275 | 1 |
| SVM (RBF) | 0.989 | 0.937 | 0.996 | 0.966 | 0.984 | 0.999 | 0.997 | 0.960 | 0.180 | 35.592 | 0.052 | 360.162 | 2 |
| Logistic Regression (L2) | 0.996 | 0.989 | 0.981 | 0.985 | 0.983 | 1.000 | 0.998 | 0.982 | 0.660 | 1.615 | 0.000 | 3.947 | 1 |
| Histogram Gradient Boosting | 0.988 | 0.933 | 0.989 | 0.960 | 0.977 | 0.999 | 0.998 | 0.954 | 0.150 | 6.256 | 0.007 | 222.005 | 3 |
| Random Forest | 0.992 | 0.967 | 0.978 | 0.972 | 0.976 | 0.999 | 0.997 | 0.967 | 0.460 | 35.633 | 0.024 | 3427.116 | 3 |
| AdaBoost | 0.992 | 0.967 | 0.978 | 0.972 | 0.976 | 0.999 | 0.996 | 0.967 | 0.490 | 79.556 | 0.028 | 104.008 | 2 |
| Shallow Neural Network | 0.991 | 0.981 | 0.959 | 0.970 | 0.963 | 0.999 | 0.997 | 0.965 | 0.200 | 19.400 | 0.001 | 149.794 | 3 |
| Decision Tree | 0.976 | 0.884 | 0.963 | 0.922 | 0.946 | 0.980 | 0.937 | 0.908 | 0.575 | 3.392 | 0.000 | 6.642 | 3 |
| K-Nearest Neighbors | 0.978 | 0.923 | 0.929 | 0.926 | 0.928 | 0.970 | 0.947 | 0.913 | 0.145 | 10.745 | 0.047 | 4006.521 | 3 |
| Quadratic Discriminant Analysis | 0.955 | 0.809 | 0.914 | 0.859 | 0.891 | 0.985 | 0.896 | 0.834 | 0.050 | 0.812 | 0.002 | 61.585 | 1 |
| Gaussian Naive Bayes | 0.937 | 0.738 | 0.900 | 0.811 | 0.862 | 0.979 | 0.876 | 0.779 | 0.150 | 0.503 | 0.001 | 5.240 | 1 |
| Multinomial Naive Bayes | 0.893 | 0.595 | 0.888 | 0.712 | 0.809 | 0.951 | 0.821 | 0.669 | 0.125 | 0.460 | 0.000 | 8.024 | 1 |

**Best screened model:** `XGBoost`. Its test F2=0.988, recall=0.993, precision=0.971, PR-AUC=0.999, ROC-AUC=1.000 and MCC=0.978. The threshold (0.295) was selected without inspecting test labels.

![Grouped metrics](visualizations/model_comparison/grouped_metric_comparison.png)

![ROC curves](visualizations/model_comparison/roc_curves_all_models.png)

![Precision recall curves](visualizations/model_comparison/precision_recall_curves_all_models.png)

![Efficiency](visualizations/model_comparison/training_time_vs_accuracy.png)

### 5.2 Cost-sensitive confusion analysis

A false negative means a synthetic fraudulent claim was recommended for approval; its illustrative loss is that claim’s INR amount. A false positive is a legitimate claim routed to review, charged at the configured ₹3,500 review/friction proxy. These are scenario assumptions, not actual insurer costs, recoveries or regulatory exposure.

- **XGBoost:** FN=2, FP=8, illustrative total cost=₹267,863.19.
- **Logistic Regression (L1):** FN=4, FP=3, illustrative total cost=₹375,117.28.
- **LightGBM:** FN=4, FP=4, illustrative total cost=₹296,434.19.
- **SVM (linear):** FN=4, FP=4, illustrative total cost=₹394,929.81.
- **SVM (RBF):** FN=1, FP=18, illustrative total cost=₹226,778.35.
- **Logistic Regression (L2):** FN=5, FP=3, illustrative total cost=₹853,818.41.
- **Histogram Gradient Boosting:** FN=3, FP=19, illustrative total cost=₹313,513.81.
- **Random Forest:** FN=6, FP=9, illustrative total cost=₹537,800.53.
- **AdaBoost:** FN=6, FP=9, illustrative total cost=₹461,338.28.
- **Shallow Neural Network:** FN=11, FP=5, illustrative total cost=₹2,126,759.01.
- **Decision Tree:** FN=10, FP=34, illustrative total cost=₹1,437,486.03.
- **K-Nearest Neighbors:** FN=19, FP=21, illustrative total cost=₹1,479,620.06.
- **Quadratic Discriminant Analysis:** FN=23, FP=58, illustrative total cost=₹1,648,901.31.
- **Gaussian Naive Bayes:** FN=27, FP=86, illustrative total cost=₹2,083,600.71.
- **Multinomial Naive Bayes:** FN=30, FP=163, illustrative total cost=₹2,293,905.52.

![Confusion matrices](visualizations/model_comparison/confusion_matrices_all_models.png)

### 5.3 Paired significance evidence

McNemar’s exact test compares paired correctness on the same test records; Wilcoxon compares five cross-validation F2 fold values. p<0.05 is a descriptive threshold only: synthetic records are generated rather than independent insurer observations, and multiple comparisons increase false-positive risk.

| reference_model | compared_model | mcnemar_discordant_leader_only | mcnemar_discordant_candidate_only | mcnemar_exact_p | wilcoxon_cv_f2_p | significant_at_0_05 |
|---|---|---|---|---|---|---|
| XGBoost | Logistic Regression (L1) | 4 | 7 | 0.549 | 1.000 | False |
| XGBoost | LightGBM | 3 | 5 | 0.727 | 0.625 | False |
| XGBoost | SVM (linear) | 4 | 6 | 0.754 | 0.812 | False |
| XGBoost | SVM (RBF) | 14 | 5 | 0.064 | 0.312 | False |
| XGBoost | Logistic Regression (L2) | 5 | 7 | 0.774 | 0.812 | False |
| XGBoost | Histogram Gradient Boosting | 13 | 1 | 0.002 | 0.625 | True |
| XGBoost | Random Forest | 10 | 5 | 0.302 | 0.125 | False |
| XGBoost | AdaBoost | 9 | 4 | 0.267 | 0.125 | False |
| XGBoost | Shallow Neural Network | 12 | 6 | 0.238 | 0.062 | False |
| XGBoost | Decision Tree | 39 | 5 | 0.000 | 0.062 | True |
| XGBoost | K-Nearest Neighbors | 37 | 7 | 0.000 | 0.062 | True |
| XGBoost | Quadratic Discriminant Analysis | 75 | 4 | 0.000 | 0.062 | True |
| XGBoost | Gaussian Naive Bayes | 107 | 4 | 0.000 | 0.062 | True |
| XGBoost | Multinomial Naive Bayes | 188 | 5 | 0.000 | 0.062 | True |

## 6. Explainability and Error Analysis

Tree split gain/importances, random-forest importances and logistic coefficient magnitudes are rendered in `visualizations/interpretability/`. These methods identify predictive association in the synthetic study, not causal evidence that a person or provider committed fraud. For a claimant-facing explanation, the system should name verifiable evidence (e.g., policy waiting-period status, itemised bill mismatch) and offer a human review path rather than expose opaque raw scores.

The synthetic error analysis should examine false negatives with very high regional treatment costs, claims near waiting-period end and incomplete documents; these may represent legitimate exceptional care. False positives may also concentrate in expensive metro/corporate treatment episodes. The correct operational response is manual evidence review, not automatic refusal.

## 7. Fairness and Ethical Analysis

Group metrics are calculated for the F2-leading model across gender, age bracket, state, income bracket and treatment type. A difference above the configured 0.10 guardrail in FPR/FNR requires investigation, sample-size checks and a mitigation comparison (for example, reweighting or removal of a proxy feature). Small groups are explicitly marked and should not support strong conclusions.

| dimension | n_groups | max_fpr_gap | max_fnr_gap | accuracy_min | accuracy_max |
|---|---|---|---|---|---|
| age_group | 5 | 0.010 | 0.029 | 0.992 | 1.000 |
| gender | 3 | 0.006 | 0.009 | 0.994 | 1.000 |
| income_bracket | 5 | 0.013 | 0.019 | 0.989 | 0.998 |
| state | 12 | 0.033 | 0.059 | 0.972 | 1.000 |
| treatment_type | 10 | 0.018 | 0.040 | 0.980 | 1.000 |

![Fairness FNR audit](visualizations/fairness/fairness_fnr_groups.png)

![Calibration](visualizations/fairness/calibration_reliability_diagram.png)

The audit supports monitoring rather than demographic rationing. Gender, disability, age and geography must not be used as shortcut reasons to reject a claim. In a production setting, protected attributes should be access-controlled and separated from scoring unless legally/ethically justified for a carefully governed fairness intervention.

## 8. Code Walkthrough and Reproduction

| File | Responsibility | Key outputs / checks |
|---|---|---|
| `src/data_loading.py` | audits the raw workbook and generates the labelled fallback population | checksum, adequacy report, synthetic data metadata |
| `src/preprocessing.py` | duplicate/outlier audit, train-only encoding/scaling/selection and splits | serialized preprocessor/selector, processed partitions |
| `src/feature_engineering.py` | deterministic domain/temporal/interaction features | documented claim-time variables |
| `src/models.py` | classifier registry, 5-fold F2 tuning and validation threshold selection | saved models and hyperparameters |
| `src/evaluate.py` | held-out metrics, INR proxy costs, fairness, calibration, significance | CSV/JSON evaluation evidence |
| `src/visualize.py` | EDA, comparison, explanation and audit plots | reusable PNG assets |
| `src/reporting.py` | derives Markdown, deck and report from live outputs | internally consistent academic artefacts |
| `src/train.py` | one-command orchestrator | end-to-end verification manifest |

Run from a fresh virtual environment: `python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python -m src.train --regenerate-data`. The fixed seed, package versions, configurations and source checksum are written alongside the artifacts. A clean rerun overwrites generated output with the new verified run; never mix files across configurations.

## 9. Limitations and Future Work

1. Labels, price distributions and fraud mechanisms are synthetic, so metric values are not estimates of operational accuracy or financial benefit.
2. Random claim-level splitting can retain related policyholders/providers across partitions; future real-data work should add temporal and entity-disjoint validation.
3. Target encoding can encode historical correlation, not causation; it must be monitored for provider/demographic proxy harm.
4. Synthetic historical/provider rates are simplified and must not be conflated with verified investigations.
5. Classifier explanations do not authenticate bills or establish legal fraud; Approach 3 is designed to add document evidence, RAG citations and human checkpoints.
6. Deployment needs calibration drift monitoring, independent audit, policy-specific coverage rules, privacy controls, appeal procedures and robust manual-review workflows.

## 10. Completion Checklist

- [x] Raw source is preserved and its inadequacy is documented.
- [x] Reproducible synthetic Indian-context fallback is generated and labelled.
- [x] Stratified 70/15/15 split and train-only transforms are saved.
- [x] Classical model benchmark, threshold optimisation, INR proxy cost and computational metrics are reported.
- [x] EDA, ROC/PR, comparison, confusion, feature-importance, fairness and calibration assets are generated.
- [x] Documentation, presentation and IEEE-style report are generated from the run.
- [ ] Independent real-insurer validation and governance approval (outside this academic synthetic study).

## References

[1] C. Phua, V. Lee, K. Smith and R. Gayler, “A comprehensive survey of data mining-based fraud detection research,” Artificial Intelligence Review, vol. 34, pp. 1–14, 2010.
[2] E. W. T. Ngai, Y. Hu, Y. H. Wong, Y. Chen and X. Sun, “The application of data mining techniques in financial fraud detection,” Decision Support Systems, vol. 50, no. 3, pp. 559–569, 2011.
[3] R. J. Bolton and D. J. Hand, “Statistical fraud detection: A review,” Statistical Science, vol. 17, no. 3, pp. 235–255, 2002.
[4] A. Abdallah, M. A. Maarof and A. Zainal, “Fraud detection system: A survey,” Journal of Network and Computer Applications, vol. 68, pp. 90–113, 2016.
[5] R. A. Bauder and T. M. Khoshgoftaar, “The detection of Medicare fraud using machine learning methods with excluded provider labels,” Health Care Management Science, vol. 20, pp. 1–15, 2017.
[6] H. Joudaki et al., “Using data mining to detect health care fraud and abuse: A review of literature,” Global Journal of Health Science, vol. 7, no. 1, pp. 194–202, 2015.
[7] N. Japkowicz and S. Stephen, “The class imbalance problem: A systematic study,” Intelligent Data Analysis, vol. 6, no. 5, pp. 429–449, 2002.
[8] N. V. Chawla, K. W. Bowyer, L. O. Hall and W. P. Kegelmeyer, “SMOTE: Synthetic minority over-sampling technique,” Journal of Artificial Intelligence Research, vol. 16, pp. 321–357, 2002.
[9] H. He and E. A. Garcia, “Learning from imbalanced data,” IEEE Transactions on Knowledge and Data Engineering, vol. 21, no. 9, pp. 1263–1284, 2009.
[10] L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.
[11] J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” Annals of Statistics, vol. 29, no. 5, pp. 1189–1232, 2001.
[12] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proc. KDD, 2016, pp. 785–794.
[13] G. Ke et al., “LightGBM: A highly efficient gradient boosting decision tree,” in Proc. NeurIPS, 2017, pp. 3146–3154.
[14] C. Cortes and V. Vapnik, “Support-vector networks,” Machine Learning, vol. 20, pp. 273–297, 1995.
[15] Y. Freund and R. E. Schapire, “A decision-theoretic generalization of on-line learning and an application to boosting,” Journal of Computer and System Sciences, vol. 55, no. 1, pp. 119–139, 1997.
[16] L. V. Utkin, “A method for processing imprecise expert judgments about parameters of probability distributions,” European Journal of Operational Research, vol. 158, no. 3, pp. 657–674, 2004.
[17] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in Proc. NeurIPS, 2017, pp. 4765–4774.
[18] J. Davis and M. Goadrich, “The relationship between Precision-Recall and ROC curves,” in Proc. ICML, 2006, pp. 233–240.
[19] M. P. Wand and M. C. Jones, Kernel Smoothing. London, U.K.: Chapman & Hall, 1995.
[20] M. Hardt, E. Price and N. Srebro, “Equality of opportunity in supervised learning,” in Proc. NeurIPS, 2016, pp. 3315–3323.
[21] F. Kamiran and T. Calders, “Data preprocessing techniques for classification without discrimination,” Knowledge and Information Systems, vol. 33, pp. 1–33, 2012.
[22] Q. McNemar, “Note on the sampling error of the difference between correlated proportions,” Psychometrika, vol. 12, pp. 153–157, 1947.
[23] Insurance Regulatory and Development Authority of India, “Protection of Policyholders’ Interests Regulations,” IRDAI, New Delhi, India, 2024.
