# Approach 1 — Evaluation Evidence and Verification

Generated: 2026-08-07 16:29 UTC. This report is derived from the benchmark files produced by `src.train`; source data are synthetic educational records.

## Protocol

- Split: stratified 70% train / 15% validation / 15% held-out test.
- Hyperparameter selection: five-fold stratified CV on training data, mean F2.
- Threshold selection: validation-only maximum F2 scan.
- Final comparison: one untouched test partition.
- Ranking: F2, then ROC-AUC.
- Cost: FN equals synthetic claim amount; FP=₹3,500 configured review/friction proxy.
- Warning: no p-value or synthetic metric establishes real-world insurer efficacy.

## Complete benchmark

| algorithm | accuracy | precision | recall | f1 | f2 | auc_roc | auc_pr | mcc | brier_score | threshold | training_time_seconds | prediction_time_per_sample_ms | model_size_kb | tuned_hyperparameters | true_negative | false_positive | false_negative | true_positive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 0.994 | 0.971 | 0.993 | 0.982 | 0.988 | 1.000 | 0.999 | 0.978 | 0.004 | 0.295 | 4.791 | 0.003 | 317.877 | 4 | 1523 | 8 | 2 | 267 |
| Logistic Regression (L1) | 0.996 | 0.989 | 0.985 | 0.987 | 0.986 | 1.000 | 0.999 | 0.985 | 0.005 | 0.645 | 1.278 | 0.000 | 3.947 | 1 | 1528 | 3 | 4 | 265 |
| LightGBM | 0.996 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 0.998 | 0.983 | 0.004 | 0.085 | 1237.405 | 0.007 | 767.413 | 4 | 1527 | 4 | 4 | 265 |
| SVM (linear) | 0.996 | 0.985 | 0.985 | 0.985 | 0.985 | 1.000 | 0.998 | 0.983 | 0.005 | 0.230 | 5.090 | 0.008 | 148.275 | 1 | 1527 | 4 | 4 | 265 |
| SVM (RBF) | 0.989 | 0.937 | 0.996 | 0.966 | 0.984 | 0.999 | 0.997 | 0.960 | 0.006 | 0.180 | 35.592 | 0.052 | 360.162 | 2 | 1513 | 18 | 1 | 268 |
| Logistic Regression (L2) | 0.996 | 0.989 | 0.981 | 0.985 | 0.983 | 1.000 | 0.998 | 0.982 | 0.006 | 0.660 | 1.615 | 0.000 | 3.947 | 1 | 1528 | 3 | 5 | 264 |
| Histogram Gradient Boosting | 0.988 | 0.933 | 0.989 | 0.960 | 0.977 | 0.999 | 0.998 | 0.954 | 0.004 | 0.150 | 6.256 | 0.007 | 222.005 | 3 | 1512 | 19 | 3 | 266 |
| Random Forest | 0.992 | 0.967 | 0.978 | 0.972 | 0.976 | 0.999 | 0.997 | 0.967 | 0.016 | 0.460 | 35.633 | 0.024 | 3427.116 | 3 | 1522 | 9 | 6 | 263 |
| AdaBoost | 0.992 | 0.967 | 0.978 | 0.972 | 0.976 | 0.999 | 0.996 | 0.967 | 0.186 | 0.490 | 79.556 | 0.028 | 104.008 | 2 | 1522 | 9 | 6 | 263 |
| Shallow Neural Network | 0.991 | 0.981 | 0.959 | 0.970 | 0.963 | 0.999 | 0.997 | 0.965 | 0.008 | 0.200 | 19.400 | 0.001 | 149.794 | 3 | 1526 | 5 | 11 | 258 |
| Decision Tree | 0.976 | 0.884 | 0.963 | 0.922 | 0.946 | 0.980 | 0.937 | 0.908 | 0.021 | 0.575 | 3.392 | 0.000 | 6.642 | 3 | 1497 | 34 | 10 | 259 |
| K-Nearest Neighbors | 0.978 | 0.923 | 0.929 | 0.926 | 0.928 | 0.970 | 0.947 | 0.913 | 0.019 | 0.145 | 10.745 | 0.047 | 4006.521 | 3 | 1510 | 21 | 19 | 250 |
| Quadratic Discriminant Analysis | 0.955 | 0.809 | 0.914 | 0.859 | 0.891 | 0.985 | 0.896 | 0.834 | 0.042 | 0.050 | 0.812 | 0.002 | 61.585 | 1 | 1473 | 58 | 23 | 246 |
| Gaussian Naive Bayes | 0.937 | 0.738 | 0.900 | 0.811 | 0.862 | 0.979 | 0.876 | 0.779 | 0.055 | 0.150 | 0.503 | 0.001 | 5.240 | 1 | 1445 | 86 | 27 | 242 |
| Multinomial Naive Bayes | 0.893 | 0.595 | 0.888 | 0.712 | 0.809 | 0.951 | 0.821 | 0.669 | 0.067 | 0.125 | 0.460 | 0.000 | 8.024 | 1 | 1368 | 163 | 30 | 239 |

## Per-model tuning evidence

### XGBoost
- CV F2: 0.971 ± 0.008; fold values: `[0.9777424483306836, 0.9649122807017544, 0.9800796812749004, 0.9571088165210484, 0.9729944400317713]`.
- Validation-selected threshold: 0.295; validation F2: 0.965.
- Best hyperparameters: `{'subsample': 0.7, 'min_child_weight': 5, 'max_depth': 5, 'colsample_bytree': 0.65}`.
- Training/tuning: 4.79 seconds; held-out latency: 0.0025 ms/claim; serialized estimator: 317.9 KB.
- Held-out confusion matrix: TN=1523, FP=8, FN=2, TP=267.

### Logistic Regression (L1)
- CV F2: 0.972 ± 0.004; fold values: `[0.9650516282764099, 0.9714512291831879, 0.9739747634069401, 0.9764705882352941, 0.9711388455538221]`.
- Validation-selected threshold: 0.645; validation F2: 0.964.
- Best hyperparameters: `{'C': 0.2}`.
- Training/tuning: 1.28 seconds; held-out latency: 0.0002 ms/claim; serialized estimator: 3.9 KB.
- Held-out confusion matrix: TN=1528, FP=3, FN=4, TP=265.

### LightGBM
- CV F2: 0.967 ± 0.015; fold values: `[0.9784345047923323, 0.9848484848484849, 0.9598393574297188, 0.942261427425822, 0.9688995215311005]`.
- Validation-selected threshold: 0.085; validation F2: 0.967.
- Best hyperparameters: `{'subsample': 0.7, 'num_leaves': 31, 'min_child_samples': 35, 'colsample_bytree': 1.0}`.
- Training/tuning: 1237.41 seconds; held-out latency: 0.0067 ms/claim; serialized estimator: 767.4 KB.
- Held-out confusion matrix: TN=1527, FP=4, FN=4, TP=265.

### SVM (linear)
- CV F2: 0.973 ± 0.006; fold values: `[0.9734513274336283, 0.9768211920529801, 0.9689922480620154, 0.9800664451827242, 0.9645232815964523]`.
- Validation-selected threshold: 0.230; validation F2: 0.962.
- Best hyperparameters: `{'C': 0.1}`.
- Training/tuning: 5.09 seconds; held-out latency: 0.0079 ms/claim; serialized estimator: 148.3 KB.
- Held-out confusion matrix: TN=1527, FP=4, FN=4, TP=265.

### SVM (RBF)
- CV F2: 0.964 ± 0.004; fold values: `[0.9602649006622517, 0.9700665188470067, 0.9643255295429208, 0.9598214285714286, 0.9634551495016611]`.
- Validation-selected threshold: 0.180; validation F2: 0.962.
- Best hyperparameters: `{'C': 1.0, 'gamma': 0.02}`.
- Training/tuning: 35.59 seconds; held-out latency: 0.0525 ms/claim; serialized estimator: 360.2 KB.
- Held-out confusion matrix: TN=1513, FP=18, FN=1, TP=268.

### Logistic Regression (L2)
- CV F2: 0.970 ± 0.009; fold values: `[0.9778305621536025, 0.9621451104100947, 0.968503937007874, 0.9818611987381703, 0.9572784810126582]`.
- Validation-selected threshold: 0.660; validation F2: 0.964.
- Best hyperparameters: `{'C': 1.0}`.
- Training/tuning: 1.62 seconds; held-out latency: 0.0002 ms/claim; serialized estimator: 3.9 KB.
- Held-out confusion matrix: TN=1528, FP=3, FN=5, TP=264.

### Histogram Gradient Boosting
- CV F2: 0.969 ± 0.012; fold values: `[0.9728867623604466, 0.945730247406225, 0.9784345047923323, 0.9777424483306836, 0.9683794466403162]`.
- Validation-selected threshold: 0.150; validation F2: 0.962.
- Best hyperparameters: `{'max_leaf_nodes': 15, 'learning_rate': 0.08, 'l2_regularization': 0.2}`.
- Training/tuning: 6.26 seconds; held-out latency: 0.0065 ms/claim; serialized estimator: 222.0 KB.
- Held-out confusion matrix: TN=1512, FP=19, FN=3, TP=266.

### Random Forest
- CV F2: 0.957 ± 0.009; fold values: `[0.9537110933758979, 0.9424920127795527, 0.958631662688942, 0.9691455696202531, 0.9593949044585988]`.
- Validation-selected threshold: 0.460; validation F2: 0.960.
- Best hyperparameters: `{'min_samples_leaf': 8, 'max_features': 'sqrt', 'max_depth': 8}`.
- Training/tuning: 35.63 seconds; held-out latency: 0.0239 ms/claim; serialized estimator: 3427.1 KB.
- Held-out confusion matrix: TN=1522, FP=9, FN=6, TP=263.

### AdaBoost
- CV F2: 0.964 ± 0.008; fold values: `[0.964, 0.9565916398713826, 0.9688995215311005, 0.9544728434504792, 0.9753184713375797]`.
- Validation-selected threshold: 0.490; validation F2: 0.960.
- Best hyperparameters: `{'learning_rate': 0.3, 'n_estimators': 160}`.
- Training/tuning: 79.56 seconds; held-out latency: 0.0279 ms/claim; serialized estimator: 104.0 KB.
- Held-out confusion matrix: TN=1522, FP=9, FN=6, TP=263.

### Shallow Neural Network
- CV F2: 0.995 ± 0.002; fold values: `[0.9927293064876958, 0.9941234084231146, 0.9945408734602463, 0.9939843312814773, 0.9980430528375733]`.
- Validation-selected threshold: 0.200; validation F2: 0.965.
- Best hyperparameters: `{'learning_rate_init': 0.001, 'hidden_layer_sizes': (64, 32), 'alpha': 0.0001}`.
- Training/tuning: 19.40 seconds; held-out latency: 0.0009 ms/claim; serialized estimator: 149.8 KB.
- Held-out confusion matrix: TN=1526, FP=5, FN=11, TP=258.

### Decision Tree
- CV F2: 0.939 ± 0.008; fold values: `[0.94140625, 0.9254901960784314, 0.9355590062111802, 0.9485407066052227, 0.9445745511319282]`.
- Validation-selected threshold: 0.575; validation F2: 0.943.
- Best hyperparameters: `{'max_depth': 7, 'min_impurity_decrease': 0.001, 'min_samples_leaf': 10}`.
- Training/tuning: 3.39 seconds; held-out latency: 0.0002 ms/claim; serialized estimator: 6.6 KB.
- Held-out confusion matrix: TN=1497, FP=34, FN=10, TP=259.

### K-Nearest Neighbors
- CV F2: 0.864 ± 0.024; fold values: `[0.886437908496732, 0.8770491803278688, 0.8702791461412152, 0.8708094848732625, 0.8174273858921162]`.
- Validation-selected threshold: 0.145; validation F2: 0.931.
- Best hyperparameters: `{'n_neighbors': 7, 'p': 2, 'weights': 'distance'}`.
- Training/tuning: 10.74 seconds; held-out latency: 0.0469 ms/claim; serialized estimator: 4006.5 KB.
- Held-out confusion matrix: TN=1510, FP=21, FN=19, TP=250.

### Quadratic Discriminant Analysis
- CV F2: 0.878 ± 0.013; fold values: `[0.8958168902920284, 0.8705882352941177, 0.8619313647246608, 0.8721389108129439, 0.8915946582875098]`.
- Validation-selected threshold: 0.050; validation F2: 0.892.
- Best hyperparameters: `{'reg_param': 0.01}`.
- Training/tuning: 0.81 seconds; held-out latency: 0.0017 ms/claim; serialized estimator: 61.6 KB.
- Held-out confusion matrix: TN=1473, FP=58, FN=23, TP=246.

### Gaussian Naive Bayes
- CV F2: 0.877 ± 0.012; fold values: `[0.8553654743390358, 0.8821233411397346, 0.8734472049689441, 0.8916218293620292, 0.8846153846153846]`.
- Validation-selected threshold: 0.150; validation F2: 0.880.
- Best hyperparameters: `{'var_smoothing': 1e-07}`.
- Training/tuning: 0.50 seconds; held-out latency: 0.0007 ms/claim; serialized estimator: 5.2 KB.
- Held-out confusion matrix: TN=1445, FP=86, FN=27, TP=242.

### Multinomial Naive Bayes
- CV F2: 0.424 ± 0.027; fold values: `[0.4647006255585344, 0.38636363636363635, 0.4293381037567084, 0.40504050405040504, 0.43555555555555553]`.
- Validation-selected threshold: 0.125; validation F2: 0.806.
- Best hyperparameters: `{'alpha': 0.05}`.
- Training/tuning: 0.46 seconds; held-out latency: 0.0003 ms/claim; serialized estimator: 8.0 KB.
- Held-out confusion matrix: TN=1368, FP=163, FN=30, TP=239.

## Imbalance strategy comparison

| strategy | training_rows_after_sampling | threshold | validation_f2 | validation_recall | validation_precision |
|---|---|---|---|---|---|
| random_under | 2514 | 0.515 | 0.970 | 0.974 | 0.953 |
| tomek | 8379 | 0.320 | 0.970 | 0.970 | 0.967 |
| class_weight | 8400 | 0.660 | 0.964 | 0.963 | 0.966 |
| smote | 14286 | 0.385 | 0.962 | 0.970 | 0.932 |
| smoteenn | 13838 | 0.500 | 0.962 | 0.970 | 0.929 |

## Pairwise significance

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

## Best-model fairness audit

| dimension | group | n | accuracy | fpr | fnr | precision | recall | selection_rate | small_group_warning |
|---|---|---|---|---|---|---|---|---|---|
| gender | Female | 828 | 0.995 | 0.004 | 0.009 | 0.974 | 0.991 | 0.139 | False |
| gender | Male | 931 | 0.994 | 0.006 | 0.007 | 0.968 | 0.993 | 0.165 | False |
| gender | Non-binary/Prefer not to say | 41 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.146 | False |
| age_group | Child (0–17) | 155 | 0.994 | 0.008 | 0.000 | 0.968 | 1.000 | 0.200 | False |
| age_group | Young adult (18–34) | 479 | 0.996 | 0.000 | 0.029 | 1.000 | 0.971 | 0.140 | False |
| age_group | Adult (35–49) | 616 | 0.992 | 0.010 | 0.000 | 0.947 | 1.000 | 0.154 | False |
| age_group | Older adult (50–64) | 436 | 0.995 | 0.005 | 0.000 | 0.970 | 1.000 | 0.154 | False |
| age_group | Senior (65+) | 114 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.132 | False |
| state | Bihar | 48 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.167 | False |
| state | Delhi | 121 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.132 | False |
| state | Gujarat | 160 | 0.994 | 0.007 | 0.000 | 0.962 | 1.000 | 0.163 | False |
| state | Karnataka | 285 | 0.993 | 0.008 | 0.000 | 0.953 | 1.000 | 0.151 | False |
| state | Kerala | 128 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.156 | False |
| state | Maharashtra | 235 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.132 | False |
| state | Odisha | 56 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.161 | False |
| state | Rajasthan | 90 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.144 | False |
| state | Tamil Nadu | 210 | 0.986 | 0.006 | 0.059 | 0.970 | 0.941 | 0.157 | False |
| state | Telangana | 168 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.149 | False |
| state | Uttar Pradesh | 193 | 0.995 | 0.006 | 0.000 | 0.969 | 1.000 | 0.166 | False |
| state | West Bengal | 106 | 0.972 | 0.033 | 0.000 | 0.842 | 1.000 | 0.179 | False |
| income_bracket | High | 89 | 0.989 | 0.013 | 0.000 | 0.917 | 1.000 | 0.135 | False |
| income_bracket | Low | 338 | 0.997 | 0.000 | 0.019 | 1.000 | 0.981 | 0.154 | False |
| income_bracket | Lower-middle | 474 | 0.992 | 0.008 | 0.013 | 0.961 | 0.987 | 0.162 | False |
| income_bracket | Middle | 582 | 0.998 | 0.002 | 0.000 | 0.989 | 1.000 | 0.151 | False |
| income_bracket | Upper-middle | 317 | 0.991 | 0.011 | 0.000 | 0.935 | 1.000 | 0.145 | False |
| treatment_type | Appendectomy | 167 | 0.988 | 0.007 | 0.040 | 0.960 | 0.960 | 0.150 | False |
| treatment_type | Ayurvedic outpatient | 173 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.127 | False |
| treatment_type | Ayurvedic panchakarma | 151 | 0.993 | 0.008 | 0.000 | 0.962 | 1.000 | 0.172 | False |
| treatment_type | Cardiac hospitalization | 161 | 0.994 | 0.007 | 0.000 | 0.962 | 1.000 | 0.161 | False |
| treatment_type | Cataract day-care | 171 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.129 | False |
| treatment_type | Dengue hospitalization | 214 | 0.995 | 0.006 | 0.000 | 0.974 | 1.000 | 0.178 | False |
| treatment_type | Dialysis day-care | 147 | 0.993 | 0.008 | 0.000 | 0.966 | 1.000 | 0.197 | False |
| treatment_type | Maternity delivery | 225 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.133 | False |
| treatment_type | Orthopaedic surgery | 196 | 0.980 | 0.018 | 0.034 | 0.903 | 0.966 | 0.158 | False |
| treatment_type | Respiratory outpatient | 195 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.133 | False |

## Verification assertions

1. All metrics in this document came from arrays generated by saved models on the test split.
2. The threshold is selected before test scoring.
3. Target encoders, imputers, scalers and feature selection are fitted to train rows only.
4. `benchmark_results.csv`, `fairness_by_group.csv`, `imbalance_strategy_comparison.csv`, `significance_tests.csv` and model metadata are the machine-readable source of truth.
5. Presentation and PDF values are derived from the same benchmark at generation time.

## Interpretation boundary

Results are valid only for this deterministic synthetic simulation. They cannot be used to approve, reject or price a real medical insurance claim. A real deployment requires external validation, insurer/legal policy mapping, data-protection controls, model-risk governance, calibrated monitoring and accountable human review.
