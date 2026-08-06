# Traditional ML Approach

## Overview
Binary fraud classification with probability and risk category.

## Data
Claim-level 4500 rows, 6% fraud, prediction unit claim.

## Pipeline
- Load via `load_claims_dataset`
- Schema validation, missing, outlier, imbalance, leakage detection
- Target mapping
- Date engineering
- Build preprocessor (numerical median+standard, categorical most_frequent+onehot)
- Split train/val/test 2925/675/900 stratified
- Candidate models: DummyClassifier, LogisticRegression, LinearSVM, CalibratedLinearSVM, KNN, GaussianNB, DecisionTree, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, AdaBoost, SVM_RBF, XGBoost optional, LightGBM optional, CatBoost optional, VotingTop3
- For each: GridSearchCV CV 5 scoring average_precision (PR-AUC), fit, optional calibration isotonic, evaluate val/test
- Threshold tuning: optimize F2 on val via `select_threshold`
- Feature importance via coef_ or feature_importances_
- SHAP attempted (shap library optional)
- Save best model joblib

## Model Selection
Primary metric PR-AUC (fraud detection). Not accuracy alone. Best model selected on val PR-AUC.

## Results (actual from evaluation/)
See `evaluation/model_comparison.csv` and `metrics_summary.json`.
On this synthetic dataset, tree models achieve near-perfect PR-AUC (1.0) suggesting high separability via PatientIncome and ClaimAmount. RandomForest and HistGradientBoosting also near-perfect. Logistic 0.94, DecisionTree 1.0 val. Dummy 0.06 baseline. This may indicate synthetic generation.

## Explainability
Top features: PatientIncome, ClaimAmount dominant (see feature_importance.csv). SHAP not installed in env, status logged.

## Skipped Models
XGBoost, LightGBM, CatBoost not available in env -> recorded as skipped with reason missing dependency. RBF SVM may be slow; if skipped due to runtime, recorded.

## Limitations
- High cardinality OHE leads to 8000+ features, may overfit
- ClaimStatus potential leakage
- Synthetic separability not realistic
- No calibration fallback fixed (sklearn 1.9 cv='prefit' deprecated)

## Reproducibility
Seed 42, pipeline learned only from train, SMOTE only inside folds (if used).

## Files
- `approaches/01_traditional_ml.py`
- `evaluation/model_comparison.csv`, `metrics_summary.json`, `threshold_analysis.csv`, `confusion_matrices/`, `calibration_results.csv`, `feature_importance.csv`
