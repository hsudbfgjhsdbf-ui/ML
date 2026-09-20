# FORM 2: Traditional ML — Model Zoo Implementation Specification
## Medical Insurance Claim Fraud Detection — Approach 1 of 3
### IIIT Dharwad | B.Tech Data Science & AI

---

**Faculty Adviser:** Prof. Ramesh Athe  
**Team Members:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  
**Date:** 01-09-2026 | **Version:** 1.0

---

## 1. Purpose

This document specifies all machine learning algorithms to be trained, their configurations, hyperparameter search spaces, and training protocols for the Medical Insurance Claim Fraud Detection project (Approach 1: Traditional ML). Every algorithm is trained on the Health Insurance Fraud Claims dataset with the preprocessing pipeline defined in Form 1.

## 2. Dataset Summary for Modeling

- **Source:** Health Insurance Fraud Claims.xlsx (4,500 rows, 19 columns)
- **Target:** ClaimLegitimacy → Fraud (1) vs Legitimate (0)
- **Fraud prevalence:** 6.0% (270 of 4,500)
- **Features post-engineering:** ~35 features after encoding and scaling
- **Split:** 70/15/15 stratified (train/validation/test)
- **Key features:** Cluster, ClaimAmount, PatientIncome, ClaimAmountPatientIncome ratio

## 3. Common Training Protocol

### 3.1 Shared Configuration
- Random seed: 42
- Cross-validation: Stratified 5-fold, repeated 2 times
- Primary metric: PR-AUC (Precision-Recall Area Under Curve)
- Secondary metrics: ROC-AUC, F1, F2, Recall@95%Precision, MCC, Brier Score
- All models receive identical training data
- All models evaluated on identical test set (one-time evaluation)

### 3.2 Class Imbalance Handling
Default strategy: class_weight='balanced' applied uniformly to all models that support it.
Additional strategies (SMOTE, ADASYN) compared as ablation experiments.

### 3.3 Early Stopping Policy
- For iterative models (gradient boosting, neural networks): patience = 50 rounds
- Validation loss monitored for early stopping
- Best iteration recorded

## 4. Model Zoo — Complete Algorithm List

### 4.1 Baseline Models

| Key | Model | Purpose | Expected PR-AUC |
|-----|-------|---------|-----------------|
| M01 | Majority Class Classifier | Sanity floor | ~0.06 |
| M02 | Logistic Regression (L2) | Linear baseline | ~0.60-0.75 |
| M03 | Logistic Regression (L1) | Sparse linear | ~0.60-0.75 |

#### M01: Majority Class Classifier
```python
# Always predicts Legitimate (class 0)
# PR-AUC equals fraud prevalence = 0.06
# Serves as absolute minimum performance floor
```

#### M02: Logistic Regression (L2 Regularized)
```python
Parameters:
  - C: [0.01, 0.1, 1.0, 10.0] (inverse regularization)
  - solver: 'lbfgs'
  - max_iter: 1000
  - class_weight: 'balanced'
Search: GridSearchCV with 5-fold CV, scoring='average_precision'
```

#### M03: Logistic Regression (L1 Regularized)
```python
Parameters:
  - C: [0.01, 0.1, 1.0, 10.0]
  - penalty: 'l1'
  - solver: 'saga'
  - max_iter: 2000
  - class_weight: 'balanced'
Purpose: Embedded feature selection via L1 sparsity
```

### 4.2 Probabilistic Models

| Key | Model | Purpose | Expected PR-AUC |
|-----|-------|---------|-----------------|
| M04 | Gaussian Naive Bayes | Probabilistic baseline | ~0.50-0.65 |
| M05 | Bernoulli Naive Bayes | Binarized features | ~0.45-0.60 |

#### M04: Gaussian Naive Bayes
```python
Parameters:
  - var_smoothing: [1e-9, 1e-8, 1e-7]
Notes: Assumes Gaussian feature distributions
```

#### M05: Bernoulli Naive Bayes
```python
Parameters:
  - alpha: [0.1, 0.5, 1.0]
  - binarize: [median of each feature]
Notes: Features binarized at median threshold
```

### 4.3 Distance and Margin Models

| Key | Model | Purpose | Expected PR-AUC |
|-----|-------|---------|-----------------|
| M06 | K-Nearest Neighbors (uniform) | Instance-based | ~0.55-0.70 |
| M07 | K-Nearest Neighbors (distance) | Weighted KNN | ~0.55-0.70 |
| M08 | Linear SVM | Linear margin | ~0.60-0.75 |
| M09 | RBF SVM | Non-linear margin | ~0.60-0.78 |

#### M06-M07: K-Nearest Neighbors
```python
Parameters:
  - n_neighbors: [5, 10, 21, 31, 51]
  - weights: ['uniform', 'distance']
  - metric: ['euclidean', 'manhattan']
Notes: Scaled features required (View A from preprocessing)
```

#### M08: Linear SVM
```python
Parameters:
  - C: [0.1, 1.0, 10.0]
  - class_weight: 'balanced'
  - probability: True (via CalibratedClassifierCV)
```

#### M09: RBF SVM
```python
Parameters:
  - C: [1.0, 10.0]
  - gamma: ['scale', 0.01, 0.001]
  - class_weight: 'balanced'
Notes: Subsample to 3000 rows if memory exceeds limits
```

### 4.4 Tree-Based Models

| Key | Model | Purpose | Expected PR-AUC |
|-----|-------|---------|-----------------|
| M10 | Decision Tree | Interpretability anchor | ~0.55-0.70 |
| M11 | Random Forest | Bagging ensemble | ~0.70-0.85 |
| M12 | Extra Trees | Variance reduction | ~0.70-0.85 |

#### M10: Decision Tree
```python
Parameters:
  - max_depth: [3, 5, 8, 12, 20]
  - min_samples_leaf: [10, 20, 50]
  - criterion: ['gini', 'entropy']
  - class_weight: 'balanced'
Notes: Export tree visualization (max depth 4 for display)
```

#### M11: Random Forest
```python
Parameters:
  - n_estimators: [200, 400, 600]
  - max_depth: [None, 10, 15]
  - min_samples_leaf: [5, 10, 20]
  - max_features: ['sqrt', 'log2']
  - class_weight: 'balanced'
  - oob_score: True
Search: RandomizedSearchCV, 40 iterations
```

#### M12: Extra Trees
```python
Parameters: Same grid as M11
Notes: Extremely randomized splits for comparison
```

### 4.5 Boosting Models (Expected Leaders)

| Key | Model | Purpose | Expected PR-AUC |
|-----|-------|---------|-----------------|
| M13 | AdaBoost | Classic boosting | ~0.65-0.78 |
| M14 | HistGradientBoosting | Fast histogram GBM | ~0.75-0.90 |
| M15 | XGBoost | Optimized GBM | ~0.80-0.95 |
| M16 | LightGBM | Leaf-wise GBM | ~0.80-0.95 |

#### M13: AdaBoost
```python
Parameters:
  - n_estimators: [100, 200, 400]
  - learning_rate: [0.05, 0.1, 0.5]
  - algorithm: ['SAMME', 'SAMME.R']
```

#### M14: Histogram Gradient Boosting
```python
Parameters:
  - learning_rate: [0.05, 0.1, 0.15]
  - max_iter: [200, 500, 1000]
  - max_depth: [3, 5, 8, None]
  - min_samples_leaf: [20, 50]
  - l2_regularization: [0, 0.1, 1.0]
  - early_stopping: True
  - validation_fraction: 0.15
```

#### M15: XGBoost
```python
Parameters:
  - max_depth: [3, 5, 7, 9]
  - learning_rate: [0.01, 0.05, 0.1, 0.2]
  - n_estimators: [200, 500, 1000]
  - subsample: [0.7, 0.8, 1.0]
  - colsample_bytree: [0.6, 0.8, 1.0]
  - reg_alpha: [0, 0.1, 1.0]
  - reg_lambda: [1.0, 5.0, 10.0]
  - scale_pos_weight: computed from class ratio
  - eval_metric: 'aucpr'
  - early_stopping_rounds: 50
Search: RandomizedSearchCV, 60 iterations
```

#### M16: LightGBM
```python
Parameters:
  - num_leaves: [31, 63, 127, 255]
  - learning_rate: [0.01, 0.05, 0.1, 0.2]
  - n_estimators: [200, 500, 1000]
  - min_child_samples: [20, 50, 100]
  - feature_fraction: [0.6, 0.8, 1.0]
  - bagging_fraction: [0.7, 0.8, 1.0]
  - bagging_freq: [1, 5]
  - reg_alpha: [0, 0.1, 1.0]
  - reg_lambda: [1.0, 5.0, 10.0]
  - is_unbalance: True
Search: RandomizedSearchCV, 60 iterations
```

### 4.6 Ensemble Meta-Models

| Key | Model | Purpose | Expected PR-AUC |
|-----|-------|---------|-----------------|
| M17 | Soft Voting Ensemble | Diversity benefit | ~0.82-0.95 |
| M18 | Stacking Ensemble | Meta-learning | ~0.85-0.96 |

#### M17: Soft Voting Ensemble
```python
Constituents: Top 3 diverse models from M02-M16
Weights: Proportional to validation PR-AUC
Voting: 'soft' (probability averaging)
```

#### M18: Stacking Ensemble
```python
Base learners: 
  - M02 (Logistic Regression - linear)
  - M11 (Random Forest - bagging)
  - M15 (XGBoost - boosting)
Meta-learner: Logistic Regression
CV: Out-of-fold predictions for meta-training
```

### 4.7 Neural Network Baseline

| Key | Model | Purpose | Expected PR-AUC |
|-----|-------|---------|-----------------|
| M19 | MLP Classifier | Bridge to Approach 2 | ~0.70-0.85 |

#### M19: Multi-Layer Perceptron
```python
Parameters:
  - hidden_layer_sizes: [(64, 32), (128, 64), (256, 128, 64)]
  - activation: 'relu'
  - alpha: [1e-4, 1e-3, 1e-2]  (L2 regularization)
  - learning_rate: 'adaptive'
  - max_iter: 500
  - early_stopping: True
  - validation_fraction: 0.15
  - batch_size: 'auto'
```

## 5. Hyperparameter Tuning Protocol

### 5.1 Search Strategy
- Small search spaces (≤3 params): GridSearchCV
- Large search spaces (>3 params): RandomizedSearchCV
- Budget: 40-60 iterations per model
- Scoring metric: average_precision (PR-AUC)
- CV: Stratified 5-fold, repeated 2 times

### 5.2 Computational Budget
| Model Family | Max Training Time | Notes |
|-------------|-------------------|-------|
| Linear/Probabilistic | 5 min each | Fast |
| KNN/SVM | 15 min each | Medium |
| Tree-based | 20 min each | Medium |
| Boosting | 40 min each | Heavy |
| Ensemble | 30 min each | Heavy |
| Neural Network | 20 min each | Medium |

Total estimated training time: ~4-6 hours

### 5.3 Tuning Results Recording
Each model's tuning produces:
- Best parameters (JSON)
- All trials table (CSV)
- Best validation score
- Training time in seconds
- Convergence status

## 6. Model Selection Policy

### 6.1 Ranking Criteria
1. **Primary:** Validation PR-AUC (descending)
2. **Secondary:** Validation F2 score (descending)
3. **Tertiary:** Training time (ascending, prefer faster)

### 6.2 Tie-Breaking Rules
- If PR-AUC difference < 0.005: treat as statistical tie
- Among ties: prefer simpler model family
- Simplicity order: Linear → Tree → Bagged → Boosted → Stacked

### 6.3 Selection Steps
1. Filter models with status='complete'
2. Rank by validation PR-AUC
3. Apply tie-breaking rules
4. Verify no leakage flags on candidate
5. Declare winner and runner-up
6. Record selection memo

## 7. Expected Results Summary

Based on the dataset characteristics (strong Cluster signal, clear amount separation):

| Rank | Expected Model | Expected PR-AUC Range |
|------|---------------|----------------------|
| 1 | XGBoost or LightGBM | 0.88-0.95 |
| 2 | HistGradientBoosting | 0.85-0.92 |
| 3 | Random Forest | 0.80-0.88 |
| 4 | Stacking Ensemble | 0.85-0.93 |
| 5 | Extra Trees | 0.78-0.86 |

The best model will be selected empirically — these are expectations, not guarantees.

## 8. Model Card Template

Each model produces a model card containing:
```
Model Key: [e.g., m15_xgboost]
Family: [e.g., Gradient Boosting]
Parameters: [final hyperparameters]
Validation PR-AUC: [value ± std]
Validation ROC-AUC: [value]
Test PR-AUC: [value] (only for winner)
Training Time: [seconds]
Feature Importance Top-5: [list]
Key Finding: [one-sentence summary]
```

## 9. References

1. scikit-learn documentation: https://scikit-learn.org/stable/
2. XGBoost documentation: https://xgboost.readthedocs.io/
3. LightGBM documentation: https://lightgbm.readthedocs.io/
4. Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. KDD.
5. Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.

---

*Last updated: 01-09-2026 | Document version: 1.0*  
*Next: Form 3 — Evaluation & Benchmarking Framework*
