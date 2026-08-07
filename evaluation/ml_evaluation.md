# Approach 1 - Traditional ML Benchmarking Report

## Overview

- Dataset rows: 7272
- Imbalance strategy: smote
- CV scoring metric: F2, folds: 5
- Best model ranked by F2 score (secondary: ROC-AUC).

## Benchmarking Table

| Algorithm | Acc | Prec | Recall | F1 | **F2** | AUC-ROC | AUC-PR | MCC | Train(s) | Pred(ms) | Size(KB) | #HP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 | 1.000 | 1.97 | 0.050 | 0.0 | 4 |
| XGBoost | 0.999 | 0.976 | 1.000 | 0.988 | **0.995** | 1.000 | 1.000 | 0.987 | 0.37 | 0.006 | 0.0 | 7 |
| Decision Tree | 0.999 | 1.000 | 0.975 | 0.987 | **0.980** | 0.988 | 0.976 | 0.987 | 0.03 | 0.003 | 0.0 | 2 |
| Gradient Boosting | 0.999 | 1.000 | 0.975 | 0.987 | **0.980** | 0.998 | 0.986 | 0.987 | 0.32 | 0.008 | 0.0 | 5 |
| LightGBM | 0.999 | 1.000 | 0.975 | 0.987 | **0.980** | 1.000 | 1.000 | 0.987 | 0.21 | 0.004 | 0.0 | 6 |
| AdaBoost | 0.999 | 1.000 | 0.975 | 0.987 | **0.980** | 0.999 | 0.989 | 0.987 | 0.62 | 0.015 | 0.0 | 2 |
| Logistic Regression | 0.987 | 0.816 | 1.000 | 0.899 | **0.957** | 0.999 | 0.984 | 0.897 | 3.51 | 0.002 | 0.0 | 3 |
| SVM | 0.987 | 0.830 | 0.975 | 0.897 | **0.942** | 0.998 | 0.968 | 0.893 | 2.52 | 0.008 | 0.0 | 3 |
| Neural Network | 0.981 | 0.776 | 0.950 | 0.854 | **0.909** | 0.996 | 0.951 | 0.849 | 0.88 | 0.003 | 0.0 | 3 |
| Naive Bayes | 0.964 | 0.625 | 1.000 | 0.769 | **0.893** | 0.993 | 0.852 | 0.775 | 0.01 | 0.003 | 0.0 | 1 |
| KNN | 0.973 | 0.712 | 0.925 | 0.804 | **0.873** | 0.990 | 0.765 | 0.798 | 0.00 | 0.225 | 0.0 | 3 |
| QDA | 0.941 | 0.500 | 1.000 | 0.667 | **0.833** | 0.982 | 0.708 | 0.684 | 0.01 | 0.002 | 0.0 | 1 |

## Ranking (by F2)

1. **Random Forest** — F2=1.0000, ROC-AUC=1.0000
2. **XGBoost** — F2=0.9950, ROC-AUC=1.0000
3. **Decision Tree** — F2=0.9799, ROC-AUC=0.9875
4. **Gradient Boosting** — F2=0.9799, ROC-AUC=0.9984
5. **LightGBM** — F2=0.9799, ROC-AUC=1.0000
6. **AdaBoost** — F2=0.9799, ROC-AUC=0.9993
7. **Logistic Regression** — F2=0.9569, ROC-AUC=0.9989
8. **SVM** — F2=0.9420, ROC-AUC=0.9978
9. **Neural Network** — F2=0.9091, ROC-AUC=0.9960
10. **Naive Bayes** — F2=0.8929, ROC-AUC=0.9931
11. **KNN** — F2=0.8726, ROC-AUC=0.9895
12. **QDA** — F2=0.8333, ROC-AUC=0.9817

## Business Impact (INR)

| Model | FN (fraud approved) | FP (valid rejected) | Estimated fraud loss (INR) |
|---|---|---|---|
| Random Forest | 0 | 0 | 0 |
| XGBoost | 0 | 1 | 0 |
| Decision Tree | 1 | 0 | 5,014 |
| Gradient Boosting | 1 | 0 | 5,014 |
| LightGBM | 1 | 0 | 5,014 |
| AdaBoost | 1 | 0 | 5,014 |
| Logistic Regression | 0 | 9 | 0 |
| SVM | 1 | 8 | 5,014 |
| Neural Network | 2 | 11 | 10,028 |
| Naive Bayes | 0 | 24 | 0 |
| KNN | 3 | 15 | 15,043 |
| QDA | 0 | 40 | 0 |

## Statistical Significance (McNemar vs best)

| Model | chi2 | p-value | p<0.05? |
|---|---|---|---|
| XGBoost | 0.000 | 1.0000 | False |
| Decision Tree | 0.000 | 1.0000 | False |
| Gradient Boosting | 0.000 | 1.0000 | False |
| LightGBM | 0.000 | 1.0000 | False |
| AdaBoost | 0.000 | 1.0000 | False |
| Logistic Regression | 7.111 | 0.0077 | True |
| SVM | 7.111 | 0.0077 | True |
| Neural Network | 11.077 | 0.0009 | True |
| Naive Bayes | 22.042 | 0.0000 | True |
| KNN | 16.056 | 0.0001 | True |
| QDA | 38.025 | 0.0000 | True |

## Optimal Thresholds

| Model | Optimal threshold |
|---|---|
| Random Forest | 0.386 |
| XGBoost | 0.822 |
| Decision Tree | 0.010 |
| Gradient Boosting | 0.010 |
| LightGBM | 0.010 |
| AdaBoost | 0.436 |
| Logistic Regression | 0.243 |
| SVM | 0.490 |
| Neural Network | 0.020 |
| Naive Bayes | 0.936 |
| KNN | 0.337 |
| QDA | 0.653 |