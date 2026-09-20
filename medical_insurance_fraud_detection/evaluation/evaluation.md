# Evaluation Report — Medical Insurance Claim Fraud Detection
## Approach 1: Traditional Machine Learning

**Date:** 01-09-2026  
**Dataset:** Health Insurance Fraud Claims (4,500 records)  
**Target:** ClaimLegitimacy (Fraud: 270, Legitimate: 4,230)  
**Fraud Rate:** 6.0%

---

## 1. Run Summary

| Property | Value |
|----------|-------|
| Total Models Trained | 16 |
| Failed Models | 1 |
| Best Model | M12_XGBoost |
| Best Val PR-AUC | 1.0000 |
| Best Val ROC-AUC | 1.0000 |
| Best Val F1 | 0.9877 |
| Random Seed | 42 |
| Split | 70/15/15 stratified |

## 2. Complete Leaderboard

| Rank | Model | Status | Val PR-AUC | Val ROC-AUC | Val F1 | Val F2 | Val Recall | Val Precision | Train Time (s) |
|------|-------|--------|-----------|-------------|--------|--------|-----------|--------------|----------------|
| 1 | M12_XGBoost | complete | 1.0000 | 1.0000 | 0.9877 | 0.9804 | 0.9756 | 1.0000 | 0.1 |
| 2 | M17_VotingEnsemble | complete | 1.0000 | 1.0000 | 0.9877 | 0.9804 | 0.9756 | 1.0000 | 1.9 |
| 3 | M10_AdaBoost | complete | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0 |
| 4 | M06_RandomForest | complete | 0.9994 | 1.0000 | 0.9880 | 0.9951 | 1.0000 | 0.9762 | 0.6 |
| 5 | M11_HistGBM | complete | 0.9988 | 0.9999 | 0.9756 | 0.9756 | 0.9756 | 0.9756 | 0.3 |
| 6 | M13_LightGBM | complete | 0.9988 | 0.9999 | 0.9880 | 0.9951 | 1.0000 | 0.9762 | 0.1 |
| 7 | M07_ExtraTrees | complete | 0.9942 | 0.9995 | 0.8889 | 0.9390 | 0.9756 | 0.8163 | 0.5 |
| 8 | M14_GBM | complete | 0.9935 | 0.9996 | 0.9756 | 0.9756 | 0.9756 | 0.9756 | 2.5 |
| 9 | M09_LinearSVM | complete | 0.9828 | 0.9986 | 0.9136 | 0.9069 | 0.9024 | 0.9250 | 0.0 |
| 10 | M02_LogReg_L2 | complete | 0.9806 | 0.9986 | 0.8791 | 0.9346 | 0.9756 | 0.8000 | 0.0 |
| 11 | M03_LogReg_L1 | complete | 0.9707 | 0.9979 | 0.8791 | 0.9346 | 0.9756 | 0.8000 | 2.2 |
| 12 | M15_MLP | complete | 0.9633 | 0.9975 | 0.9048 | 0.9179 | 0.9268 | 0.8837 | 0.6 |
| 13 | M05_DecisionTree | complete | 0.9533 | 0.9869 | 0.9756 | 0.9756 | 0.9756 | 0.9756 | 0.0 |
| 14 | M08_KNN | complete | 0.9160 | 0.9937 | 0.8148 | 0.8088 | 0.8049 | 0.8250 | 0.0 |
| 15 | M04_GaussianNB | complete | 0.8083 | 0.9914 | 0.8889 | 0.9390 | 0.9756 | 0.8163 | 0.0 |
| 16 | M01_MajorityClass | complete | 0.0607 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 |

## 3. Winner Selection

**Winner:** M12_XGBoost

### Validation Metrics
- PR-AUC: 1.0000
- ROC-AUC: 1.0000
- F1 Score: 0.9877
- F2 Score: 0.9804
- Precision: 1.0000
- Recall: 0.9756
- MCC: 0.9870
- Brier Score: 0.0009

### Test Metrics (One-Time Evaluation)
- PR-AUC: 0.9919
- ROC-AUC: 0.9993
- F1 Score: 0.9750

## 4. Key Findings

1. **Cluster is the strongest signal:** Cluster 1 contains 267 of 270 fraud cases (98.9%)
2. **ClaimAmount separation:** Fraudulent claims average 8,734 INR vs 4,777 INR for legitimate
3. **ClaimAmountPatientIncome ratio:** Novel engineered feature captures financial anomaly
4. **Boosting models dominate:** XGBoost and LightGBM achieve highest PR-AUC
5. **Class imbalance handled:** 6% fraud rate managed via class_weight='balanced'

## 5. Data Quality Summary

| Gate | Status |
|------|--------|
| Completeness (0 nulls) | PASS |
| Schema (19 columns) | PASS |
| Uniqueness (claim_id) | PASS |
| Target values | PASS |
| Range checks | PASS |
| Prevalence (6%) | PASS |

## 6. Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Total Records | 4,500 |
| Features (raw) | 19 |
| Features (model-ready) | 44 |
| Fraud Cases | 270 (6.0%) |
| Legitimate Cases | 4230 (94.0%) |
| Class Imbalance Ratio | 1:15.7 |

## 7. Feature Importance (Top 10)

Based on the best model's feature importance:

1. Cluster — strongest single predictor (Cluster 1 = high fraud risk)
2. ClaimAmount — fraudulent claims significantly higher
3. ClaimAmountPatientIncome — financial anomaly ratio
4. ClusterIs1 — binary flag for high-fraud cluster
5. HighClaimAmt — binary flag for high-value claims
6. Cluster1_HighAmt — interaction of two strongest signals
7. LogClaimAmount — log-transformed amount
8. PatientIncome — income context
9. ClaimAmountZScore — standardized amount
10. ProviderSpecialty — specialty-level variation

## 8. Business Impact Analysis

### Cost Assumptions (Indian Context)
- Average fraudulent claim: ~8,734 INR
- Manual review cost per claim: ~500 INR
- False negative cost: 8,734 INR (fraud paid)
- False positive cost: 500 INR (review overhead)

### Estimated Impact (per 4,500 claims)
- True Positives detected: 263
- Fraud prevented: TP × 8,734 INR
- Review costs: (TP + FP) × 500 INR
- Net savings: Significant positive

## 9. Fairness Summary

The model was evaluated across demographic slices:
- **Gender:** No significant disparity in fraud detection rates
- **Age Groups:** Consistent performance across age bands
- **Employment Status:** Balanced detection across categories
- **Claim Types:** Uniform performance across claim types

## 10. Limitations

1. **Synthetic dataset:** The data is generated, not from real claims
2. **Cluster leakage concern:** Cluster feature is extremely predictive — in practice, cluster assignment may not be available at claim time
3. **No temporal validation:** We use random split, not time-based split
4. **No document analysis:** Approach 1 only handles tabular features
5. **Limited to 4,500 records:** Larger datasets may reveal different patterns

## 11. Comparison with Other Approaches

| Approach | Method | Expected Strength |
|----------|--------|-------------------|
| Approach 1 (This) | Traditional ML | Interpretable, fast, strong baseline |
| Approach 2 | Deep Learning + XAI | Automatic feature learning |
| Approach 3 | Agent AI Multi-Agent | Document understanding, NLP explanations |

---

*Generated by run_pipeline.py on 01-09-2026*  
*Faculty Adviser: Prof. Ramesh Athe*  
*IIIT Dharwad — B.Tech Data Science & AI*
