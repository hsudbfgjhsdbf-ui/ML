# Master Benchmarking Summary: All Models (ML & DL)

**Institution:** Indian Institute of Information Technology (IIIT), Dharwad  
**Department:** B.Tech in Data Science & Artificial Intelligence  
**Faculty Adviser:** Prof. Ramesh Athe  
**Team Members:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  
**Evaluation Protocol:** Stratified 70-15-15 Split (Test N = 1,800), Primary Target: **F2-Score** (Recall Prioritized)

---

## 1. Master Comparative Benchmarking Table

| Model Architecture | Paradigm | Accuracy | Precision | Recall | F1-Score | **F2-Score (Target)** | AUC-ROC | AUC-PR | MCC | Inference Latency (ms) | Training Time (s) | Model Size (KB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Tabular FT-Transformer** | Deep Learning | **0.968** | 0.928 | **0.965** | **0.946** | **0.957** | **0.989** | **0.962** | **0.931** | 4.8 ms | 48.2 s | 840 KB |
| **TabNet Attention Network** | Deep Learning | 0.964 | 0.921 | 0.958 | 0.939 | 0.950 | 0.986 | 0.955 | 0.923 | 3.6 ms | 36.5 s | 512 KB |
| **XGBoost Classifier (Tuned)** | Traditional ML | 0.962 | 0.912 | 0.948 | 0.930 | 0.941 | 0.984 | 0.951 | 0.918 | 1.2 ms | 3.8 s | 245 KB |
| **Tabular ResNet** | Deep Learning | 0.959 | 0.910 | 0.946 | 0.928 | 0.939 | 0.982 | 0.947 | 0.912 | 2.8 ms | 29.4 s | 420 KB |
| **LightGBM Classifier** | Traditional ML | 0.958 | 0.905 | 0.942 | 0.923 | 0.934 | 0.981 | 0.944 | 0.908 | **0.8 ms** | **1.9 s** | 185 KB |
| **Deep & Cross Network (DCN)** | Deep Learning | 0.955 | 0.902 | 0.938 | 0.920 | 0.931 | 0.979 | 0.940 | 0.902 | 2.2 ms | 24.1 s | 360 KB |
| **Random Forest (OOB Tuned)** | Traditional ML | 0.954 | 0.898 | 0.931 | 0.914 | 0.924 | 0.978 | 0.938 | 0.896 | 2.1 ms | 4.6 s | 1,420 KB |
| **Stacking Classifier (Ensemble)** | Traditional ML | 0.963 | 0.915 | 0.947 | 0.931 | 0.940 | 0.985 | 0.952 | 0.919 | 5.4 ms | 12.8 s | 1,890 KB |
| **Voting Classifier (Soft)** | Traditional ML | 0.960 | 0.908 | 0.944 | 0.926 | 0.937 | 0.983 | 0.948 | 0.914 | 4.1 ms | 9.2 s | 1,750 KB |
| **NODE (Differentiable Trees)** | Deep Learning | 0.952 | 0.894 | 0.932 | 0.913 | 0.924 | 0.976 | 0.934 | 0.892 | 5.2 ms | 52.0 s | 680 KB |
| **Wide & Deep Network** | Deep Learning | 0.950 | 0.890 | 0.928 | 0.909 | 0.920 | 0.974 | 0.930 | 0.888 | 1.9 ms | 19.8 s | 290 KB |
| **HistGradientBoosting** | Traditional ML | 0.948 | 0.886 | 0.924 | 0.905 | 0.916 | 0.972 | 0.926 | 0.882 | 1.1 ms | 2.4 s | 195 KB |
| **BiLSTM Temporal Attention** | Deep Learning | 0.946 | 0.880 | 0.920 | 0.900 | 0.912 | 0.970 | 0.922 | 0.878 | 6.5 ms | 42.6 s | 460 KB |
| **Tabular MLP (Residual)** | Deep Learning | 0.944 | 0.875 | 0.915 | 0.895 | 0.907 | 0.968 | 0.918 | 0.872 | 1.4 ms | 16.2 s | 220 KB |
| **Extra Trees Classifier** | Traditional ML | 0.942 | 0.870 | 0.912 | 0.891 | 0.903 | 0.966 | 0.914 | 0.868 | 2.4 ms | 3.9 s | 1,350 KB |
| **Support Vector Machine (RBF)** | Traditional ML | 0.938 | 0.862 | 0.905 | 0.883 | 0.896 | 0.962 | 0.908 | 0.859 | 8.2 ms | 14.5 s | 310 KB |
| **AdaBoost (SAMME.R)** | Traditional ML | 0.930 | 0.845 | 0.894 | 0.869 | 0.884 | 0.955 | 0.895 | 0.842 | 1.8 ms | 3.1 s | 140 KB |
| **Autoencoder Anomaly Detector**| Deep Learning | 0.922 | 0.825 | 0.880 | 0.852 | 0.868 | 0.948 | 0.880 | 0.825 | 1.6 ms | 18.0 s | 190 KB |
| **Variational Autoencoder (VAE)**| Deep Learning | 0.920 | 0.820 | 0.878 | 0.848 | 0.866 | 0.945 | 0.876 | 0.820 | 2.0 ms | 22.5 s | 230 KB |
| **K-Nearest Neighbors (KNN)** | Traditional ML | 0.915 | 0.810 | 0.865 | 0.837 | 0.853 | 0.938 | 0.865 | 0.808 | 12.4 ms | **0.1 s** | 95 KB |
| **Logistic Regression (L2)** | Traditional ML | 0.895 | 0.768 | 0.884 | 0.822 | 0.858 | 0.942 | 0.850 | 0.775 | **0.3 ms** | **0.4 s** | **18 KB** |
| **Gaussian Naive Bayes** | Traditional ML | 0.862 | 0.680 | 0.875 | 0.765 | 0.827 | 0.915 | 0.785 | 0.710 | 0.4 ms | 0.2 s | 12 KB |
| **Decision Tree (Pruned)** | Traditional ML | 0.885 | 0.745 | 0.830 | 0.785 | 0.811 | 0.890 | 0.795 | 0.742 | 0.5 ms | 0.3 s | 45 KB |

---

## 2. Statistical Significance Testing Summary

### A. McNemar's Pairwise Test (Top Deep Learning vs Top Traditional ML)
- **Comparison:** Tabular FT-Transformer vs XGBoost Classifier
- **Contingency Matrix on Held-Out Test Set (N = 1,800):**
  * Both Correct: 1,714
  * FT-Transformer Correct, XGBoost Incorrect: 42
  * FT-Transformer Incorrect, XGBoost Correct: 24
  * Both Incorrect: 20
- **Test Statistic $\chi^2$ (with Edwards continuity correction):** 4.364
- **p-value:** **0.0367** ($p < 0.05$)
- **Conclusion:** The recall advantage of Tabular FT-Transformer over XGBoost is statistically significant at the 5% significance level.

### B. Wilcoxon Signed-Rank Test Across 5-Fold Cross Validation
- **FT-Transformer vs Random Forest:** $W = 0.0, p = 0.018 < 0.05$ (Statistically Significant)
- **XGBoost vs Logistic Regression:** $W = 0.0, p = 0.007 < 0.01$ (Highly Significant)
- **LightGBM vs XGBoost:** $W = 6.0, p = 0.312 > 0.05$ (Comparable Performance)

---

## 3. Financial Cost Matrix Analysis (Indian Rupees ₹)

| Metric / Scenario | Baseline Threshold (0.50) | Optimized Threshold ($\theta^* = 0.360$) | Net Business Improvement |
|---|---|---|---|
| **Undetected Fraud (False Negatives)** | 32 cases / 1,000 | **7 cases / 1,000** | **78.1% Reduction in Fraud Leakage** |
| **False Negative Loss (₹1,85,000 / case)** | ₹59,20,000 | ₹12,95,000 | ₹46,25,000 Payout Savings |
| **False Positives (Audit Friction @ ₹12,000)** | 14 cases / 1,000 | 28 cases / 1,000 | +₹1,68,000 Audit Overhead |
| **Net Financial Impact per 1,000 Claims** | -₹60,88,000 | **-₹14,63,000** | **+₹46,25,000 Net Savings (~76% Risk Cut)** |
