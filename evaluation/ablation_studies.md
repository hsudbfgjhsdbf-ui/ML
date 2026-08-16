# Empirical Ablation Studies & Sensitivity Analysis

**Institution:** IIIT Dharwad — Department of Data Science & AI  
**Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 1. Feature Component Ablation
To quantify the individual contribution of each engineered feature group, we conducted systematic ablation on our top tabular model (FT-Transformer).

| Feature Set Configuration | Test Accuracy | Precision | Recall | **F2-Score (Target)** | AUC-ROC | $\Delta$ F2 Impact |
|---|---|---|---|---|---|---|
| **Full Engineered Feature Set (All)** | **0.968** | **0.928** | **0.965** | **0.957** | **0.989** | Baseline |
| *Without Treatment Cost Deviation* | 0.942 | 0.875 | 0.910 | 0.902 | 0.965 | **-0.055** |
| *Without Claim-to-Premium Ratio* | 0.948 | 0.884 | 0.922 | 0.914 | 0.970 | **-0.043** |
| *Without Provider Rejection Aggregations*| 0.953 | 0.895 | 0.934 | 0.926 | 0.975 | **-0.031** |
| *Without Temporal & Waiting Period Delta*| 0.956 | 0.902 | 0.940 | 0.932 | 0.978 | **-0.025** |
| *Raw Features Only (No Engineering)* | 0.892 | 0.760 | 0.852 | 0.832 | 0.925 | **-0.125** |

**Conclusion:** Domain-specific feature engineering accounts for a **+12.5% boost in F2-Score**, with Treatment Cost Deviation and Claim-to-Premium Ratio contributing the largest predictive signals.

---

## 2. Loss Function Ablation on Class Imbalance
We evaluated the impact of loss formulation across identical tabular deep architectures:

| Loss Function Configuration | Precision | Recall | **F2-Score** | PR-AUC | Hard Fraud Case Coverage |
|---|---|---|---|---|---|
| Standard Binary Cross-Entropy (BCE) | 0.882 | 0.885 | 0.884 | 0.905 | 68.4% |
| Weighted BCE (Class Weight = 8.5) | 0.895 | 0.938 | 0.929 | 0.938 | 84.2% |
| **Focal Loss ($\gamma=2.0, \alpha=0.25$)** | **0.928** | **0.965** | **0.957** | **0.962** | **94.8%** |
| Focal Loss ($\gamma=3.5, \alpha=0.25$) | 0.915 | 0.958 | 0.949 | 0.954 | 92.1% |

**Conclusion:** Focal Loss ($\gamma=2.0$) down-weights the vast majority of easy legitimate claims and increases hard fraud case detection from 68.4% to **94.8%**.

---

## 3. Data Augmentation Ablation
| Augmentation Strategy | Test Accuracy | F2-Score | ECE Calibration |
|---|---|---|---|
| No Augmentation | 0.958 | 0.942 | 0.048 |
| Gaussian Noise ($\sigma=0.05$) | 0.961 | 0.947 | 0.035 |
| Mixup Augmentation ($\alpha=0.2$) | 0.965 | 0.952 | 0.024 |
| **Combined SMOTE + Mixup + Feature Masking** | **0.968** | **0.957** | **0.019** |
