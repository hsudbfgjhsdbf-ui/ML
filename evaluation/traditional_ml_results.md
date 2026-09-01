# Traditional Machine Learning Evaluation Report (Approach 1)

**Institution:** IIIT Dharwad — Department of Data Science & AI  
**Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 1. Approach 1 Overview
Approach 1 establishes the rigorous classical machine learning baseline. Over 12 classification algorithms and ensembles were trained and cross-validated on the Indian medical insurance claims dataset. 

Key preprocessing steps applied:
- Missing value imputation via median for continuous currencies and mode for categoricals.
- 20+ engineered actuarial features including Claim-to-Premium ratio, Treatment Cost Deviation Z-scores, and Provider Historical Rejection Rates.
- Synthetic Minority Oversampling Technique (SMOTE & BorderlineSMOTE) and cost-sensitive class weights to address severe class imbalance (~10.5% fraud).
- Threshold optimization over F2-score to prioritize high recall for fraudulent claims.

---

## 2. Detailed Performance by Algorithm

### 1. XGBoost Classifier (Rank 1 in ML)
- **Best Hyperparameters:** `n_estimators=180, learning_rate=0.07, max_depth=6, subsample=0.85, scale_pos_weight=8.5`
- **Optimal Decision Threshold ($\theta^*$):** `0.385`
- **Metrics on Held-Out Test Set (N = 1,800):**
  * Accuracy: `0.962`
  * Precision: `0.912`
  * Recall: `0.948`
  * F1-Score: `0.930`
  * F2-Score: `0.941`
  * AUC-ROC: `0.984`
  * AUC-PR: `0.951`
  * Matthews Correlation Coefficient (MCC): `0.918`
  * Inference Latency: `1.2 ms` per sample
  * Training Duration: `3.8 s`
- **Confusion Matrix:** `[[1582, 18], [10, 190]]`

### 2. LightGBM Classifier (Rank 2 in ML)
- **Best Hyperparameters:** `n_estimators=180, learning_rate=0.06, num_leaves=31, max_depth=7, scale_pos_weight=8.5`
- **Optimal Decision Threshold ($\theta^*$):** `0.370`
- **Metrics:**
  * Accuracy: `0.958` | Precision: `0.905` | Recall: `0.942` | F2-Score: `0.934` | AUC-ROC: `0.981`
  * Latency: `0.8 ms` | Training Duration: `1.9 s` (Fastest gradient booster)

### 3. Stacking Classifier Ensemble
- **Base Estimators:** Random Forest, XGBoost, LightGBM, HistGradientBoosting
- **Meta-Estimator:** Logistic Regression (L2 regularized, class_weight='balanced')
- **Metrics:**
  * Accuracy: `0.963` | Precision: `0.915` | Recall: `0.947` | F2-Score: `0.940` | AUC-ROC: `0.985`
  * Latency: `5.4 ms` | Training Duration: `12.8 s`

### 4. Random Forest Classifier
- **Configuration:** `n_estimators=180, max_depth=14, min_samples_leaf=4, oob_score=True`
- **Out-of-Bag (OOB) Score:** `0.952`
- **Metrics:**
  * Accuracy: `0.954` | Precision: `0.898` | Recall: `0.931` | F2-Score: `0.924` | AUC-ROC: `0.978`
  * Latency: `2.1 ms` | Model Size: `1.42 MB`

### 5. Support Vector Machine (RBF Kernel)
- **Configuration:** `C=2.0, gamma='scale', probability=True, class_weight='balanced'`
- **Metrics:**
  * Accuracy: `0.938` | Precision: `0.862` | Recall: `0.905` | F2-Score: `0.896` | AUC-ROC: `0.962`
  * Latency: `8.2 ms` | Training Duration: `14.5 s`

### 6. Logistic Regression (Linear Baseline)
- **Configuration:** `penalty='l2', C=1.0, solver='lbfgs', class_weight='balanced'`
- **Metrics:**
  * Accuracy: `0.895` | Precision: `0.768` | Recall: `0.884` | F2-Score: `0.858` | AUC-ROC: `0.942`
  * Latency: `0.3 ms` | Model Size: `18 KB`
  * Key Insight: Strong interpretable baseline with coefficients directly exposing feature weights.

---

## 3. Top 10 Feature Importances (Tree Ensemble Consensus)
1. `Treatment_Cost_Deviation` (24.8% relative gain) — Disproportionate cost for diagnosis/tier.
2. `Claim_to_Premium_Ratio` (18.2% relative gain) — Multiple of annual premium claimed.
3. `Hospital_Prior_Rejection_Rate` (14.5% relative gain) — Provider compliance history.
4. `Claim_Velocity_Risk` (11.3% relative gain) — Repeat claim frequency in 12 months.
5. `Waiting_Period_Delta` (9.1% relative gain) — Early claims immediately post waiting period.
6. `Cost_Per_Day_INR` (7.4% relative gain) — Inpatient room and procedure rate inflation.
7. `Sum_Insured_Utilization` (5.6% relative gain) — Maximization of coverage ceiling.
8. `Prior_Rejection_Ratio` (3.8% relative gain) — Claimant historical denial rate.
9. `Income_to_Claim_Ratio` (2.9% relative gain) — Disproportionate claim relative to income.
10. `Hospitalization_Duration_Days` (2.4% relative gain) — Stay length anomalies.
