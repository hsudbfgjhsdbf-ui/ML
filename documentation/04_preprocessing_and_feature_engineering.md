# Chapter 4: Data Preprocessing and Feature Engineering

**Institution:** IIIT Dharwad — Department of Data Science & AI  
**Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 4.1 Modular Preprocessing Pipeline
To guarantee reproducibility and prevent data leakage, all statistical transformations were fitted strictly on the training partition:
1. **Missing Value Imputation:**
   - Continuous numerical variables (e.g., Claim Amounts, Incomes): Median imputation.
   - Normal numeric variables: Mean imputation.
   - Categorical attributes: Mode imputation with out-of-vocabulary handling.
2. **Outlier Treatment:** IQR-based soft bounds ($[Q_1 - 3 \cdot \text{IQR}, Q_3 + 3 \cdot \text{IQR}]$) applied carefully to prevent discarding valid fraud anomalies while eliminating corrupt entries.
3. **Categorical Encodings:**
   - Binary variables: Standard Label Encoding.
   - High-cardinality nominals: Target Encoding with smoothing.
   - Ordinal tiers (Tier 1, Tier 2, Tier 3): Monotonic integer mapping.
4. **Feature Scaling:** Standard scaling ($\mu=0, \sigma=1$) for linear models/neural nets and Robust scaling for distance-based estimators.
5. **Class Imbalance Mitigation:** Evaluated SMOTE, BorderlineSMOTE, ADASYN, RandomUnderSampler, and SMOTEENN.

---

## 4.2 Mathematical Formulations of Engineered Features

### 1. Claim-to-Premium Ratio (CPR)
$$\text{CPR} = \frac{\text{Claim\_Amount\_INR}}{\text{Annual\_Premium\_INR} + 1}$$
*Intuition:* Moral hazard is strongly correlated with claims that exceed many multiples of the policyholder's annual premium contribution.

### 2. Treatment Cost Deviation (TCD)
$$\text{TCD} = \frac{\text{Claim\_Amount\_INR} - \mu_{\text{tier, diag}}}{\sigma_{\text{tier, diag}} + \epsilon}$$
*Intuition:* Computes the standard score ($Z$) against the expected clinical tariff for that specific surgical intervention in that specific hospital tier. $Z > 2.0$ signifies strong tariff inflation.

### 3. Cost Per Day (CPD)
$$\text{CPD} = \frac{\text{Claim\_Amount\_INR}}{\text{Hospitalization\_Duration\_Days} + 1}$$
*Intuition:* Flags excessive daily ICU and bed charges that deviate from standard schedule of charges (SOC).

### 4. Waiting Period Delta (WPD)
$$\text{WPD} = \text{Policy\_Duration\_Months} - \text{Waiting\_Period\_Months}$$
*Intuition:* Claims filed within 1–2 months of waiting period expiration ($\text{WPD} \in [0, 2]$) exhibit high fraud risk due to pre-existing condition suppression.

### 5. Claim Velocity Risk (CVR)
$$\text{CVR} = \text{Prior\_Claims\_Count} \times \log(1 + \text{Total\_Prior\_Claimed\_INR})$$
*Intuition:* Measures longitudinal claim frequency and cumulative monetary volume per policyholder.
