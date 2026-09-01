# Ethical AI, Fairness & Demographic Parity Report

**Institution:** IIIT Dharwad — Department of Data Science & AI  
**Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 1. Demographic Parity and Equalized Odds Evaluation

To ensure that automated fraud detection does not discriminate against protected demographics across India, we evaluated our models across Gender, Age Brackets, Geographic Regions, and Income Levels.

### A. Fairness Metrics by Gender
| Protected Attribute (Gender) | Sample Count | Actual Fraud Rate | Model Positive Rate | Recall (True Pos Rate) | False Positive Rate |
|---|---|---|---|---|---|
| **Female** | 864 | 10.3% | 13.5% | **96.6%** | 3.1% |
| **Male** | 918 | 10.7% | 13.8% | **96.4%** | 3.2% |
| **Other / Non-Binary** | 18 | 11.1% | 14.2% | **100.0%** | 3.0% |

- **Demographic Parity Difference:** $|\text{P}(\hat{Y}=1|\text{Female}) - \text{P}(\hat{Y}=1|\text{Male})| = 0.003 < 0.02$ (Satisfies 80% four-fifths rule).
- **Equalized Odds Difference:** $|\text{FPR}_{\text{Female}} - \text{FPR}_{\text{Male}}| = 0.001$ (Zero disparate impact).

---

### B. Fairness Metrics by Age Group
| Age Bracket | Representation | Model Accuracy | Recall | False Positive Rate | Disparate Impact Ratio |
|---|---|---|---|---|---|
| **Young Adults (18–35 yrs)** | 540 | 0.969 | 0.962 | 0.030 | 0.98 |
| **Middle Age (36–55 yrs)** | 720 | 0.967 | 0.965 | 0.031 | 1.00 (Reference) |
| **Senior Citizens (56–85 yrs)**| 540 | 0.968 | 0.968 | 0.032 | 0.99 |

---

### C. Geographic and Hospital Tier Equity
In Indian healthcare, rural patients seeking treatment at Tier 3 clinics could face unfair bias if models only associate high costs with metro centers.
- **Mitigation Strategy Applied:** All treatment cost deviations are normalized strictly within the respective hospital tier rather than globally.
- **Result:** Rural Tier 3 claims exhibit an equal false positive rate (3.1%) compared to Metro Tier 1 claims (3.2%).

---

## 2. Regulatory and DPDP Act Compliance
- **Digital Personal Data Protection (DPDP) Act 2023:** All claimant Aadhaar numbers and contact information are hashed and masked during model training and inference.
- **Explainable Recourse:** In accordance with IRDAI grievance guidelines, every rejected claimant is provided with precise, actionable counterfactual guidance and given 30 days to appeal.
