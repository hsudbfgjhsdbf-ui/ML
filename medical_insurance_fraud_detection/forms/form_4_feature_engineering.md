# FORM 4: Traditional ML — Feature Engineering & Selection Specification
## Medical Insurance Claim Fraud Detection — Approach 1 of 3
### IIIT Dharwad | B.Tech Data Science & AI

---

**Faculty Adviser:** Prof. Ramesh Athe  
**Team Members:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  
**Date:** 01-09-2026 | **Version:** 1.0

---

## 1. Purpose

This document specifies every feature engineering decision, feature selection method, and the final frozen feature list used for modeling. All features are derived from the Health Insurance Fraud Claims dataset (4,500 rows, 19 columns).

## 2. Feature Engineering Pipeline

### 2.1 Input Features (After Dropping Identifiers)

Raw features retained from preprocessing:
- ClaimAmount (float)
- PatientAge (int)
- PatientGender (categorical: F, M)
- ProviderSpecialty (categorical: 5 levels)
- PatientIncome (float)
- PatientMaritalStatus (categorical: 4 levels)
- PatientEmploymentStatus (categorical: 4 levels)
- ClaimType (categorical: 4 levels)
- ClaimSubmissionMethod (categorical: 3 levels)
- Cluster (int: 0-3)
- ClaimDate → temporal features extracted

### 2.2 Ratio Features

| ID | Feature Name | Formula | Rationale |
|----|-------------|---------|-----------|
| RF01 | ClaimAmountPatientIncome | ClaimAmount / PatientIncome | Financial burden ratio — high values suggest unusual claims relative to income |
| RF02 | ClaimAmountPerAge | ClaimAmount / (PatientAge + 1) | Age-normalized claim severity |
| RF03 | IncomePerAge | PatientIncome / (PatientAge + 1) | Life-stage income normalization |
| RF04 | LogClaimAmount | log1p(ClaimAmount) | Reduces right-skew in claim amounts |
| RF05 | LogPatientIncome | log1p(PatientIncome) | Reduces skew in income distribution |
| RF06 | ClaimAmountZScore | (ClaimAmount - mean) / std | Standardized claim amount |
| RF07 | IncomeZScore | (PatientIncome - mean) / std | Standardized income |

### 2.3 Temporal Features (from ClaimDate)

| ID | Feature Name | Extraction | Rationale |
|----|-------------|-----------|-----------|
| TF01 | ClaimMonth | month(ClaimDate) | Seasonal patterns |
| TF02 | ClaimDayOfWeek | dayofweek(ClaimDate) | Weekend vs weekday patterns |
| TF03 | ClaimYear | year(ClaimDate) | Year-over-year trends |
| TF04 | ClaimIsWeekend | 1 if Sat/Sun else 0 | Weekend filing patterns |
| TF05 | ClaimQuarter | quarter(ClaimDate) | Quarterly patterns |

### 2.4 Binned Features

| ID | Feature Name | Bins | Encoding |
|----|-------------|------|----------|
| BF01 | AgeGroup | [0,17,30,45,60,99] → [0,1,2,3,4] | Ordinal |
| BF02 | IncomeGroup | Low(<50K), Med(50K-100K), High(>100K) → [0,1,2] | Ordinal |
| BF03 | ClaimAmountGroup | Low(<3K), Med(3K-7K), High(>7K) → [0,1,2] | Ordinal |

### 2.5 Interaction Features

| ID | Feature Name | Formula | Rationale |
|----|-------------|---------|-----------|
| IF01 | ClusterIs1 | 1 if Cluster==1 else 0 | Binary flag for high-fraud cluster |
| IF02 | HighClaimAmt | 1 if ClaimAmount > 6400 else 0 | Threshold flag |
| IF03 | Cluster1_HighAmt | ClusterIs1 × HighClaimAmt | Interaction of two strongest signals |
| IF04 | LowIncome_HighClaim | (IncomeGroup==0) × (ClaimAmountGroup==2) | Financial anomaly flag |

### 2.6 Categorical Encoding Results

After encoding, the feature expansion produces:

| Original Feature | Encoding | Resulting Columns |
|-----------------|----------|-------------------|
| PatientGender | Label (F=0, M=1) | 1 column |
| ProviderSpecialty | One-Hot | 5 columns |
| PatientMaritalStatus | One-Hot | 4 columns |
| PatientEmploymentStatus | One-Hot | 4 columns |
| ClaimType | One-Hot | 4 columns |
| ClaimSubmissionMethod | One-Hot | 3 columns |
| **Total Categorical** | | **21 columns** |

### 2.7 Complete Feature List (Frozen)

| # | Feature | Type | Source |
|---|---------|------|--------|
| 1 | ClaimAmount | numeric | raw |
| 2 | PatientAge | numeric | raw |
| 3 | PatientIncome | numeric | raw |
| 4 | Cluster | numeric | raw |
| 5 | PatientGender | binary | encoded |
| 6 | ProviderSpecialty_Cardiology | binary | one-hot |
| 7 | ProviderSpecialty_GeneralPractice | binary | one-hot |
| 8 | ProviderSpecialty_Neurology | binary | one-hot |
| 9 | ProviderSpecialty_Orthopedics | binary | one-hot |
| 10 | ProviderSpecialty_Pediatrics | binary | one-hot |
| 11 | PatientMaritalStatus_Divorced | binary | one-hot |
| 12 | PatientMaritalStatus_Married | binary | one-hot |
| 13 | PatientMaritalStatus_Single | binary | one-hot |
| 14 | PatientMaritalStatus_Widowed | binary | one-hot |
| 15 | PatientEmploymentStatus_Employed | binary | one-hot |
| 16 | PatientEmploymentStatus_Retired | binary | one-hot |
| 17 | PatientEmploymentStatus_Student | binary | one-hot |
| 18 | PatientEmploymentStatus_Unemployed | binary | one-hot |
| 19 | ClaimType_Emergency | binary | one-hot |
| 20 | ClaimType_Inpatient | binary | one-hot |
| 21 | ClaimType_Outpatient | binary | one-hot |
| 22 | ClaimType_Routine | binary | one-hot |
| 23 | ClaimSubmissionMethod_Online | binary | one-hot |
| 24 | ClaimSubmissionMethod_Paper | binary | one-hot |
| 25 | ClaimSubmissionMethod_Phone | binary | one-hot |
| 26 | ClaimAmountPatientIncome | numeric | engineered |
| 27 | ClaimAmountPerAge | numeric | engineered |
| 28 | IncomePerAge | numeric | engineered |
| 29 | LogClaimAmount | numeric | engineered |
| 30 | LogPatientIncome | numeric | engineered |
| 31 | ClaimAmountZScore | numeric | engineered |
| 32 | IncomeZScore | numeric | engineered |
| 33 | ClaimMonth | numeric | temporal |
| 34 | ClaimDayOfWeek | numeric | temporal |
| 35 | ClaimYear | numeric | temporal |
| 36 | ClaimIsWeekend | binary | temporal |
| 37 | ClaimQuarter | numeric | temporal |
| 38 | AgeGroup | ordinal | binned |
| 39 | IncomeGroup | ordinal | binned |
| 40 | ClaimAmountGroup | ordinal | binned |
| 41 | ClusterIs1 | binary | interaction |
| 42 | HighClaimAmt | binary | interaction |
| 43 | Cluster1_HighAmt | binary | interaction |
| 44 | LowIncome_HighClaim | binary | interaction |

**Total frozen features: 44**

### 2.8 Dropped Columns — Justification

| Column | Reason for Dropping |
|--------|-------------------|
| ClaimID | Unique identifier — no generalizable pattern |
| PatientID | Unique identifier — one claim per patient |
| ProviderID | Unique identifier — no aggregation possible |
| DiagnosisCode | 4,495 unique values — near-unique codes |
| ProcedureCode | 4,495 unique values — near-unique codes |
| ClaimStatus | Post-decision outcome — potential leakage |
| ProviderLocation | 3,876 unique values — near-unique locations |

## 3. Feature Selection

### 3.1 Methods Applied

| Method | Description | Features Removed |
|--------|-------------|-----------------|
| Variance Filter | Remove near-constant features (threshold < 1e-6) | 0 features |
| Correlation Prune | Remove one of pairs with |r| > 0.98 | TBD after training |
| Model Importance | Gradient-boosted feature importance ranking | TBD after training |
| Mutual Information | Non-linear dependency ranking | TBD after training |

### 3.2 Expected High-Importance Features

Based on EDA findings:
1. Cluster (especially ClusterIs1)
2. ClaimAmount
3. ClaimAmountPatientIncome (ratio)
4. PatientIncome
5. LogClaimAmount
6. HighClaimAmt
7. Cluster1_HighAmt

### 3.3 Feature Lineage Documentation

Every feature has documented:
- Source column(s)
- Transform applied
- Reason for inclusion
- Relation to fraud domain knowledge

## 4. Data Pipeline Summary

```
Raw Data (19 columns)
    │
    ├── Drop 7 columns (identifiers, leakage, high-cardinality)
    │   → 12 raw features retained
    │
    ├── Engineer 8 new features (ratios, temporal, binned)
    │   → 20 base features
    │
    ├── Engineer 4 interaction features
    │   → 24 features
    │
    ├── Encode categoricals (one-hot, label)
    │   → 44 final features
    │
    ├── Scale numerics (RobustScaler, StandardScaler)
    │   → 44 model-ready features
    │
    └── Split (70/15/15 stratified)
        → Train: 3,150 × 44
        → Validation: 675 × 44
        → Test: 675 × 44
```

## 5. Leakage Prevention

### 5.1 Leakage Audit Table

| Feature | Knowable at Decision Time? | Decision |
|---------|---------------------------|----------|
| ClaimAmount | Yes | KEEP |
| PatientAge | Yes | KEEP |
| Cluster | Yes (pre-assigned) | KEEP |
| ClaimStatus | NO — determined after investigation | DROP |
| All engineered features | Derived from kept features only | KEEP |

### 5.2 Watchdog Rule
If any single feature has importance > 0.60 gain, trigger mandatory leakage re-review.

## 6. Feature Freeze Protocol

- Feature list frozen at 44 features
- Freeze recorded in `config/frozen_features.yaml`
- Any post-freeze change requires deviation log entry
- Hash of feature list recorded in run manifest

---

*Last updated: 01-09-2026 | Document version: 1.0*  
*Next: Form 5 — Explainability, Visualization & Reporting*
