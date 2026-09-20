# FORM 1: Traditional ML — Dataset Acquisition, EDA & Preprocessing Specification
## Medical Insurance Claim Fraud Detection — Approach 1 of 3
### IIIT Dharwad | B.Tech Data Science & AI

---

**Faculty Adviser:** Prof. Ramesh Athe  
**Team Members:**  
- B Varshith (23BDS011)  
- M Jagadeshwar (23BDS033)  
- J Ganesh (23BDS024)  

**Date:** 01-09-2026  
**Document Version:** 1.0  
**Status:** Complete

---

## 1. Purpose and Scope

This document specifies the complete dataset acquisition, exploratory data analysis (EDA), and preprocessing pipeline for Approach 1 of the Medical Insurance Claim Fraud Detection project. It serves as the foundational specification that all downstream modeling, evaluation, and documentation work depends upon. Every design decision documented here is binding for the entire Approach 1 implementation.

## 2. Dataset Source and Acquisition

### 2.1 Primary Dataset

The dataset used for this project is the **Health Insurance Fraud Claims** dataset, sourced from the team's public GitHub repository at:

- **URL:** https://github.com/hsudbfgjhsdbf-ui/ML/blob/main/Health%20Insurance%20Fraud%20Claims.xlsx
- **Source repository:** hsudbfgjhsdbf-ui/ML
- **File format:** Microsoft Excel (.xlsx)
- **Retrieval date:** 01-09-2026

### 2.2 Dataset Overview

| Property | Value |
|----------|-------|
| Total rows | 4,500 |
| Total columns | 19 |
| Fraud cases | 270 (6.0%) |
| Legitimate cases | 4,230 (94.0%) |
| Class imbalance ratio | 1:15.67 |
| Missing values | 0 (complete dataset) |
| Unique claimants | 4,500 (one claim per patient) |
| Unique providers | 4,500 |
| Date range | 2022-07-09 to 2024-07-08 |

### 2.3 Column Inventory

| # | Column Name | Data Type | Unique Values | Description |
|---|-------------|-----------|---------------|-------------|
| 1 | ClaimID | string/UUID | 4,500 | Unique claim identifier |
| 2 | PatientID | string/UUID | 4,500 | Unique patient identifier |
| 3 | ProviderID | string/UUID | 4,500 | Unique provider identifier |
| 4 | ClaimAmount | float64 | 4,490 | Claimed amount (INR range: 100–9,997) |
| 5 | ClaimDate | datetime | 731 | Date of claim submission |
| 6 | DiagnosisCode | string | 4,495 | ICD-10 style diagnosis code |
| 7 | ProcedureCode | string | 4,495 | Procedure catalog identifier |
| 8 | PatientAge | int64 | 100 | Patient age in years (0–99) |
| 9 | PatientGender | string | 2 | F (Female), M (Male) |
| 10 | ProviderSpecialty | string | 5 | Cardiology, General Practice, Neurology, Orthopedics, Pediatrics |
| 11 | ClaimStatus | string | 3 | Approved, Denied, Pending |
| 12 | PatientIncome | float64 | 4,500 | Annual patient income (INR: 20,007–149,958) |
| 13 | PatientMaritalStatus | string | 4 | Single, Married, Divorced, Widowed |
| 14 | PatientEmploymentStatus | string | 4 | Employed, Student, Unemployed, Retired |
| 15 | ProviderLocation | string | 3,876 | Provider geographic location |
| 16 | ClaimType | string | 4 | Inpatient, Outpatient, Emergency, Routine |
| 17 | ClaimSubmissionMethod | string | 3 | Paper, Online, Phone |
| 18 | Cluster | int64 | 4 | Cluster assignment (0, 1, 2, 3) |
| 19 | ClaimLegitimacy | string | 2 | Legitimate, Fraud (TARGET) |

### 2.4 Target Variable

The target variable is `ClaimLegitimacy` with two classes:
- **Legitimate (0):** 4,230 claims (94.0%)
- **Fraud (1):** 270 claims (6.0%)

### 2.5 Critical Data Insight: Cluster-Fraud Relationship

The `Cluster` column is the single strongest predictor of fraud:

| Cluster | Legitimate | Fraud | Fraud Rate |
|---------|-----------|-------|------------|
| 0 | 1,144 | 0 | 0.0% |
| 1 | 833 | 267 | 24.3% |
| 2 | 1,104 | 0 | 0.0% |
| 3 | 1,149 | 3 | 0.3% |

**Key finding:** 267 of 270 fraud cases (98.9%) belong to Cluster 1. The remaining 3 fraud cases are in Cluster 3. Clusters 0 and 2 contain zero fraud cases.

### 2.6 ClaimAmount Distribution by Fraud Status

| Metric | Legitimate | Fraud |
|--------|-----------|-------|
| Mean | 4,776.80 INR | 8,733.57 INR |
| Std | 2,787.01 | 722.10 |
| Median | 4,718.37 INR | 8,751.36 INR |
| Min | 100.12 INR | 6,468.55 INR |
| Max | 9,997.20 INR | 9,978.43 INR |

**Key finding:** Fraudulent claims have significantly higher amounts (mean 8,734 INR vs 4,777 INR) and lower variance, concentrated in a narrow high range.

## 3. Data Quality Assessment

### 3.1 Completeness
- All 19 columns have zero missing values
- No null entries detected in any field
- Data completeness score: 100%

### 3.2 Consistency Checks
- **ClaimID uniqueness:** All 4,500 values unique — PASS
- **PatientID uniqueness:** All 4,500 values unique — PASS (one claim per patient)
- **ProviderID uniqueness:** All 4,500 values unique — PASS
- **Date range validity:** All dates within 2022-07-09 to 2024-07-08 — PASS
- **Age range:** 0 to 99 years — PASS
- **ClaimAmount positivity:** All values > 0 — PASS
- **PatientIncome positivity:** All values > 0 — PASS

### 3.3 Accuracy Validation
- DiagnosisCode format: 2 letters + 3 digits — consistent pattern
- ProcedureCode format: 2 letters + 3 digits — consistent pattern
- No negative monetary amounts — PASS
- No impossible age values — PASS

### 3.4 Data Quality Verdict
**OVERALL: PASS** — The dataset is clean with no missing values, no duplicates, and consistent data types. No imputation is required.

## 4. Exploratory Data Analysis (EDA)

### 4.1 Univariate Analysis

#### Numeric Features
| Feature | Mean | Std | Min | 25% | 50% | 75% | Max | Skewness |
|---------|------|-----|-----|-----|-----|-----|-----|----------|
| ClaimAmount | 5,014.20 | 2,866.29 | 100.12 | 2,509.07 | 5,053.77 | 7,462.45 | 9,997.20 | ~0.0 |
| PatientAge | 50.22 | 28.34 | 0 | 26 | 51 | 75 | 99 | ~0.0 |
| PatientIncome | 84,384.28 | 37,085.91 | 20,007 | 52,792 | 84,061 | 115,768 | 149,958 | ~0.0 |
| Cluster | 1.50 | 1.13 | 0 | 0 | 2 | 3 | 3 | ~0.0 |

#### Categorical Features
| Feature | Levels | Top Category | Top Share |
|---------|--------|-------------|-----------|
| PatientGender | 2 | F / M | ~50% each |
| ProviderSpecialty | 5 | Orthopedics | ~22% |
| ClaimStatus | 3 | Pending | ~34% |
| PatientMaritalStatus | 4 | Married | ~26% |
| PatientEmploymentStatus | 4 | Employed | ~26% |
| ClaimType | 4 | Inpatient | ~26% |
| ClaimSubmissionMethod | 3 | Paper | ~35% |

### 4.2 Bivariate Analysis: Feature-Target Relationships

#### Key Fraud Signals Identified:
1. **Cluster = 1:** Fraud rate 24.3% vs <0.3% for other clusters
2. **ClaimAmount > 6,400 INR:** Fraud probability sharply increases
3. **ClaimAmount × Cluster interaction:** Highest fraud when Cluster=1 AND ClaimAmount is high
4. **PatientIncome:** Fraud claims tend to have different income distributions
5. **ClaimStatus:** Denied claims show higher fraud correlation

### 4.3 Correlation Analysis

Top feature correlations with target (ClaimLegitimacy encoded as 0/1):
- Cluster: Strong positive correlation with fraud
- ClaimAmount: Moderate positive correlation
- PatientIncome: Weak negative correlation

### 4.4 High-Cardinality Features

The following features have extremely high cardinality and require special handling:
- **ClaimID:** 4,500 unique values → Drop (identifier only)
- **PatientID:** 4,500 unique values → Drop (identifier only)
- **ProviderID:** 4,500 unique values → Drop (identifier only)
- **DiagnosisCode:** 4,495 unique values → Drop (near-unique, no pattern)
- **ProcedureCode:** 4,495 unique values → Drop (near-unique, no pattern)
- **ProviderLocation:** 3,876 unique values → Drop (near-unique)
- **ClaimDate:** 731 unique values → Extract temporal features only

## 5. Preprocessing Pipeline Specification

### 5.1 Column Disposition

| Column | Action | Rationale |
|--------|--------|-----------|
| ClaimID | DROP | Identifier, no predictive value |
| PatientID | DROP | Identifier, no predictive value |
| ProviderID | DROP | Identifier, no predictive value |
| ClaimAmount | KEEP + SCALE | Strong fraud signal |
| ClaimDate | DROP (extract features) | Extract month, day_of_week, year features |
| DiagnosisCode | DROP | Near-unique, no generalizable pattern |
| ProcedureCode | DROP | Near-unique, no generalizable pattern |
| PatientAge | KEEP + SCALE | Demographic signal |
| PatientGender | KEEP + ENCODE | Binary categorical |
| ProviderSpecialty | KEEP + ENCODE | 5-level categorical |
| ClaimStatus | DROP | Potential leakage (post-decision) |
| PatientIncome | KEEP + SCALE | Strong fraud signal when combined with ClaimAmount |
| PatientMaritalStatus | KEEP + ENCODE | 4-level categorical |
| PatientEmploymentStatus | KEEP + ENCODE | 4-level categorical |
| ProviderLocation | DROP | Near-unique |
| ClaimType | KEEP + ENCODE | 4-level categorical |
| ClaimSubmissionMethod | KEEP + ENCODE | 3-level categorical |
| Cluster | KEEP | Strongest single predictor |
| ClaimLegitimacy | TARGET | Encode: Fraud=1, Legitimate=0 |

### 5.2 Feature Engineering

#### New Features to Create:
1. **ClaimAmountPatientIncome** = ClaimAmount / PatientIncome (ratio)
2. **ClaimDate_Month** = Month of claim (1–12)
3. **ClaimDate_DayOfWeek** = Day of week (0–6)
4. **ClaimDate_Year** = Year of claim (2022, 2023, 2024)
5. **ClaimDate_IsWeekend** = 1 if Saturday/Sunday, 0 otherwise
6. **AgeGroup** = Binned age: 0-17, 18-30, 31-45, 46-60, 61+
7. **IncomeGroup** = Binned income: Low, Medium, High
8. **LogClaimAmount** = log1p(ClaimAmount)

### 5.3 Encoding Strategy

| Feature | Encoding Method | Details |
|---------|----------------|---------|
| PatientGender | Label Encoding | F=0, M=1 |
| ProviderSpecialty | One-Hot Encoding | 5 columns |
| PatientMaritalStatus | One-Hot Encoding | 4 columns |
| PatientEmploymentStatus | One-Hot Encoding | 4 columns |
| ClaimType | One-Hot Encoding | 4 columns |
| ClaimSubmissionMethod | One-Hot Encoding | 3 columns |
| AgeGroup | Ordinal Encoding | 0, 1, 2, 3, 4 |
| IncomeGroup | Ordinal Encoding | 0, 1, 2 |

### 5.4 Scaling Strategy

- **RobustScaler** for ClaimAmount, PatientIncome (handles outliers)
- **StandardScaler** for PatientAge, ClaimAmountPatientIncome ratio
- **No scaling** for encoded categorical features and Cluster

### 5.5 Train-Validation-Test Split

```
Split ratio: 70% train / 15% validation / 15% test
Strategy: Stratified by ClaimLegitimacy
Random seed: 42
Expected sizes:
  - Train: 3,150 samples (189 fraud, 2,961 legitimate)
  - Validation: 675 samples (41 fraud, 634 legitimate)
  - Test: 675 samples (40 fraud, 635 legitimate)
```

### 5.6 Class Imbalance Handling

Multiple strategies will be compared:
1. **No resampling** (baseline)
2. **SMOTE** (Synthetic Minority Oversampling) on training data only
3. **class_weight='balanced'** in model parameters
4. **ADASYN** (Adaptive Synthetic Sampling)

The best strategy will be selected based on validation PR-AUC.

## 6. Pipeline Architecture

```
┌─────────────────────────────────────────────────────┐
│                DATA ACQUISITION                       │
│  Health Insurance Fraud Claims.xlsx (4,500 × 19)     │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              DATA QUALITY GATES                       │
│  Completeness ✓  Consistency ✓  Accuracy ✓           │
│  No missing values, no duplicates                     │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│           EXPLORATORY DATA ANALYSIS                   │
│  Univariate → Bivariate → Correlation → Insights     │
│  48+ visualizations generated                         │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│           COLUMN DISPOSITION & CLEANING               │
│  Drop identifiers, high-cardinality, leakage cols    │
│  Result: 12 raw features retained                     │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│            FEATURE ENGINEERING                        │
│  Ratios, temporal features, bins, log transforms     │
│  Result: ~20 engineered features                      │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│           ENCODING & SCALING                          │
│  One-hot, ordinal, label encoding                    │
│  RobustScaler, StandardScaler                         │
│  Result: ~35 model-ready features                     │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│           STRATIFIED SPLITTING                        │
│  70/15/15 train/validation/test                      │
│  Stratified by ClaimLegitimacy                        │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│          CLASS IMBALANCE HANDLING                     │
│  Compare: None, SMOTE, class_weight, ADASYN          │
│  Select best by validation PR-AUC                     │
└─────────────────────────────────────────────────────┘
```

## 7. Data Contracts

### 7.1 Input Contract (Stage S1)
- File: Health Insurance Fraud Claims.xlsx
- Expected columns: 19 (as listed in Section 2.3)
- Expected rows: 4,500 (±5%)
- Expected dtypes: As specified in Section 2.3

### 7.2 Output Contract (Stage S5)
- Format: pandas DataFrame
- Rows: 4,500 (before split)
- Columns: ~35 encoded features + 1 target
- All numeric, no nulls, no infinities
- Target: binary (0/1)

### 7.3 Split Contract (Stage S6)
- Three DataFrames: train, validation, test
- No patient overlap between splits
- Class ratios preserved within ±0.5%

## 8. Reproducibility

- Random seed: 42 (propagated to all stochastic operations)
- Python version: 3.11+
- Library versions: Pinned in requirements.txt
- Dataset hash: SHA256 computed and recorded in evaluation/

## 9. References

1. Dataset source: https://github.com/hsudbfgjhsdbf-ui/ML
2. IRDAI Annual Report 2023-24: Insurance fraud statistics in India
3. sklearn documentation: Preprocessing and encoding utilities
4. imbalanced-learn documentation: SMOTE and ADASYN implementations

## 10. Sign-Off

| Role | Name | Date | Status |
|------|------|------|--------|
| Team Lead | B Varshith | 01-09-2026 | ✅ Approved |
| Data Engineer | M Jagadeshwar | 01-09-2026 | ✅ Approved |
| ML Engineer | J Ganesh | 01-09-2026 | ✅ Approved |
| Faculty Adviser | Prof. Ramesh Athe | Pending | ⏳ Review |

---

*Last updated: 01-09-2026 | Document version: 1.0*  
*Next: Form 2 — Model Zoo Implementation Specification*
