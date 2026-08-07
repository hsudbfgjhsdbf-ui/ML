# Data Dictionary — Health Insurance Fraud Claims

This document describes every feature in the raw dataset
`Health Insurance Fraud Claims.xlsx` (4,500 claims, one claim per row).

| # | Feature | Data Type | Description | Valid Range | Relevance to Fraud |
|---|---------|-----------|-------------|-------------|--------------------|
| 1 | `ClaimID` | string (UUID) | Unique identifier of each claim | — | Identifier (dropped) |
| 2 | `PatientID` | string (UUID) | Unique identifier of the patient/claimant | — | Identifier (dropped) |
| 3 | `ProviderID` | string (UUID) | Unique identifier of the healthcare provider/hospital | — | Identifier (dropped) |
| 4 | `ClaimAmount` | float (INR) | Amount claimed in Indian Rupees | 100.12 – 9,997.20 | **High** — inflated claims signal fraud |
| 5 | `ClaimDate` | datetime | Date the claim was submitted | 2022-07-09 – 2024-07-08 | Medium — temporal patterns |
| 6 | `DiagnosisCode` | string | Diagnosis code (high cardinality) | 4,495 unique | Medium — diagnosis-treatment mismatch |
| 7 | `ProcedureCode` | string | Medical procedure code (high cardinality) | 4,495 unique | Medium — procedure cost anomalies |
| 8 | `PatientAge` | int (years) | Age of the patient | 0 – 99 | Medium — age-based cost profiles |
| 9 | `PatientGender` | string | Patient gender | F, M | Low — bias-monitoring attribute |
| 10 | `ProviderSpecialty` | string | Medical specialty of the provider | 5 specialties | **High** — cost baselines per specialty |
| 11 | `ClaimStatus` | string | Status of the claim | Pending, Denied, Approved | Medium — outcome patterns |
| 12 | `PatientIncome` | float (INR) | Annual income of the patient | 20,006 – 149,957 | Medium — claim-to-income ratio |
| 13 | `PatientMaritalStatus` | string | Marital status | Single, Married, Divorced, Widowed | Low |
| 14 | `PatientEmploymentStatus` | string | Employment status | Employed, Student, Unemployed, Retired | Low — income verification |
| 15 | `ProviderLocation` | string | Location/city of provider (high cardinality) | 3,876 unique | Medium — regional cost variance |
| 16 | `ClaimType` | string | Type of claim | Inpatient, Outpatient, Emergency, Routine | Medium — type-based cost norms |
| 17 | `ClaimSubmissionMethod` | string | How the claim was submitted | Paper, Online, Phone | Low — submission risk |
| 18 | `Cluster` | int | Segment/cluster label | 0–3 | Medium |
| 19 | `ClaimLegitimacy` | string | **Target variable** | Legitimate, Fraud | — Target (Fraud = positive) |

## Engineered features
Derived during feature engineering (see `src/feature_engineering.py`):
- Temporal: `ClaimYear`, `ClaimMonth`, `ClaimDayOfWeek`, `ClaimDayOfYear`,
  `ClaimIsWeekend`, `ClaimSeason`.
- Ratios: `ClaimToIncome`, `ClaimAmountLog`, `IncomeLog`, `AgeGroup`.
- Interactions: `AmountPerAge`, `AmountPerIncome`, `AgeXIncome`,
  `AmountSq`, `AgeSq`.

## Features dropped to prevent data leakage
The following raw features are **not used as model inputs** because they either
encode the target directly or are near-unique identifiers:

| Feature | Reason for dropping |
|---------|---------------------|
| `DiagnosisCode` | ~4,495 unique values across 4,500 rows → target-encoding would leak the label |
| `ProcedureCode` | Same as above (near-unique) |
| `ProviderLocation` | ~3,876 unique values → near-unique, no reliable encoding |
| `Cluster` | **Direct leak** — 267 of 270 frauds fall in Cluster 1, making it a near-perfect fraud proxy |

## Class balance
- **Legitimate**: 4,230 (94%)
- **Fraud**: 270 (6%)
This is an imbalanced binary classification problem (6% positive rate).
