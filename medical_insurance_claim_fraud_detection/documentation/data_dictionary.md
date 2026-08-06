# Data Dictionary

Source: `data/data_dictionary.csv`

| Column | Type | Description | Example | IsTarget | IsSensitive |
|--------|------|-------------|---------|----------|-------------|
| ClaimID | string | Unique claim ID | 4d76c7f7-... | No | No |
| PatientID | string | Patient ID | 19cf2638-... | No | Yes |
| ProviderID | string | Provider ID | a3d0cc80-... | No | No |
| ClaimAmount | float | Claimed amount USD | 7820.52 | No | No |
| ClaimDate | datetime | Claim submitted date | 2024-07-08 | No | No |
| DiagnosisCode | string | ICD-like | Ta150 | No | No |
| ProcedureCode | string | CPT-like | iO013 | No | No |
| PatientAge | int | Age | 96 | No | Yes |
| PatientGender | string | M/F | F | No | Yes |
| ProviderSpecialty | string | Specialty | Orthopedics | No | No |
| ClaimStatus | string | Pending/Approved/Denied | Pending | No | No |
| PatientIncome | float | Income | 57595.11 | No | Yes |
| PatientMaritalStatus | string | Single/Married etc | Single | No | Yes |
| PatientEmploymentStatus | string | Employed etc | Employed | No | Yes |
| ProviderLocation | string | City | New Alishaview | No | No |
| ClaimType | string | Inpatient/Outpatient/Emergency/Routine | Inpatient | No | No |
| ClaimSubmissionMethod | string | Paper/Online/Phone | Paper | No | No |
| Cluster | int | Derived cluster 0-3 | 3 | No | No |
| ClaimLegitimacy | string | Target Legitimate/Fraud | Legitimate | Yes | No |

## Additional Engineered Features (pipeline)
- ClaimDate_year, month, day, dayofweek, quarter, ordinal

## Synthetic Document Fields (for OCR demo)
- document_type, claim_id, patient_name (REDACTED), provider_name, bill_date, discharge_date, total_amount, items, diagnosis_code, procedure_code

## Knowledge Base Docs (RAG)
- policy_rules.txt, exclusion_clauses.txt, fraud_indicators.txt, coverage_rules.txt, claim_guidelines.txt

## API Contract
See `api/README.md`
