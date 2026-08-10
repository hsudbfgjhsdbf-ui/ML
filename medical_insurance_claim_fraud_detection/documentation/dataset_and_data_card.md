# Dataset and Data Card

See `../data/data_card.md` for full data card.

## Summary
- Name: Health Insurance Fraud Claims
- Rows: 4500, Cols: 19
- Source: Local Excel provided, structurally similar to CMS/Kaggle healthcare fraud datasets
- License: Academic use, synthetic anonymized
- Access: 2026-08-06
- Prediction Unit: claim-level
- Target: ClaimLegitimacy (Legitimate 4230, Fraud 270, 6% fraud rate)
- Missing: 0
- Features: ClaimAmount, ClaimDate, DiagnosisCode, ProcedureCode, PatientAge, Gender, ProviderSpecialty, ClaimStatus, Income, Marital, Employment, Location, Type, SubmissionMethod, Cluster, IDs

## Data Dictionary
See `../data/data_dictionary.csv` and `data_dictionary.md`

## Data Quality
- No missing, but imputation in pipeline for robustness
- Outliers: 0 via IQR for ClaimAmount, Age, Income (synthetic)
- Leakage heuristic: ClaimStatus, Marital, Employment flagged as potential post-decision; ClaimStatus especially may leak because denied claims correlate with fraud but it's administrative outcome.

## Cleaning
- Drop target column for features, map to binary
- Date engineering: year, month, day, dayofweek, quarter, ordinal
- No rows dropped

## Class Imbalance
- Ratio minority/majority 0.0638, imbalanced True
- Use class_weight balanced and SMOTE optionally inside training folds only

## Bias & Privacy
- Sensitive: PatientID, Age, Gender, Income, Marital
- Potential bias: income, gender, location, specialty
- Privacy: UUIDs synthetic, ANONYMIZE_PII true, no external API transmission by default

## Limitations
- Synthetic separable (income + amount high importance suggests artificial separability)
- No free-text notes, limited doc images (we use synthetic JSON)
- Modest size for DL
- Single time window July 2024
- No provider longitudinal history

## References
- CMS Public Use Files: https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files
- Kaggle Healthcare Provider Fraud: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
- Kaggle Provider Fraud (example): https://www.kaggle.com/datasets/itsmohitsharma/medicare-provider-fraud-detection-dataset
- Imbalanced Learn: https://imbalanced-learn.org/stable/
- SHAP: https://shap.readthedocs.io/
