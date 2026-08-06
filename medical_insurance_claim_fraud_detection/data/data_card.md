# Data Card — Medical Insurance Fraud Claims

## Dataset Name
Health Insurance Fraud Claims (provided academic dataset)

## Source URL
- Local file: `Health_Insurance_Fraud_Claims.xlsx` provided in `/home/user/ML/`
- Structurally aligned with public sources:
  - CMS Data: https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files
  - Kaggle: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis (example; similar structure)
  - CMS Provider Fraud: https://www.kaggle.com/datasets/itsmohitsharma/medicare-provider-fraud-detection-dataset (example reference)

## License
- Provided for academic use in IIIT Dharwad coursework.
- Public analogous datasets: CC0 / Public Domain (CMS) or MIT (Kaggle contributors). Verify before redistribution.

## Access Date
2026-08-06

## Dataset Version
1.0 (local copy). Original file modification timestamp preserved in manifest.

## Number of Rows and Columns
- Rows: 4500
- Columns: 19 (including target)

## Prediction Unit
**Claim-level**: Each row represents a single medical insurance claim submitted by a patient via a provider.

NOT provider-level. We explicitly document claim-level.

## Target Variable
- Column: `ClaimLegitimacy`
- Type: Categorical binary
- Values: `Legitimate` (4230) , `Fraud` (270)
- Binary encoding: Legitimate=0, Fraud=1
- Fraud rate: 6.0%

## Feature Descriptions
See `data_dictionary.csv`.

## Class Distribution
- Legitimate: 4230 (94.0%)
- Fraud: 270 (6.0%)
- Imbalance ratio minority/majority: 0.0638

## Missing-Value Information
- Raw dataset: 0 missing values across all 19 columns.
- Preprocessing still includes median/most_frequent imputation for robustness.

## Potential Bias
- Age distribution skew? Patients aged 0-100 present; need to check fairness.
- Income correlated with fraud? Must audit.
- Gender: M/F distribution should be checked.
- ProviderLocation: Geographic bias possible.
- ProviderSpecialty: Different specialties may have different fraud rates.
- ClaimAmount: High-value claims may be over-penalized.

## Privacy Considerations
- IDs are UUIDs synthetic — no real PII.
- PatientAge, Income, Gender are quasi-identifiers; still anonymized.
- For project: enable ANONYMIZE_PII, minimize logging IDs, never transmit real records to external APIs.
- Synthetic document fixtures used for OCR.

## Known Limitations
- Synthetic nature not fully representative of real-world complex fraud schemes.
- No free-text clinical notes; limited textual features.
- Provider history not longitudinally rich (no time series).
- No images of bills; we use synthetic bills for OCR demo.
- 4500 rows is modest — limited for deep learning.

## Data-Cleaning Operations
- Loaded via pandas read_excel.
- Converted ClaimDate to datetime.
- Engineered date features: year, month, day, dayofweek, quarter, ordinal.
- Checked leakage: ClaimStatus may be post-decision but retained with note.
- No rows dropped; zero missing.

## Label-Generation Procedure
- Labels provided in raw file as ClaimLegitimacy.
- No synthetic label generation.
- No invented labels.

## Intended Use
- Supervised binary fraud classification.
- Anomaly detection comparison.
- Document validation demo uses synthetic bills, not this tabular dataset.
- Decision-support prototype, not autonomous adjudication.

## Out-of-Scope Use
- Not to be used as legal or medical determinant.
- Not for production denial without human review.

## Ethical Notes
- False positives harm legitimate patients; optimize for high recall but maintain precision.
- False negatives cause financial loss.
- Human review mandatory.
