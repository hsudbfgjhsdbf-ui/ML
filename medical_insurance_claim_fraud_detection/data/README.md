# Data Folder — Medical Insurance Claim Fraud Detection

## Overview
This folder contains the dataset used for the academic project "Medical Insurance Claim Fraud Detection — AI-Driven Claim Verification & Explainable Fraud Detection Platform".

## Structure
- `raw/` : Original raw file(s) as provided. The primary dataset is `Health_Insurance_Fraud_Claims.xlsx` (4500 rows, 19 columns). Claim-level prediction.
- `interim/` : Intermediate cleaning outputs (not committed if large).
- `processed/` : Cleaned CSV version `claims_processed.csv` and model artifacts under `artifacts/`.
- `sample/` : Small 100-row sample for quick tests and synthetic document fixtures.

## Dataset Summary
- **Rows**: 4500
- **Columns**: 19
- **Target**: `ClaimLegitimacy` with values `Legitimate` (4230, 94%) and `Fraud` (270, 6%).
- **Prediction unit**: Individual medical insurance claim (not provider-level).
- **Imbalance**: Highly imbalanced (fraud rate 6%).
- **Missing values**: None in raw file, but code handles missing gracefully.

## Source
- Local file provided alongside repo: `Health Insurance Fraud Claims.xlsx`.
- Structurally similar to public CMS / Kaggle healthcare fraud datasets, e.g.:
  - CMS Public Use Files: https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files
  - Kaggle Healthcare Provider Fraud: https://www.kaggle.com/datasets/multiple (search "health insurance fraud")
- For this project we document as locally provided synthetic-but-realistic claims dataset.

## Label Meaning
- `Legitimate` = claim assessed as non-fraudulent in ground truth.
- `Fraud` = claim flagged as fraudulent.

Do NOT misrepresent as provider-level unless explicitly stated. This dataset is claim-level.

## Privacy
- All IDs are anonymized synthetic UUIDs.
- No real PII present; still treat as sensitive and apply minimization.

## Usage
```bash
python ../approaches/01_traditional_ml.py --data_path data/raw/Health_Insurance_Fraud_Claims.xlsx
```

## Data Card
See `data_card.md` for detailed documentation.

## Dictionary
See `data_dictionary.csv`.

## Manifest
See `dataset_manifest.json`.

## Synthetic Fixtures
`sample/` also contains synthetic document fixtures for OCR/VLM testing:
- `sample_100.csv`
- `knowledge_base/` contains policy rules and fraud indicators (see below)

Synthetic data must NEVER be mixed silently with real benchmark data.

## Knowledge Base for RAG
`data/sample/knowledge_base/` contains:
- `policy_rules.txt`
- `exclusion_clauses.txt`
- `fraud_indicators.txt`
- `coverage_rules.txt`
- `claim_guidelines.txt`

These are created for demonstration and are not legal policies.
