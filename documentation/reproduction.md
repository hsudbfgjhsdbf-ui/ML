# Reproduction guide

**Purpose:** run the complete Approach 1 pipeline from a clean checkout.  
**Run generated:** `run_20260807_151423`.  
**Last updated:** 07-08-2026 15:14:23 UTC.

## Fresh environment

1. Use Python 3.11 or newer.
2. Create a virtual environment.
3. Install the pinned dependencies from `requirements.txt`.
4. Confirm `data/raw/health_insurance_fraud_claims.xlsx` exists.
5. Run `python scripts/run_pipeline.py --config config/default.yaml`.

The command overwrites the latest lightweight artifacts and writes a timestamped
run manifest under `evaluation/runs/`. Historical runs should be copied out if
long-term archival is needed. Use `--dry-run` to print the plan and
`--self-test` to validate imports and a toy metric calculation.

## Verification

- Compare the SHA-256 workbook checksum in `data/metadata/raw_manifest.json`.
- Open `evaluation/leaderboard.csv` and `evaluation/evaluation.md`.
- Confirm the selected model card points to the same run id.
- Open `presentation/approach_1_traditional_ml.pptx`.
- Open `reports/approach_1_project_report.pdf` and `reports/approach_1_ieee_paper.pdf`.
- Run `python scripts/verify_artifacts.py`.

The data is a supplied academic snapshot. Do not claim the reported scores
represent all Indian insurers or deploy the model for automatic denial.
