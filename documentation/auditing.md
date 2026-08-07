# Auditing guide

1. Read `evaluation/run_manifest.json` and record its run id.
2. Verify the input checksum in `data/metadata/raw_manifest.json`.
3. Open `evaluation/split_summary.csv` and confirm 70/15/15 counts.
4. Open `evaluation/feature_lineage.csv` and confirm the target and identifiers are absent from model inputs.
5. Compare `evaluation/leaderboard.csv` with `evaluation/complete_evaluation_record.md`.
6. Follow the selected model key into `evaluation/metrics/`, `evaluation/curves/`, `evaluation/tuning/`, and `evaluation/model_cards/`.
7. Inspect the threshold memo and test unlock log before reading test metrics.
8. Inspect calibration, fairness, permutation importance, and concept-image provenance.
9. Open the PPTX and both PDFs; verify adviser, team, institution, and limitations.
10. Rerun `python scripts/verify_artifacts.py` and `python -m pytest -q`.

The model is a review-prioritization aid. An audit that finds a mismatch should
open a new deviation, rerun the pipeline, and regenerate all documents.
