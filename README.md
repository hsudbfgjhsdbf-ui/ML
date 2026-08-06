# Medical Insurance Claim Fraud Detection — Approach 1

**AI-Driven Claim Verification and Explainable Fraud Detection**  
IIIT Dharwad, B.Tech Data Science and AI  
B Varshith (23BDS011) · M Jagadeshwar (23BDS033) · J Ganesh (23BDS024)  
**Faculty Adviser: Prof. Ramesh Athe**

This reproducible academic codebase builds a synthetic, Indian-context medical-claim fraud triage benchmark. It predicts a fraud probability and a yes/no flag; it is decision support, never an automatic denial system.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py --config config/default.yaml
```

The single command creates validated synthetic data, EDA charts, group-disjoint splits, a classical model benchmark, test metrics, a selected-model artifact, PPTX, and two PDF communications. Generated outputs are intentionally ignored by Git because results must come from an actual local run rather than pre-filled claims.

## Reproducibility and data

- Fixed master seed: `42`; all shipped results are generated fresh.
- Default dataset: 50,000 wholly synthetic records. Set `data.rows: 120000` to use the specified full fallback size.
- `fraud_type`, identifiers, dates, labels, and post-decision fields are excluded from features.
- Selection prioritizes validation PR-AUC. The test partition is evaluated after selection.
- Currency is INR and date formatting uses DD-MM-YYYY.

## Repository guide

| Path | Purpose |
|---|---|
| `run_pipeline.py` | S0–S14 master entry point |
| `config/default.yaml` | pinned, commented reproducibility settings |
| `src/` | generation, validation, features, models, evaluation, reporting |
| `documentation/implementation_specification.md` | 2,000-line implementation and traceability record |
| `evaluation/` | generated run manifests, metrics, leaderboard, calibration assets |
| `images/` | generated EDA/model diagrams and figures |
| `presentation/` | generated `approach_1_traditional_ml.pptx` |
| `reports/` | generated project report and IEEE-style paper PDFs |

## Responsible use

The synthetic dataset is not representative of all Indian policyholders or medical settings. Scores must be reviewed with source documents, appeal mechanisms, and human investigator oversight. See the documentation specification for limitations, data contracts, fairness checks, and audit procedure.

## License

Academic project scaffold. Add the institution-approved license before public release.
