# Medical Insurance Claim Fraud Detection

## Approach 1 — Traditional Machine Learning

This repository contains a reproducible, artifact-driven baseline for medical
insurance claim fraud screening in an Indian healthcare context. It is built
for the IIIT Dharwad mini-project team:

- **B Varshith** — 23BDS011
- **M Jagadeshwar** — 23BDS033
- **J Ganesh** — 23BDS024
- **Faculty adviser:** Prof. Ramesh Athe
- **Institution:** IIIT Dharwad, Department of Data Science and AI

The model answers the narrow academic question: **does this structured claim
record resemble the supplied fraud class, yes or no?** It produces a fraud
probability and triage band. It is **not** an automatic claim-denial system.

## What is included

| Area | Delivered artifact |
| --- | --- |
| Dataset | `data/raw/health_insurance_fraud_claims.xlsx`, checksum manifest, dataset card, dictionary |
| Code | Modular `src/` package plus `scripts/run_pipeline.py` |
| Model zoo | 20 baselines/algorithms under one common metric and threshold protocol |
| Evaluation | `evaluation/leaderboard.csv`, per-model metrics, curves, calibration, fairness, test intervals |
| Visuals | EDA figures, model comparison figures, concept images, high-resolution PNGs |
| Presentation | `presentation/approach_1_traditional_ml.pptx` and speaker notes |
| Reports | `reports/approach_1_project_report.pdf` and `reports/approach_1_ieee_paper.pdf` |
| Audit | Run manifest, split membership, feature lineage, selection memo, threshold memo |

## Pipeline architecture

```text
supplied_xlsx
    |
    v
load + schema gates --> EDA figures/tables
    |
    v
feature engineering --> stratified 70/15/15 split
    |
    v
train-only imputer + one-hot + scaler
    |
    v
20-model zoo + stratified CV search
    |
    v
validation F2 threshold --> validation leaderboard --> winner
    |
    v
winner refit on train+validation --> one-time locked test evaluation
    |
    v
metrics + calibration + fairness + explanations
    |
    v
Markdown + PPTX + project PDF + IEEE paper PDF
```

The final run artifacts in `evaluation/` are the source of truth for every
number appearing in the generated presentation and reports.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --config config/default.yaml
python scripts/verify_artifacts.py
```

The command reads `data/raw/health_insurance_fraud_claims.xlsx`, runs the
complete baseline, and regenerates the output files. Use `--dry-run` to print
the ordered stages and `--self-test` to check imports and metric semantics.

```bash
python scripts/run_pipeline.py --dry-run
python scripts/run_pipeline.py --self-test
```

## Dataset honesty statement

The repository contains a single 4,500-row workbook. It does not contain the
larger multi-table Medicare schema described in some planning documents. The
workbook has no verified Indian state, policy wording, sum insured, hospital
tier, document, or clinical reference fields. The project therefore does not
invent those values and does not attribute the data to Kaggle, IRDAI, or a
government source without evidence. The original license and redistribution
terms require confirmation from the data owner. Read `data/dataset_card.md`
before redistributing the repository.

The source snapshot is small and appears synthetic or de-identified. Reported
scores are valid only for the supplied snapshot and declared protocol. They are
not evidence of performance for all Indian insurers, policyholders, hospitals,
or claim types.

## Responsible-use boundary

A high score is a review signal. Before a real adverse decision, a licensed
claims professional must verify policy clauses, medical documents, provider
records, and claimant information. Explanations must be specific and neutral;
claimants need a correction and appeal pathway. Sensitive fields are audited
for disparate error patterns and are not used as identifier-like model inputs.

## Where to read next

1. [`goal.md`](goal.md) — end-to-end goals, milestones, and acceptance checklist.
2. [`documentation/00_project_overview.md`](documentation/00_project_overview.md) — actual run overview.
3. [`data/dataset_card.md`](data/dataset_card.md) — provenance and limitations.
4. [`documentation/methodology.md`](documentation/methodology.md) — pipeline and leakage controls.
5. [`documentation/eda_report.md`](documentation/eda_report.md) — EDA observations and figure index.
6. [`documentation/models.md`](documentation/models.md) — model families and search spaces.
7. [`evaluation/evaluation.md`](evaluation/evaluation.md) — generated comparative results.
8. [`presentation/approach_1_traditional_ml.pptx`](presentation/approach_1_traditional_ml.pptx) — 20-slide defense deck.
9. [`reports/approach_1_project_report.pdf`](reports/approach_1_project_report.pdf) — long project report.
10. [`reports/approach_1_ieee_paper.pdf`](reports/approach_1_ieee_paper.pdf) — compact IEEE-inspired manuscript.

## Development checks

```bash
python -m compileall -q src scripts tests
python -m pytest -q
python scripts/verify_artifacts.py
```

Optional external boosters are listed in `requirements-optional.txt`. The core
pipeline intentionally does not require them, GPU access, paid APIs, or an
internet connection during training.

## Approach 2 — Deep learning with XAI

Approach 2 is now implemented under `src_dl/` and consumes the frozen Approach 1
split. Install the pinned stack and run:

```bash
python -m pip install -r requirements-dl.txt
python scripts/run_deep_learning.py --config config_dl/default.yaml
python scripts/verify_deep_artifacts.py
```

Delivered deep artifacts include five architectures (MLP, TabNet-style,
1D-CNN, autoencoder hybrid, and feature-token transformer), three seeded runs,
epoch telemetry, checkpoints, calibration, fairness, occlusion importance,
faithfulness, stability, a 22-slide deck, a 35-page project report, and a
6-page IEEE-inspired paper. The deep run is still bounded by the supplied
4,500-row workbook and does not claim production or population-level validity.

Approach 3 agentic document reasoning is intentionally not claimed as completed
until its Gemini/LangGraph/Next.js artifacts are built and evaluated separately.

## Repository hygiene

The serialized model bundle and comparative model binaries under
`artifacts/models/` are included in the release because the approach
specification requires saved model state. Temporary caches belong in
`workspace/`. Do not commit API keys, credentials, real claimant data, or raw
identifiers in new examples.
