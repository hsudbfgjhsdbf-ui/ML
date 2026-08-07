# Medical Insurance Claim Fraud Detection

**IIIT Dharwad — Department of Data Science and AI**  
**Faculty Adviser:** Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)

A reproducible academic project comparing three approaches to medical-insurance claim fraud screening in an Indian context:

1. **Approach 1 — Traditional ML:** an implemented, evaluated and explainable tabular baseline.
2. **Approach 2 — Deep Learning:** planned to evaluate learned tabular/temporal representations against the same frozen split.
3. **Approach 3 — Agent AI:** planned evidence-grounded document, policy, anomaly and human-review workflow.

> **Important safety and data statement:** The supplied workbook is preserved as a reference but is too small and insufficiently Indian-contextual for the stated study. Approach 1 therefore uses an explicitly labelled, reproducible **synthetic educational dataset**. Results are not operational insurance performance claims. A model score is a screening recommendation for trained human review—never an automatic claim denial, settlement, pricing decision or diagnosis.

## Repository guide

| Location | Purpose |
|---|---|
| `GOAL.md` | Scope, order of work, definition of done and quality gates for all approaches. |
| `data/raw/` | Unmodified copy of the supplied workbook plus source adequacy/checksum audit. |
| `data/synthetic/` | Deterministically generated Indian-context synthetic claim population. |
| `data/processed/` | Duplicate-cleaned stratified train/validation/test partitions and quality report. |
| `src/` | Modular, typed pipeline: loading, preprocessing, feature engineering, modelling, evaluation, visualization and reporting. |
| `models/` | Serialized preprocessing/selector/model artifacts and machine-readable model metadata. |
| `evaluation/` | Benchmark, held-out predictions, sampling comparison, fairness, calibration, paired tests and verification manifest. |
| `documentation/` | Data dictionary and methodology/results narrative generated from the run. |
| `visualizations/` | EDA, comparison, interpretability, fairness and technical diagrams. |
| `presentation/` | 20-slide Approach-1 academic defence deck. |
| `reports/` | IEEE-inspired Approach-1 research report PDF. |

## Quick start — Approach 1

Python 3.11 is recommended. Create an isolated environment, install the pinned packages, then run the complete pipeline:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.train --regenerate-data
```

This command will:

1. checksum and audit the original `Health Insurance Fraud Claims.xlsx` without changing it;
2. generate the documented synthetic fallback population (12,000 unique claims plus deliberate duplicate-quality-test rows);
3. remove exact duplicates, report missingness/outlier candidates, generate domain features and create a 70/15/15 stratified split;
4. fit all learned preprocessing on training data only;
5. compare requested imbalance strategies on validation data;
6. train/tune 15 classifiers with five-fold stratified CV and F2 scoring;
7. choose an operating threshold from validation data, then evaluate the untouched test split once;
8. write metrics, INR-cost proxy, fairness, calibration and significance analyses;
9. generate plots, a 20-slide PPTX, an IEEE-inspired report PDF and a verification manifest.

A quicker **development-only** run is available, but it deliberately uses three folds and reduced search breadth. Do not cite it in an academic report:

```bash
.venv/bin/python -m src.train --fast
```

## Current verified Approach-1 run

The full five-fold run is recorded in `evaluation/verification_manifest.json`. The F2-leading test-set model is **XGBoost**. Exact metrics, thresholds, confusion counts, latency, model size and hyperparameters are in `evaluation/benchmark_results.csv`; do not rely on a copied value when the pipeline has been rerun.

The generated academic artefacts are:

- `documentation/approach_1_traditional_ml_documentation.md`
- `evaluation/approach_1_evaluation_report.md`
- `presentation/approach_1_traditional_ml_presentation.pptx`
- `reports/approach_1_ieee_style_research_report.pdf`

## Data decision and reproducibility

The original workbook contains 4,500 rows, below the required 10,000-record threshold, generic/non-Indian locations and no policy duration/waiting-period/co-pay/history fields. It is therefore not silently relabelled or treated as a real Indian insurer dataset. `src.data_loading.generate_indian_synthetic_claims` records every modelling assumption: INR amounts are right-skewed; policy types include individual/family floater/employer group/Ayushman Bharat/ECHS; providers vary by state/city/tier; and labels are probabilistic rather than a deterministic feature rule.

The fixed seed, package pins, source checksum, data metadata, split sizes, selected features, best hyperparameters and output validation all live in generated metadata. Train-only transformations are serialized to support consistent future synthetic claim scoring.

## Evaluation protocol

- **Selection metric:** F2, to prioritise fraud recall while retaining precision reporting.
- **Cross-validation:** five-fold stratified CV on training partition for every model.
- **Split:** 70% train, 15% validation and 15% held-out test, stratified by synthetic fraud label.
- **Threshold:** validation-only F2 scan; no universal 0.50 assumption.
- **Metrics:** accuracy, precision, recall, F1, F2, ROC-AUC, PR-AUC, MCC, Brier score, confusion counts, training time, inference latency and artifact size.
- **Cost illustration:** false-negative cost equals the associated synthetic claim amount; false-positive cost is an explicit ₹3,500 review/friction proxy. It is not a real insurer cost model.
- **Fairness audit:** accuracy/FPR/FNR/precision/recall/selection rate by gender, age band, state, income band and treatment type. Small groups are flagged.
- **Statistical tests:** McNemar exact comparisons on paired test correctness and Wilcoxon tests on CV F2 folds. Synthetic data limits their external interpretation.

## Responsible use

This project is educational. Before any live use, it would require independently labelled and legally governed data, external and temporal validation, clinical/policy expertise, data-protection controls, calibrated thresholds, fairness monitoring, drift monitoring, secured access, explanation/appeal procedures and accountable human claim-review authority. Demographic characteristics must never become shortcut reasons for adverse decisions.

## Next work

The traditional baseline is deliberately first. Approach 2 must reuse its frozen population and partitions for a fair neural comparison. Approach 3 must be offline-first and auditable, using only synthetic/demo data unless a secure, authorised environment is supplied. See `GOAL.md` for the full cross-approach delivery charter.
