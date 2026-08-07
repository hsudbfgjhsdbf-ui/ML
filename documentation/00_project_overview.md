# Medical insurance claim fraud detection — Approach 1

**Purpose:** reproducible traditional machine-learning baseline for screening supplied claim records.  
**Owner:** B Varshith, M Jagadeshwar, and J Ganesh.  
**Faculty adviser:** Prof. Ramesh Athe.  
**Institution:** IIIT Dharwad, Department of Data Science and AI.  
**Generated:** 07-08-2026 15:14:23 UTC.  
**Run identifier:** `run_20260807_151423`.  

## Executive result

The validation-only selection policy ranks complete models by F2 score and uses
PR-AUC as the first tie-breaker. The selected model is **Soft voting ensemble**. This
is a dataset- and protocol-specific result, not evidence that one algorithm is
universally best. The supplied workbook contains **4,500** claims,
with **270** fraud rows (6.0%) and
**4,230** legitimate rows.

| Item | Value |
| --- | --- |
| Source file | `data/raw/health_insurance_fraud_claims.xlsx` |
| Claims | 4,500 |
| Source columns | 19 |
| Fraud prevalence | 6.0% |
| Split | 70% train / 15% validation / 15% test |
| Primary selection metric | Validation F2; PR-AUC tie-breaker |
| Operating decision | approve / flag / reject triage; human adjudication remains mandatory |
| Run | `run_20260807_151423` |

## Reading path

1. [Project goals](../goal.md) — acceptance checklist and milestone plan.
2. [Dataset card](../data/dataset_card.md) — provenance, scope, and limitations.
3. [Methodology](methodology.md) — data flow, leakage controls, and model protocol.
4. [EDA report](eda_report.md) — figure index and observations before training.
5. [Feature engineering](feature_engineering.md) — formulas and exclusion decisions.
6. [Models](models.md) — algorithm families and search spaces.
7. [Explainability](explainability.md) — global, local, calibration, and reason templates.
8. [Fairness](fairness.md) — demographic slice audit and limitations.
9. [Reproduction](reproduction.md) — exact commands and artifact checks.
10. [Evaluation hub](../evaluation/evaluation.md) — single source of comparative results.

## Responsible-use boundary

This project is an academic decision-support baseline. It must not be used to
automatically deny claims, infer claimant intent, or replace a licensed claims
professional. The model flags records for proportionate review; a claimant must
receive a specific explanation, an opportunity to correct documentation, and an
appeal or grievance route. Sensitive fields are audited for fairness but are
not included as direct model identifiers. The workbook is a supplied snapshot,
not a verified national Indian insurance population.

## Artifact principle

Numbers in this file are generated from evaluation artifacts. Result figures are
saved under `images/`, tables under `evaluation/`, and model state under
`artifacts/models/`. If the workbook, configuration, or code changes, rerun the
pipeline and treat the previous run as historical rather than editing it.

_Last updated: 07-08-2026 15:14:23 UTC_
