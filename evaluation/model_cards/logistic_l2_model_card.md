# Model card — Logistic regression (L2)

- **Key:** `logistic_l2`
- **Family:** `linear`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `grid`; trials recorded: `3`
- **Parameters:** `{"C": 10.0}`
- **Validation F2:** `0.9479`
- **Validation PR-AUC:** `0.9747`
- **Test F2:** `0.9615`
- **Training seconds:** `0.995`
- **Threshold:** `0.5300`

## Intended use

This estimator is a comparative fraud-screening baseline for the supplied
academic snapshot. It should prioritize human review and must not make an
automatic denial decision.

## Caveats

 The source workbook is
small and lacks policy/document context. The metrics are valid only under the
run's split and preprocessing contract.

## Evidence

- Metrics: `evaluation/metrics/logistic_l2_metrics.json`
- Search: `evaluation/tuning/logistic_l2_search.csv`
- Curves: `evaluation/curves/logistic_l2_roc.csv` and `_pr.csv`
