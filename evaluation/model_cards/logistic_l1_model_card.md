# Model card — Logistic regression (L1)

- **Key:** `logistic_l1`
- **Family:** `linear`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `grid`; trials recorded: `3`
- **Parameters:** `{"C": 1.0}`
- **Validation F2:** `0.9662`
- **Validation PR-AUC:** `0.9775`
- **Test F2:** `0.9709`
- **Training seconds:** `1.911`
- **Threshold:** `0.6500`

## Intended use

This estimator is a comparative fraud-screening baseline for the supplied
academic snapshot. It should prioritize human review and must not make an
automatic denial decision.

## Caveats

 The source workbook is
small and lacks policy/document context. The metrics are valid only under the
run's split and preprocessing contract.

## Evidence

- Metrics: `evaluation/metrics/logistic_l1_metrics.json`
- Search: `evaluation/tuning/logistic_l1_search.csv`
- Curves: `evaluation/curves/logistic_l1_roc.csv` and `_pr.csv`
