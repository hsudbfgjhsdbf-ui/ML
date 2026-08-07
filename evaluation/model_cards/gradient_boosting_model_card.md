# Model card — Gradient boosting

- **Key:** `gradient_boosting`
- **Family:** `boosting`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `random`; trials recorded: `4`
- **Parameters:** `{"learning_rate": 0.1, "max_depth": 2, "n_estimators": 140}`
- **Validation F2:** `1.0000`
- **Validation PR-AUC:** `1.0000`
- **Test F2:** `0.9799`
- **Training seconds:** `6.831`
- **Threshold:** `0.9900`

## Intended use

This estimator is a comparative fraud-screening baseline for the supplied
academic snapshot. It should prioritize human review and must not make an
automatic denial decision.

## Caveats

 The source workbook is
small and lacks policy/document context. The metrics are valid only under the
run's split and preprocessing contract.

## Evidence

- Metrics: `evaluation/metrics/gradient_boosting_metrics.json`
- Search: `evaluation/tuning/gradient_boosting_search.csv`
- Curves: `evaluation/curves/gradient_boosting_roc.csv` and `_pr.csv`
