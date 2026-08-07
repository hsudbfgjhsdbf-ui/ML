# Model card — Histogram gradient boosting

- **Key:** `hist_gradient_boosting`
- **Family:** `boosting`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `random`; trials recorded: `4`
- **Parameters:** `{"l2_regularization": 1.0, "learning_rate": 0.1, "max_iter": 180, "max_leaf_nodes": 15}`
- **Validation F2:** `1.0000`
- **Validation PR-AUC:** `1.0000`
- **Test F2:** `0.9799`
- **Training seconds:** `1.134`
- **Threshold:** `0.9700`

## Intended use

This estimator is a comparative fraud-screening baseline for the supplied
academic snapshot. It should prioritize human review and must not make an
automatic denial decision.

## Caveats

 The source workbook is
small and lacks policy/document context. The metrics are valid only under the
run's split and preprocessing contract.

## Evidence

- Metrics: `evaluation/metrics/hist_gradient_boosting_metrics.json`
- Search: `evaluation/tuning/hist_gradient_boosting_search.csv`
- Curves: `evaluation/curves/hist_gradient_boosting_roc.csv` and `_pr.csv`
