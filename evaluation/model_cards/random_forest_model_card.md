# Model card — Random forest

- **Key:** `random_forest`
- **Family:** `bagging`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `random`; trials recorded: `4`
- **Parameters:** `{"max_depth": null, "min_samples_leaf": 1, "n_estimators": 220}`
- **Validation F2:** `1.0000`
- **Validation PR-AUC:** `1.0000`
- **Test F2:** `0.9950`
- **Training seconds:** `3.221`
- **Threshold:** `0.2000`

## Intended use

This estimator is a comparative fraud-screening baseline for the supplied
academic snapshot. It should prioritize human review and must not make an
automatic denial decision.

## Caveats

Out-of-bag estimate is retained when bootstrap is enabled. The source workbook is
small and lacks policy/document context. The metrics are valid only under the
run's split and preprocessing contract.

## Evidence

- Metrics: `evaluation/metrics/random_forest_metrics.json`
- Search: `evaluation/tuning/random_forest_search.csv`
- Curves: `evaluation/curves/random_forest_roc.csv` and `_pr.csv`
