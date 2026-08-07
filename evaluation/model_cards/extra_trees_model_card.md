# Model card — Extra trees

- **Key:** `extra_trees`
- **Family:** `bagging`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `random`; trials recorded: `4`
- **Parameters:** `{"max_depth": null, "min_samples_leaf": 1, "n_estimators": 140}`
- **Validation F2:** `0.9709`
- **Validation PR-AUC:** `0.9874`
- **Test F2:** `0.9701`
- **Training seconds:** `2.564`
- **Threshold:** `0.4000`

## Intended use

This estimator is a comparative fraud-screening baseline for the supplied
academic snapshot. It should prioritize human review and must not make an
automatic denial decision.

## Caveats

 The source workbook is
small and lacks policy/document context. The metrics are valid only under the
run's split and preprocessing contract.

## Evidence

- Metrics: `evaluation/metrics/extra_trees_metrics.json`
- Search: `evaluation/tuning/extra_trees_search.csv`
- Curves: `evaluation/curves/extra_trees_roc.csv` and `_pr.csv`
