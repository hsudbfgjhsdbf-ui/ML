# Model card — Decision tree

- **Key:** `decision_tree`
- **Family:** `tree`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `grid`; trials recorded: `12`
- **Parameters:** `{"max_depth": 3, "min_samples_leaf": 3}`
- **Validation F2:** `0.9804`
- **Validation PR-AUC:** `0.9771`
- **Test F2:** `0.9799`
- **Training seconds:** `0.368`
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

- Metrics: `evaluation/metrics/decision_tree_metrics.json`
- Search: `evaluation/tuning/decision_tree_search.csv`
- Curves: `evaluation/curves/decision_tree_roc.csv` and `_pr.csv`
