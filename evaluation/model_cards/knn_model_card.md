# Model card — K-nearest neighbors

- **Key:** `knn`
- **Family:** `instance`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `grid`; trials recorded: `6`
- **Parameters:** `{"n_neighbors": 5, "weights": "distance"}`
- **Validation F2:** `0.8520`
- **Validation PR-AUC:** `0.8219`
- **Test F2:** `0.8028`
- **Training seconds:** `0.339`
- **Threshold:** `0.3700`

## Intended use

This estimator is a comparative fraud-screening baseline for the supplied
academic snapshot. It should prioritize human review and must not make an
automatic denial decision.

## Caveats

 The source workbook is
small and lacks policy/document context. The metrics are valid only under the
run's split and preprocessing contract.

## Evidence

- Metrics: `evaluation/metrics/knn_metrics.json`
- Search: `evaluation/tuning/knn_search.csv`
- Curves: `evaluation/curves/knn_roc.csv` and `_pr.csv`
