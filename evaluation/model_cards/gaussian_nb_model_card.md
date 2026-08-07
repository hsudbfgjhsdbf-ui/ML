# Model card — Gaussian naive Bayes

- **Key:** `gaussian_nb`
- **Family:** `probabilistic`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `grid`; trials recorded: `3`
- **Parameters:** `{"var_smoothing": 1e-07}`
- **Validation F2:** `0.8511`
- **Validation PR-AUC:** `0.6629`
- **Test F2:** `0.8368`
- **Training seconds:** `0.166`
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

- Metrics: `evaluation/metrics/gaussian_nb_metrics.json`
- Search: `evaluation/tuning/gaussian_nb_search.csv`
- Curves: `evaluation/curves/gaussian_nb_roc.csv` and `_pr.csv`
