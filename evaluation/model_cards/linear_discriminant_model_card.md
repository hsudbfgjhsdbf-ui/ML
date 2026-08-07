# Model card — Linear discriminant analysis

- **Key:** `linear_discriminant`
- **Family:** `probabilistic`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `grid`; trials recorded: `2`
- **Parameters:** `{"solver": "lsqr"}`
- **Validation F2:** `0.9569`
- **Validation PR-AUC:** `0.9133`
- **Test F2:** `0.9569`
- **Training seconds:** `0.189`
- **Threshold:** `0.8700`

## Intended use

This estimator is a comparative fraud-screening baseline for the supplied
academic snapshot. It should prioritize human review and must not make an
automatic denial decision.

## Caveats

 The source workbook is
small and lacks policy/document context. The metrics are valid only under the
run's split and preprocessing contract.

## Evidence

- Metrics: `evaluation/metrics/linear_discriminant_metrics.json`
- Search: `evaluation/tuning/linear_discriminant_search.csv`
- Curves: `evaluation/curves/linear_discriminant_roc.csv` and `_pr.csv`
