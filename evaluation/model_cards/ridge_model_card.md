# Model card — Calibrated ridge classifier

- **Key:** `ridge`
- **Family:** `linear`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `grid`; trials recorded: `3`
- **Parameters:** `{"estimator__alpha": 1.0}`
- **Validation F2:** `0.9259`
- **Validation PR-AUC:** `0.9501`
- **Test F2:** `0.9242`
- **Training seconds:** `0.269`
- **Threshold:** `0.2600`

## Intended use

This estimator is a comparative fraud-screening baseline for the supplied
academic snapshot. It should prioritize human review and must not make an
automatic denial decision.

## Caveats

 The source workbook is
small and lacks policy/document context. The metrics are valid only under the
run's split and preprocessing contract.

## Evidence

- Metrics: `evaluation/metrics/ridge_metrics.json`
- Search: `evaluation/tuning/ridge_search.csv`
- Curves: `evaluation/curves/ridge_roc.csv` and `_pr.csv`
