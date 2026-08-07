# Model card — Quadratic discriminant analysis

- **Key:** `qda`
- **Family:** `probabilistic`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `grid`; trials recorded: `4`
- **Parameters:** `{"reg_param": 0.0}`
- **Validation F2:** `0.8952`
- **Validation PR-AUC:** `0.7000`
- **Test F2:** `0.8547`
- **Training seconds:** `0.217`
- **Threshold:** `0.0800`

## Intended use

This estimator is a comparative fraud-screening baseline for the supplied
academic snapshot. It should prioritize human review and must not make an
automatic denial decision.

## Caveats

 The source workbook is
small and lacks policy/document context. The metrics are valid only under the
run's split and preprocessing contract.

## Evidence

- Metrics: `evaluation/metrics/qda_metrics.json`
- Search: `evaluation/tuning/qda_search.csv`
- Curves: `evaluation/curves/qda_roc.csv` and `_pr.csv`
