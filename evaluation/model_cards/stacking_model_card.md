# Model card — Stacking ensemble

- **Key:** `stacking`
- **Family:** `ensemble`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `none`; trials recorded: `1`
- **Parameters:** `{}`
- **Validation F2:** `1.0000`
- **Validation PR-AUC:** `1.0000`
- **Test F2:** `0.9950`
- **Training seconds:** `1.261`
- **Threshold:** `0.5000`

## Intended use

This estimator is a comparative fraud-screening baseline for the supplied
academic snapshot. It should prioritize human review and must not make an
automatic denial decision.

## Caveats

Out-of-fold meta-learning; higher complexity than a single booster. The source workbook is
small and lacks policy/document context. The metrics are valid only under the
run's split and preprocessing contract.

## Evidence

- Metrics: `evaluation/metrics/stacking_metrics.json`
- Search: `evaluation/tuning/stacking_search.csv`
- Curves: `evaluation/curves/stacking_roc.csv` and `_pr.csv`
