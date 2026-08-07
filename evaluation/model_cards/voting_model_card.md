# Model card — Soft voting ensemble

- **Key:** `voting`
- **Family:** `ensemble`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `none`; trials recorded: `1`
- **Parameters:** `{}`
- **Validation F2:** `1.0000`
- **Validation PR-AUC:** `1.0000`
- **Test F2:** `0.9799`
- **Training seconds:** `0.459`
- **Threshold:** `0.4900`

## Intended use

This estimator is a comparative fraud-screening baseline for the supplied
academic snapshot. It should prioritize human review and must not make an
automatic denial decision.

## Caveats

Diversity demonstration; base estimators use frozen reference settings. The source workbook is
small and lacks policy/document context. The metrics are valid only under the
run's split and preprocessing contract.

## Evidence

- Metrics: `evaluation/metrics/voting_metrics.json`
- Search: `evaluation/tuning/voting_search.csv`
- Curves: `evaluation/curves/voting_roc.csv` and `_pr.csv`
