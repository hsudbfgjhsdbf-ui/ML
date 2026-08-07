# Model card — Majority-class baseline

- **Key:** `majority`
- **Family:** `baseline`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `none`; trials recorded: `1`
- **Parameters:** `{}`
- **Validation F2:** `0.2443`
- **Validation PR-AUC:** `0.0607`
- **Test F2:** `0.2395`
- **Training seconds:** `0.001`
- **Threshold:** `0.0500`

## Intended use

This estimator is a comparative fraud-screening baseline for the supplied
academic snapshot. It should prioritize human review and must not make an
automatic denial decision.

## Caveats

Sanity floor; predicts no fraud at the default threshold. The source workbook is
small and lacks policy/document context. The metrics are valid only under the
run's split and preprocessing contract.

## Evidence

- Metrics: `evaluation/metrics/majority_metrics.json`
- Search: `evaluation/tuning/majority_search.csv`
- Curves: `evaluation/curves/majority_roc.csv` and `_pr.csv`
