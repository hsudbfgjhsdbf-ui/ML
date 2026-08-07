# Model card — Passive-aggressive classifier

- **Key:** `passive_aggressive`
- **Family:** `online_linear`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `grid`; trials recorded: `3`
- **Parameters:** `{"estimator__C": 0.1}`
- **Validation F2:** `0.9479`
- **Validation PR-AUC:** `0.9524`
- **Test F2:** `0.9198`
- **Training seconds:** `0.294`
- **Threshold:** `0.2700`

## Intended use

This estimator is a comparative fraud-screening baseline for the supplied
academic snapshot. It should prioritize human review and must not make an
automatic denial decision.

## Caveats

 The source workbook is
small and lacks policy/document context. The metrics are valid only under the
run's split and preprocessing contract.

## Evidence

- Metrics: `evaluation/metrics/passive_aggressive_metrics.json`
- Search: `evaluation/tuning/passive_aggressive_search.csv`
- Curves: `evaluation/curves/passive_aggressive_roc.csv` and `_pr.csv`
