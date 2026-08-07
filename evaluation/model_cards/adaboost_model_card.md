# Model card — AdaBoost

- **Key:** `adaboost`
- **Family:** `boosting`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `random`; trials recorded: `4`
- **Parameters:** `{"learning_rate": 0.05, "n_estimators": 80}`
- **Validation F2:** `1.0000`
- **Validation PR-AUC:** `1.0000`
- **Test F2:** `0.9799`
- **Training seconds:** `4.523`
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

- Metrics: `evaluation/metrics/adaboost_metrics.json`
- Search: `evaluation/tuning/adaboost_search.csv`
- Curves: `evaluation/curves/adaboost_roc.csv` and `_pr.csv`
