# Model card — Bernoulli naive Bayes

- **Key:** `bernoulli_nb`
- **Family:** `probabilistic`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `grid`; trials recorded: `3`
- **Parameters:** `{"alpha": 5.0}`
- **Validation F2:** `0.8032`
- **Validation PR-AUC:** `0.4821`
- **Test F2:** `0.7937`
- **Training seconds:** `0.175`
- **Threshold:** `0.9700`

## Intended use

This estimator is a comparative fraud-screening baseline for the supplied
academic snapshot. It should prioritize human review and must not make an
automatic denial decision.

## Caveats

 The source workbook is
small and lacks policy/document context. The metrics are valid only under the
run's split and preprocessing contract.

## Evidence

- Metrics: `evaluation/metrics/bernoulli_nb_metrics.json`
- Search: `evaluation/tuning/bernoulli_nb_search.csv`
- Curves: `evaluation/curves/bernoulli_nb_roc.csv` and `_pr.csv`
