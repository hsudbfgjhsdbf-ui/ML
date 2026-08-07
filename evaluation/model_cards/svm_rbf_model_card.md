# Model card — Support vector machine (RBF)

- **Key:** `svm_rbf`
- **Family:** `margin`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `random`; trials recorded: `4`
- **Parameters:** `{"C": 10.0, "gamma": 0.01}`
- **Validation F2:** `0.9804`
- **Validation PR-AUC:** `0.9820`
- **Test F2:** `0.9406`
- **Training seconds:** `1.295`
- **Threshold:** `0.4300`

## Intended use

This estimator is a comparative fraud-screening baseline for the supplied
academic snapshot. It should prioritize human review and must not make an
automatic denial decision.

## Caveats

Probability fitting adds runtime; the run records it explicitly. The source workbook is
small and lacks policy/document context. The metrics are valid only under the
run's split and preprocessing contract.

## Evidence

- Metrics: `evaluation/metrics/svm_rbf_metrics.json`
- Search: `evaluation/tuning/svm_rbf_search.csv`
- Curves: `evaluation/curves/svm_rbf_roc.csv` and `_pr.csv`
