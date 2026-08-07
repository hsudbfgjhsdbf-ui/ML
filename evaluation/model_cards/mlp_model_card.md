# Model card — Multi-layer perceptron

- **Key:** `mlp`
- **Family:** `neural_baseline`
- **Run:** `run_20260807_151423`
- **Status:** `complete`
- **Search:** `random`; trials recorded: `4`
- **Parameters:** `{"alpha": 0.01, "hidden_layer_sizes": [64, 32], "learning_rate_init": 0.003}`
- **Validation F2:** `0.9615`
- **Validation PR-AUC:** `0.9755`
- **Test F2:** `0.9615`
- **Training seconds:** `1.459`
- **Threshold:** `0.2300`

## Intended use

This estimator is a comparative fraud-screening baseline for the supplied
academic snapshot. It should prioritize human review and must not make an
automatic denial decision.

## Caveats

Classical MLP baseline; deep-learning approach remains separate. The source workbook is
small and lacks policy/document context. The metrics are valid only under the
run's split and preprocessing contract.

## Evidence

- Metrics: `evaluation/metrics/mlp_metrics.json`
- Search: `evaluation/tuning/mlp_search.csv`
- Curves: `evaluation/curves/mlp_roc.csv` and `_pr.csv`
