# Deep Learning Approach

## Framework
sklearn_mlp_fallback

## Architecture
Hidden layers [128, 64, 32]
Dropout 0.3
## Results
Val PR-AUC 0.8728729314454975 test PR-AUC 0.9026157321644857
Threshold 0.05

## Observation
On tabular insurance fraud data with 4500 rows and 6% fraud, deep learning often underperforms tree-based models due to lack of inductive bias, limited data, and class imbalance. Tree ensembles handle categorical splits better. DL may still be useful with embeddings for high-cardinality codes and with larger data.
