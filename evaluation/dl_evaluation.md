# Approach 2 - Deep Learning Benchmarking Report

## Overview

- 10 neural architectures, same data/test set as Approach 1.
- Loss: weighted BCE; epochs=60; batch=128.
- Autoencoder/VAE are one-class detectors trained on legitimate claims only.
- Best ranked by F2 score.

## Benchmarking Table

| Architecture | Acc | Prec | Recall | F1 | **F2** | AUC-ROC | AUC-PR | MCC | Params | Train(s) | Pred(ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WideAndDeep | 0.987 | 0.844 | 0.950 | 0.894 | **0.927** | 0.998 | 0.966 | 0.889 | 17,010 | 4.1 | 0.002 |
| Transformer | 0.981 | 0.765 | 0.975 | 0.857 | **0.924** | 0.997 | 0.948 | 0.854 | 103,297 | 4.7 | 0.008 |
| DCN | 0.979 | 0.760 | 0.950 | 0.844 | **0.905** | 0.996 | 0.951 | 0.839 | 21,664 | 2.3 | 0.002 |
| TabNet | 0.963 | 0.615 | 1.000 | 0.762 | **0.889** | 0.996 | 0.937 | 0.769 | 13,936 | 1.8 | 0.004 |
| MLP | 0.975 | 0.725 | 0.925 | 0.813 | **0.877** | 0.997 | 0.965 | 0.807 | 54,401 | 3.8 | 0.003 |
| ResNet | 0.966 | 0.644 | 0.950 | 0.768 | **0.868** | 0.991 | 0.878 | 0.767 | 106,881 | 3.7 | 0.005 |
| NODE | 0.960 | 0.607 | 0.925 | 0.733 | **0.837** | 0.987 | 0.832 | 0.731 | 3,233 | 2.6 | 0.003 |
| LSTM | 0.956 | 0.578 | 0.925 | 0.712 | **0.826** | 0.988 | 0.842 | 0.711 | 10,401 | 2.0 | 0.002 |
| Autoencoder | 0.890 | 0.302 | 0.650 | 0.413 | **0.528** | 0.901 | 0.475 | 0.393 | 8,255 | 0.6 | 0.001 |
| VAE | 0.884 | 0.289 | 0.650 | 0.400 | **0.520** | 0.896 | 0.462 | 0.381 | 9,295 | 0.8 | 0.001 |

## Ranking (by F2)

1. **WideAndDeep** — F2=0.9268, ROC-AUC=0.9977
2. **Transformer** — F2=0.9242, ROC-AUC=0.9967
3. **DCN** — F2=0.9048, ROC-AUC=0.9957
4. **TabNet** — F2=0.8889, ROC-AUC=0.9962
5. **MLP** — F2=0.8768, ROC-AUC=0.9969
6. **ResNet** — F2=0.8676, ROC-AUC=0.9909
7. **NODE** — F2=0.8371, ROC-AUC=0.9869
8. **LSTM** — F2=0.8259, ROC-AUC=0.9876
9. **Autoencoder** — F2=0.5285, ROC-AUC=0.9014
10. **VAE** — F2=0.5200, ROC-AUC=0.8960

## Bootstrap 95% Confidence Intervals (AUC)

| Architecture | AUC-ROC (95% CI) | AUC-PR (95% CI) |
|---|---|---|
| WideAndDeep | 0.998 [0.995,1.000] | 0.965 [0.921,0.995] |
| Transformer | 0.997 [0.993,0.999] | 0.947 [0.884,0.987] |
| DCN | 0.996 [0.991,0.999] | 0.949 [0.896,0.985] |
| TabNet | 0.996 [0.992,0.999] | 0.938 [0.869,0.985] |
| MLP | 0.997 [0.994,0.999] | 0.965 [0.924,0.993] |
| ResNet | 0.991 [0.985,0.996] | 0.876 [0.778,0.942] |
| NODE | 0.987 [0.979,0.994] | 0.830 [0.708,0.927] |
| LSTM | 0.988 [0.980,0.994] | 0.840 [0.729,0.931] |
| Autoencoder | 0.900 [0.857,0.939] | 0.488 [0.326,0.650] |
| VAE | 0.894 [0.851,0.935] | 0.475 [0.313,0.630] |

## Notes

- Wide & Deep and Transformer are the strongest deep classifiers, closely matching the best tree-based ML baseline while learning features automatically.
- Autoencoder/VAE (one-class anomaly detectors) reach AUC-ROC ≈ 0.90, confirming they capture the legitimate distribution; reconstruction error is a softer signal than a supervised boundary, hence lower recall-oriented F2.
- See visualizations/dl/ for learning curves, ROC/PR, t-SNE and SHAP plots.