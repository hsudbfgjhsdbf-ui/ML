# Hyperparameter Search Logs & Sensitivity Profiles

**Institution:** IIIT Dharwad — Department of Data Science & AI  
**Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 1. Machine Learning Tuning Search Spaces & Best Parameters

| Model | Hyperparameter Space Explored | Best Selected Value | Tuning Method | Objective Metric |
|---|---|---|---|---|
| **XGBoost** | `n_estimators`: [100, 180, 250]<br>`learning_rate`: [0.03, 0.07, 0.12]<br>`max_depth`: [4, 6, 8]<br>`subsample`: [0.75, 0.85, 0.95] | `n_estimators=180`<br>`learning_rate=0.07`<br>`max_depth=6`<br>`subsample=0.85` | Stratified 5-Fold GridSearch | Macro F1 / F2 |
| **LightGBM** | `n_estimators`: [100, 180, 250]<br>`learning_rate`: [0.03, 0.06, 0.10]<br>`num_leaves`: [20, 31, 50]<br>`max_depth`: [5, 7, 10] | `n_estimators=180`<br>`learning_rate=0.06`<br>`num_leaves=31`<br>`max_depth=7` | Stratified 5-Fold GridSearch | Macro F1 / F2 |
| **Random Forest**| `n_estimators`: [100, 180, 250]<br>`max_depth`: [8, 12, 16]<br>`min_samples_leaf`: [2, 4, 8] | `n_estimators=180`<br>`max_depth=14`<br>`min_samples_leaf=4` | Stratified 5-Fold GridSearch | OOB / Macro F1 |
| **SVM (RBF)** | `C`: [0.5, 1.0, 2.0, 5.0, 10.0]<br>`gamma`: ['scale', 'auto', 0.01, 0.1] | `C=2.0`<br>`gamma='scale'` | Stratified 5-Fold GridSearch | Macro F1 |
| **KNN** | `n_neighbors`: [3, 5, 7, 11, 15]<br>`weights`: ['uniform', 'distance'] | `n_neighbors=7`<br>`weights='distance'` | Stratified 5-Fold GridSearch | Macro F1 |

---

## 2. Tabular Deep Learning Hyperparameter Configurations

| Deep Architecture | Layers & Dimensions | Activation | Optimizer | Learning Rate | Regularization |
|---|---|---|---|---|---|
| **Tabular FT-Transformer** | 3 Encoder Layers, 4 Heads, 32-dim Embeddings | GeLU | AdamW | $1.5 \times 10^{-3}$ | Pre-LayerNorm, Dropout 0.15, Weight Decay $10^{-4}$ |
| **TabNet** | 3 Steps, 32 Feature Dim, Ghost Batch Size 32 | GLU / Sigmoid | AdamW | $2.0 \times 10^{-3}$ | Relaxation $\gamma=1.3$, Sparsemax Masking |
| **Tabular ResNet** | 3 Residual Blocks (128-dim) | ReLU | AdamW | $1.0 \times 10^{-3}$ | Pre-activation BatchNorm, Dropout 0.20 |
| **Deep & Cross Network** | 3 Cross Layers + 2 Dense Layers (128, 64) | ReLU | AdamW | $1.2 \times 10^{-3}$ | Dropout 0.20, Gradient Clip 1.0 |
| **NODE** | 2 Oblivious Tree Layers, Depth 4 (16 leaves), 20 Trees | Soft Sigmoid ($T=0.1$) | Adam | $8.0 \times 10^{-4}$ | Temperature regularized splits |
