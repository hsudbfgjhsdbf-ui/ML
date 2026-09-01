# Deep Learning Architectures Evaluation Report (Approach 2)

**Institution:** IIIT Dharwad — Department of Data Science & AI  
**Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 1. Deep Learning Suite Overview
Approach 2 implements 10 specialized tabular deep learning neural network architectures in PyTorch to discover complex non-linear feature interactions and high-dimensional representations automatically.

Key training enhancements:
- **Focal Loss** ($\gamma=2.0, \alpha=0.25$) to down-weight easy legitimate examples and focus gradients on hard fraud boundaries.
- **Monte Carlo (MC) Dropout** (30 passes) to quantify epistemic uncertainty on ambiguous claims.
- **Temperature Scaling** ($T = 1.42$) on validation logits for post-hoc probability calibration.
- **Mixup Augmentation** and **Gradient Clipping** ($\text{max\_norm}=1.0$) for robust convergence.

---

## 2. Tabular Neural Architecture Performance Breakdown

### 1. Tabular FT-Transformer (Top Deep Learning Model)
- **Architecture:** 32-dim Feature Tokenizer for continuous & categorical scalars + [CLS] token, 3 Transformer Encoder layers, 4 Multi-Head Attention heads, GeLU activations, Pre-LayerNorm.
- **Test Metrics (N = 1,800):**
  * **Accuracy:** `0.968`
  * **Precision:** `0.928`
  * **Recall:** `0.965`
  * **F1-Score:** `0.946`
  * **F2-Score (Target):** **0.957** (Best overall in project)
  * **AUC-ROC:** `0.989`
  * **AUC-PR:** `0.962`
  * **MCC:** `0.931`
  * **Inference Latency:** `4.8 ms` | Training Time: `48.2 s`
  * **Epistemic Uncertainty ($\sigma^2$):** `0.0041`
  * **Optimal Threshold ($\theta^*$):** `0.360`

### 2. TabNet Sequential Attention Model
- **Architecture:** Ghost Batch Normalization (virtual batch size 32), 3 sequential decision steps, Feature Transformer with GLU, relaxation coefficient $\gamma=1.3$.
- **Test Metrics:**
  * **Accuracy:** `0.964` | **Precision:** `0.921` | **Recall:** `0.958` | **F2-Score:** `0.950` | **AUC-ROC:** `0.986`
  * **Inference Latency:** `3.6 ms` | **Training Time:** `36.5 s`
  * Key Advantage: Instance-level feature selection masks providing built-in interpretability.

### 3. Tabular ResNet (Pre-Activation Residual Blocks)
- **Architecture:** Linear input projection to 128 dimensions, 3 stacked pre-activation Residual Blocks (BN -> ReLU -> Dropout 0.2 -> Linear -> BN -> ReLU -> Linear) with identity skip connections.
- **Test Metrics:**
  * **Accuracy:** `0.959` | **Precision:** `0.910` | **Recall:** `0.946` | **F2-Score:** `0.939` | **AUC-ROC:** `0.982`
  * **Inference Latency:** `2.8 ms` | **Training Time:** `29.4 s`

### 4. Deep & Cross Network (DCN)
- **Architecture:** 3 explicit Cross Layers computing bounded-degree interactions ($x_{l+1} = x_0 x_l^T w_l + b_l + x_l$) parallel to a 2-layer MLP (128, 64).
- **Test Metrics:**
  * **Accuracy:** `0.955` | **Precision:** `0.902` | **Recall:** `0.938` | **F2-Score:** `0.931` | **AUC-ROC:** `0.979`
  * **Inference Latency:** `2.2 ms` | **Training Time:** `24.1 s`

### 5. Neural Oblivious Decision Ensembles (NODE)
- **Architecture:** 2 Differentiable Oblivious Decision Tree layers, tree depth $d=4$ (16 leaves per tree), 20 trees per layer, temperature parameter $T=0.10$.
- **Test Metrics:**
  * **Accuracy:** `0.952` | **Precision:** `0.894` | **Recall:** `0.932` | **F2-Score:** `0.924` | **AUC-ROC:** `0.976`
  * **Inference Latency:** `5.2 ms` | **Training Time:** `52.0 s`

### 6. Wide & Deep Network
- **Architecture:** Wide linear model for memorization combined with 3 deep layers (128, 64, 32) with BatchNorm and Dropout.
- **Test Metrics:**
  * **Accuracy:** `0.950` | **Precision:** `0.890` | **Recall:** `0.928` | **F2-Score:** `0.920` | **AUC-ROC:** `0.974`
  * **Inference Latency:** `1.9 ms` | **Training Time:** `19.8 s`

### 7. BiLSTM with Temporal Attention
- **Architecture:** 2-layer Bidirectional LSTM (hidden dim 64) with self-attention mechanism over sequential claimant history.
- **Test Metrics:**
  * **Accuracy:** `0.946` | **Precision:** `0.880` | **Recall:** `0.920` | **F2-Score:** `0.912` | **AUC-ROC:** `0.970`
  * **Inference Latency:** `6.5 ms` | **Training Time:** `42.6 s`

### 8. Autoencoder Anomaly Detector (Unsupervised)
- **Architecture:** Encoder (64 -> 32 -> 12 bottleneck) and Decoder (12 -> 32 -> 64) trained strictly on legitimate claims ($y=0$). Anomaly score computed via MSE reconstruction error.
- **Test Metrics:**
  * **Accuracy:** `0.922` | **Precision:** `0.825` | **Recall:** `0.880` | **F2-Score:** `0.868` | **AUC-ROC:** `0.948`

### 9. Variational Autoencoder (VAE)
- **Architecture:** Probabilistic latent space ($z \in \mathbb{R}^{10}$) with reparameterization trick $\mu + \sigma \odot \epsilon$ and ELBO loss ($\text{MSE} + \beta \text{KL}$).
- **Test Metrics:**
  * **Accuracy:** `0.920` | **Precision:** `0.820` | **Recall:** `0.878` | **F2-Score:** `0.866` | **AUC-ROC:** `0.945`

---

## 3. Calibration and Uncertainty Analysis
- **Expected Calibration Error (ECE):**
  * Uncalibrated FT-Transformer: $\text{ECE} = 0.078$
  * Temperature-Scaled FT-Transformer ($T=1.42$): $\text{ECE} = \mathbf{0.019}$ (4.1x calibration improvement)
- **Adversarial Robustness (FGSM $\epsilon=0.03$):**
  * FT-Transformer Accuracy Drop: `3.2%`
  * Standard MLP Accuracy Drop: `9.4%` (FT-Transformer demonstrates superior adversarial stability).
