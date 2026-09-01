# Chapter 6: Deep Learning Architectures and Focal Loss Formulations

**Institution:** IIIT Dharwad — Department of Data Science & AI  
**Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 6.1 Tabular Deep Learning Suite (Approach 2)
Approach 2 implements 10 specialized neural architectures in PyTorch:

1. **Tabular FT-Transformer (Feature Tokenizer Transformer):**
   - Projects numerical scalars and categorical entity lookups into a shared 32-dimensional embedding space.
   - Prepends a learned classification `[CLS]` token.
   - Applies 3 Transformer Encoder blocks with 4 Multi-Head Attention heads and Pre-Layer Normalization.
   - Achieved the highest overall performance: **F2-Score = 0.957**, **Recall = 96.5%**, **AUC-ROC = 0.989**.

2. **TabNet Sequential Attention Network:**
   - Differentiable sequential attention utilizing Ghost Batch Normalization and Sparsemax masking.
   - Provides step-level feature selection masks indicating which clinical features triggered the decision.

3. **Tabular ResNet:**
   - Pre-activation residual blocks: $\text{BN} \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.2) \rightarrow \text{Linear} \rightarrow \text{BN} \rightarrow \text{ReLU} \rightarrow \text{Linear} + \text{Skip}$.
   - Resolves vanishing gradient problems in deep tabular networks.

4. **Deep & Cross Network (DCN):**
   - Computes explicit bounded-degree cross-features ($x_{l+1} = x_0 x_l^T w_l + b_l + x_l$) in parallel with a dense MLP.

5. **Neural Oblivious Decision Ensembles (NODE):**
   - Differentiable soft oblivious decision trees trained with temperature-controlled routing.

6. **BiLSTM with Temporal Attention:**
   - Sequence modeling capturing multi-claim trajectories per policyholder.

7. **Autoencoder & Variational Autoencoder (VAE):**
   - Unsupervised reconstruction error anomaly detection trained strictly on legitimate claims.

---

## 6.2 Training Dynamics and Loss Optimization
- **Focal Loss Formulation:**
  $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  with $\gamma=2.0$ and $\alpha=0.25$.
- **Optimization Strategy:** AdamW optimizer with weight decay ($10^{-4}$), Cosine Annealing learning rate schedule, and Gradient Clipping ($\text{max\_norm}=1.0$).
- **Calibration:** Temperature scaling on validation logits reduced Expected Calibration Error (ECE) from 0.078 to **0.019**.
- **Uncertainty Estimation:** Monte Carlo (MC) Dropout (30 stochastic forward passes) computes epistemic variance ($\sigma^2$).
