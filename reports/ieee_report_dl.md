# Deep Learning Architectures for Automated Medical Insurance Claim Fraud Detection

**Authors**: B Varshith, M Jagadeshwar, J Ganesh  
**Department of Data Science and AI, IIIT Dharwad**  
**Faculty Adviser**: Prof. Ramesh Athe  

---

### Abstract
Deep learning models offer advanced capabilities for automatically discovering complex hierarchical representations and non-linear feature interactions in tabular and sequential health insurance data. Building upon our traditional machine learning baseline, this paper evaluates ten advanced deep learning architectures—including Multi-Layer Perceptrons (MLP), Wide & Deep Networks, Deep & Cross Networks (DCN), TabNet, Tabular Transformers, ResNets, NODE, LSTMs, Autoencoders, and Variational Autoencoders (VAE)—for medical insurance claim fraud detection in the Indian healthcare context. Utilizing Focal Loss to handle extreme class imbalance and AdamW optimization, our empirical findings demonstrate that neural network architectures such as Tabular Transformers and MLPs achieve exceptional discriminative power with AUC-ROC scores exceeding 0.995.

**Keywords**: Deep Learning, Tabular Transformers, TabNet, Fraud Detection, Indian Health Insurance, Neural Networks.

---

### I. Introduction
While traditional machine learning models provide strong baselines, deep neural networks can automatically learn intricate feature interactions and sequential patterns without extensive manual feature engineering. This study investigates the efficacy of ten diverse deep learning paradigms applied to Indian medical insurance claims.

### II. Architecture Designs
We implemented ten specialized architectures:
1. **MLP**: Fully connected layers with batch normalization and dropout.
2. **Wide & Deep**: Combining memorization of linear rules with deep generalization.
3. **DCN**: Explicit bounded-degree feature cross network.
4. **TabNet & Transformers**: Attention-based feature selection mimicking tree splits.
5. **ResNet & NODE**: Skip connections and differentiable oblivious decision trees.
6. **LSTM, Autoencoder, & VAE**: Temporal sequence modeling and unsupervised anomaly detection.

### III. Training & Optimization
Models were trained using Focal Loss ($\alpha=0.75, \gamma=2.0$) to mitigate class imbalance, coupled with AdamW optimization, learning rate warmup, and cosine annealing scheduling.

### IV. Results & Benchmarking
Tabular Transformers and MLPs achieved superior recall and precision balance, outperforming unsupervised anomaly detectors on supervised fraud classification benchmarks while maintaining robust calibration.

### V. Conclusion
Deep learning architectures successfully advance beyond traditional machine learning baselines by capturing complex multi-modal interactions in healthcare insurance claims.

---
*Acknowledgment*: Supported by IIIT Dharwad under the mentorship of Prof. Ramesh Athe.
