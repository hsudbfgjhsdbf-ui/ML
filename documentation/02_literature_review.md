# Chapter 2: Literature Review and Related Work

**Institution:** IIIT Dharwad — Department of Data Science & AI  
**Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 2.1 Critical Survey of Existing Literature

The literature on healthcare fraud detection spans three evolutionary epochs: statistical anomaly detection, classical machine learning ensembles, and modern deep tabular architectures with cognitive agents.

### 1. Gorishniy et al. (NeurIPS 2021) — *Revisiting Deep Learning Models for Tabular Data*
- **Methodology:** Introduced Feature Tokenizer Transformer (FT-Transformer) which transforms continuous and categorical variables into embeddings followed by multi-head self-attention.
- **Dataset:** Standard open-source tabular benchmarks (Higgs, Covertype, California Housing).
- **Key Findings:** FT-Transformer outperforms or matches top tree-based models on complex tabular datasets with rich feature interactions.
- **Limitation / Gap:** Evaluated only on generic datasets; does not consider asymmetric class imbalance losses like Focal Loss or Indian healthcare economics.

### 2. Arik & Pfister (AAAI 2021) — *TabNet: Attentive Interpretable Tabular Learning*
- **Methodology:** Sequential multi-step attention using sparsemax feature selection masks and Ghost Batch Normalization.
- **Key Findings:** Enables end-to-end gradient descent with instance-level feature selection masks.
- **Gap:** Computationally intensive for real-time high-throughput claim screening compared to gradient-boosted trees.

### 3. Lundberg & Lee (NeurIPS 2017) — *A Unified Approach to Interpreting Model Predictions (SHAP)*
- **Methodology:** Game-theoretic Shapley values providing additive feature attribution (TreeExplainer, DeepExplainer).
- **Key Findings:** Establishes local accuracy and consistency properties for complex machine learning models.
- **Gap:** Computes static feature attributions; cannot generate actionable natural language counterfactual recommendations.

### 4. Chen & Guestrin (ACM KDD 2016) — *XGBoost: A Scalable Tree Boosting System*
- **Methodology:** Regularized gradient boosting with second-order Taylor expansion and weighted quantile sketch.
- **Key Findings:** Dominant benchmark for structured tabular data.
- **Gap:** Requires explicit manual feature engineering to capture high-order non-linear interactions.

### 5. Lin et al. (IEEE TPAMI 2020) — *Focal Loss for Dense Object Detection*
- **Methodology:** Modulated cross-entropy loss $\text{FL}(p_t) = -\alpha (1-p_t)^\gamma \log(p_t)$ down-weighting well-classified easy examples.
- **Key Findings:** Dramatically resolves extreme class imbalance (1:1000).
- **Our Adaptation:** Successfully applied to tabular insurance fraud where legitimate claims comprise ~90% of samples.

### 6. Borisov et al. (IEEE TNNLS 2022) — *Deep Neural Networks and Tabular Data: A Survey*
- **Survey Findings:** Categorizes tabular deep architectures into differentiable trees (NODE), transformer embeddings, and regularization methods.
- **Gap:** Highlights the lack of domain-specific benchmarks incorporating financial cost matrices.

### 7. IRDAI Regulations & Circulars (2020–2023) — *Master Circular on Health Insurance Claims*
- **Regulatory Mandate:** Enforces a 30-day settlement mandate and requires written evidence-backed rejection clauses.
- **Our Contribution:** Direct grounding of our Multi-Agent RAG knowledge base in these statutory clauses.

### 8. NHA PM-JAY Anti-Fraud Guidelines (Govt of India, 2021)
- **Typologies Documented:** Hospital upcoding, phantom surgeries, and collusive doctor networks.
- **Our Contribution:** Injected these authentic Indian healthcare fraud topologies into our synthetic dataset and rulebook.

### 9. Hancock & Khoshgoftaar (Journal of Big Data 2020) — *CatBoost for Big Data*
- **Key Findings:** Effective ordered target encoding for high-cardinality categorical variables.

### 10. Baesens et al. (Decision Support Systems 2015) — *Benchmarking State-of-the-Art Classification Algorithms for Credit Scoring*
- **Findings:** Emphasizes that AUC and F-beta metrics must supersede raw accuracy in fraud contexts.

### 11–18. Additional Foundational Studies:
- **Popov et al. (ICLR 2020):** Neural Oblivious Decision Ensembles (NODE).
- **Wang et al. (ACM WebConf 2021):** DCN-V2 for improved cross network interactions.
- **Ribeiro et al. (KDD 2016):** LIME for local model-agnostic explanations.
- **Gal & Ghahramani (ICML 2016):** Dropout as a Bayesian approximation for uncertainty estimation.
- **Guo et al. (ICML 2017):** Temperature scaling on logits for probability calibration.
- **Kotsiantis et al. (2006):** Handling class imbalance in financial anomaly detection.
- **Chawla et al. (JAIR 2002):** SMOTE for synthetic minority oversampling.
- **He et al. (CVPR 2016):** Deep Residual Learning for pre-activation skip connections.

---

## 2.2 Comparative Literature Matrix

| Study / Reference | Core Methodology | Dataset Domain | Focus on Explainability | Asymmetric Cost Matrix | Indian Healthcare Context |
|---|---|---|---|---|---|
| Gorishniy et al. (2021) | FT-Transformer | Generic Tabular | Medium (Attention) | No | No |
| Arik & Pfister (2021) | TabNet | US Census / Forest | High (Masks) | No | No |
| Chen & Guestrin (2016) | XGBoost | Multi-domain | Low (Feature Gain) | Partial (Weights) | No |
| Baesens et al. (2015) | Classical ML Benchmarks| Credit / Banking | Low | Yes | No |
| NHA PM-JAY (2021) | Rule-based Audit | Indian Public Schemes | High (Manual) | Yes | **Yes** |
| **Our Study (IIIT Dharwad)**| **Tripartite ML + DL + Multi-Agent RAG** | **Indian Healthcare (12k+ Records)** | **Ultra-High (Bilingual XAI & SHAP)** | **Yes (₹ INR Optimization)** | **Yes (16 States & Tiers)** |
