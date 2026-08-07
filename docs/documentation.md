# Medical Insurance Claim Fraud Detection — Project Documentation

**IIIT Dharwad · Department of Data Science and AI**
**Faculty Adviser:** Ramesh Athe · **Team:** B Varshith, M Jagadeshwar, J Ganesh

---

## 1. Introduction

Medical insurance claim fraud — submitting fabricated, inflated or otherwise
false claims — causes significant financial losses for Indian insurers and
raises premiums for genuine policyholders. This project develops an automated
system that classifies each claim as **Fraud** or **Legitimate**, evaluated
across three complementary approaches:

1. **Traditional Machine Learning** — interpretable classical models.
2. **Deep Learning** — neural architectures that learn feature hierarchies.
3. **Agent AI** — an explainable multi-agent system.

The scope is the **Indian healthcare & insurance landscape**: family-floater and
group policies, government schemes such as Ayushman Bharat, region and
hospital-tier cost variance, and all demographics (women, children, elderly,
differently-abled).

## 2. Dataset

`data/raw/Health Insurance Fraud Claims.xlsx` — 4,500 synthetic claims.
See `data/raw/data_dictionary.md` for the full feature description.

- **Target:** `ClaimLegitimacy` (`Fraud` = 6%, `Legitimate` = 94%).
- **Splits:** stratified 70 / 15 / 15 (train / validation / test).
- The dataset is imbalanced, motivating SMOTE and cost-sensitive evaluation.

## 3. Approach 1 — Traditional Machine Learning

`run_ml_pipeline.py` implements the full pipeline:

- **Preprocessing:** missing-value reporting, deduplication, IQR outlier
  flagging, one-hot encoding (low cardinality), target encoding (high
  cardinality), standard scaling.
- **Feature engineering:** temporal (year, month, weekday, weekend, season),
  ratios (`ClaimToIncome`, `ClaimAmountLog`), and interactions
  (`AmountPerAge`, `AgeXIncome`).
- **Class imbalance:** SMOTE oversampling on the training split only.
- **Algorithms (12):** Logistic Regression, Decision Tree, Random Forest,
  Gradient Boosting, XGBoost, LightGBM, SVM, KNN, Naive Bayes, Neural Network
  (MLP), AdaBoost, QDA.
- **Tuning:** stratified 5-fold CV, grid/randomised search optimising **F2**.
- **Evaluation:** accuracy, precision, recall, F1, F2, AUC-ROC, AUC-PR, MCC;
  optimal-threshold selection; McNemar's test; business-impact (INR);
  fairness / bias analysis.
- **Outputs:** `models/ml/*.pkl`, `evaluation/ml_evaluation.md`,
  `visualizations/ml/*.png`.

## 4. Approach 2 — Deep Learning

`run_dl_pipeline.py` trains ten neural architectures on the same data/test set:

- MLP, Wide & Deep, Deep & Cross Network (DCN), TabNet, Transformer, ResNet,
  NODE, LSTM, Autoencoder, VAE.
- Weighted-BCE / focal-style losses for imbalance; AdamW + cosine LR schedule;
  batch norm; dropout; gradient clipping; early stopping (patience 15).
- Autoencoder/VAE provide reconstruction-based anomaly detection.
- **Outputs:** `models/dl/`, `evaluation/dl_evaluation.md`,
  `visualizations/dl/*.png`, comparison with the Approach-1 baseline.

## 5. Approach 3 — Agent AI Multi-Agent System

`run_agent_pipeline.py` orchestrates a coordinator over five agents:

| Agent | Responsibility |
|-------|----------------|
| Eligibility | Completeness & plausibility of claim fields |
| Policy | Coverage, waiting-period and cost-baseline checks |
| Anomaly | Cost/age/temporal deviation detection |
| Historical | Provider-level & percentile risk signals |
| Reasoning | Synthesises findings into an explainable verdict |

Each agent emits structured findings (status, confidence, evidence). The
reasoning agent aggregates a risk score and returns **Approved / Flagged /
Rejected** with a natural-language explanation. The interfaces mirror an
LLM-backed (LangChain/LangGraph + Gemini) implementation and can be swapped in
when credentials are available.

- **Outputs:** `evaluation/agent_evaluation.md`, `reports/agent_sample_decision.json`.

## 6. Evaluation Metrics

| Metric | Meaning |
|--------|---------|
| Accuracy | Overall correctness (misleading when imbalanced) |
| Precision | Of flagged claims, how many are truly fraud |
| Recall | Of actual fraud, how many are caught |
| F1 | Harmonic mean of precision & recall |
| **F2** | Weighted toward recall (primary ranking criterion) |
| AUC-ROC | Discrimination across thresholds |
| AUC-PR | Precision-recall area (informative for rare fraud) |
| MCC | Balanced correlation measure |

## 7. Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_ml_pipeline.py
python run_dl_pipeline.py
python run_agent_pipeline.py
python make_deliverables.py
```

All random seeds are fixed; splits are stratified with `random_state=42`;
preprocessing is fitted on training data only to prevent leakage.
