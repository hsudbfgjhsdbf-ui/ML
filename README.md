# Medical Insurance Claim Fraud Detection

**Indian Healthcare & Insurance Context — IIIT Dharwad, Department of Data Science and AI**

A complete end-to-end project that detects fraudulent medical insurance claims
using **three complementary approaches**:

1. **Approach 1 — Traditional Machine Learning** (`run_ml_pipeline.py`)
   — 12 classical classifiers with preprocessing, feature engineering,
     class-imbalance handling, hyperparameter tuning, statistical evaluation,
     fairness analysis and rich visualisations.
2. **Approach 2 — Deep Learning** (`run_dl_pipeline.py`)
   — 10 neural architectures (MLP, Wide & Deep, DCN, TabNet, Transformer,
     ResNet, NODE, LSTM, Autoencoder, VAE) trained on the same data.
3. **Approach 3 — Agent AI Multi-Agent System** (`run_agent_pipeline.py`)
   — a coordinator orchestrating 5 specialised agents that produce an
     **explainable** verdict (Approved / Flagged / Rejected) for each claim.

The goal is a binary classification of each insurance claim as **Fraud** or
**Legitimate**, evaluated on real statistical metrics (Accuracy, Precision,
Recall, **F2**, AUC-ROC, AUC-PR, MCC).

---

## Dataset

`data/raw/Health Insurance Fraud Claims.xlsx` — 4,500 synthetic claims
reflecting Indian insurance patterns. Target = `ClaimLegitimacy`
(**Fraud = 6%**, Legitimate = 94% → imbalanced). See
[data/raw/data_dictionary.md](data/raw/data_dictionary.md).

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run everything (single command)

```bash
source .venv/bin/activate
python run_ml_pipeline.py      # Approach 1
python run_dl_pipeline.py      # Approach 2
python run_agent_pipeline.py   # Approach 3
python make_deliverables.py    # PPTs, PDF reports, research paper, images
```

The final `make_deliverables.py` bundles the generated results into:

| Deliverable | Location |
|---|---|
| PPT presentation (Approach 1) | `presentation/presentation_ml.pptx` |
| PPT presentation (Approach 2) | `presentation/presentation_dl.pptx` |
| PPT presentation (Approach 3) | `presentation/presentation_agent.pptx` |
| PDF report (Approach 1, IEEE) | `reports/report_ml.pdf` |
| PDF report (Approach 2, IEEE) | `reports/report_dl.pdf` |
| PDF report (Approach 3, IEEE) | `reports/report_agent.pdf` |
| Research paper (IEEE) | `reports/research_paper.pdf` |
| Benchmark reports | `evaluation/*.md` |
| Visualisations | `visualizations/**` |
| AI-generated images | `assets/` |

---

## Project structure

```
config/        YAML configuration for each approach
data/          raw + processed data and data dictionary
src/           source modules (ml, dl, agent, visualise, etc.)
notebooks/     exploratory notebooks
models/        saved models + checkpoints + results
evaluation/    benchmark markdown reports
visualizations/ generated plots (ml, dl, eda, agent)
docs/          detailed project documentation
reports/       PDF reports + research paper
presentation/  PPT files
assets/        AI-generated images
```

## Results snapshot

All models are evaluated on the same held-out test set (675 claims, 6% fraud),
ranked by **F2** (recall-oriented). Full tables are in `evaluation/*.md`.

**Approach 1 — Traditional ML** (best): Random Forest **F2=1.000**,
XGBoost F2=0.995, LightGBM/AdaBoost/GBM F2=0.980 (AUC-ROC ≥0.99 across models).

**Approach 2 — Deep Learning** (best): Wide & Deep **F2=0.927** (AUC-ROC 0.998),
Transformer F2=0.924, DCN F2=0.905, TabNet F2=0.889.
*Reconstruction-based anomaly models (Autoencoder/VAE) underperform on this
dataset — a documented negative result.*

**Approach 3 — Agent AI**: 5-agent pipeline on 675 held-out claims
→ Accuracy 0.548, Precision 0.079, Recall 0.625, F2 0.263 (heuristic
baseline; LLM-backed agents are the intended production path).

### Honesty note on data leakage
The source data contains near-identifier columns (`DiagnosisCode`,
`ProcedureCode`, `ProviderLocation`) and a `Cluster` feature that is a direct
target proxy (267/270 frauds in Cluster 1). These are **dropped** before
modelling so reported metrics reflect genuinely learnable signal
(see `data/raw/data_dictionary.md`).

## Team
- B Varshith
- M Jagadeshwar
- J Ganesh

**Faculty Adviser:** Ramesh Athe — IIIT Dharwad
