# Limitations and future work

**Run:** `run_20260807_151423`.  

The dataset is small relative to the requested ten-thousand-plus target, single
row per unique patient/provider in the supplied snapshot, and has a six-percent
fraud prevalence. It lacks policy, hospital-tier, sum-insured, treatment-cost,
document, and verified Indian geography fields. The observed cluster and
financial separation may be synthetic shortcuts. The work therefore supports a
reproducible baseline and a software submission, not production underwriting.

Future work should: (1) obtain a license-cleared, claim-level Indian dataset;
(2) add temporal history and provider aggregates with strict prior windows;
(3) validate with external time and geography splits; (4) compare optional
XGBoost/LightGBM/CatBoost adapters; (5) add SHAP and counterfactual audits;
(6) calibrate on a truly independent calibration set; (7) conduct fairness
analysis with sufficient positive examples; and (8) integrate the later deep
learning and document/agent approaches without changing the frozen test set.
