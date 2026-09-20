# FORM 5: Traditional ML — Explainability, Visualization & Reporting Specification
## Medical Insurance Claim Fraud Detection — Approach 1 of 3
### IIIT Dharwad | B.Tech Data Science & AI

---

**Faculty Adviser:** Prof. Ramesh Athe  
**Team Members:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  
**Date:** 01-09-2026 | **Version:** 1.0

---

## 1. Purpose

This document specifies all explainability analyses, visualization requirements, and reporting deliverables for Approach 1. It ensures the model's decisions are transparent, interpretable, and communicated effectively through visual and narrative means.

## 2. Explainability Framework

### 2.1 Global Explainability (Model-Level)

#### 2.1.1 Feature Importance (Tree-Based Models)
- **Method:** Native Gini importance (gain-based) from tree models
- **Output:** Horizontal bar chart of top 25 features
- **Models:** Random Forest, Extra Trees, XGBoost, LightGBM, HistGradientBoosting
- **File:** `images/models/feature_importance_{model_key}.png`

#### 2.1.2 SHAP (SHapley Additive exPlanations) Analysis
- **Method:** TreeExplainer for tree models, KernelExplainer for others
- **Plots Required:**
  - SHAP Summary Plot (beeswarm) — shows feature impact direction
  - SHAP Bar Plot — mean absolute SHAP values
  - SHAP Dependence Plots — top 5 features showing non-linear effects
- **Files:**
  - `images/xai/shap_summary_{model_key}.png`
  - `images/xai/shap_bar_{model_key}.png`
  - `images/xai/shap_dependence_{feature}_{model_key}.png`

#### 2.1.3 SHAP Value Interpretation

For the best model, the SHAP analysis reveals:
- **Positive SHAP values** → push prediction toward fraud
- **Negative SHAP values** → push prediction toward legitimate
- **Feature direction:** Shows whether high/low values increase fraud risk

### 2.2 Local Explainability (Claim-Level)

#### 2.2.1 Individual Claim Explanations
Five claims selected for narrative explanation:
- 2 True Positives (correctly detected fraud)
- 1 True Negative (correctly approved legitimate)
- 1 False Positive (wrongly flagged legitimate)
- 1 False Negative (missed fraud)

#### 2.2.2 Explanation Template
```
CLAIM ID: CLM-XXXX
PREDICTION: [Fraud/Legitimate] with [XX.X%] confidence
TOP 3 DRIVERS:
  1. [Feature Name] = [Value] → contributes [+/−] [SHAP value]
  2. [Feature Name] = [Value] → contributes [+/−] [SHAP value]
  3. [Feature Name] = [Value] → contributes [+/−] [SHAP value]
INTERPRETATION:
  This claim was [flagged/approved] because [plain-language explanation].
NEXT STEPS:
  [For flagged claims: documents to provide, review timeline, contact info]
```

#### 2.2.3 Force Plots
- SHAP force plots for each of the 5 selected claims
- File: `images/xai/force_plot_{claim_id}.png`

### 2.3 Partial Dependence Plots (PDP)

For the top 8 features, generate PDPs showing how fraud probability changes as the feature varies:

| Feature | Expected Relationship |
|---------|----------------------|
| Cluster | Sharp spike at Cluster=1 |
| ClaimAmount | Increasing probability with higher amounts |
| ClaimAmountPatientIncome | Increasing probability at high ratios |
| PatientIncome | Decreasing probability at higher income |
| PatientAge | Possible U-shape (very young and very old) |
| ProviderSpecialty | Variation across specialties |
| ClaimType | Higher for Inpatient/Emergency |
| ClaimSubmissionMethod | Variation by method |

Files: `images/models/pdp_{feature}_{model_key}.png`

## 3. Visualization Requirements — Complete Inventory

### 3.1 EDA Visualizations (Set A: Distributions)

| ID | Plot | Description | File |
|----|------|-------------|------|
| A01 | Histogram | ClaimAmount distribution by class | `eda/claim_amount_distribution.png` |
| A02 | Histogram | PatientAge distribution by class | `eda/patient_age_distribution.png` |
| A03 | Histogram | PatientIncome distribution by class | `eda/patient_income_distribution.png` |
| A04 | Histogram | LogClaimAmount distribution | `eda/log_claim_amount_distribution.png` |
| A05 | Violin | ClaimAmount by ProviderSpecialty | `eda/claim_amount_by_specialty.png` |
| A06 | Scatter | ClaimAmount vs PatientIncome colored by fraud | `eda/claim_vs_income_scatter.png` |
| A07 | Bar | ClaimLegitimacy class distribution | `eda/class_distribution.png` |
| A08 | Bar | Cluster distribution by class | `eda/cluster_distribution.png` |
| A09 | Bar | PatientGender distribution | `eda/gender_distribution.png` |
| A10 | Bar | ClaimType distribution by class | `eda/claim_type_distribution.png` |
| A11 | Bar | ProviderSpecialty distribution | `eda/specialty_distribution.png` |
| A12 | Bar | ClaimSubmissionMethod distribution | `eda/submission_method_distribution.png` |
| A13 | Bar | PatientMaritalStatus distribution | `eda/marital_status_distribution.png` |
| A14 | Bar | PatientEmploymentStatus distribution | `eda/employment_distribution.png` |
| A15 | Box | ClaimAmount by class | `eda/claim_amount_boxplot.png` |
| A16 | Box | PatientAge by class | `eda/patient_age_boxplot.png` |
| A17 | Box | PatientIncome by class | `eda/patient_income_boxplot.png` |
| A18 | Pie | Class distribution pie chart | `eda/class_pie_chart.png` |
| A19 | Bar | ClaimAmount by ClaimType and class | `eda/claim_amount_by_type.png` |
| A20 | Bar | Fraud rate by ClaimMonth | `eda/fraud_rate_by_month.png` |
| A21 | Bar | Fraud rate by ClaimDayOfWeek | `eda/fraud_rate_by_dayofweek.png` |
| A22 | Bar | Fraud rate by AgeGroup | `eda/fraud_rate_by_age_group.png` |
| A23 | Bar | Fraud rate by IncomeGroup | `eda/fraud_rate_by_income_group.png` |
| A24 | Bar | Fraud rate by ClaimAmountGroup | `eda/fraud_rate_by_claim_amount_group.png` |

### 3.2 Relationship Visualizations (Set B)

| ID | Plot | Description | File |
|----|------|-------------|------|
| B01 | Heatmap | Correlation matrix (all numeric features) | `eda/correlation_heatmap.png` |
| B02 | Bar | Fraud rate by Cluster with counts | `eda/fraud_rate_by_cluster.png` |
| B03 | Bar | Fraud rate by ClaimAmount decile | `eda/fraud_rate_by_amount_decile.png` |
| B04 | Scatter | ClaimAmount vs Cluster colored by fraud | `eda/amount_vs_cluster_scatter.png` |
| B05 | Heatmap | Cross-tab: Cluster × ClaimType fraud rates | `eda/cluster_x_claimtype_heatmap.png` |
| B06 | Heatmap | Cross-tab: AgeGroup × ClaimType fraud rates | `eda/agegroup_x_claimtype_heatmap.png` |
| B07 | Bar | ClaimAmountPatientIncome by class | `eda/ratio_by_class.png` |
| B08 | Scatter | ClaimAmount vs PatientIncome (fraud highlighted) | `eda/amount_income_fraud_highlight.png` |
| B09 | Bar | Fraud rate by ProviderSpecialty | `eda/fraud_rate_by_specialty.png` |
| B10 | Bar | Fraud rate by ClaimSubmissionMethod | `eda/fraud_rate_by_submission.png` |

### 3.3 Model Performance Visualizations

| ID | Plot | Description | File |
|----|------|-------------|------|
| C01 | Overlay | ROC curves all models | `models/roc_all_models_overlay.png` |
| C02 | Overlay | PR curves all models | `models/pr_all_models_overlay.png` |
| C03 | Grouped Bar | All metrics comparison | `models/all_metrics_comparison.png` |
| C04 | Radar | Top 5 models radar chart | `models/radar_top5.png` |
| C05 | Scatter | Training time vs PR-AUC | `models/time_vs_performance.png` |
| C06 | Heatmap | Prediction correlation matrix | `models/prediction_correlation.png` |
| C07 | Bar | Feature importance (best model) | `models/feature_importance_best.png` |
| C08 | Confusion Matrix | Best model test set | `models/confusion_matrix_best.png` |

### 3.4 XAI Visualizations

| ID | Plot | Description | File |
|----|------|-------------|------|
| D01 | Beeswarm | SHAP summary (best model) | `xai/shap_summary_best.png` |
| D02 | Bar | SHAP bar plot (best model) | `xai/shap_bar_best.png` |
| D03 | Dependence | SHAP dependence (top 5 features) | `xai/shap_dep_{feature}.png` |
| D04 | Force | SHAP force plot (5 claims) | `xai/force_{claim_id}.png` |
| D05 | PDP | Partial dependence (top 8 features) | `xai/pdp_{feature}.png` |

### 3.5 Visualization Standards

- **Resolution:** Minimum 150 DPI for all plots
- **Color scheme:** Fraud = red/orange tones; Legitimate = teal/blue tones
- **Colorblind-safe:** Use colorblind-friendly palettes (e.g., seaborn colorblind)
- **Font size:** Minimum 10pt for axis labels, 12pt for titles
- **Figure size:** 10×6 inches for single plots, 12×8 for overlays
- **Grid:** Light major gridlines only
- **Legend:** Always present when multiple series plotted
- **Caption:** Every figure has a caption in documentation

## 4. Reporting Deliverables

### 4.1 Documentation Folder

| File | Description | Min Length |
|------|-------------|-----------|
| documentation/overview.md | Project at-a-glance | 120 lines |
| documentation/problem_statement.md | Problem deep-dive | 150 lines |
| documentation/dataset.md | Data sourcing and schema | 200 lines |
| documentation/methodology.md | Pipeline stages explained | 250 lines |
| documentation/preprocessing.md | Transform justifications | 200 lines |
| documentation/models.md | Algorithm descriptions | 250 lines |
| documentation/evaluation.md | Results hub (links to eval/) | 150 lines |
| documentation/explainability.md | XAI methods and findings | 200 lines |
| documentation/fairness.md | Fairness analysis | 150 lines |
| documentation/decisions.md | Decision register | 100 lines |
| documentation/data_contracts.md | Stage boundary schemas | 150 lines |
| documentation/code_walkthrough.md | File-by-file explanation | 200 lines |
| documentation/reproduction.md | Rerun instructions | 120 lines |
| documentation/limitations.md | Known weaknesses | 100 lines |
| documentation/future_work.md | Approach 2 and 3 preview | 100 lines |
| documentation/glossary.md | Term definitions | 120 lines |
| documentation/references.md | Citations (15+ entries) | 80 lines |
| documentation/lessons_learned.md | Retrospective | 100 lines |

### 4.2 PPT Specification (~20 Slides)

| Slide | Title | Content |
|-------|-------|---------|
| 1 | Title Slide | Project name, team, adviser Prof. Ramesh Athe, IIIT Dharwad |
| 2 | Problem Context | Insurance fraud in India, financial impact, motivation |
| 3 | Objectives | Binary classification, benchmarking, explainability |
| 4 | Dataset Overview | Source, size, features, class distribution |
| 5 | Data Relationships | ER diagram, key entity connections |
| 6 | EDA Headline Findings | Top 5 insights with 2-3 key figures |
| 7 | Class Imbalance | 6% fraud rate, handling strategy |
| 8 | Pipeline Architecture | S0-S14 stage diagram |
| 9 | Preprocessing & Features | Feature engineering summary, 44 features |
| 10 | Leakage Controls | What was dropped and why |
| 11 | Model Zoo Overview | 19 models across 6 families |
| 12 | Benchmarking Protocol | Fair comparison methodology |
| 13 | Leaderboard Results | Top 5 models table |
| 14 | Winner Spotlight | Best model parameters and metrics |
| 15 | Calibration & Threshold | Operating point story |
| 16 | Explainability | SHAP findings, claim narrative example |
| 17 | Fairness Analysis | Slice results, equity discussion |
| 18 | Limitations | Honest threats to validity |
| 19 | Bridge to Approaches 2 & 3 | What comes next |
| 20 | Thank You & References | Acknowledgements, Q&A |

**Each slide includes 60-120 words of speaker notes.**

### 4.3 Project Report PDF

- **Format:** IEEE two-column, A4
- **Sections:** Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, Fairness, Limitations, Conclusion, References
- **Length:** 30-40 pages including appendices
- **Figures:** Minimum 10 figures with captions
- **Tables:** Minimum 5 tables with titles
- **References:** Minimum 15 IEEE-formatted citations
- **Title page:** Team names, roll numbers, adviser Prof. Ramesh Athe

### 4.4 IEEE Research Paper PDF

- **Format:** IEEE conference paper, two-column
- **Sections:** Abstract (200 words), Keywords, Introduction, Related Work, Method, Experiments, Results, Discussion, Conclusion
- **Length:** 6-8 pages
- **References:** Minimum 15 IEEE citations
- **Title page:** Team names, adviser Prof. Ramesh Athe

## 5. Consistency Rules

1. Every number in PPT must match evaluation/ CSV/JSON
2. Every number in PDF must match evaluation/ CSV/JSON
3. Every figure in PPT/PDF must exist in images/ folder
4. Prof. Ramesh Athe credited on PPT title slide and PDF title pages
5. Team members listed with roll numbers on all documents
6. Currency in INR, dates in DD-MM-YYYY format
7. Color scheme consistent across all visualizations

---

*Last updated: 01-09-2026 | Document version: 1.0*  
*Next: Form 6 — Code Structure, Pipeline & Deployment Specification*
