# FORM 8: Technical Approach — Implementation Details & Image Gallery
## Medical Insurance Claim Fraud Detection — Approach 1 of 3
### IIIT Dharwad | B.Tech Data Science & AI

---

**Faculty Adviser:** Prof. Ramesh Athe  
**Team Members:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  
**Date:** 01-09-2026 | **Version:** 1.0

---

## 1. Technical Approach Summary

### 1.1 Approach Philosophy

Approach 1 applies classical machine learning algorithms to the structured tabular data of health insurance claims. The core technical decisions are:

1. **Data Source:** Health Insurance Fraud Claims dataset (4,500 rows, 19 columns) from team GitHub repository
2. **Target:** Binary classification — Fraud (1) vs Legitimate (0) — 6% fraud rate
3. **Key Signal:** Cluster assignment (Cluster 1 contains 267 of 270 fraud cases) + ClaimAmount separation
4. **Algorithm Coverage:** 19 models across 7 families (baseline, probabilistic, distance/margin, tree, boosting, ensemble, neural)
5. **Primary Metric:** PR-AUC (appropriate for imbalanced data)
6. **Explainability:** SHAP values + feature importance for model transparency

### 1.2 Technical Innovation Points

| Innovation | Description |
|-----------|-------------|
| Comprehensive Benchmarking | 19 models compared under identical protocol |
| Indian Context | All amounts in INR, Indian insurance domain |
| ClaimAmountPatientIncome Ratio | Novel engineered feature capturing financial anomaly |
| Cluster-Fraud Analysis | Deep analysis of cluster assignment as fraud signal |
| Multi-Threshold Analysis | F2-optimal threshold with business cost justification |
| Fairness Slicing | Demographic group analysis for bias detection |

## 2. Implementation Architecture

### 2.1 Code Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    run_pipeline.py (Entry Point)                      │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    CONFIGURATION LAYER                           │ │
│  │  config/default_config.yaml                                      │ │
│  │  • project settings, data paths, model configs                  │ │
│  │  • seed=42, train/val/test=0.70/0.15/0.15                      │ │
│  └────────────────────────────┬────────────────────────────────────┘ │
│                               │                                       │
│  ┌────────────────────────────▼────────────────────────────────────┐ │
│  │                    DATA LAYER                                    │ │
│  │                                                                   │ │
│  │  src/data/loader.py                                              │ │
│  │  ├── load_dataset() → reads .xlsx                               │ │
│  │  ├── validate_schema() → checks 19 columns                      │ │
│  │  └── compute_checksum() → SHA256 verification                   │ │
│  │                                                                   │ │
│  │  src/data/quality_gates.py                                       │ │
│  │  ├── check_completeness() → 0 nulls PASS                        │ │
│  │  ├── check_ranges() → all within bounds PASS                    │ │
│  │  └── check_uniqueness() → claim_id unique PASS                  │ │
│  └────────────────────────────┬────────────────────────────────────┘ │
│                               │                                       │
│  ┌────────────────────────────▼────────────────────────────────────┐ │
│  │                 FEATURE ENGINEERING LAYER                        │ │
│  │                                                                   │ │
│  │  src/features/engineering.py                                     │ │
│  │  ├── create_ratio_features() → ClaimAmountPatientIncome, etc.   │ │
│  │  ├── create_temporal_features() → Month, DayOfWeek, IsWeekend   │ │
│  │  ├── create_binned_features() → AgeGroup, IncomeGroup           │ │
│  │  └── create_interaction_features() → ClusterIs1, HighClaimAmt   │ │
│  │                                                                   │ │
│  │  src/features/encoding.py                                        │ │
│  │  ├── one_hot_encode() → ProviderSpecialty, ClaimType, etc.      │ │
│  │  ├── label_encode() → PatientGender                             │ │
│  │  └── scale_features() → RobustScaler, StandardScaler            │ │
│  └────────────────────────────┬────────────────────────────────────┘ │
│                               │                                       │
│  ┌────────────────────────────▼────────────────────────────────────┐ │
│  │                    MODELING LAYER                                 │ │
│  │                                                                   │ │
│  │  src/models/                                                     │ │
│  │  ├── baselines.py → M01 Majority, M02-M03 Logistic             │ │
│  │  ├── probabilistic.py → M04-M05 Naive Bayes                    │ │
│  │  ├── neighbors.py → M06-M09 KNN, SVM                           │ │
│  │  ├── trees.py → M10-M12 DecisionTree, RF, ExtraTrees           │ │
│  │  ├── boosting.py → M13-M16 Ada, HGBM, XGB, LGBM              │ │
│  │  ├── ensembles.py → M17-M18 Voting, Stacking                   │ │
│  │  └── neural.py → M19 MLP                                        │ │
│  │                                                                   │ │
│  │  Common Interface:                                               │ │
│  │  ├── train(X_train, y_train, config) → fitted_model             │ │
│  │  ├── predict(model, X) → predictions                            │ │
│  │  ├── predict_proba(model, X) → probabilities                    │ │
│  │  └── get_feature_importance(model) → importance_dict             │ │
│  └────────────────────────────┬────────────────────────────────────┘ │
│                               │                                       │
│  ┌────────────────────────────▼────────────────────────────────────┐ │
│  │                  EVALUATION LAYER                                 │ │
│  │                                                                   │ │
│  │  src/evaluation/                                                 │ │
│  │  ├── metrics.py → precision, recall, F1, F2, ROC-AUC, PR-AUC   │ │
│  │  ├── benchmark.py → leaderboard assembly & ranking              │ │
│  │  ├── calibration.py → Platt scaling, isotonic                   │ │
│  │  └── fairness.py → slice metrics across demographics            │ │
│  └────────────────────────────┬────────────────────────────────────┘ │
│                               │                                       │
│  ┌────────────────────────────▼────────────────────────────────────┐ │
│  │                 REPORTING LAYER                                   │ │
│  │                                                                   │ │
│  │  src/reporting/                                                  │ │
│  │  ├── visualizations.py → All matplotlib/seaborn plots           │ │
│  │  ├── document_builder.py → Markdown, PPT, PDF generation        │ │
│  │  └── model_cards.py → Per-model summary cards                   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                   UTILITY LAYER                                   │ │
│  │  src/utils/config.py, seeds.py, logging_utils.py, paths.py      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Algorithm Decision Flow

```
                    ┌─────────────────────┐
                    │   START: Load Data   │
                    │   4,500 × 19 cols    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Target Distribution │
                    │  Fraud: 270 (6%)    │
                    │  Legit: 4,230 (94%) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Key Signal: Cluster │
                    │  Cluster 1 → 267/270│
                    │  fraud cases (99%)   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Key Signal: Amount  │
                    │  Fraud mean: 8,734  │
                    │  Legit mean: 4,777  │
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │    FEATURE ENGINEERING           │
              │    • 44 features total           │
              │    • Ratios, temporal, binned    │
              │    • One-hot encoded categoricals│
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │    STRATIFIED SPLIT              │
              │    70% train / 15% val / 15% test│
              └────────────────┬────────────────┘
                               │
         ┌─────────────────────▼─────────────────────┐
         │         TRAIN ALL 19 MODELS                 │
         │                                             │
         │  Baselines ──► Probabilistic ──► Neighbors  │
         │  Trees ──► Boosting ──► Ensembles ──► MLP  │
         │                                             │
         │  Each model:                                │
         │  • Grid/Random search for best params       │
         │  • 5-fold stratified CV                     │
         │  • All 10 metrics computed                  │
         └─────────────────────┬─────────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │     LEADERBOARD ASSEMBLY         │
              │     Rank by Validation PR-AUC    │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │     SELECT WINNER                │
              │     Tune hyperparameters         │
              │     Final train on train+val     │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │     ONE-TIME TEST EVALUATION     │
              │     Report final metrics         │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │     CALIBRATE & SET THRESHOLD    │
              │     Generate explanations (SHAP) │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │     GENERATE ALL REPORTS         │
              │     • Evaluation markdown        │
              │     • 34+ visualizations         │
              │     • PPT (20 slides)            │
              │     • IEEE paper PDF             │
              │     • Project report PDF         │
              └─────────────────────────────────┘
```

## 3. Image Gallery Description

### 3.1 EDA Images (Generated in images/eda/)

| Image | Description | Key Insight |
|-------|-------------|-------------|
| `class_distribution.png` | Bar chart of Legitimate vs Fraud | 6% fraud rate (270/4500) |
| `cluster_distribution.png` | Cluster counts by class | Cluster 1 holds 99% of frauds |
| `claim_amount_distribution.png` | Histogram of amounts by class | Fraud amounts concentrated 6,500-9,900 INR |
| `patient_income_distribution.png` | Income distribution by class | Overlapping distributions |
| `claim_amount_boxplot.png` | Boxplot by fraud status | Fraud claims have higher median |
| `correlation_heatmap.png` | Feature correlation matrix | ClaimAmount-Income ratio correlates with fraud |
| `fraud_rate_by_cluster.png` | Bar chart of fraud rate per cluster | Cluster 1: 24.3%, others <1% |
| `claim_vs_income_scatter.png` | Scatter plot colored by fraud | Fraud claims cluster in high-amount region |
| `fraud_rate_by_amount_decile.png` | Fraud rate across amount deciles | Top deciles show higher fraud rates |
| `eda_summary_dashboard.png` | Multi-panel EDA overview | Complete dataset summary |

### 3.2 Model Performance Images (Generated in images/models/)

| Image | Description | Key Insight |
|-------|-------------|-------------|
| `roc_all_models_overlay.png` | All ROC curves on one plot | XGBoost/LightGBM lead |
| `pr_all_models_overlay.png` | All PR curves on one plot | Boosting models dominate |
| `confusion_matrix_best.png` | Best model confusion matrix | High TP, low FN |
| `feature_importance_best.png` | Top features horizontal bar | Cluster, ClaimAmount lead |
| `all_metrics_comparison.png` | Grouped bar chart all models | Visual comparison |
| `time_vs_performance.png` | Scatter: time vs PR-AUC | Trade-off visualization |
| `radar_top5.png` | Radar chart of top 5 models | Multi-dimensional comparison |
| `learning_curves_best.png` | Learning curves for winner | Convergence check |

### 3.3 XAI Images (Generated in images/xai/)

| Image | Description | Key Insight |
|-------|-------------|-------------|
| `shap_summary_best.png` | SHAP beeswarm plot | Feature impact directions |
| `shap_bar_best.png` | SHAP bar plot | Mean |SHAP| ranking |
| `shap_dependence_cluster.png` | SHAP dependence for Cluster | Clear step function at Cluster=1 |
| `shap_dependence_amount.png` | SHAP dependence for ClaimAmount | Increasing fraud probability |
| `pdp_cluster.png` | Partial dependence: Cluster | Sharp spike at Cluster=1 |
| `pdp_claim_amount.png` | Partial dependence: ClaimAmount | Monotone increase |

### 3.4 Architecture Diagrams (Generated in images/diagrams/)

| Image | Description |
|-------|-------------|
| `pipeline_architecture.png` | Full pipeline S0-S14 |
| `data_flow_diagram.png` | Data transformation flow |
| `er_diagram.png` | Entity-relationship diagram |
| `module_dependency.png` | Code module dependencies |
| `artifact_flow.png` | Artifact dependency graph |
| `deployment_architecture.png` | Serving architecture |

## 4. Performance Characteristics

### 4.1 Expected Runtime Profile

```
Stage S0-S2:    ████░░░░░░░░░░░░░░░░  2 min   (5%)
Stage S3:       ██████████░░░░░░░░░░  10 min  (20%)
Stage S4-S6:    ██████░░░░░░░░░░░░░░  4 min   (8%)
Stage S7-S8:    ████████████████████  240 min (78%)
Stage S9-S12:   ███░░░░░░░░░░░░░░░░░  15 min  (5%)
Stage S13-S14:  ██░░░░░░░░░░░░░░░░░░  15 min  (5%)
─────────────────────────────────────────────────
TOTAL:          ~276 min (~4.6 hours)
```

### 4.2 Memory Profile

```
Data loading:          ~200 MB
EDA generation:        ~400 MB
Feature engineering:   ~300 MB
Model training:        ~2-4 GB (peak during boosting)
Visualization:         ~500 MB
Total peak:            ~4 GB
```

### 4.3 Output Artifact Count

| Category | Count | Location |
|----------|-------|----------|
| EDA plots | 24+ | images/eda/ |
| Model plots | 16+ | images/models/ |
| XAI plots | 10+ | images/xai/ |
| Diagrams | 6+ | images/diagrams/ |
| Metric files | 19 JSON | evaluation/metrics/ |
| Curve files | 38 CSV | evaluation/curves/ |
| Model cards | 19 MD | evaluation/model_cards/ |
| Documentation | 18 MD | documentation/ |
| Total artifacts | 150+ | Throughout |

## 5. Risk Mitigation Strategy

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Dataset unavailable | Low | High | Dataset already in repo |
| Models fail to converge | Low | Medium | Multiple fallback configs |
| Memory overflow | Low | Medium | Subsample for heavy models |
| Runtime exceeds limit | Medium | Medium | Skip expensive models if needed |
| Low model performance | Low | High | Cluster feature provides strong signal |
| Class imbalance issues | Medium | Medium | SMOTE + class_weight comparison |

## 6. Success Criteria

| Criterion | Target | Measured By |
|-----------|--------|-------------|
| PR-AUC | > 0.80 | Validation set |
| Recall@90% precision | > 0.60 | Threshold sweep |
| All models trained | 19 models | Leaderboard |
| Runtime | < 6 hours | Run manifest |
| Documentation | 18+ files | File count |
| Visualizations | 50+ plots | Image count |
| Zero code errors | 0 errors | Pipeline exit code |

---

*Last updated: 01-09-2026 | Document version: 1.0*  
*This completes all 8 specification forms (6 core + 2 technical approach) for Approach 1.*

---

## Summary of All Forms

| Form | Title | Pages | Status |
|------|-------|-------|--------|
| Form 1 | Dataset, EDA & Preprocessing | ~10 | ✅ Complete |
| Form 2 | Model Zoo Implementation | ~12 | ✅ Complete |
| Form 3 | Evaluation & Benchmarking | ~10 | ✅ Complete |
| Form 4 | Feature Engineering & Selection | ~8 | ✅ Complete |
| Form 5 | Explainability, Visualization & Reporting | ~10 | ✅ Complete |
| Form 6 | Code Structure, Pipeline & Deployment | ~10 | ✅ Complete |
| Form 7 | Technical Architecture (with diagrams) | ~10 | ✅ Complete |
| Form 8 | Technical Approach (with image gallery) | ~10 | ✅ Complete |

**Total specification coverage:** All aspects of Approach 1 are documented across these 8 forms, from data acquisition through model training, evaluation, and reporting.
