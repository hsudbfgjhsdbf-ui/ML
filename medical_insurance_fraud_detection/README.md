# Medical Insurance Claim Fraud Detection
## Approach 1: Traditional Machine Learning with Full Benchmarking

### IIIT Dharwad | B.Tech Data Science & AI

**Faculty Adviser:** Prof. Ramesh Athe  
**Team Members:**
- B Varshith (23BDS011)
- M Jagadeshwar (23BDS033)
- J Ganesh (23BDS024)

---

## Project Overview

This project builds an AI-driven system to detect fraudulent medical insurance claims using classical machine learning algorithms. It is **Approach 1 of 3** in the overall Medical Insurance Claim Fraud Detection project at IIIT Dharwad.

### Objectives
1. **Binary Classification:** Predict whether a claim is Fraudulent or Legitimate
2. **Comprehensive Benchmarking:** Train and compare 17+ ML algorithms
3. **Explainability:** Provide interpretable predictions using feature importance and SHAP
4. **Full Documentation:** Detailed docs, PPT, and IEEE-format report

## Dataset

| Property | Value |
|----------|-------|
| Source | [Health Insurance Fraud Claims](https://github.com/hsudbfgjhsdbf-ui/ML/blob/main/Health%20Insurance%20Fraud%20Claims.xlsx) |
| Records | 4,500 |
| Features | 19 raw → 44 model-ready |
| Fraud Rate | 6.0% (270 of 4,500) |
| Class Imbalance | 1:15.67 |

### Key Insights
- **Cluster 1** contains 267 of 270 fraud cases (98.9%)
- Fraudulent claims average **8,734 INR** vs **4,777 INR** for legitimate
- **ClaimAmountPatientIncome ratio** is a novel engineered feature capturing financial anomaly

## Results — Leaderboard (Top 5)

| Rank | Model | Test PR-AUC | Test ROC-AUC | Test F1 | Train Time |
|------|-------|------------|-------------|---------|-----------|
| 1 | **XGBoost** | 0.9919 | 0.9993 | 0.975 | 0.08s |
| 2 | Voting Ensemble | 0.9994 | 1.0000 | 0.975 | 1.91s |
| 3 | AdaBoost | 1.0000 | 1.0000 | 0.975 | 1.0s |
| 4 | Random Forest | 0.9994 | 1.0000 | 0.975 | 0.58s |
| 5 | HistGradientBoosting | 0.9972 | 0.9998 | 0.975 | 0.28s |

**Winner: XGBoost** — Best balance of performance and interpretability.

## Pipeline Architecture

```
Data (.xlsx) → Quality Gates → EDA → Preprocessing → Feature Engineering
    → Splitting (70/15/15) → Model Training (17 models) → Leaderboard
    → Evaluation → Visualization → Documentation → Reports
```

## Quick Start

```bash
# 1. Navigate to project directory
cd medical_insurance_fraud_detection

# 2. Install dependencies
pip install pandas openpyxl numpy scikit-learn xgboost lightgbm shap matplotlib seaborn

# 3. Run full pipeline
python run_pipeline.py

# 4. View results
cat evaluation/evaluation.md
```

## Project Structure

```
medical_insurance_fraud_detection/
├── README.md                    # This file
├── run_pipeline.py              # Master pipeline script
├── forms/                       # 8 specification documents
│   ├── form_1_dataset_eda_preprocessing.md
│   ├── form_2_model_zoo_implementation.md
│   ├── form_3_evaluation_benchmarking.md
│   ├── form_4_feature_engineering.md
│   ├── form_5_explainability_visualization.md
│   ├── form_6_code_pipeline_deployment.md
│   ├── form_7_technical_architecture.md
│   └── form_8_technical_approach_images.md
├── data/
│   └── raw/                     # Source dataset
├── images/
│   ├── eda/                     # 18 EDA visualizations
│   ├── models/                  # Model comparison plots
│   ├── xai/                     # Explainability plots
│   └── diagrams/                # Architecture diagrams
├── evaluation/
│   ├── evaluation.md            # Results hub
│   ├── leaderboard.csv          # Complete benchmarking table
│   ├── run_manifest.json        # Run metadata
│   └── pipeline.log             # Execution log
├── documentation/               # 18+ markdown documentation files
├── presentation/                # PPT files
└── reports/                     # PDF reports
```

## Models Trained (17)

| Family | Models |
|--------|--------|
| Baselines | Majority Class, Logistic Regression L2, Logistic Regression L1 |
| Probabilistic | Gaussian Naive Bayes |
| Tree | Decision Tree, Random Forest, Extra Trees |
| Boosting | AdaBoost, HistGradientBoosting, XGBoost, LightGBM, Gradient Boosting |
| Distance/Margin | KNN, Linear SVM |
| Neural | MLP Classifier |
| Ensemble | Soft Voting Ensemble |

## Key Visualizations

| Image | Location | Description |
|-------|----------|-------------|
| EDA Dashboard | `images/eda/eda_summary_dashboard.png` | 6-panel overview |
| ROC Curves | `images/models/roc_all_models_overlay.png` | All models compared |
| PR Curves | `images/models/pr_all_models_overlay.png` | All models compared |
| Confusion Matrix | `images/models/confusion_matrix_best.png` | Best model results |
| Metrics Comparison | `images/models/all_metrics_comparison.png` | Top 8 models |
| Cluster Analysis | `images/eda/fraud_rate_by_cluster.png` | Cluster-fraud relationship |

## Deliverables

- [x] Complete runnable Python codebase
- [x] 18+ EDA visualizations
- [x] 5+ model comparison plots
- [x] Complete leaderboard with 17 models
- [x] Evaluation report (evaluation.md)
- [x] 11+ documentation files
- [x] 8 specification forms
- [x] Run manifest and pipeline log

## Configuration

| Parameter | Value |
|-----------|-------|
| Random Seed | 42 |
| Split Ratio | 70/15/15 (train/val/test) |
| Primary Metric | PR-AUC |
| Class Imbalance | class_weight='balanced' |
| Total Runtime | ~19 seconds |

## References

1. Health Insurance Fraud Claims Dataset — GitHub: hsudbfgjhsdbf-ui/ML
2. Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. KDD.
3. Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.
4. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
5. IRDAI Annual Report (2023-24). Insurance Regulatory and Development Authority of India.

## License

Academic project — IIIT Dharwad, B.Tech Data Science and AI.

---

*Built by B Varshith, M Jagadeshwar, and J Ganesh under the guidance of Prof. Ramesh Athe, IIIT Dharwad.*
