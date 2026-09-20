# FORM 6: Traditional ML — Code Structure, Pipeline Automation & Deployment Specification
## Medical Insurance Claim Fraud Detection — Approach 1 of 3
### IIIT Dharwad | B.Tech Data Science & AI

---

**Faculty Adviser:** Prof. Ramesh Athe  
**Team Members:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  
**Date:** 01-09-2026 | **Version:** 1.0

---

## 1. Purpose

This document specifies the complete code structure, pipeline automation, single-command execution, and deployment considerations for Approach 1. The entire project must run end-to-end from a single entry point.

## 2. Repository Structure

```
medical_insurance_fraud_detection/
│
├── README.md                           # Project overview & quick-start
├── requirements.txt                    # Pinned dependencies
├── run_pipeline.py                     # MASTER SCRIPT — single entry point
├── config/
│   ├── default_config.yaml             # Pipeline configuration
│   └── frozen_features.yaml            # Frozen feature list
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                   # Dataset loading & validation
│   │   └── quality_gates.py            # Data quality checks
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py              # Feature creation
│   │   ├── encoding.py                 # Categorical encoding
│   │   └── selection.py                # Feature selection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baselines.py                # M01-M03 (majority, logistic)
│   │   ├── probabilistic.py            # M04-M05 (naive bayes)
│   │   ├── neighbors.py                # M06-M09 (KNN, SVM)
│   │   ├── trees.py                    # M10-M12 (DT, RF, ET)
│   │   ├── boosting.py                 # M13-M16 (Ada, HGBM, XGB, LGBM)
│   │   ├── ensembles.py               # M17-M18 (voting, stacking)
│   │   └── neural.py                   # M19 (MLP)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                  # Metric computation
│   │   ├── benchmark.py                # Leaderboard assembly
│   │   ├── calibration.py              # Probability calibration
│   │   └── fairness.py                 # Slice analysis
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── visualizations.py           # All plot generation
│   │   ├── document_builder.py         # Markdown/PPT/PDF generation
│   │   └── model_cards.py              # Model card generator
│   └── utils/
│       ├── __init__.py
│       ├── config.py                   # Configuration management
│       ├── seeds.py                    # Random seed management
│       ├── logging_utils.py            # Logging setup
│       └── paths.py                    # Path resolution
│
├── scripts/
│   ├── fetch_dataset.py                # Download dataset from GitHub
│   └── generate_report.py             # Generate PPT and PDF
│
├── data/
│   ├── raw/                            # Original dataset (fetched)
│   ├── interim/                        # Validated data
│   └── processed/                      # Model-ready features
│
├── images/
│   ├── eda/                            # EDA visualizations
│   ├── models/                         # Model performance plots
│   ├── xai/                            # Explainability plots
│   └── diagrams/                       # Architecture diagrams
│
├── documentation/                      # 18+ markdown files
├── evaluation/                         # Results hub
├── presentation/                       # PPT file
├── reports/                            # PDF files
├── notebooks/                          # Exploration notebooks
└── forms/                              # These specification documents
```

## 3. Master Pipeline Script (run_pipeline.py)

### 3.1 Single-Command Execution

```bash
# Full pipeline run (default configuration)
python run_pipeline.py

# Resume from specific stage
python run_pipeline.py --from-stage 8

# Dry run (print plan without executing)
python run_pipeline.py --dry-run

# Custom configuration
python run_pipeline.py --config config/custom_config.yaml

# Self-test with small data
python run_pipeline.py --self-test
```

### 3.2 Pipeline Stages

| Stage | Name | Description | Est. Time |
|-------|------|-------------|-----------|
| S0 | Environment Bootstrap | Verify dependencies, log versions | 30 sec |
| S1 | Data Acquisition | Fetch dataset, verify checksums | 1 min |
| S2 | Data Quality Gates | Validate schema, types, ranges | 1 min |
| S3 | Exploratory Analysis | Generate all EDA plots and report | 10 min |
| S4 | Preprocessing | Encode, scale, engineer features | 2 min |
| S5 | Feature Freeze | Finalize feature list, export schema | 30 sec |
| S6 | Data Splitting | Stratified 70/15/15 split | 30 sec |
| S7 | Baseline Training | Train M01-M03 baselines | 2 min |
| S8 | Model Zoo Training | Train M04-M19 (all algorithms) | 3-4 hours |
| S9 | Leaderboard Assembly | Collect metrics, rank models | 1 min |
| S10 | Winner Tuning | Hyperparameter optimization | 1-2 hours |
| S11 | Final Training | Train winner on train+val | 10 min |
| S12 | Test Evaluation | One-time test set evaluation | 2 min |
| S13 | Calibration | Probability calibration, threshold | 5 min |
| S14 | Report Generation | Docs, PPT, PDFs, all artifacts | 15 min |

### 3.3 Configuration File (default_config.yaml)

```yaml
# Pipeline Configuration
# Medical Insurance Claim Fraud Detection — Approach 1

project:
  name: "Medical Insurance Claim Fraud Detection"
  approach: 1
  approach_name: "Traditional Machine Learning"
  seed: 42

data:
  source: "data/raw/Health_Insurance_Fraud_Claims.xlsx"
  target_column: "ClaimLegitimacy"
  positive_class: "Fraud"
  negative_class: "Legitimate"

splitting:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  stratify: true
  group_column: null

resampling:
  strategies: ["none", "smote", "class_weight"]
  primary_metric: "average_precision"

models:
  include_all: true
  timeout_per_model: 2400  # seconds
  n_jobs: -1

tuning:
  cv_folds: 5
  cv_repeats: 2
  search_iterations: 60
  scoring: "average_precision"

metrics:
  primary: "pr_auc"
  secondary: ["roc_auc", "f1", "f2", "mcc", "brier"]
  
reporting:
  generate_ppt: true
  generate_pdf: true
  figure_dpi: 150
  colorblind_safe: true
```

## 4. Dependency Management

### 4.1 requirements.txt

```
# Core Data Science
numpy==1.26.4
pandas==2.2.1
openpyxl==3.1.2
scipy==1.12.0

# Machine Learning
scikit-learn==1.4.1
xgboost==2.0.3
lightgbm==4.3.0
shap==0.44.1
imbalanced-learn==0.12.2

# Visualization
matplotlib==3.8.3
seaborn==0.13.2
plotly==5.18.0

# Reporting
python-pptx==0.6.23
reportlab==4.1.0
markdown==3.5.2
jinja2==3.1.3

# Utilities
pyyaml==6.0.1
joblib==1.3.2
tqdm==4.66.2
```

### 4.2 Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import sklearn; import xgboost; import lightgbm; print('All OK')"

# Fetch dataset
python scripts/fetch_dataset.py

# Run full pipeline
python run_pipeline.py
```

## 5. Logging Specification

### 5.1 Log Levels
| Level | Usage |
|-------|-------|
| DEBUG | Detailed variable values, intermediate calculations |
| INFO | Stage start/end, data loaded, model trained, evaluation complete |
| WARNING | Recoverable issues, skipped models, high quarantine rates |
| ERROR | Stage failures, data quality gate failures |

### 5.2 Log Format
```
2026-09-01 10:30:45 | INFO | Stage S8 | Training model m15_xgboost...
2026-09-01 10:35:12 | INFO | Stage S8 | m15_xgboost complete. Val PR-AUC: 0.9123. Time: 267.3s
```

### 5.3 Log Output
- Console: INFO and above (human-readable)
- File: DEBUG and above (evaluation/run_log.txt)

## 6. Error Handling

### 6.1 Error Categories
| Category | Action |
|----------|--------|
| Data not found | Halt with clear message: "Run scripts/fetch_dataset.py first" |
| Schema mismatch | Halt with column comparison |
| Model convergence failure | Log warning, mark model as 'failed', continue |
| Memory error | Reduce data subset, log warning |
| Config error | Halt with specific key identified |

### 6.2 Recovery Policy
- Individual model failure: Log and continue with next model
- Stage failure: Halt pipeline, report issue
- Data quality failure: Halt pipeline, suggest remediation

## 7. Reproducibility

### 7.1 Seed Management
```python
# Master seed propagated everywhere
MASTER_SEED = 42

# Derived seeds for each stochastic component
numpy.random.seed(MASTER_SEED)
random.seed(MASTER_SEED)
# sklearn: random_state=MASTER_SEED
# xgboost: seed=MASTER_SEED
# lightgbm: random_seed=MASTER_SEED
```

### 7.2 Determinism Guarantees
- All random operations use MASTER_SEED
- Data splits are deterministic
- Model initialization is deterministic
- Hyperparameter search order is deterministic
- Results should be identical across runs on same hardware

### 7.3 Run Manifest
Each pipeline run generates `evaluation/run_manifest.json`:
```json
{
  "run_id": "run_20260901_103000",
  "start_time": "2026-09-01T10:30:00",
  "end_time": "2026-09-01T14:45:00",
  "config_hash": "abc123...",
  "data_hash": "def456...",
  "python_version": "3.11.2",
  "library_versions": {...},
  "hardware": {"cpu": "...", "ram_gb": 16},
  "stages_completed": ["S0", "S1", ..., "S14"],
  "artifacts_generated": [...]
}
```

## 8. Deployment Considerations

### 8.1 Model Serving Architecture
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Claim Input  │────▶│ Preprocessing│────▶│   Model      │
│  (JSON/REST)  │     │  Pipeline    │     │  Prediction  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                    ┌──────────────┐     ┌────────▼───────┐
                    │  Response    │◀────│  Explanation   │
                    │  (JSON/REST) │     │  Generator     │
                    └──────────────┘     └────────────────┘
```

### 8.2 Model Persistence
- Best model saved as: `evaluation/final_model.pkl`
- Preprocessing pipeline saved as: `evaluation/preprocessing_pipeline.pkl`
- Feature list saved as: `config/frozen_features.yaml`
- All three required for inference on new data

### 8.3 Inference API Specification
```python
# Input: JSON with claim features
{
  "ClaimAmount": 8500.00,
  "PatientAge": 45,
  "PatientIncome": 75000.00,
  "PatientGender": "M",
  "ProviderSpecialty": "Cardiology",
  "Cluster": 1,
  "ClaimType": "Inpatient",
  ...
}

# Output: JSON prediction
{
  "prediction": "Fraud",
  "probability": 0.87,
  "confidence_band": "high",
  "action": "escalate",
  "top_drivers": [
    {"feature": "Cluster", "value": 1, "contribution": "+0.35"},
    {"feature": "ClaimAmount", "value": 8500, "contribution": "+0.22"},
    {"feature": "ClaimAmountPatientIncome", "value": 0.113, "contribution": "+0.15"}
  ]
}
```

## 9. Quick-Start Guide (for README)

```bash
# 1. Clone the repository
git clone https://github.com/hsudbfgjhsdbf-ui/ML.git
cd ML/medical_insurance_fraud_detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Fetch dataset
python scripts/fetch_dataset.py

# 5. Run full pipeline (generates all artifacts)
python run_pipeline.py

# 6. View results
# Open evaluation/evaluation.md for results
# Open images/ for visualizations
# Open documentation/index.md for docs
```

## 10. References

1. Cookiecutter Data Science: https://drivendata.github.io/cookiecutter-data-science/
2. scikit-learn pipeline documentation
3. XGBoost and LightGBM documentation
4. Python packaging best practices

---

*Last updated: 01-09-2026 | Document version: 1.0*  
*This completes all 6 specification forms for Approach 1.*
