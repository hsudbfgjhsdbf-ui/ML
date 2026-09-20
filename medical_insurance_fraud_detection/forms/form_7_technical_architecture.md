# FORM 7: Technical Architecture — System Design & Pipeline Diagrams
## Medical Insurance Claim Fraud Detection — Approach 1 of 3
### IIIT Dharwad | B.Tech Data Science & AI

---

**Faculty Adviser:** Prof. Ramesh Athe  
**Team Members:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  
**Date:** 01-09-2026 | **Version:** 1.0

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

The Medical Insurance Claim Fraud Detection system follows a modular pipeline architecture where data flows through sequential stages, each producing well-defined artifacts. The architecture is designed for reproducibility, auditability, and extensibility across three approaches.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MEDICAL INSURANCE FRAUD DETECTION SYSTEM              │
│                    Approach 1: Traditional Machine Learning              │
│                    IIIT Dharwad — Prof. Ramesh Athe                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  DATA    │───▶│ PREPROC  │───▶│ FEATURE  │───▶│  MODEL   │          │
│  │ ACQUIRE  │    │ & CLEAN  │    │ ENGINEER │    │  TRAIN   │          │
│  │  (S0-S2) │    │  (S3-S4) │    │  (S5-S6) │    │  (S7-S8) │          │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘          │
│       │               │               │               │                │
│       ▼               ▼               ▼               ▼                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  Excel   │    │ 44 Clean │    │  44+     │    │  19 ML   │          │
│  │  Dataset │    │ Features │    │ Engineered│    │ Models   │          │
│  │ 4500×19  │    │ Encoded  │    │  Features │    │ Trained  │          │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘          │
│                                                       │                │
│                                                       ▼                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  PPT &   │◀───│  REPORT  │◀───│EXPLAIN & │◀───│EVALUATE  │          │
│  │  PDF     │    │ GENERATE │    │ VISUALIZE│    │ & BENCH  │          │
│  │ (S14)   │    │  (S14)   │    │ (S13)    │    │ (S9-S12) │          │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Detailed Pipeline DAG

The pipeline is a directed acyclic graph (DAG) of 15 stages (S0-S14). Each stage has defined inputs, outputs, and validation gates.

```
S0 ──▶ S1 ──▶ S2 ──▶ S3 ──▶ S4 ──▶ S5 ──▶ S6 ──▶ S7 ──▶ S8
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       ▼
│       │       │       │       │       │       │       │      S9
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       ▼
│       │       │       │       │       │       │       │      S10
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       ▼
│       │       │       │       │       │       │       │      S11
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       ▼
│       │       │       │       │       │       │       │      S12
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       ▼
│       │       │       │       │       │       │       │      S13
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       ▼
│       │       │       │       │       │       │       │      S14
▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼
[Run Manifest — records all stage outputs, timestamps, and checksums]
```

### 1.3 Stage Details

| Stage | Name | Input | Output | Duration |
|-------|------|-------|--------|----------|
| S0 | Environment Bootstrap | requirements.txt | Verified environment | 30s |
| S1 | Data Acquisition | GitHub URL | Raw .xlsx + checksums | 1 min |
| S2 | Data Quality Gates | Raw data | Validated data or halt | 1 min |
| S3 | Exploratory Analysis | Validated data | EDA report + 34+ figures | 10 min |
| S4 | Preprocessing | Train split | Fitted transformers | 2 min |
| S5 | Feature Engineering | Transformed data | Frozen feature schema | 1 min |
| S6 | Data Splitting | Feature matrix | train/val/test splits | 30s |
| S7 | Baseline Training | Train data | Baseline metrics | 2 min |
| S8 | Model Zoo Training | Train/val data | Per-model artifacts | 3-4 hrs |
| S9 | Leaderboard Assembly | All model metrics | Ranked leaderboard | 1 min |
| S10 | Winner Tuning | Winner config space | Tuned parameters | 1-2 hrs |
| S11 | Final Training | Train+val data | Final model artifact | 10 min |
| S12 | Test Evaluation | Test split (once) | Final test metrics | 2 min |
| S13 | Calibration & Threshold | Val probabilities | Operating point | 5 min |
| S14 | Report Generation | All artifacts | Docs, PPT, PDFs | 15 min |

## 2. Data Flow Architecture

### 2.1 Data Flow Diagram

```
                    ┌─────────────────────────┐
                    │  Health Insurance Fraud  │
                    │   Claims.xlsx (GitHub)   │
                    │   4,500 rows × 19 cols   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     DATA ACQUISITION     │
                    │  • Download from GitHub  │
                    │  • Verify SHA256 hash    │
                    │  • Record retrieval date │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    QUALITY VALIDATION    │
                    │  • Schema check (19 cols)│
                    │  • Type validation       │
                    │  • Range validation      │
                    │  • Null check (0 found)  │
                    │  • Uniqueness check      │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │          DATA TRANSFORMATION          │
              │                                       │
              │  ┌─────────────┐  ┌──────────────┐   │
              │  │ Drop 7 cols │  │ Extract date │   │
              │  │ (IDs, leak) │  │ features     │   │
              │  └──────┬──────┘  └──────┬───────┘   │
              │         │                │            │
              │  ┌──────▼──────────────▼──────┐      │
              │  │    12 retained columns      │      │
              │  └──────────────┬──────────────┘      │
              │                 │                      │
              │  ┌──────────────▼──────────────┐      │
              │  │  Feature Engineering (+12)   │      │
              │  │  • Ratios (ClaimAmt/Income)  │      │
              │  │ • Log transforms             │      │
              │  │ • Binned features            │      │
              │  │ • Interaction flags          │      │
              │  └──────────────┬──────────────┘      │
              │                 │                      │
              │  ┌──────────────▼──────────────┐      │
              │  │  Categorical Encoding (+20)  │      │
              │  │  • One-hot (5+4+4+4+3=20)   │      │
              │  │  • Label (Gender)            │      │
              │  └──────────────┬──────────────┘      │
              │                 │                      │
              │  ┌──────────────▼──────────────┐      │
              │  │   44 Model-Ready Features    │      │
              │  └──────────────┬──────────────┘      │
              └─────────────────┼────────────────────┘
                                │
              ┌─────────────────▼────────────────────┐
              │         STRATIFIED SPLITTING          │
              │                                       │
              │   ┌─────────┐ ┌─────────┐ ┌────────┐ │
              │   │ TRAIN   │ │   VAL   │ │  TEST  │ │
              │   │ 3,150   │ │  675    │ │  675   │ │
              │   │ 189 F   │ │  41 F   │ │  40 F  │ │
              │   │ 2,961 L │ │  634 L  │ │ 635 L  │ │
              │   └────┬────┘ └────┬────┘ └───┬────┘ │
              └────────┼──────────┼────────────┼─────┘
                       │          │            │
              ┌────────▼──────────▼────────────▼─────┐
              │          MODEL TRAINING                │
              │                                        │
              │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │
              │  │ M01  │ │ M02  │ │ M03  │ │ ...  │ │
              │  │Base  │ │LR-L2 │ │LR-L1 │ │ M19  │ │
              │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ │
              │     │        │        │        │      │
              │     └────────┴────────┴────────┘      │
              │                  │                     │
              │         ┌───────▼────────┐             │
              │         │   LEADERBOARD  │             │
              │         │  (19 models)   │             │
              │         └───────┬────────┘             │
              └─────────────────┼─────────────────────┘
                                │
              ┌─────────────────▼─────────────────────┐
              │        EVALUATION & REPORTING           │
              │                                        │
              │  ┌──────────────────────────────┐      │
              │  │ • 10 metrics per model       │      │
              │  │ • ROC + PR curves            │      │
              │  │ • SHAP explainability        │      │
              │  │ • Fairness slices            │      │
              │  │ • Business impact analysis   │      │
              │  └──────────────┬───────────────┘      │
              │                 │                       │
              │  ┌──────────────▼───────────────┐      │
              │  │ • Evaluation Markdown         │      │
              │  │ • PPT (20 slides)            │      │
              │  │ • IEEE Paper PDF             │      │
              │  │ • Project Report PDF         │      │
              │  └──────────────────────────────┘      │
              └────────────────────────────────────────┘
```

### 2.2 Entity-Relationship Diagram

```
┌───────────────────┐       ┌───────────────────┐
│    POLICYHOLDER    │       │      POLICY        │
│───────────────────│       │───────────────────│
│ • PatientID (PK)  │──1:N──│ • PolicyID (PK)   │
│ • PatientAge      │       │ • ClaimType       │
│ • PatientGender   │       │ • ProviderSpec    │
│ • PatientIncome   │       │ • ProviderLoc     │
│ • MaritalStatus   │       │ • Cluster         │
│ • EmployStatus    │       │ • ClaimAmount     │
└───────────────────┘       └────────┬──────────┘
                                     │
                                     │ 1:N
                                     │
                            ┌────────▼──────────┐
                            │       CLAIM        │
                            │───────────────────│
                            │ • ClaimID (PK)    │
                            │ • ClaimDate       │
                            │ • DiagnosisCode   │
                            │ • ProcedureCode   │
                            │ • ClaimStatus     │
                            │ • SubmissionMethod│
                            │ • ClaimLegitimacy │◀── TARGET
                            └───────────────────┘
```

## 3. Module Architecture

### 3.1 Source Code Module Dependency Graph

```
                    ┌─────────────────┐
                    │  run_pipeline.py │  (Master Script)
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
    ┌───────▼───────┐ ┌─────▼─────┐ ┌───────▼───────┐
    │  src/data/    │ │src/features│ │ src/models/   │
    │  • loader.py  │ │• engineer  │ │ • baselines   │
    │  • quality    │ │• encoding  │ │ • probabilist │
    └───────┬───────┘ │• selection │ │ • neighbors   │
            │         └─────┬─────┘ │ • trees       │
            │               │       │ • boosting    │
            │               │       │ • ensembles   │
            │               │       │ • neural      │
            │               │       └───────┬───────┘
            │               │               │
    ┌───────▼───────────────▼───────────────▼───────┐
    │                src/evaluation/                  │
    │  • metrics.py  • benchmark.py                  │
    │  • calibration.py  • fairness.py               │
    └───────────────────────┬───────────────────────┘
                            │
    ┌───────────────────────▼───────────────────────┐
    │                src/reporting/                   │
    │  • visualizations.py  • document_builder.py    │
    │  • model_cards.py                              │
    └───────────────────────┬───────────────────────┘
                            │
    ┌───────────────────────▼───────────────────────┐
    │                 src/utils/                      │
    │  • config.py  • seeds.py  • logging_utils.py  │
    │  • paths.py                                    │
    └────────────────────────────────────────────────┘
```

### 3.2 Data Flow Between Modules

```
loader.py ──► DataFrame(raw)
    │
    ▼
quality_gates.py ──► DataFrame(validated)
    │
    ▼
engineering.py ──► DataFrame(engineered features)
    │
    ▼
encoding.py ──► DataFrame(encoded features)
    │
    ▼
selection.py ──► DataFrame(frozen features)
    │
    ├──► baselines.py ──► metrics.json
    ├──► probabilistic.py ──► metrics.json
    ├──► neighbors.py ──► metrics.json
    ├──► trees.py ──► metrics.json
    ├──► boosting.py ──► metrics.json
    ├──► ensembles.py ──► metrics.json
    └──► neural.py ──► metrics.json
                │
                ▼
         benchmark.py ──► leaderboard.csv
                │
                ▼
         calibration.py ──► calibrated model
                │
                ▼
         visualizations.py ──► *.png files
                │
                ▼
         document_builder.py ──► evaluation.md, PPT, PDF
```

## 4. Artifact Flow Architecture

### 4.1 Artifact Dependency Graph

```
┌────────────────────────────────────────────────────────────────┐
│                      ARTIFACT FLOW                              │
│                                                                 │
│  raw_data.xlsx                                                  │
│       │                                                         │
│       ▼                                                         │
│  validated_data.parquet ──► data_quality_report.json            │
│       │                                                         │
│       ▼                                                         │
│  eda_report.md ──► 34+ EDA images                               │
│       │                                                         │
│       ▼                                                         │
│  preprocessed_data.parquet ──► preprocessing_pipeline.pkl       │
│       │                                                         │
│       ▼                                                         │
│  feature_matrix.parquet ──► frozen_features.yaml                │
│       │                                                         │
│       ├──► train.parquet                                        │
│       ├──► validation.parquet                                   │
│       └──► test.parquet                                         │
│              │                                                  │
│              ▼                                                  │
│  model_artifacts/ ──► 19 × metrics.json                        │
│              │      ──► 19 × roc.csv, pr.csv                   │
│              │      ──► 19 × confusion_matrix.png              │
│              ▼                                                  │
│  leaderboard.csv ──► selection_memo.md                          │
│              │                                                  │
│              ▼                                                  │
│  test_results.json ──► threshold_memo.md                        │
│              │                                                  │
│              ▼                                                  │
│  evaluation.md ──► PPT (20 slides) ──► Project Report PDF      │
│              │                      ──► IEEE Paper PDF          │
│              ▼                                                  │
│  run_manifest.json (records everything)                         │
└────────────────────────────────────────────────────────────────┘
```

## 5. Technology Stack

### 5.1 Technology Radar

| Technology | Category | Ring | Rationale |
|-----------|----------|------|-----------|
| Python 3.11 | Language | ADOPT | Standard for ML |
| scikit-learn 1.4 | ML Framework | ADOPT | Most algorithms |
| pandas 2.2 | Data Processing | ADOPT | DataFrame operations |
| numpy 1.26 | Numerical | ADOPT | Array operations |
| XGBoost 2.0 | Boosting | ADOPT | Top performer |
| LightGBM 4.3 | Boosting | ADOPT | Fast, accurate |
| SHAP 0.44 | Explainability | ADOPT | Model interpretation |
| matplotlib 3.8 | Visualization | ADOPT | Plot generation |
| seaborn 0.13 | Visualization | ADOPT | Statistical plots |
| imbalanced-learn | Resampling | TRIAL | SMOTE handling |
| python-pptx | Reporting | TRIAL | PPT generation |
| PyYAML | Configuration | ADOPT | Config files |

### 5.2 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB |
| Storage | 2 GB free | 5 GB free |
| GPU | Not required | Not required |
| OS | Linux/Windows/Mac | Linux preferred |
| Python | 3.9+ | 3.11 |

## 6. Security & Privacy Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                   SECURITY ARCHITECTURE                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  DATA PRIVACY                                           │   │
│  │  • No real patient data (synthetic/public dataset)      │   │
│  │  • No PII: names, phone, Aadhaar excluded               │   │
│  │  • UUIDs for patient/provider identifiers               │   │
│  │  • All data generated or publicly sourced               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  MODEL ETHICS                                           │   │
│  │  • Fairness analysis across demographic groups           │   │
│  │  • Bias detection and mitigation                         │   │
│  │  • Decision-support only (not auto-denial)              │   │
│  │  • Human investigators remain decision owners           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  CODE INTEGRITY                                         │   │
│  │  • Git version control                                   │   │
│  │  • Deterministic seeds (42)                              │   │
│  │  • No hardcoded paths                                    │   │
│  │  • No credentials in repository                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

*Last updated: 01-09-2026 | Document version: 1.0*
