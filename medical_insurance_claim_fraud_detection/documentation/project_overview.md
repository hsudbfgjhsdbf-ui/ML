# Project Overview

## Title
Medical Insurance Claim Fraud Detection — AI-Driven Claim Verification & Explainable Fraud Detection Platform

## Motivation
Medical insurance fraud causes billions in losses annually, increases premiums, and burdens legitimate patients. Manual verification is slow and error-prone. AI can assist but must be explainable and human-in-the-loop.

## Central Objectives (from reference PDF)
1. Collect claimant, policy, incident, and supporting-document information.
2. Detect potentially fraudulent medical insurance claims.
3. Extract information from medical bills, prescriptions, discharge summaries, and related documents using OCR and optional Vision-Language Model APIs.
4. Validate claims against policy rules, historical claims, medical information, and fraud indicators.
5. Experiment with classical ML, deep learning, anomaly detection, document intelligence, agentic AI, and RAG-based approaches.
6. Benchmark approaches using appropriate evaluation metrics.
7. Produce transparent, human-readable explanations for approval, manual review, or rejection.

## System Components
- **Data Layer**: Raw claims Excel, processed CSV, sample fixtures, knowledge base policy rules
- **Preprocessing**: Date engineering, scaling, OHE, imputation, SMOTE, leakage detection
- **Approaches**:
  - 01 Traditional ML: benchmarking 14+ classifiers
  - 02 Deep Learning: MLP with PyTorch/TF or sklearn fallback
  - 03 Anomaly Detection: IsolationForest, LOF, OneClassSVM, Robust, Autoencoder, Ensemble
  - 04 Document Intelligence: OCR (Tesseract/EasyOCR/PaddleOCR) + VLM optional interface + validation
  - 05 Agentic RAG: 7 agents + RAG over policy docs + grounded explanations
  - 06 Hybrid End-to-End: Fusion + decision synthesis + risk categories
- **Evaluation**: PR-AUC primary, ROC-AUC, F1/F2, MCC, Balanced Accuracy, Precision@k, Calibration, Runtime
- **Explainability**: Feature importance, SHAP attempted, human-readable auditable summaries
- **Deployment**: FastAPI/Flask optional, JSON contract for Next.js

## Architecture
See `system_architecture.md` and `images/architecture_diagram.png`

## Ethics & Privacy
- No real PHI
- Synthetic IDs
- External APIs disabled by default
- Human review mandatory for REJECT_OR_ESCALATE
- See `ethics_privacy_and_limitations.md`

## Team
IIIT Dharwad, B.Tech Data Science & AI.

## Repository Structure
```
medical_insurance_claim_fraud_detection/
├── approaches/ (6 files)
├── common/ (utilities)
├── data/
├── evaluation/
├── images/
├── relations/
├── documentation/
├── presentation/
├── report/
├── api/
├── tests/
├── config.yaml
├── requirements.txt
└── run_pipeline.py
```

## How Started
- Started with dataset inspection (Health Insurance Fraud Claims.xlsx)
- Validated schema, target distribution (94% legit, 6% fraud)
- Built reusable common utilities for config, logging, seed, preprocessing, metrics
- Implemented traditional ML first as baseline, then DL, anomaly, document, RAG, hybrid
- Generated evaluations, visualizations, docs, presentation, report

## Key Results
- Best traditional model: DecisionTree / RandomForest / HistGradientBoosting with PR-AUC ~1.0 on this synthetic dataset (suggests high separability, possibly income feature leakage-like).
- Deep learning (sklearn MLP fallback): PR-AUC 0.87 val, 0.90 test - does NOT outperform tree models on tabular data.
- Anomaly: LOF best PR-AUC 0.147, highlighting limitation of unsupervised for fraud.
- Document: Bill total mismatch detection works.
- Hybrid: Combines signals, conservative manual review zone.

## Limitations & Future Work
- Dataset synthetic, small (4500)
- No real clinical notes
- Need temporal provider history, graph features for collusion
- Need calibrated probabilities, fairness audit
- Production needs FHIR integration, secure storage, audit trails
