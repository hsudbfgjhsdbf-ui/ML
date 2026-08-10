# Data Lineage

## Source
- Original: `Health Insurance Fraud Claims.xlsx` provided locally at `/home/user/ML/Health Insurance Fraud Claims.xlsx`
- Copied to `data/raw/Health_Insurance_Fraud_Claims.xlsx` (gitignored content but manifest preserved)
- Converted to CSV `data/processed/claims_processed.csv` (4500 rows, 19 cols)
- Sample 100 rows `data/sample/sample_100.csv`

## Processing Steps
1. **Load**: pandas read_excel
2. **Validate**: schema check, missing, class distribution
3. **Target Encoding**: Legitimate=0, Fraud=1
4. **Date Engineering**: ClaimDate -> year, month, day, dayofweek, quarter, ordinal (via `engineer_date_features`)
5. **Split**: Stratified train 65% (2925), val 15% (675), test 20% (900), random_state 42
6. **Preprocessing Pipeline**:
   - Numerical: median impute + StandardScaler
   - Categorical: most_frequent impute + OneHotEncoder(handle_unknown=ignore)
   - Learned only on training data (no leakage)
   - Transformations applied to val/test via pipeline
7. **SMOTE**: Optional, applied only inside training folds if configured
8. **Model Training**: Multiple classifiers, CV 5-fold, scoring average_precision (PR-AUC)
9. **Calibration**: Isotonic/Sigmoid (attempted, fallback if fails)
10. **Threshold Tuning**: Optimize F2 on validation
11. **Evaluation**: Metrics on test, confusion matrix, calibration curve, runtime
12. **Artifacts**: Best model saved to `data/processed/artifacts/best_traditional_ml_model.joblib`

## Derived Artifacts
- `evaluation/model_comparison.csv`
- `evaluation/metrics_summary.json/csv`
- `evaluation/threshold_analysis.csv`
- `evaluation/feature_importance.csv`
- `images/*.png` visualizations

## Document Intelligence Lineage
- Synthetic fixtures in `data/sample/synthetic_*.json`
- No real PHI used
- OCR fallback reads JSON directly
- Validation output -> `evaluation/document_intelligence_sample_output.json`

## RAG Lineage
- Knowledge base `data/sample/knowledge_base/*.txt` (policy_rules, fraud_indicators, etc)
- Chunking 500 words overlap 50
- TF-IDF vectorization (fallback) or sentence-transformers if available
- Similarity search top_k 5

## Hybrid Lineage
- Combines best ML model, anomaly models, doc validation, RAG
- Output `evaluation/hybrid_sample_result.json` and `api/sample_response.json`

## Reproducibility
- Random seed 42 set globally (python, numpy, torch, tf if available)
- Config in `config.yaml`
- Requirements in `requirements.txt`

## Privacy
- UUIDs synthetic
- No real PII transmitted externally
- ENABLE_EXTERNAL_API_CALLS=false by default
- ANONYMIZE_PII=true
