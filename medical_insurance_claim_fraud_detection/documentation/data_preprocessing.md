# Data Preprocessing

## Steps
1. **Load**: Excel via pandas read_excel, fallback CSV
2. **Schema Inspection**: rows, cols, missing, target distribution
3. **Target Mapping**: Legitimate->0, Fraud->1
4. **Date Engineering** (`common/preprocessing.py:engineer_date_features`):
   - Parse ClaimDate to datetime
   - Derive year, month, day, dayofweek, quarter, ordinal days
   - Drop original ClaimDate (to avoid leakage, use engineered)
5. **Feature Typing** from config.yaml:
   - Numerical: ClaimAmount, PatientAge, PatientIncome, Cluster + engineered date cols
   - Categorical: PatientGender, ProviderSpecialty, ClaimStatus, PatientMaritalStatus, PatientEmploymentStatus, ProviderLocation, ClaimType, ClaimSubmissionMethod, DiagnosisCode, ProcedureCode
   - Date: ClaimDate (engineered before pipeline)
   - Drop: ClaimID, PatientID, ProviderID (IDs not predictive but could leak)
6. **Pipeline** (`build_preprocessor`):
   - Numerical: SimpleImputer(median) + StandardScaler (or RobustScaler)
   - Categorical: SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown=ignore, sparse_output=False)
   - ColumnTransformer remainder=drop, learned ONLY from training data
   - No leakage: fit on train, transform val/test
7. **Class Imbalance Handling**:
   - Check imbalance ratio (0.0638)
   - Use class_weight balanced for models supporting it
   - Optional SMOTE: applied only inside training folds or on train resampled (k_neighbors 5)
   - Documented in data_quality_report.md
8. **Outlier Analysis**: IQR method for numerical, no removal but logged
9. **Leakage Detection**: Heuristic keyword scan for status/label/outcome columns; ClaimStatus flagged; documented but retained with caution note
10. **Splitting**: Stratified train 65%, val 15%, test 20%, random_state 42, shuffle
    - Time-aware note: ClaimDate within short window (July 2024), so stratified suffices; future work could sort by ordinal for time-aware split
    - Group-aware: could group by ProviderID to avoid provider leak, but dataset small, so stratified used with note
11. **Missing**: None in raw, but pipeline includes imputation for robustness

## Reproducibility
- Seed 42 via `set_global_seed`
- Configurable paths, no hard-coded absolute paths

## Artifacts
- Processed CSV `data/processed/claims_processed.csv`
- Preprocessor joblib saved in artifacts for inference

## Limitations
- OHE on high-cardinality DiagnosisCode, ProcedureCode, ProviderLocation creates 8000+ features, memory heavy, sparse would be better but we use dense for simplicity
- No text embedding for codes (future: learned embeddings)
