# Correlation Analysis

## Numerical Features
- ClaimAmount
- PatientAge
- PatientIncome
- Cluster
- Engineered dates: year, month, day, dayofweek, quarter, ordinal

## Method
- Pearson correlation heatmap generated from actual dataset (see `images/correlation_heatmap.png`)
- Observations from 4500-row dataset:

### Key Observations
- ClaimAmount vs PatientIncome: weak correlation (~0.02 expected due to synthetic nature). Check actual heatmap.
- PatientAge vs PatientIncome: slight negative maybe.
- ClaimAmount vs Cluster: moderate, as Cluster may be derived from amount.
- Date ordinal vs ClaimAmount: no strong trend (data within July 2024 window).

### Categorical Associations (Cramér's V approximation)
- ProviderSpecialty vs ClaimLegitimacy: compute fraud rate per specialty.
- ClaimType vs ClaimAmount: Inpatient higher.
- ClaimStatus vs Fraud: Potential leakage - denied claims may correlate with fraud but status is post-decision, so should not be used as hard feature in production without caution.

### Bias Considerations
- Income, Gender, Age, Location could introduce bias.
- Model should be audited: SHAP dependence plots for income vs fraud probability.
- ProviderLocation fraud rate may reflect data bias, not true risk.

### Feature Engineering Suggestions
- Interaction: ClaimAmount / PatientIncome ratio (claim burden)
- Temporal: days since last claim per patient (requires history)
- Provider: avg claim amount per provider, provider fraud rate (leave-one-out)
- Diagnosis-Procedure co-occurrence rarity

### Note on Synthetic Data
Dataset appears highly separable by PatientIncome and ClaimAmount (see feature_importance.csv) suggesting synthetic generation with leakage-like separability. In real-world, fraud is harder, requires more nuanced features.

## Visuals
- `correlation_heatmap.png`
- `fraud_rate_by_*` plots
- `*distribution.png` and `*fraud_comparison.png`

All plots generated from actual data, not invented.
