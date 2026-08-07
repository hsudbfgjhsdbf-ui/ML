# Feature engineering and leakage register

**Purpose:** document every model input and every deliberate exclusion.  
**Run:** `run_20260807_151423`.  
**Last updated:** 07-08-2026 15:14:23 UTC.

## Inclusion lineage

| Feature | Source | Transform | Fraud-detection relevance | Available at screening |
| --- | --- | --- | --- | --- |
| claim_amount_inr | ClaimAmount | renamed | Claim value is the primary financial signal. | yes |
| patient_age_years | PatientAge | renamed | Age supports demographic slice auditing and risk context. | yes |
| patient_income_inr | PatientIncome | renamed | Income contextualizes the requested amount. | yes |
| cluster_code | Cluster | numeric category code | Supplied cluster is treated as a source feature, not a label. | yes |
| log_claim_amount | ClaimAmount | log1p | Reduces the effect of right-skewed claim values. | yes |
| log_patient_income | PatientIncome | log1p | Stabilizes income scale for linear and distance models. | yes |
| claim_to_income_ratio | ClaimAmount, PatientIncome | safe ratio | High claim relative to income can prioritize review. | yes |
| claim_minus_income_scaled | ClaimAmount, PatientIncome | scaled difference | Captures financial mismatch without unbounded subtraction. | yes |
| claim_amount_per_age | ClaimAmount, PatientAge | safe ratio | Captures amount-age interaction without dividing by zero. | yes |
| income_age_interaction | PatientIncome, PatientAge | safe ratio | Represents income context by age. | yes |
| claim_year | ClaimDate | calendar year | Captures broad temporal drift. | yes |
| claim_month | ClaimDate | calendar month | Captures seasonal claim patterns. | yes |
| claim_day_of_week | ClaimDate | weekday index | Captures operational timing patterns. | yes |
| claim_day_of_month | ClaimDate | day of month | Captures billing-cycle timing. | yes |
| claim_is_weekend | ClaimDate | binary calendar flag | Weekend activity is an auditable process signal. | yes |
| claim_month_sin | ClaimDate | cyclic month encoding | Preserves December-to-January continuity. | yes |
| claim_month_cos | ClaimDate | cyclic month encoding | Preserves seasonal phase information. | yes |
| age_band | PatientAge | ordinal bins | Provides interpretable age-group audits. | yes |
| cluster_category | Cluster | nominal category | Allows nonlinear cluster effects without ordinal assumptions. | yes |
| patientgender | PatientGender | normalized category | Retains source context with unknown-category safety. | yes |
| providerspecialty | ProviderSpecialty | normalized category | Retains source context with unknown-category safety. | yes |
| patientmaritalstatus | PatientMaritalStatus | normalized category | Retains source context with unknown-category safety. | yes |
| patientemploymentstatus | PatientEmploymentStatus | normalized category | Retains source context with unknown-category safety. | yes |
| claimtype | ClaimType | normalized category | Retains source context with unknown-category safety. | yes |
| claimsubmissionmethod | ClaimSubmissionMethod | normalized category | Retains source context with unknown-category safety. | yes |

## Exclusion register

| Source field | Action | Reason |
| --- | --- | --- |
| `ClaimID` | drop | Identifier; no predictive meaning and no generalization. |
| `PatientID` | drop | Sensitive/identifier-like field; unique in supplied snapshot. |
| `ProviderID` | drop | Identifier-like; provider aggregation is not available from one-row-per-provider data. |
| `DiagnosisCode` | drop | 4,495 unique values among 4,500 rows; direct memorization risk. |
| `ProcedureCode` | drop | 4,495 unique values among 4,500 rows; direct memorization risk. |
| `ProviderLocation` | drop | 3,876 unique values; supplied values are not verified Indian geography. |
| `ClaimStatus` | drop | May be updated after review; retaining it risks outcome leakage. |
| `ClaimDate` | transform | Calendar components preserve timing without raw datetime dtype. |
| `ClaimLegitimacy` | target only | Target is never included in the feature matrix. |

## Feature families

1. **Financial:** raw amount, log amount, income, log income, claim-to-income ratio, scaled difference, amount-per-age.
2. **Demographic audit context:** age, age band, gender, marital status, employment status.
3. **Clinical/provider category context:** specialty and claim type; no raw high-cardinality code memorization.
4. **Temporal:** year, month, weekday, day of month, weekend flag, cyclic month terms.
5. **Source latent structure:** cluster numeric and nominal views, retained with a shortcut-risk note.

## Leakage test

The target column is explicitly absent from the engineered features. The fitted
transformer is trained on `x_train` only. A reviewer can verify this with
`python -m scripts.verify_artifacts` after a run. Current lineage row count:
**25**. Any future change must add a new decision-register
entry and rerun the full benchmark.
