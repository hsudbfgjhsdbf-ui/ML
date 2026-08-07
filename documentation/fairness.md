# Fairness and demographic audit

**Purpose:** audit disparate error patterns without using sensitive fields as
model identifiers.  
**Run:** `run_20260807_151423`.  
**Last updated:** 07-08-2026 15:14:23 UTC.

Age, gender, claim type, and employment status are evaluated as audit slices.
The model matrix excludes IDs and raw location strings. Slice metrics are
unstable when a group has few positive examples; small cells remain visible but
are not used for a strong disparity claim.

| Slice | Value | Rows | Fraud rows | TPR | FPR | Precision | Stability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PatientGender | F | 356 | 25 | 0.9600 | 0.0000 | 1.0000 | stable |
| PatientGender | M | 319 | 15 | 1.0000 | 0.0000 | 1.0000 | stable |
| age_band | 0_17 | 103 | 5 | 1.0000 | 0.0000 | 1.0000 | stable |
| age_band | 18_30 | 95 | 7 | 1.0000 | 0.0000 | 1.0000 | stable |
| age_band | 31_45 | 97 | 8 | 0.8750 | 0.0000 | 1.0000 | stable |
| age_band | 46_60 | 115 | 4 | 1.0000 | 0.0000 | 1.0000 | stable |
| age_band | 61_75 | 102 | 5 | 1.0000 | 0.0000 | 1.0000 | stable |
| age_band | 76_plus | 163 | 11 | 1.0000 | 0.0000 | 1.0000 | stable |
| ClaimType | Emergency | 185 | 14 | 1.0000 | 0.0000 | 1.0000 | stable |
| ClaimType | Inpatient | 159 | 10 | 1.0000 | 0.0000 | 1.0000 | stable |
| ClaimType | Outpatient | 151 | 6 | 0.8333 | 0.0000 | 1.0000 | stable |
| ClaimType | Routine | 180 | 10 | 1.0000 | 0.0000 | 1.0000 | stable |
| PatientEmploymentStatus | Employed | 177 | 11 | 1.0000 | 0.0000 | 1.0000 | stable |
| PatientEmploymentStatus | Retired | 157 | 15 | 0.9333 | 0.0000 | 1.0000 | stable |
| PatientEmploymentStatus | Student | 160 | 10 | 1.0000 | 0.0000 | 1.0000 | stable |
| PatientEmploymentStatus | Unemployed | 181 | 4 | 1.0000 | 0.0000 | 1.0000 | stable |

## Interpretation policy

A gap greater than five percentage points is a review trigger, not a finding of
discrimination. The supplied workbook is not a representative Indian
population sample, and demographic variables may be generated or incomplete.
Mitigations to explore in a future validated dataset include reweighting,
removing shortcut fields, group-aware threshold analysis with legal review, and
additional data collection. Sensitive attributes are retained for auditing only.
The human review pathway must be available to every group.
