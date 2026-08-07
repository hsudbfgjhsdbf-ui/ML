# Dataset card — supplied health-insurance fraud workbook

## Motivation

The workbook is used to demonstrate a reproducible binary fraud-screening
pipeline for an academic project at IIIT Dharwad. It is not presented as a
national Indian claims sample.

## Composition

- File: `data/raw/health_insurance_fraud_claims.xlsx`
- Rows: 4,500; source columns: 19
- Fraud rows: 270; legitimate rows: 4,230; prevalence: 0.0600
- Exact duplicates: 0; duplicate ClaimID: 0
- Missing cells: 0
- Checksum: `952b6e23c9845b9086994c08bc81323575968c318f845ea69e3b9609614b6a45`

## Collection and provenance

The workbook was present in the repository supplied for this task. The original
creator, collection process, license, and redistribution permissions are not
verified in the checkout. The project preserves the source copy and records the
unknown-license status rather than attributing it to Kaggle or a government
source without evidence.

## Preprocessing

The target is `ClaimLegitimacy`; `Fraud` maps to 1. IDs and near-unique codes
are audit-only or excluded. Numeric and categorical transformations are fitted
on the training partition. See `data/data_dictionary.csv` and
`documentation/feature_engineering.md`.

## Uses and non-uses

Use for academic reproducibility, code review, and baseline comparison. Do not
use for real claim denial, risk pricing, claimant profiling, or regulatory
submission. No real patient-identifying fields were observed, but identifiers
are still treated as sensitive and omitted from printed examples.

## Known limitations and biases

The sample is small, has one row per unique patient/provider in this snapshot,
contains no policy or document evidence, lacks verified Indian geography, and
may contain synthetic shortcuts such as Cluster. Demographic performance is
therefore an audit exercise, not proof of fairness in production.

## FAQ

**Why not use the large Medicare table plan?** The supplied repository contains
this workbook; no other claims tables are available.  
**Is the workbook public and license-cleared?** That is not verifiable from the
checkout and must be confirmed by the data owner.  
**Are amounts confirmed INR?** No; the pipeline does not invent a conversion.  
**Can the result be generalized to Indian insurers?** No; external validation is required.  
**What is the target?** `Fraud` is positive and `Legitimate` is negative.
