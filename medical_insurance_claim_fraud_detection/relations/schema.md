# Data Schema and Entity Relationships

## Overview
The project uses claim-level medical insurance data. In production, the full system would encompass multiple entities; the current dataset provides a subset.

## Entities

### Claimant / Patient
- PatientID (PK, anonymized)
- PatientAge, PatientGender, PatientIncome, PatientMaritalStatus, PatientEmploymentStatus
- Related to Policyholder (may be same as patient or family)
- One patient can have many claims (one-to-many)

### Policyholder
- Not explicitly in dataset (assumed same as patient for demo)
- In production: PolicyholderID, PolicyNumber, Coverage, Premium, Active status
- Relationship: Policyholder owns Policy (1-many)

### Policy
- PolicyNumber (not in dataset, synthetic in docs)
- Coverage rules, exclusion clauses, deductible, limits
- Relationship: Policy covers many Claims (1-many)

### Provider
- ProviderID (PK)
- ProviderSpecialty, ProviderLocation
- Relationship: Provider submits many Claims (1-many)
- Provider has historical fraud rate (derived)

### Claim
- ClaimID (PK)
- ClaimAmount, ClaimDate, DiagnosisCode, ProcedureCode, ClaimType, ClaimSubmissionMethod, ClaimStatus, Cluster
- FK: PatientID, ProviderID, PolicyNumber (implicit)
- Relationship: One Claim has many Documents (1-many)
- Relationship: One Claim has one Diagnosis, one Procedure (many-to-one to code tables)

### Diagnosis
- DiagnosisCode (PK, ICD-like)
- Description
- Relationship: Many Claims share one Diagnosis (many-to-one)

### Procedure
- ProcedureCode (PK)
- Description, Cost baseline
- Relationship: Many Claims share one Procedure

### Bill / Document
- DocumentID, DocumentType (medical_bill, prescription, discharge_summary, investigation_report, identity_document, policy_document)
- BillTotal, BillDate, Hospital, ProviderName
- FK: ClaimID
- Relationship: Duplicate detection via hash
- Validation: Bill total vs ClaimAmount

### Historical Claim
- Partition of Claim: past claims for peer comparison
- Used for anomaly detection: amount distribution per specialty, frequency per patient

### Fraud Label
- ClaimLegitimacy: Legitimate / Fraud (ground truth)
- Not real-time; labels for supervised training
- Distinguish: Confirmed label (dataset) vs Model prediction vs Anomaly indicator vs Rule violation

### Review Decision
- Decision: APPROVE, FLAG_FOR_MANUAL_REVIEW, REJECT_OR_ESCALATE
- Human reviewer, timestamp, evidence references
- Audit log

## Production Schema Proposal (vs Actual Dataset)

Actual dataset contains:
- Claim, Patient simplified, Provider simplified, DiagnosisCode, ProcedureCode, Bill amount/date (via ClaimAmount, ClaimDate), Fraud label

Missing in dataset but proposed for production:
- Policy table with coverage, exclusions, pre-auth rules
- Document table with OCR extracted fields (we simulate via synthetic JSON fixtures)
- Review decision audit table
- Provider history longitudinal
- Patient policy mapping

## Feature Relationships (CSV)
See `feature_relationships.csv`

## Correlation Analysis
See `correlation_analysis.md`

## Data Lineage
See `data_lineage.md`

## Diagram
See `entity_relationship_diagram.mmd` and `.png`
