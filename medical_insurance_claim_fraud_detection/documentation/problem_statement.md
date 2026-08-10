# Problem Statement

## Background
Medical insurance claim fraud includes:
- Upcoding, unbundling, duplicate billing
- Fake or altered documents
- Medically unnecessary procedures
- Identity theft, provider collusion

## Problem
Given:
- Claimant demographics (age, gender, income, marital, employment)
- Policy information (coverage, exclusions) 
- Incident details (diagnosis, procedure, provider, location, date, amount, type)
- Supporting documents (bills, prescriptions, discharge summaries)

Determine:
- Fraud risk (probability 0-1)
- Risk category (LOW/MEDIUM/HIGH)
- Recommended action (APPROVE, FLAG_FOR_MANUAL_REVIEW, REJECT_OR_ESCALATE)
- Transparent explanation with evidence references

## Why Important
- Financial: reduces losses, lowers premiums
- Operational: speeds manual review, prioritizes suspicious claims
- Patient: protects legitimate claims from delayed denial, prevents identity fraud
- Compliance: audit trail, explainability for regulators

## Prediction Unit Clarification
- This project: **claim-level** prediction (each row = one claim). Target `ClaimLegitimacy` legitimate/fraud.
- Provider-level fraud (e.g., CMS provider fraud) aggregates claims per provider-period. If using such dataset, must clearly state prediction unit is provider, not claim. We do NOT misrepresent.

## Challenges
- Highly imbalanced (6% fraud)
- Potential leakage: ClaimStatus is post-decision; need to document and handle
- High-cardinality categorical: DiagnosisCode, ProcedureCode, ProviderLocation
- Missing temporal history
- Need explainability, not black-box auto-rejection
- Privacy: health data sensitive

## Success Criteria
- PR-AUC primary (fraud precision-recall), not accuracy alone
- Good recall at acceptable precision (F2 prioritized)
- Threshold analysis documented, conservative manual review zone
- Explainable outputs with evidence
- System runs without paid APIs (fallback behavior)
- Human review mandatory for high-impact decisions
