# Ethics, Privacy, and Limitations

## Sensitive Health Information
- Dataset contains health-related: diagnosis codes, procedure codes, age, gender
- All IDs synthetic UUIDs, not real PII
- Still treat as sensitive: minimization, no logging IDs in plaintext (but for demo we log counts)
- No real medical records transmitted externally by default

## PII
- PatientID, ProviderID, ClaimID anonymized
- PatientIncome, Age, Gender quasi-identifiers
- For OCR demo: patient_name REDACTED in synthetic fixtures
- ANONYMIZE_PII=true config, privacy_safeguard field in doc output

## Data Minimization
- Only needed columns used for modeling
- IDs dropped for training
- Document text preview truncated to 1000 chars

## Encryption and Access Control
- Future production: encrypt at rest, TLS in transit, RBAC for reviewers, audit logs
- Current prototype: local files, no encryption implemented, documented as future work

## API Data-Sharing Risks
- VLM_API_KEY, LLM_API_KEY never in source, via .env
- ENABLE_EXTERNAL_API_CALLS=false by default
- WARNING: Do not send real medical records to external VLM/LLM APIs
- Fallback deterministic rules used when API disabled

## Fairness
- Potential bias across demographic/geographic groups
- Check gender, income, location, specialty, marital status for disparate impact
- Fraud rate by ProviderSpecialty, PatientGender visualizations provided
- Model may penalize high income? Need audit SHAP dependence
- Recommendation: fairness metrics per subgroup, equal opportunity, reduce threshold bias

## False Positives
- Flagging legitimate claim as fraud delays care, causes financial stress, harms patient trust
- Mitigation: conservative manual review zone, high precision threshold, human reviewer mandatory, appeal mechanism
- F2 score optimization prioritizes recall but maintain precision

## False Negatives
- Missing fraud causes financial loss, premium increase
- Mitigation: high recall target, anomaly detection as additional signal, monitoring

## Human Review
- System is decision-support prototype, NOT autonomous legal/medical/insurance decision-maker
- Human reviewer must remain involved, especially for REJECT_OR_ESCALATE or high-impact claims
- UI must show explanation, evidence refs, disclaimer prominently
- Appeal and correction mechanism required for production

## Appeal and Correction
- Production should have: patient notification, reason for flag, evidence, reviewer contact, timeline for appeal, correction of data errors
- Current prototype: JSON includes explanation, missing info, evidence refs for reviewer

## Danger of Auto-Rejection
- Auto-reject based only on model output is dangerous, illegal in many jurisdictions
- Must have human-in-the-loop, audit, compliance, fairness checks

## Limitations
- Synthetic dataset not fully representative
- 4500 rows modest, no temporal provider history, no free-text notes, no real doc images
- High-cardinality OHE memory heavy, not sparse
- ClaimStatus potential leakage
- Near-perfect traditional ML suggests synthetic separability, not real-world performance
- Anomaly detection low PR-AUC, high FP
- Deep learning underperforms trees on tabular
- OCR fallback not real OCR, needs Tesseract/EasyOCR/PaddleOCR for production
- RAG TFIDF fallback less semantic than dense embeddings
- No graph-based collusion detection
- No real-time FHIR integration

## Future Improvements
- Larger real-world CMS data with proper licensing
- Time-aware splitting, provider group split
- Embeddings for diagnosis/procedure codes
- Graph features for provider-patient networks
- Fairness audit per subgroup
- Calibration + cost-sensitive optimization
- Encrypted storage, secure API, audit trails
- Next.js frontend with reviewer workflow
- Model monitoring, drift detection, retraining pipeline
