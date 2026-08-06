# Hybrid End-to-End Pipeline

## Components
- Best traditional ML model
- Deep learning where useful (fallback)
- Anomaly scores
- Document validation
- Policy checks via RAG
- Explainability
- Human-review rules

## Decision thresholds
{'approve_max_prob': 0.3, 'review_min_prob': 0.3, 'review_max_prob': 0.7, 'reject_min_prob': 0.7}

Weights: ML 0.5, DL 0.2, Anomaly 0.15, Document 0.15

## Outcomes
APPROVE, FLAG_FOR_MANUAL_REVIEW, REJECT_OR_ESCALATE

Conservative manual review zone 0.3-0.7

## Sample result
{
  "claim_id": "6eea92b2-bd25-484d-94bd-278706e7f11c",
  "model_version": "hybrid_v1.0",
  "fraud_probability": 0.0,
  "fraud_prediction": 0,
  "anomaly_score": 0.404009656492165,
  "document_validation_status": "FAILED",
  "policy_validation_status": "PASSED",
  "risk_category": "HIGH",
  "recommended_decision": "FLAG_FOR_MANUAL_REVIEW",
  "key_risk_signals": [
    "Missing required documents: prescription, discharge_summary",
    "Document validation FAILED",
    "Bill total 6000.0 != claimed 1703.04",
    "Diagnosis mismatch doc Fo766 vs claim Bj740"
  ],
  "positive_evidence": [
    "No strong historical anomaly"
  ],
  "missing_or_inconsistent_info": [
    "prescription",
    "discharge_summary",
    "Validation error: Bill total 7820.52 does not match claimed amount 1703.04 diff 6117.48"
  ],
  "explanation": "Fraud risk probability estimated at 0.00 (HIGH risk). Key risk signals: Missing required documents: prescription, discharge_summary; Document validation FAILED; Bill total 6000.0 != claimed 1703.04; Diagnosis mismatch doc Fo766 vs claim Bj740. Positive evidence: No strong historical anomaly. Evidence sources: document_intelligence, /home/user/ML/medical_insurance_claim_fraud_detection/data/sample/synthetic_bill_1.json, /home/user/ML/medical_insurance_claim_fraud_detection/data/sample/synthetic_bill_mismatch.json. Recommended operational decision: FLAG_FOR_MANUAL_REVIEW. Flagged for manual review due to moderate risk or validation issues; reviewer should verify documents and policy coverage. Disclaimer: This result is decision support and not a final legal or insurance determination. Human reviewer must remain involved.",
  "evidence_references": [
    {
      "type": "missing_docs",
      "details": [
        "prescription",
        "discharge_summary"
      ],
      "source": "document_intelligence"
    },
    {
      "type": "doc_error",
      "doc": "medical_bill",
      "errors": [
        "Bill total 7820.52 does not match claimed amount 1703.04 diff 6117.48"
      ],
      "source": "/home/user/ML/medical_insurance_claim_fraud_detection/data/sample/synthetic_bill_1.json"
    },
    {
      "type": "doc_error",
      "doc": "medical_bill",
      "errors": [
        "Bill total 6000.0 does not match claimed amount 1703.04 diff 4296.96"
      ],
      "source": "/home/user/ML/medical_insurance_claim_fraud_detection/data/sample/synthetic_bill_mismatch.json"
    },
    {
      "text": "Policy Rules - Medical Insurance 1. Policy must be active at time of service. 2. Claim must be submitted within 30 days of discharge. 3. Pre-authorization required for procedures costing > $10,000. 4. Only licensed providers in network are covered; out-of-network requires special approval. 5. Duplicate claim IDs are not allowed. 6. Claim amount must match sum of bill line items within $5 tolerance. 7. Diagnosis must justify procedure per medical necessity table. 8. Patient age must be consistent",
      "source": "/home/user/ML/medical_insurance_claim_fraud_detection/data/sample/knowledge_base/policy_rules.txt",
      "score": 0.2390698597515511
    },
    {
      "text": "Coverage Rules - Inpatient coverage includes room, surgery, anesthesia up to policy limit. - Outpatient: consultations and labs covered at 80% after deductible. - Emergency: 100% covered if genuine emergency, else 70%. - Deductible $500 per year, out-of-pocket max $5000. - Pre-authorization: required for MRI, CT, major surgery. - Generic prescriptions preferred; branded need justification.",
      "source": "/home/user/ML/medical_insurance_claim_fraud_detection/data/sample/knowledge_base/coverage_rules.txt",
      "score": 0.20187423788636477
    },
    {
      "text": "Claim Processing Guidelines 1. Verify policy active and premium paid. 2. Validate documents: bill, prescription, discharge summary, ID proof. 3. Check duplicates via ClaimID and PatientID+Date+Amount. 4. Cross-check diagnosis and procedure codes for medical necessity. 5. Run fraud model; if probability >0.7 escalate. 6. If document validation fails, request resubmission or flag for manual review. 7. If anomaly score in top 5% vs history, manual review. 8. For flagged claims, retrieve policy rule",
      "source": "/home/user/ML/medical_insurance_claim_fraud_detection/data/sample/knowledge_base/claim_guidelines.txt",
      "score": 0.18700590220218852
    },
    {
      "text": "Fraud Indicators - Unusually high claim amount vs peer group (>3 std dev). - Upcoding: billing for more expensive procedure than performed. - Duplicate billing same date same procedure. - Patient age inconsistent with diagnosis. - Provider billing far outside specialty (e.g., orthopedics billing cardiology). - Claim submission date before service date. - Inconsistent provider location vs patient location without referral. - Missing discharge summary for inpatient claims. - Altered documents, mis",
      "source": "/home/user/ML/medical_insurance_claim_fraud_detection/data/sample/knowledge_base/fraud_indi