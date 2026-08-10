# Agentic RAG Reasoning

## Agents
- Document Verification
- Policy Rule Matching
- Claim Consistency
- Historical Pattern/Anomaly
- Evidence Retrieval
- Decision Synthesis
- Explanation Generation

## RAG
KB dir /home/user/ML/medical_insurance_claim_fraud_detection/data/sample/knowledge_base chunks 5
Chunk size 500 overlap 50 TFIDF fallback or sentence-transformers if configured.

## Grounding
Every explanation grounded in retrieved evidence, extracted fields, model outputs, rules.

## LLM
Optional, controlled via LLM_ENABLED env. Deterministic fallback used by default.

## Sample Output
{
  "claim_id": "6eea92b2-bd25-484d-94bd-278706e7f11c",
  "model_version": "agentic_rag_v1.0",
  "fraud_probability": 0.65,
  "anomaly_score": 0.7,
  "agent_results": [
    {
      "agent": "DocumentVerificationAgent",
      "status": "PASSED",
      "confidence": 0.9,
      "risk_signals": [],
      "evidence": [],
      "positive_evidence": [
        "All required documents present"
      ]
    },
    {
      "agent": "PolicyRuleMatchingAgent",
      "status": "PASSED",
      "confidence": 0.7,
      "risk_signals": [],
      "evidence": [
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
          "source": "/home/user/ML/medical_insurance_claim_fraud_detection/data/sample/knowledge_base/fraud_indicators.txt",
          "score": 0.1549826115393207
        },
        {
          "text": "Exclusion Clauses - Cosmetic surgery not covered unless post-trauma reconstructive. - Experimental treatments not covered. - Self-inflicted injuries excluded. - Claims with fraudulent documents automatically flagged. - Pre-existing conditions have 90-day waiting period. - Dental procedures only covered under dental rider.",
          "source": "/home/user/ML/medical_insurance_claim_fraud_detection/data/sample/knowledge_base/exclusion_clauses.txt",
          "score": 0.0
        }
      ],
      "positive_evidence": [],
      "retrieved": [
        {
          "id": "policy_rules.txt_chunk_0",
          "text": "Policy Rules - Medical Insurance 1. Policy must be active at time of service. 2. Claim must be submitted within 30 days of discharge. 3. Pre-authorization required for procedures costing > $10,000. 4. Only licensed providers in network are covered; out-of-network requires special approval. 5. Duplicate claim IDs are not allowed. 6. Claim amount must match sum of bill line items within $5 tolerance. 7. Diagnosis must justify procedure per medical necessity table. 8. Patient age must be consistent with procedure (e.g., pediatric codes not for adults >18 unless exception). 9. Routine checkups limited to 2 per year. 10. Emergency claims require ER documentation.",
          "source": "/home/user/ML/medical_insurance_claim_fraud_detection/data/sample/knowledge_base/policy_rules.txt",
          "source_name": "policy_rules.txt",
          "chunk_index": 0,
          "score": 0.2390698597515511
        },
        {
          "id": "coverage_rules.txt_chunk_0",
          "text": "Coverage Rules - Inpatient coverage includes room, surgery, anesthesia up to policy limit. - Outpatient: 