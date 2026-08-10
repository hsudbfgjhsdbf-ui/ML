# Document Intelligence Approach

OCR engine: fallback
VLM enabled: False

## Supported document types
Medical bills, prescriptions, discharge summaries, investigation reports, ID/policy docs

## Pipeline
1. OCR extraction (Tesseract/EasyOCR/PaddleOCR/fallback)
2. Type identification (keyword + structured)
3. Field extraction regex + structured
4. Validation: bill total, date consistency, provider, policyholder, duplicate, missing docs
5. Risk scoring
6. Optional VLM interface (env controlled)

## Privacy
- No API keys in code
- External calls disabled by default
- Synthetic fixtures for testing
- PII redaction

## Output JSON
Contains extracted_fields, confidences, validation errors, risk indicators, document_risk LOW/MEDIUM/HIGH

## Sample result
{
  "claim_id": "6eea92b2-bd25-484d-94bd-278706e7f11c",
  "documents_processed": 4,
  "document_results": [
    {
      "document_path": "/home/user/ML/medical_insurance_claim_fraud_detection/data/sample/synthetic_bill_1.json",
      "document_hash": "292def6eeced8379d569ce9f6f9dc671b08674e4d6cd50fbb31aad3c1af59710",
      "document_type": "medical_bill",
      "document_type_confidence": 0.99,
      "ocr_text_preview": "{\n  \"document_type\": \"medical_bill\",\n  \"claim_id\": \"4d76c7f7-d36a-4139-b451-a9a4ad10d7d5\",\n  \"patient_name\": \"REDACTED\",\n  \"provider_name\": \"General Hospital - New Alishaview\",\n  \"bill_date\": \"2024-07-08\",\n  \"discharge_date\": \"2024-07-07\",\n  \"total_amount\": 7820.52,\n  \"items\": [\n    {\n      \"description\": \"Room Charges\",\n      \"amount\": 3000\n    },\n    {\n      \"description\": \"Surgery\",\n      \"amount\": 4000\n    },\n    {\n      \"description\": \"Lab\",\n      \"amount\": 820.52\n    }\n  ],\n  \"diagnosis_code\": \"Ta150\",\n  \"procedure_code\": \"iO013\"\n}",
      "ocr_confidence": 0.95,
      "extracted_fields": {
        "claim_number": "4d76c7f7-d36a-4139-b451-a9a4ad10d7d5",
        "provider_name": "General Hospital - New Alishaview",
        "bill_total": 7820.52,
        "bill_date": "2024-07-08",
        "diagnosis_code": "Ta150",
        "procedure_code": "iO013"
      },
      "field_confidences": {
        "claim_number": 0.95,
        "provider_name": 0.95,
        "bill_total": 0.95,
        "bill_date": 0.95,
        "diagnosis_code": 0.95,
        "procedure_code": 0.95
      },
      "validation": {
        "status": "FAILED",
        "errors": [
          "Bill total 7820.52 does not match claimed amount 1703.04 diff 6117.48"
        ],
        "warnings": [],
        "risk_indicators": [
          "amount_mismatch"
        ],
        "validation_confidence": 0.9500000000000001
      },
      "vlm": {
        "vlm_used": false,
        "reason": "VLM disabled via config/env"

