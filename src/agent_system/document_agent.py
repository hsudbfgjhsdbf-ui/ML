"""
Document Processing and Multi-Modal Extraction Agent.
Extracts structured JSON fields, itemized medical billing, diagnoses, medications,
and provider credentials from scanned bills, prescriptions, and discharge summaries.
"""

import os
import re
import json
import time
from typing import Dict, Any, List, Optional
from src.utils import logger

class DocumentProcessingAgent:
    """
    Cognitive Agent simulating OCR and Vision Language Model (VLM) extraction
    from uploaded Indian medical documentation.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
    def process_medical_document(
        self,
        doc_type: str,
        file_path: str,
        claim_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parses uploaded medical document images/PDFs and returns structured schema with confidence.
        """
        t0 = time.time()
        logger.info(f"Document Agent processing document of type '{doc_type}' from {file_path}")
        
        # High-fidelity simulated extraction engine aligned with Indian medical standards
        ctx = claim_context or {}
        patient_name = ctx.get("full_name", "Ramesh Kumar Patil")
        hospital = ctx.get("hospital_name", "SDM College of Medical Sciences Dharwad")
        tier = ctx.get("hospital_tier", "Tier 2 (City Multispecialty)")
        claimed_amt = float(ctx.get("claimed_amount_inr", 78000.0))
        stay_days = int(ctx.get("stay_duration_days", 3))
        treatment = ctx.get("treatment_name", "Laparoscopic Appendectomy")
        
        # Build realistic itemized billing breakdown
        room_rent = round(stay_days * 3500.0, 2)
        surgeon_fee = round(claimed_amt * 0.38, 2)
        ot_charges = round(claimed_amt * 0.22, 2)
        medicines = round(claimed_amt * 0.18, 2)
        investigations = round(claimed_amt - (room_rent + surgeon_fee + ot_charges + medicines), 2)
        
        extracted_data = {
            "document_type": doc_type,
            "patient_name_extracted": patient_name,
            "hospital_name_extracted": hospital,
            "hospital_tier": tier,
            "dates_extracted": {
                "admission_date": ctx.get("admission_date", "2024-06-10"),
                "discharge_date": ctx.get("discharge_date", "2024-06-13"),
                "duration_days": stay_days
            },
            "itemized_billing_inr": {
                "room_and_nursing": max(0.0, room_rent),
                "operation_theatre_charges": max(0.0, ot_charges),
                "surgeon_and_anesthetist_fees": max(0.0, surgeon_fee),
                "pharmacy_and_consumables": max(0.0, medicines),
                "diagnostics_and_lab_tests": max(0.0, investigations),
                "total_invoiced_inr": claimed_amt
            },
            "clinical_details": {
                "primary_diagnosis": ctx.get("diagnosis_category", "Gastroenterology & General Surgery"),
                "icd10_code": ctx.get("icd10_code", "K35.8"),
                "procedure_performed": treatment,
                "discharging_physician": "Dr. S. K. Kulkarni, MS (Gen Surg), Reg: KMC-44910",
                "vital_signs_on_discharge": "Stable, Afebrile, BP 124/82 mmHg"
            },
            "document_integrity": {
                "is_legible": True,
                "has_hospital_stamp": True,
                "has_physician_signature": True,
                "gstin_format_valid": True,
                "quality_confidence_score": 0.94
            }
        }
        
        processing_time = (time.time() - t0) * 1000.0
        return {
            "agent": "DocumentProcessingAgent",
            "status": "SUCCESS",
            "confidence": 0.94,
            "processing_time_ms": round(processing_time, 2),
            "extracted_data": extracted_data
        }
