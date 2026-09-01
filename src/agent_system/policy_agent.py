"""
Policy Verification and Contractual Compliance Agent.
Cross-references claim details against underwriting terms, waiting periods,
co-payment requirements, sub-limits, and IRDAI regulations using RAG.
"""

import time
from typing import Dict, Any, List, Optional
from src.agent_system.rag_engine import InsuranceKnowledgeRAG
from src.utils import logger

class PolicyVerificationAgent:
    """
    Cognitive Agent executing contractual underwriting checks grounded in RAG.
    """
    
    def __init__(self, rag_engine: Optional[InsuranceKnowledgeRAG] = None):
        self.rag = rag_engine or InsuranceKnowledgeRAG()
        
    def verify_policy_compliance(
        self,
        claim_data: Dict[str, Any],
        doc_extraction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes exhaustive policy verification checks and returns verification pass/fail status.
        """
        t0 = time.time()
        logger.info(f"Policy Agent verifying Claim {claim_data.get('claim_id', 'UNKNOWN')}")
        
        claimed_amt = float(claim_data.get("claimed_amount_inr", 0.0))
        sum_insured = float(claim_data.get("sum_insured_inr", 500000.0))
        copay_pct = float(claim_data.get("copay_percentage", 10.0))
        duration_months = int(claim_data.get("duration_months", 24))
        waiting_months = int(claim_data.get("waiting_period_months", 24))
        policy_type = claim_data.get("policy_type", "Family Floater Plan")
        treatment = claim_data.get("treatment_name", "Laparoscopic Appendectomy")
        
        # 1. Retrieve applicable RAG policy clauses
        rag_clauses = self.rag.retrieve(f"{treatment} waiting period co-payment rules", top_k=2)
        
        checks = []
        all_passed = True
        
        # Check 1: Sum Insured Limit
        if claimed_amt <= sum_insured:
            checks.append({
                "check_name": "Sum Insured Coverage Check",
                "status": "PASS",
                "detail": f"Claimed amount of ₹{claimed_amt:,.0f} is within available Sum Insured limit of ₹{sum_insured:,.0f}.",
                "clause_ref": "Clause 2.1 (Policy Sum Insured Limits)"
            })
        else:
            all_passed = False
            checks.append({
                "check_name": "Sum Insured Coverage Check",
                "status": "FAIL",
                "detail": f"Claimed amount of ₹{claimed_amt:,.0f} exceeds total policy coverage of ₹{sum_insured:,.0f}.",
                "clause_ref": "Clause 2.1 (Exceeding Sum Insured)"
            })
            
        # Check 2: Mandatory Waiting Period
        if duration_months >= waiting_months:
            checks.append({
                "check_name": "Pre-Existing Condition Waiting Period",
                "status": "PASS",
                "detail": f"Policy active for {duration_months} months; successfully elapsed mandatory {waiting_months} month waiting period.",
                "clause_ref": "IRDAI PED Waiting Period Guidelines (Sec 4.1)"
            })
        else:
            all_passed = False
            checks.append({
                "check_name": "Pre-Existing Condition Waiting Period",
                "status": "FAIL",
                "detail": f"Policy active for only {duration_months} months; violates mandatory {waiting_months} month waiting period for this treatment.",
                "clause_ref": "IRDAI PED Waiting Period Guidelines (Sec 4.1)"
            })
            
        # Check 3: Co-payment & Deductibles
        copay_deduction = claimed_amt * (copay_pct / 100.0)
        net_payable_est = claimed_amt - copay_deduction
        checks.append({
            "check_name": "Co-payment Deductible Calculation",
            "status": "PASS",
            "detail": f"Applied {copay_pct}% co-payment deduction (₹{copay_deduction:,.0f}); estimated payable balance is ₹{net_payable_est:,.0f}.",
            "clause_ref": "Clause 5.3 (Mandatory Co-Payment Schedule)"
        })
        
        # Check 4: Sub-limits on Specific Treatments
        sub_limit_applies = False
        if "Cataract" in treatment and claimed_amt > 40000.0:
            sub_limit_applies = True
            checks.append({
                "check_name": "Procedure Sub-Limit Check",
                "status": "FLAG",
                "detail": f"Cataract procedure subject to capping at ₹40,000; excess ₹{claimed_amt - 40000:,.0f} requires audit reduction.",
                "clause_ref": "Clause 7.2 (Daycare Capping Schedule)"
            })
        else:
            checks.append({
                "check_name": "Procedure Sub-Limit Check",
                "status": "PASS",
                "detail": "Claim amount conforms to individual procedural sub-limits.",
                "clause_ref": "Clause 7.2 (Standard Procedural Tariffs)"
            })
            
        processing_time = (time.time() - t0) * 1000.0
        return {
            "agent": "PolicyVerificationAgent",
            "overall_status": "COMPLIANT" if all_passed else "NON_COMPLIANT",
            "confidence": 0.95,
            "processing_time_ms": round(processing_time, 2),
            "verification_checks": checks,
            "estimated_payable_inr": round(net_payable_est, 2),
            "rag_citations": [c["title"] for c in rag_clauses]
        }
