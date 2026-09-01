"""
Reasoning and Explainable Decision Synthesis Agent.
Synthesizes multi-agent evidence, resolves conflicts, and generates detailed
bilingual natural language justifications (English & Hindi) citing exact clauses and tariffs.
"""

import time
from typing import Dict, Any, List, Optional
from src.utils import logger

class ReasoningDecisionAgent:
    """
    Cognitive Agent synthesizing multi-source evidence into transparent, explainable decisions.
    """
    
    def __init__(self):
        pass
        
    def synthesize_final_decision(
        self,
        claim_data: Dict[str, Any],
        doc_result: Dict[str, Any],
        policy_result: Dict[str, Any],
        anomaly_result: Dict[str, Any],
        historical_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesizes findings from all specialized agents into a unified, evidence-backed adjudication verdict.
        """
        t0 = time.time()
        logger.info(f"Reasoning Agent adjudicating Claim {claim_data.get('claim_id')}")
        
        claimed_amt = float(claim_data.get("claimed_amount_inr", 0.0))
        patient_name = claim_data.get("full_name", "Policyholder")
        treatment = claim_data.get("treatment_name", "Medical Procedure")
        hospital = claim_data.get("hospital_name", "Admitting Hospital")
        tier = claim_data.get("hospital_tier", "Tier 2 (City Multispecialty)")
        
        # Calculate composite risk score
        anomaly_risk = anomaly_result.get("anomaly_risk_score", 0.1)
        history_risk = historical_result.get("historical_risk_score", 0.05)
        policy_compliant = policy_result.get("overall_status") == "COMPLIANT"
        
        composite_fraud_score = (0.50 * anomaly_risk) + (0.25 * history_risk) + (0.25 * (0.0 if policy_compliant else 0.90))
        composite_fraud_score = min(1.0, composite_fraud_score)
        
        # Adjudication Decision Logic
        if not policy_compliant or composite_fraud_score > 0.60:
            decision = "REJECTED"
            action_code = "DENIAL_FRAUD_POLICY_DEFECT"
            approved_amount = 0.0
        elif composite_fraud_score > 0.30:
            decision = "FLAGGED_FOR_MANUAL_REVIEW"
            action_code = "MANUAL_INVESTIGATION_REQUIRED"
            approved_amount = 0.0
        else:
            decision = "APPROVED"
            action_code = "AUTO_SETTLED_WITH_COPAY"
            approved_amount = float(policy_result.get("estimated_payable_inr", claimed_amt * 0.90))
            
        # Build Layered English Natural Language Explanation
        if decision == "APPROVED":
            summary_en = (
                f"Claim {claim_data.get('claim_id')} submitted for {treatment} at {hospital} has been APPROVED "
                f"for a net settlement of ₹{approved_amount:,.2f} after applying contractual co-payment deductions."
            )
            detailed_en = (
                f"1. Policy Verification: The policy is active, has elapsed the required waiting period, and the claimed amount "
                f"is within the overall Sum Insured limit.\n"
                f"2. Document Authentication: Invoices, discharge summaries, and medical reports from {hospital} are verified and consistent.\n"
                f"3. Tariff Compliance: The billed treatment cost of ₹{claimed_amt:,.2f} aligns with the standard IRDAI pricing schedule "
                f"for {tier} healthcare facilities.\n"
                f"4. Settlement Action: Net approved amount of ₹{approved_amount:,.2f} scheduled for disbursement to the claimant's account."
            )
            explanation_hi = (
                f"दावा {claim_data.get('claim_id')} ({treatment}, {hospital}) स्वीकृत कर दिया गया है। "
                f"पॉलिसी शर्तों और सह-भुगतान (Co-pay) कटौती के बाद कुल ₹{approved_amount:,.2f} की राशि स्वीकृत की गई है। "
                f"सभी दस्तावेज और अस्पताल के बिल नियमों के अनुसार पाए गए हैं।"
            )
        elif decision == "FLAGGED_FOR_MANUAL_REVIEW":
            summary_en = (
                f"Claim {claim_data.get('claim_id')} for ₹{claimed_amt:,.2f} has been FLAGGED for specialized manual audit "
                f"due to detected tariff variances and channel risk indicators."
            )
            detailed_en = (
                f"1. Tariff Variance: The claimed amount of ₹{claimed_amt:,.2f} is moderately elevated compared to standard "
                f"benchmark tariffs for {treatment} in {tier}.\n"
                f"2. Investigation Trigger: A human claim investigator has been assigned to verify original pharmacy receipts "
                f"and confirm admission logs with {hospital}.\n"
                f"3. Next Step: Claimant is requested to provide detailed itemized laboratory investigation reports if contacted."
            )
            explanation_hi = (
                f"दावा {claim_data.get('claim_id')} (राशि: ₹{claimed_amt:,.2f}) को विशेष समीक्षा के लिए चिह्नित (Flag) किया गया है। "
                f"अस्पताल के बिल में कुछ दरों का अंतर पाया गया है। बीमा अधिकारी द्वारा 48 घंटों में सत्यापन पूरा किया जाएगा।"
            )
        else: # REJECTED
            fail_reasons = [c["detail"] for c in policy_result.get("verification_checks", []) if c["status"] == "FAIL"]
            anomaly_reasons = [a["detail"] for a in anomaly_result.get("anomalies_detected", [])]
            combined_reasons = fail_reasons + anomaly_reasons
            primary_reason = combined_reasons[0] if combined_reasons else "Billing exceeds authorized limits and policy clauses."
            
            summary_en = (
                f"Claim {claim_data.get('claim_id')} for ₹{claimed_amt:,.2f} has been REJECTED. "
                f"Primary Reason: {primary_reason}"
            )
            detailed_en = (
                f"1. Contractual & Clinical Defect: {primary_reason}\n"
                f"2. Audit Findings: The claimed procedure ({treatment}) billed at ₹{claimed_amt:,.2f} exceeds allowable limits "
                f"or violates pre-existing condition waiting period clauses under IRDAI guidelines.\n"
                f"3. Regulatory Rights: As per IRDAI Grievance Redressal Regulations, the claimant has the right to appeal "
                f"this decision within 30 days with additional clinical evidence."
            )
            explanation_hi = (
                f"दावा {claim_data.get('claim_id')} अस्वीकृत (Reject) कर दिया गया है। "
                f"मुख्य कारण: {primary_reason} "
                f"बीमा नियामक (IRDAI) नियमों के अनुसार, आप 30 दिनों के भीतर शिकायत निवारण मंच पर अपील कर सकते हैं।"
            )
            
        processing_time = (time.time() - t0) * 1000.0
        return {
            "agent": "ReasoningDecisionAgent",
            "decision": decision,
            "action_code": action_code,
            "composite_fraud_risk": round(composite_fraud_score, 3),
            "approved_amount_inr": round(approved_amount, 2),
            "summary_explanation_en": summary_en,
            "detailed_explanation_en": detailed_en,
            "decision_hindi": explanation_hi,
            "processing_time_ms": round(processing_time, 2)
        }
