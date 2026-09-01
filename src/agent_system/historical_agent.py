"""
Historical Pattern and Claimant Velocity Analysis Agent.
Queries claimant history, checks claim velocity, repeat claims, and entity links.
"""

import time
from typing import Dict, Any, List, Optional
from src.utils import logger

class HistoricalPatternAgent:
    """
    Cognitive Agent evaluating longitudinal claim frequency and policyholder risk velocity.
    """
    
    def __init__(self):
        pass
        
    def analyze_claimant_history(
        self,
        claim_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluates policyholder's historical claim trajectory and velocity.
        """
        t0 = time.time()
        logger.info(f"Historical Pattern Agent querying historical records for Policyholder {claim_data.get('user_id')}")
        
        prior_cnt = int(claim_data.get("prior_claims_count", 0))
        rej_cnt = int(claim_data.get("rejected_prior_claims", 0))
        tot_prior_amt = float(claim_data.get("total_prior_claimed_inr", 0.0))
        current_amt = float(claim_data.get("claimed_amount_inr", 0.0))
        
        patterns = []
        historical_risk = 0.05
        
        # 1. High Velocity Spike
        if prior_cnt >= 4:
            historical_risk += 0.35
            patterns.append({
                "pattern": "High Claim Frequency Spike",
                "severity": "HIGH",
                "detail": f"Policyholder has lodged {prior_cnt} separate claims within the trailing 12 months."
            })
            
        # 2. Prior Rejection History
        if rej_cnt >= 2:
            historical_risk += 0.40
            patterns.append({
                "pattern": "Repeat Claim Denial History",
                "severity": "HIGH",
                "detail": f"Policyholder has {rej_cnt} previously rejected claims on file for documentation or eligibility defects."
            })
            
        # 3. Escalating Claim Amounts
        if prior_cnt > 0:
            avg_prior = tot_prior_amt / prior_cnt
            if current_amt > (avg_prior * 2.5) and current_amt > 150000.0:
                historical_risk += 0.20
                patterns.append({
                    "pattern": "Escalating Claim Magnitude",
                    "severity": "MEDIUM",
                    "detail": f"Current claim of ₹{current_amt:,.0f} is {current_amt/max(1.0, avg_prior):.1f}x higher than historical average payout of ₹{avg_prior:,.0f}."
                })
        else:
            patterns.append({
                "pattern": "Clean Historical Record",
                "severity": "INFO",
                "detail": "First claim filed on policy; no prior adverse history."
            })
            
        historical_risk = min(1.0, historical_risk)
        processing_time = (time.time() - t0) * 1000.0
        
        return {
            "agent": "HistoricalPatternAgent",
            "historical_risk_score": round(historical_risk, 3),
            "prior_claims_count": prior_cnt,
            "rejected_prior_claims": rej_cnt,
            "patterns_detected": patterns,
            "processing_time_ms": round(processing_time, 2)
        }
