"""Schema validation utilities."""
from typing import Dict, List, Any
import pandas as pd

def check_required_columns(df: pd.DataFrame, required: List[str]) -> Dict[str, Any]:
    missing = [c for c in required if c not in df.columns]
    return {"missing": missing, "ok": len(missing)==0}

def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    miss_pct = df.isna().mean().sort_values(ascending=False)
    return pd.DataFrame({"missing_pct": miss_pct, "high_missing": miss_pct > threshold})

def check_class_imbalance(y: pd.Series) -> Dict[str, Any]:
    vc = y.value_counts()
    total = len(y)
    ratio = vc.min()/vc.max() if len(vc)>1 else 1.0
    return {
        "counts": vc.to_dict(),
        "ratio_minority_majority": float(ratio),
        "fraud_rate": float((y==1).mean()) if set(y.unique()).issubset({0,1}) else None,
        "is_imbalanced": ratio < 0.2
    }

def detect_potential_leakage(df: pd.DataFrame, target_col: str = "ClaimLegitimacy") -> List[str]:
    """Heuristic leakage detection: look for columns that are post-decision."""
    leak_keywords = ["status", "legitim", "fraud", "outcome", "decision", "review", "label"]
    candidates = []
    for col in df.columns:
        lower = col.lower()
        for kw in leak_keywords:
            if kw in lower and col != target_col:
                # ClaimStatus could be leakage if denied correlated with fraud but may be legit field
                candidates.append(col)
                break
    return candidates

def validate_claim_result_schema(result: Dict[str, Any]) -> List[str]:
    """Validate final hybrid output JSON schema."""
    required = [
        "claim_id", "model_version", "fraud_probability", "fraud_prediction",
        "anomaly_score", "document_validation_status", "policy_validation_status",
        "risk_category", "recommended_decision", "key_risk_signals", "positive_evidence",
        "missing_or_inconsistent_info", "explanation", "evidence_references",
        "timestamp", "disclaimer"
    ]
    missing = [k for k in required if k not in result]
    return missing
