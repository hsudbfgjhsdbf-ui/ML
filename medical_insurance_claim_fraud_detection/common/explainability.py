"""Explainability helpers."""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

def get_feature_importance(model, feature_names: List[str] = None) -> pd.DataFrame:
    """Extract feature importance if available."""
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        importances = np.abs(coef)
    else:
        return pd.DataFrame()
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importances))]
    # Align lengths
    if len(feature_names) != len(importances):
        # Might be due to OHE expansion; fallback to generic
        feature_names = [f"f{i}" for i in range(len(importances))]
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False)
    return df

def shap_explain(model, X_train, X_test, background_samples: int = 100):
    """Attempt SHAP explanation; returns shap values if shap available."""
    try:
        import shap
    except ImportError:
        return None, "shap_not_installed"
    try:
        # Sample background
        bg = shap.sample(X_train, background_samples) if len(X_train) > background_samples else X_train
        # For sklearn models, use Explainer
        explainer = shap.Explainer(model, bg)
        shap_values = explainer(X_test[:50])  # limit for speed
        return shap_values, "ok"
    except Exception as e:
        return None, f"shap_failed: {e}"

def generate_human_explanation(
    fraud_prob: float,
    risk_category: str,
    top_features: List[Tuple[str, float]],
    doc_errors: List[str],
    policy_violations: List[str],
    anomaly_signals: List[str],
    missing_info: List[str]
) -> str:
    """Generate concise auditable explanation."""
    parts = []
    parts.append(f"Fraud risk probability estimated at {fraud_prob:.2f} ({risk_category} risk).")
    if top_features:
        feats = ", ".join([f"{k} ({v:.2f})" for k,v in top_features[:5]])
        parts.append(f"Key contributing features: {feats}.")
    if doc_errors:
        parts.append(f"Document validation issues detected: {'; '.join(doc_errors[:3])}.")
    if policy_violations:
        parts.append(f"Policy rule checks: {'; '.join(policy_violations[:3])}.")
    if anomaly_signals:
        parts.append(f"Anomaly indicators: {'; '.join(anomaly_signals[:3])}.")
    if missing_info:
        parts.append(f"Missing or inconsistent information: {'; '.join(missing_info[:3])}.")
    if risk_category == "HIGH":
        parts.append("Recommended action: Flag for manual review and escalation. Human reviewer must verify documents and policy coverage before determination.")
    elif risk_category == "MEDIUM":
        parts.append("Recommended action: Manual review required due to moderate risk signals.")
    else:
        parts.append("Recommended action: No strong fraud signals; may approve if document and policy checks pass, with routine audit.")
    parts.append("Disclaimer: This is a decision-support risk indicator, not a final legal or insurance determination. Human review is mandatory for high-impact decisions.")
    return " ".join(parts)
