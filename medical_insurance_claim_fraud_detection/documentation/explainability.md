# Explainability

## Methods
- Feature importance: coef_ or feature_importances_ from sklearn models
- SHAP: attempted via shap library, sampled background 100, limited to 50 test rows for speed; fallback if not installed
- Human-readable explanations generated via `common/explainability.py:generate_human_explanation`
- Document validation errors, policy rule matches, anomaly indicators, missing evidence

## Example Explanation
"The claim was flagged for manual review because the submitted amount is substantially higher than the learned peer pattern, the bill total does not match the claimed amount, and the submitted document contains an inconsistent treatment date. This result is a risk indicator and requires human review."

Not vague like "The AI thinks this claim is fraudulent."

## Distinction Among
- Confirmed label in training dataset (ground truth)
- Model prediction (binary 0/1 with prob)
- Anomaly indicator (score, not calibrated prob)
- Rule violation (policy rule check)
- Missing evidence (document missing)
- Human review decision (final operational)

## Auditable Summary (RAG)
Returns:
- Observed evidence
- Applied rule
- Risk signal
- Model result
- Recommended action
- Source reference

No hidden chain-of-thought.

## Model Cards
Each major model should have:
- Intended use: fraud-risk decision support prototype, not final determination
- Out-of-scope: autonomous rejection, legal/medical determination
- Training data: 2925 claims, 6% fraud, synthetic
- Evaluation data: 900 test claims
- Metrics: PR-AUC, etc from evaluation/
- Known limitations: high-cardinality OHE, synthetic separability, potential leakage ClaimStatus
- Bias: audit income, gender, location, specialty
- Explainability: feature importance, SHAP
- Threshold policy: 0.3 approve max, 0.7 reject min, manual review zone
- Failure modes: high FP for anomaly, low confidence when missing docs
- Version: v1.0

## Visuals
- `images/feature_importance.png`
- SHAP summary, dependence (if shap available)
- `images/confusion_matrix.png`
- `images/threshold_performance.png`
