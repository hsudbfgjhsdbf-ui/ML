# Explainability, calibration, and decision communication

**Purpose:** make the selected model inspectable without claiming causal explanations.  
**Run:** `run_20260807_151423`.  
**Selected model:** Soft voting ensemble.  
**Last updated:** 07-08-2026 15:14:23 UTC.

## Global explanation

The final model receives permutation importance on the validation matrix. The
importance is the decrease in average precision after shuffling one transformed
column at a time. One-hot columns are shown as encoded features and then mapped
back to source families in the feature registry. A positive importance means
the shuffled column helped the model on the sampled validation rows; it does not
mean the feature causes fraud. The top-20 figure is saved at
`images/models/feature_importance_permutation.png`.

## Local explanation template

For a scored claim, the operational explanation should contain:

1. the probability and operating band;
2. up to three strongest model signals, with direction if available;
3. the evidence values in the record and a neutral comparison phrase;
4. uncertainty or missing-data caveats;
5. a human-review next step and claimant appeal route.

The language must say “this record contains a pattern associated with higher
model risk” rather than “the claimant committed fraud.” Only investigators can
establish facts after document and policy review.

## Calibration

The selected model's validation probabilities are compared before and after a
validation-fitted isotonic calibration mapping. Brier score and expected
calibration error are stored in `evaluation/calibration/`. Calibration improves
probability interpretation but does not remove dataset shift.

## Example reason codes

| Code | Neutral explanation theme | Follow-up |
| --- | --- | --- |
| `R-BILLHIGH` | Claim amount is high relative to the supplied financial context. | Verify bill line items and authorization. |
| `R-CLUSTER` | Supplied cluster context is associated with elevated validation risk. | Check whether the cluster is a stable operational field. |
| `R-EARLY` | Claim timing falls in a high-risk temporal segment. | Verify policy effective dates and submission timeline. |
| `R-DOC` | Structured documentation fields are incomplete or inconsistent. | Request the missing document; do not infer intent. |
| `R-REVIEW` | Model confidence is close to the operating threshold. | Route to a human reviewer. |

## Limitations

Permutation importance is global and model-agnostic, not a proof of an
individual claimant's reason. A future release may add SHAP with a pinned
version and a separate faithfulness audit. The current approach is deliberately
honest and lightweight for a 4,500-row workbook.
