# Decision register

**Purpose:** append-only design decisions for auditability.  
**Run:** `run_20260807_151423`.  
**Last updated:** 07-08-2026 15:14:23 UTC.

| Date | Decision | Rationale | Impact |
| --- | --- | --- | --- |
| 07-08-2026 | Use the supplied Excel workbook as the primary snapshot. | It is the dataset available in the repository; no external download is required. | Results are tied to 4,500 supplied rows and are not a national benchmark. |
| 07-08-2026 | Treat `Fraud` as positive class 1. | Consistent with fraud-screening conventions. | All precision, recall, F1, F2, PR-AUC and confusion matrices refer to fraud. |
| 07-08-2026 | Exclude IDs, raw high-cardinality codes, location, and claim status. | Prevent memorization, invalid geography claims, and post-decision leakage. | Matrix is smaller and more defensible for unseen claims. |
| 07-08-2026 | Fit imputation, one-hot encoding, and scaling on train only. | Prevent validation/test information leakage. | Reproduction must use the serialized preprocessor. |
| 07-08-2026 | Select threshold by validation F2 with a precision floor. | Missing fraud is operationally costly; a floor avoids flag-everything behavior. | Test threshold is frozen before test evaluation. |
| 07-08-2026 | Declare `voting` using validation F2 and PR-AUC tie-breaker. | Selection is made before the locked test evaluation. | The winner is the only model refit on train plus validation. |

## Change control

Any change to the data snapshot, target semantics, exclusion list, split seed,
metric formula, threshold constraint, or leaderboard ranking requires a new run
identifier and a new decision row. Hand-editing result numbers is prohibited.
