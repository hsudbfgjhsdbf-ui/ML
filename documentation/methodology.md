# Methodology and data flow

**Purpose:** explain the reproducible stages that transform the supplied
workbook into an evaluated binary classifier.  
**Owner:** project team.  
**Run:** `run_20260807_151423`.  
**Last updated:** 07-08-2026 15:14:23 UTC.

## Stage map

| Stage | Input | Output | Gate |
| --- | --- | --- | --- |
| S0 | Python environment | package and hardware snapshot | required imports |
| S1 | supplied Excel workbook | validated raw dataframe | schema, types, labels |
| S2 | validated rows | source profile and missingness table | no duplicate ClaimID |
| S3 | raw rows | EDA tables and figures | EDA precedes model fitting |
| S4 | engineered rows | 70/15/15 split indices | stratified label proportions |
| S5 | training feature rows | fitted imputer, encoder, scaler | train-only fit |
| S6 | transformed train | model zoo and search records | fixed CV budget |
| S7 | validation probabilities | threshold, leaderboard, calibration | test remains locked |
| S8 | frozen validation policy | one test evaluation per model | no test-informed selection |
| S9 | final outputs | markdown, PPTX, project PDF, IEEE paper | link and number checks |

## Reproducible split

The input contains one row per supplied claim and unique patient/provider IDs.
The split uses stratified random sampling with seed 42. The requested fractions
are 0.70, 0.15, and 0.15. Labels are encoded as `1 = Fraud` and `0 = Legitimate`.
Because the supplied patient IDs are unique in this snapshot, a group overlap
check is vacuously satisfied; if a future snapshot repeats a patient across
claims, a group-aware split must replace this split before modeling.

## Preprocessing contract

Numeric features use median imputation and standard scaling. Categorical
features use most-frequent imputation followed by `OneHotEncoder` with
`handle_unknown=ignore`. The transformer is fit only on the training partition.
Validation and test data are transformed with the frozen object. This protects
against missing-value and category leakage. No `ClaimID`, `PatientID`,
`ProviderID`, raw diagnosis/procedure code, provider location, or claim status
enters the model matrix. The last two exclusions are important: high-cardinality
location/codes are nearly unique, and status may be observed after a decision.

## Model selection

Every model receives the same transformed training rows, validation rows, target
semantics, and random seed policy. Search procedures use stratified three-fold
cross-validation on the training partition and average precision as the search
score where a search is defined. Hyperparameter ranges are intentionally small
and explicit to remain affordable on a laptop. After fitting each complete
model, the operating threshold is selected on validation data by maximizing F2
subject to a 0.50 precision floor when a candidate exists. Selection ranks by
validation F2, then validation PR-AUC, then training time.

## Business translation

A fraud probability is not a verdict. In the generated triage template,
probabilities below 0.30 are candidates for routine processing, scores from
0.30 to the selected threshold are candidates for manual review, and scores at
or above the selected threshold receive priority review. An insurer may choose
a different operating point after measuring investigator capacity and costs.
False negatives represent fraud that can be paid; false positives represent
legitimate policyholders who may face delays. Both errors are quantified.

## Audit trail

Each model has JSON metrics, threshold sweep CSV, confusion matrix CSV, and a
serialized estimator. The run manifest stores the config, environment, input
checksum, split sizes, feature registry, and artifact list. The evaluation hub
is generated from these artifacts and should be regenerated rather than hand
edited. See [auditing.md](auditing.md) for a reviewer walkthrough.
