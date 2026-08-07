# Speaker notes

## Slide 1

Introduce the project as a decision-support baseline for Indian health-insurance claims. Emphasize that the model prioritizes suspicious records for human review rather than making autonomous denial decisions. Credit Prof. Ramesh Athe and the three team members.

## Slide 2

The core question is simple, but the cost of the two error types is not symmetric. This is why recall-aware F2 and precision-recall analysis are more useful than accuracy alone.

## Slide 3

These are the acceptance criteria. The selected model is chosen from validation evidence only. The test set is reserved for a final locked evaluation after the selection memo is written.

## Slide 4

Be transparent: the workbook is the dataset available in the repository, not the larger Medicare multi-table benchmark described in the planning prompts. Its fields are useful for a baseline, but not sufficient to claim production or national representativeness.

## Slide 5

Use the class-balance figure to explain why the majority baseline remains in the benchmark. It is an honesty anchor, not a competitor. The plot is generated from the same raw data used by the code.

## Slide 6

This slide is central to the academic contribution. A high score from a leaked identifier is not useful. The feature-engineering register records every inclusion and exclusion with a reason.

## Slide 7

Explain that feature engineering converts raw columns into stable model inputs without inventing policy or clinical variables. The cluster feature is a deliberate audit point because it is strongly concentrated in fraud rows in this supplied snapshot.

## Slide 8

Walk left to right. The orange block is where validation-based selection and thresholding happen. The lock happens before the test metrics are computed. The final document stage reads the same leaderboard CSV and metric JSON files.

## Slide 9

The goal is not to promise every named library. The core suite is installable with scikit-learn and covers distinct algorithmic assumptions. Optional XGBoost, LightGBM, and CatBoost adapters can be added without changing the evaluator contract.

## Slide 10

Hyperparameter tuning is part of the evidence chain, not an opaque command. Every search writes parameters, trial count, and the selected validation metrics to the evaluation folder.

## Slide 11

This is the first results slide. Read the winner from the artifact-derived table and say the protocol phrase: best on this dataset under this protocol. Avoid universal claims.

## Slide 12

The winner is not declared from test performance. It is chosen using validation F2, with PR-AUC as a tie-breaker. Only after the decision is frozen is the winner refit on train plus validation data.

## Slide 13

Emphasize the threshold memo. Probabilities are useful only when the operating point is explicit. The model recommends queue priority; claims staff still verify documents, policy clauses, and medical evidence.

## Slide 14

The PR overlay is more informative than an ROC overlay at six-percent prevalence. The full high-resolution figure and raw curve points are stored in images and evaluation/curves.

## Slide 15

Explainability is about accountability. A feature ranking can help an investigator inspect a claim, but it cannot prove intent. The current baseline uses permutation importance because it is dependency-light and auditable.

## Slide 16

The fairness audit keeps demographic fields out of the model matrix and uses them only after scoring to look for disparate error patterns. Because positive counts are limited and the workbook is synthetic-looking, conclusions are cautious.

## Slide 17

This is the responsible-AI slide. The implementation deliberately avoids a fully automatic reject action. The model is a prioritization layer inside a regulated workflow.

## Slide 18

Present limitations as a roadmap. The next approaches add representation learning and document-grounded reasoning, but must reuse the same test protocol when comparison is claimed.

## Slide 19

Show the repository structure so examiners know where evidence lives. The README links these artifacts and explains that generated metrics must never be hand-edited.

## Slide 20

Close by returning to the headline: the system prioritizes suspicious claims for explainable human review. Invite questions about the data, leakage controls, threshold, or limitations.
