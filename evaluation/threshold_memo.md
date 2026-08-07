# Operating threshold memo

The threshold was selected on validation probabilities using F2 and a preferred
precision floor of 0.50. Selected threshold: **0.4900**.

- Below 0.30: routine-processing candidate, subject to business rules.
- 0.30 through the selected threshold: manual-review candidate.
- At or above the selected threshold: priority investigation queue.

These bands are decision-support conventions. They must be recalibrated against
investigator capacity, false-negative cost, claimant protection, and regulatory
requirements before any deployment. Test labels were not used to choose the
threshold.
