# Approach 3 - Agent AI Multi-Agent System Report

## Overview

- Coordinator orchestrates 5 specialised agents (eligibility, policy,
  anomaly, historical, reasoning).
- Each agent emits structured findings with confidence & evidence.
- Reasoning agent synthesises findings into Approved/Flagged/Rejected
  verdict with a natural-language explanation.
- Tested on 675 held-out claims.

## Aggregate Performance

| Accuracy | Precision | Recall | F2 |
|---|---|---|---|
| 0.548 | 0.079 | 0.625 | 0.263 |

## Verdict Distribution

- Approved: 360
- Flagged: 315