# Selection memo

**Run:** `run_20260807_151423`  
**Policy frozen before test evaluation:** rank validation F2 descending; break
practical ties with validation PR-AUC, then lower training time.  

## Verdict

The validation leaderboard selects **Soft voting ensemble** (`voting`).
Its validation F2 is **1.0000**, validation PR-AUC is
**1.0000**, and selected threshold is **0.4900**.
This means it is the best under this dataset and protocol, not universally.

## Rejected alternatives

The complete leaderboard remains in `leaderboard.csv`. Lower-ranked models are
not discarded: their metrics, search parameters, curves, and caveats remain
available for scientific comparison. The selection decision is made without
reading test labels.

## Test unlock

After this memo was generated, the winner was refit on train plus validation
rows and the test set was evaluated once. The unlock record is
`evaluation/test_unlock.log`.
