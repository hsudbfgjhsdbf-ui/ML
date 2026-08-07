# Approach 2 selection memo

Run: `run_20260807_155540`  
Selection policy: mean validation PR-AUC, then mean validation F2, then lower
training time. Test metrics are not used to select the model.

## Verdict

The selected deep model is **Feature-token transformer** (`dl_e_transformer`). Its
mean validation PR-AUC is **0.9800**, mean validation F2 is
**0.9709**, and validation PR-AUC standard deviation across
three seeds is **0.0058**.

The winner is a deep-learning comparison anchor, not a universal champion. The
same test split and target semantics are required for the later cross-approach
comparison.
