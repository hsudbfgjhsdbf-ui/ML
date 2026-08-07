# Deviations log

**Run:** `run_20260807_151423`.

- The repository supplies a 4,500-row single workbook rather than the larger multi-table public Medicare schema described in the planning prompts. The pipeline uses the available workbook and reports this limitation explicitly.
- The workbook contains fictional-looking non-Indian location values and no verified INR/policy fields. No invented localization is applied; Indian context is limited to responsible framing and future-work recommendations.
- Optional external boosters (XGBoost, LightGBM, CatBoost) are not required for the core run so the baseline remains installable from pinned open-source dependencies. The built-in histogram gradient booster provides a reproducible boosting comparator.

_Last updated: 07-08-2026 15:14:23 UTC_
