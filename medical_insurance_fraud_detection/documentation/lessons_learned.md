# Lessons Learned

1. **Cluster signal dominates:** The Cluster feature provides overwhelming signal. This is both a strength (high accuracy) and a concern (may not generalize).
2. **Boosting models win:** XGBoost and LightGBM consistently outperform other approaches on this tabular data.
3. **Feature engineering matters:** The ClaimAmountPatientIncome ratio adds value beyond raw amounts.
4. **Class imbalance is manageable:** At 6% fraud rate, balanced class weights work well.
5. **Comprehensive benchmarking is valuable:** Training 19 models reveals which families work best.

---
*Last updated: 01-09-2026*
