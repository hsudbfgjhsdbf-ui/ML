# Chapter 9: Evaluation and Benchmarking Suite

**Institution:** IIIT Dharwad — Department of Data Science & AI  
**Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 9.1 Primary Metric: F2-Score Optimization
In medical insurance fraud detection, missing a fraudulent claim (False Negative) results in substantial financial loss (~₹1,85,000 avg payout), whereas flagging a legitimate claim for audit (False Positive) imposes only a minor administrative cost (~₹12,000).

Therefore, model evaluation strictly prioritizes the **F2-Score** ($\beta=2.0$), weighting recall twice as heavily as precision:
$$F_2 = \frac{5 \cdot \text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$

---

## 9.2 Complete Benchmark Summary
- **Top Deep Learning Architecture:** Tabular FT-Transformer ($F_2 = \mathbf{0.957}$, $\text{Recall} = \mathbf{96.5\%}$, $\text{AUC-ROC} = \mathbf{0.989}$).
- **Top Traditional ML Algorithm:** XGBoost Classifier ($F_2 = \mathbf{0.941}$, $\text{Recall} = \mathbf{94.8\%}$, $\text{AUC-ROC} = \mathbf{0.984}$).
- **Statistical Superiority:** McNemar's Test ($\chi^2 = 4.364, p = 0.0367 < 0.05$) confirms that FT-Transformer's recall advantage over XGBoost is statistically significant.
- **Economic Value:** Financial cost analysis proves that our optimized threshold saves ₹14.2 Lakhs in fraud leakage per 1,000 claims processed.
