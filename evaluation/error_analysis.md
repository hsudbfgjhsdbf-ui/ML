# Error Pattern Analysis and Case Studies

**Institution:** IIIT Dharwad — Department of Data Science & AI  
**Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 1. Confusion Matrix Breakdown on Held-Out Test Set (N = 1,800)

For our best model (**Tabular FT-Transformer** at threshold $\theta^* = 0.360$):

```
                        Predicted Legitimate (0)    Predicted Fraud (1)
Actual Legitimate (0):           1,550                      50           (Total Legitimate: 1,600)
Actual Fraudulent (1):              7                      193           (Total Fraudulent: 200)
```

- **True Positives (TP):** 193 (Successfully intercepted fraud claims)
- **True Negatives (TN):** 1,550 (Clean legitimate approvals)
- **False Positives (FP):** 50 (Legitimate claims flagged for audit friction)
- **False Negatives (FN):** 7 (Undetected fraudulent claims)
- **Sensitivity / Recall:** **96.5%**
- **Specificity:** **96.9%**

---

## 2. Qualitative Analysis of Residual Errors

### False Negative Case Study (Undetected Fraud)
- **Case Profile:** Claim `CLM-IND-88421` — Laparoscopic Appendectomy at a Tier 2 hospital in Pune.
- **Claimed Amount:** ₹82,000 (Within standard statistical range ₹45,000 - ₹95,000).
- **Reason for Model Failure:** The fraud syndicate used genuine procedural tariffs but submitted duplicate bills across two different insurance carriers simultaneously. Tabular single-claim models cannot detect cross-insurer duplicate submissions without inter-carrier identity federation.
- **Multi-Agent AI Solution:** The Document Processing Agent extracted matching invoice timestamps and flagged duplicate bill IDs during historical graph queries.

### False Positive Case Study (Legitimate Claim Flagged)
- **Case Profile:** Claim `CLM-IND-44109` — Total Knee Replacement at Manipal Hospital Bangalore.
- **Claimed Amount:** ₹4,20,000 (Exceeded standard Tier 1 average of ₹3,50,000).
- **Reason for Model Flag:** High tariff deviation ($Z = +2.4$).
- **Clinical Reality:** Patient had severe post-operative diabetic complications requiring 48 hours of extended ICU monitoring, legitimately increasing billing.
- **Human-in-the-Loop Resolution:** Flagged status paused automated rejection and routed claim to a medical officer who approved settlement after reviewing ICU chart notes.
