# Chapter 11: Ethical Considerations, Fairness and Bias Analysis

**Institution:** IIIT Dharwad — Department of Data Science & AI  
**Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 11.1 Fairness Across Protected Demographics
In automated healthcare fraud detection, unfair bias could lead to discriminatory claim rejections against marginalized groups or geographic areas.

We evaluated our models across:
1. **Gender Equity:** Equal positive prediction rates and recall across Female (96.6%) and Male (96.4%) claimants ($\Delta < 0.5\%$).
2. **Age Equity:** Senior citizens (>55 years) and pediatric patients exhibit consistent recall without disparate impact.
3. **Geographic Fairness:** Cost baselines are normalized within hospital tiers to prevent penalizing rural or small-town claimants who travel to metropolitan tertiary hospitals.

---

## 11.2 Privacy and Digital Personal Data Protection (DPDP) Act
- All sensitive identifying information (such as 12-digit Aadhaar numbers and phone numbers) are irreversibly hashed using SHA-256 with salt.
- Model decisions are paired with transparent justifications in English and Hindi to guarantee claimant rights and ensure algorithmic accountability.
