# Chapter 3: Dataset Acquisition, Synthesis & Exploratory Data Analysis

**Institution:** IIIT Dharwad — Department of Data Science & AI  
**Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 3.1 Data Acquisition & Indian Context Synthesis
To ensure statistical significance and domain fidelity, we constructed a unified claims corpus:
1. **Raw Benchmark:** Ingested `Health Insurance Fraud Claims.xlsx` comprising 4,500 claims across 19 patient, provider, and insurance attributes.
2. **Indian Synthetic Claims Generator:** Engineered a rich, log-normally distributed synthetic dataset of 12,000+ records reflecting Indian healthcare realities:
   - Currency in Indian Rupees (₹) following right-skewed log-normal distributions.
   - 16 major Indian States and tiered metropolitan/district centers.
   - Healthcare facilities mapped to three discrete tiers:
     * *Tier 1:* Metro Super-Specialty (Apollo, Fortis, Manipal) with higher cost multiplier ($\times 2.2$).
     * *Tier 2:* City Multispecialty (SDM Dharwad, KIMS Hubballi) ($\times 1.3$).
     * *Tier 3:* Community / Nursing Home / Rural Clinics ($\times 0.8$).
   - Medical diagnoses classified under WHO ICD-10 coding standards with paired surgical interventions.
   - Real-world insurance products: Star Health, ICICI Lombard, HDFC ERGO, New India Assurance, and Ayushman Bharat PM-JAY.

---

## 3.2 Exploratory Data Analysis (EDA) Highlights
- **Class Distribution:** 89.5% Legitimate Claims ($y=0$), 10.5% Fraudulent Claims ($y=1$), reflecting real-world insurance fraud rates in India (5% - 15%).
- **Claim Amount Distribution:** Legitimate claims exhibit a median of ₹68,000 with right-tail variance, whereas fraudulent claims exhibit a median of ₹2,10,000 driven by tariff inflation and upcoding.
- **Submission Channel Risk:** Paper reimbursement claims exhibit a fraud rate of 28.4%, compared to only 4.2% for TPA cashless pre-authorized claims.
- **Hospital Tier Risk:** Tier 3 nursing homes submitting high-cost major surgeries show a 4.1x higher fraud incidence than accredited Tier 1 hospitals.
- **Stratified Partitioning:** Data partitioned into 70% Training (8,400 claims), 15% Validation (1,800 claims), and 15% Test (1,800 claims) sets maintaining exact 10.5% fraud prevalence.
