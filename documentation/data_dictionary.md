# Data Dictionary

> **Data status:** educational synthetic Indian-context claims. It must not be interpreted as records from an insurer or used to make a real coverage decision.

| Feature | Type | Valid range / values | Description | Fraud relevance |
|---|---|---|---|---|
| claim_id | object | CLM-202400000, CLM-202400001, CLM-202400002, CLM-202400003, CLM-202400004, CLM-202400005, CLM-202400006, CLM-202400007 … | Synthetic claim identifier; excluded from modelling. | Potential contextual signal; not proof of fraud. |
| policyholder_id | object | PH-00000, PH-00001, PH-00002, PH-00003, PH-00004, PH-00005, PH-00006, PH-00007 … | Synthetic policyholder identifier; excluded from modelling. | Potential contextual signal; not proof of fraud. |
| provider_id | object | PR-0000, PR-0001, PR-0002, PR-0003, PR-0004, PR-0005, PR-0006, PR-0007 … | Synthetic provider identifier; target encoded using training labels only. | Potential contextual signal; not proof of fraud. |
| claim_date | object | 2024-01-01, 2024-01-02, 2024-01-03, 2024-01-04, 2024-01-05, 2024-01-06, 2024-01-07, 2024-01-08 … | Claim event date. | Potential contextual signal; not proof of fraud. |
| state | object | Bihar, Delhi, Gujarat, Karnataka, Kerala, Maharashtra, Odisha, Rajasthan … | Claimant residence state in India. | Potential contextual signal; not proof of fraud. |
| city | object | Ahmedabad, Bengaluru, Bhubaneswar, Chennai, Coimbatore, Dharwad, Hyderabad, Jaipur … | Claimant residence city in India. | Potential contextual signal; not proof of fraud. |
| age | int64 | 0 to 90 | Claimant age in years (0–90). | Potential contextual signal; not proof of fraud. |
| gender | object | Female, Male, Non-binary/Prefer not to say | Self-reported gender category, retained for fairness audit. | Potential contextual signal; not proof of fraud. |
| income_bracket | object | High, Low, Lower-middle, Middle, Upper-middle | Synthetic monthly-income category, retained for fairness audit. | Potential contextual signal; not proof of fraud. |
| monthly_income_inr | float64 | 1.07e+04 to 5e+05 | Synthetic estimated monthly income in INR. | Potential contextual signal; not proof of fraud. |
| occupation_type | object | Agricultural, Daily-wage, Homemaker, Retired, Salaried, Self-employed, Student | Synthetic occupation category. | Potential contextual signal; not proof of fraud. |
| disability_accommodation | object | Hearing, Mobility, No disclosed disability, Other, Visual | Accommodation status; fairness/audit field, not a decision rule. | Potential contextual signal; not proof of fraud. |
| policy_type | object | Ayushman Bharat, ECHS, Employer group, Family floater, Individual | Individual, family floater, employer group, Ayushman Bharat or ECHS. | Potential contextual signal; not proof of fraud. |
| insurer | object | Government scheme, HDFC ERGO, ICICI Lombard, New India Assurance, Star Health | Synthetic insurer/product issuer. | Potential contextual signal; not proof of fraud. |
| sum_insured_inr | int64 | 3e+05 to 2.5e+06 | Policy coverage limit in INR. | Potential contextual signal; not proof of fraud. |
| annual_premium_inr | float64 | 3.61e+03 to 1.45e+05 | Annual premium in INR. | Potential contextual signal; not proof of fraud. |
| policy_start_date | object | 2020-01-01, 2020-01-02, 2020-01-03, 2020-01-04, 2020-01-05, 2020-01-06, 2020-01-07, 2020-01-08 … | Policy inception date. | Potential contextual signal; not proof of fraud. |
| policy_duration_days | int64 | 18 to 2.18e+03 | Days elapsed from inception to claim. | Potential contextual signal; not proof of fraud. |
| waiting_period_days | int64 | 180 to 730 | Applicable waiting period in days. | Potential contextual signal; not proof of fraud. |
| waiting_period_completed | object | No, Yes | Whether the waiting period is complete at claim date. | Potential contextual signal; not proof of fraud. |
| copay_percent | int64 | 0 to 30 | Policyholder co-payment percentage. | Potential contextual signal; not proof of fraud. |
| claim_amount_inr | float64 | 1.07e+03 to 2.61e+06 | Claimed amount in Indian Rupees. | Potential contextual signal; not proof of fraud. |
| claim_type | object | Day-care, Hospitalization, Outpatient, Pre-authorization | Hospitalization, day-care, outpatient or pre-authorization. | Potential contextual signal; not proof of fraud. |
| treatment_type | object | Appendectomy, Ayurvedic outpatient, Ayurvedic panchakarma, Cardiac hospitalization, Cataract day-care, Dengue hospitalization, Dialysis day-care, Maternity delivery … | Synthetic clinical treatment category. | Potential contextual signal; not proof of fraud. |
| medical_practice | object | Allopathic, Ayurvedic | Allopathic or Ayurvedic treatment practice. | Potential contextual signal; not proof of fraud. |
| diagnosis_code | object | A90, H25.9, I25.1, J18.9, K35.8, M17.0, M54.5, N18.6 … | Synthetic ICD-style diagnosis code. | Potential contextual signal; not proof of fraud. |
| procedure_code | object | Ayurvedic consultation, CABG/angioplasty, Consultation and medicines, Haemodialysis, Knee arthroscopy, Laparoscopic appendectomy, Normal/C-section delivery, Panchakarma therapy … | Procedure description/code. | Potential contextual signal; not proof of fraud. |
| hospitalization_days | int64 | 0 to 10 | Length of stay; zero for outpatient care. | Potential contextual signal; not proof of fraud. |
| procedure_count | int64 | 1 to 6 | Number of procedures on claim. | Potential contextual signal; not proof of fraud. |
| doctor_credential | object | BAMS, DNB, MBBS, MD/MS, Registration unavailable, Visiting consultant | Synthetic credential/documentation category. | Potential contextual signal; not proof of fraud. |
| hospital_name | object | Ahmedabad AYUSH-centre Hospital 120, Ahmedabad Corporate Hospital 009, Ahmedabad Corporate Hospital 016, Ahmedabad Corporate Hospital 178, Ahmedabad Government Hospital 153, Ahmedabad Nursing-home Hospital 020, Ahmedabad Nursing-home Hospital 116, Ahmedabad Nursing-home Hospital 155 … | Synthetic provider name. | Potential contextual signal; not proof of fraud. |
| hospital_tier | object | AYUSH centre, Corporate, Government, Nursing home, Tier-2 private | Government, nursing home, tier-2 private, corporate or AYUSH centre. | Potential contextual signal; not proof of fraud. |
| hospital_state | object | Bihar, Delhi, Gujarat, Karnataka, Kerala, Maharashtra, Odisha, Rajasthan … | Provider state. | Potential contextual signal; not proof of fraud. |
| network_hospital | object | No, Yes | Network status for the synthetic policy. | Potential contextual signal; not proof of fraud. |
| distance_to_hospital_km | float64 | 0 to 1.04e+03 | Approximate claimant-to-provider travel distance. | Potential contextual signal; not proof of fraud. |
| time_since_last_claim_days | int64 | 1 to 1.81e+03 | Elapsed days since prior claim. | Potential contextual signal; not proof of fraud. |
| claims_past_12_months | int64 | 0 to 7 | Prior 12-month claim count. | Potential contextual signal; not proof of fraud. |
| total_historical_claims | int64 | 0 to 9 | Total prior claims. | Potential contextual signal; not proof of fraud. |
| historical_claimed_amount_inr | float64 | 2e+03 to 6.2e+06 | Total prior claimed amount in INR. | Potential contextual signal; not proof of fraud. |
| historical_average_claim_inr | float64 | 2e+03 to 1.51e+06 | Prior mean claim amount in INR. | Potential contextual signal; not proof of fraud. |
| historical_claim_std_inr | float64 | 253 to 8.95e+05 | Prior claim amount standard deviation in INR. | Potential contextual signal; not proof of fraud. |
| historical_max_claim_inr | float64 | 2.38e+03 to 2.85e+06 | Prior maximum claim amount in INR. | Potential contextual signal; not proof of fraud. |
| rejected_claim_count | int64 | 0 to 7 | Prior rejected-claim count proxy. | Potential contextual signal; not proof of fraud. |
| provider_rejection_rate | float64 | 0.0009 to 0.533 | Historical provider rejection-rate proxy, generated independent of current label. | Potential contextual signal; not proof of fraud. |
| provider_average_claim_inr | float64 | 2.38e+03 to 5.18e+05 | Provider-level average claim proxy. | Potential contextual signal; not proof of fraud. |
| provider_unique_patient_count | int64 | 25 to 1.2e+03 | Provider patient volume proxy. | Potential contextual signal; not proof of fraud. |
| regional_treatment_baseline_inr | float64 | 2.2e+03 to 4.8e+05 | Expected regional/tier treatment cost baseline. | Potential contextual signal; not proof of fraud. |
| season | object | Monsoon, Post-monsoon, Summer, Winter | Claim season in India. | Potential contextual signal; not proof of fraud. |
| claim_submission_method | object | Branch, Cashless, Reimbursement, TPA portal | Cashless, reimbursement, portal or branch submission. | Potential contextual signal; not proof of fraud. |
| gst_amount_inr | float64 | 0 to 1.15e+05 | Synthetic GST component in INR where applicable. | Potential contextual signal; not proof of fraud. |
| document_completeness_score | float64 | 0.556 to 1 | Synthetic document-completeness score (0–1). | Potential contextual signal; not proof of fraud. |
| is_fraud | int64 | 0 to 1 | Binary synthetic target: 1 fraudulent, 0 legitimate. | Target |
| claim_legitimacy | object | Fraudulent, Legitimate | Human-readable form of synthetic target. | Potential contextual signal; not proof of fraud. |
