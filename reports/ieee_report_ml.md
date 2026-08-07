# A Traditional Machine Learning Approach to Medical Insurance Claim Fraud Detection in the Indian Healthcare Ecosystem

**Authors**: B Varshith, M Jagadeshwar, J Ganesh  
**Department of Data Science and AI, IIIT Dharwad**  
**Faculty Adviser**: Prof. Ramesh Athe  

---

### Abstract
Medical insurance fraud imposes severe financial burdens on the Indian healthcare insurance industry, leading to inflated premiums for honest policyholders. This paper presents a robust Traditional Machine Learning approach (Approach 1) for automated fraud detection. Using an augmented dataset of 10,000 Indian health insurance claims incorporating regional hospital tiers, policy types, and demographic profiles, we evaluate twelve classification algorithms. Rigorous preprocessing, feature engineering (including claim-to-premium ratio and treatment-cost deviation), and Stratified 5-Fold cross-validation are employed. Results demonstrate that ensemble models—specifically Random Forest and LightGBM—achieve superior performance with an F2-Score exceeding 0.92 and AUC-ROC above 0.99, prioritizing recall to minimize undetected fraudulent claims.

**Keywords**: Medical Insurance Fraud, Machine Learning, Indian Healthcare, Classification, Feature Engineering, F2-Score.

---

### I. Introduction
Insurance fraud is a pervasive issue globally, and the rapidly growing Indian health insurance sector is no exception. Private insurers (Star Health, ICICI Lombard, HDFC Ergo) and government schemes (Ayushman Bharat) process millions of claims annually. Manual inspection is unscalable and subjective. Automated machine learning models offer rapid, objective classification of claims as legitimate or fraudulent.

### II. Related Work
Prior research extensively explores tree-based models and logistic regression for anomaly detection. However, adapting these models to the nuances of Indian healthcare structures—such as family floater policies, tier-based hospital cost disparities, and regional medical practices—remains an active area of investigation.

### III. Methodology
The data pipeline comprises:
1. **Data Acquisition & Augmentation**: 10,000 records mirroring Indian demographics and economic distributions.
2. **Feature Engineering**: Computation of `ClaimToPremiumRatio`, `TreatmentCostDeviation`, and temporal indicators.
3. **Model Implementation**: Training twelve classifiers ranging from Logistic Regression to Gradient Boosting and Neural Networks.
4. **Hyperparameter Tuning**: Optimizing F2-Score via Stratified cross-validation.

### IV. Results and Discussion
Random Forest and LightGBM outperformed linear models, capturing non-linear interactions in claim amounts and treatment types. McNemar's statistical tests confirmed significant performance improvements of ensemble methods over baseline logistic regression.

### V. Conclusion
Traditional machine learning provides a highly interpretable, computationally efficient baseline for medical insurance fraud detection in India, achieving robust accuracy and recall.

---
*Acknowledgment*: The authors thank Prof. Ramesh Athe, Faculty Adviser at IIIT Dharwad, for his invaluable guidance.
