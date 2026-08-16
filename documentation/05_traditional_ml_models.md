# Chapter 5: Traditional Machine Learning Models & Benchmarking

**Institution:** IIIT Dharwad — Department of Data Science & AI  
**Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 5.1 Overview of Implemented Algorithms (Approach 1)
Approach 1 explores 12+ distinct machine learning classifiers evaluated with Stratified 5-Fold Cross-Validation:

1. **Logistic Regression (L1 & L2 Regularized):**
   $$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta^T X)}}$$
   Provides direct coefficient interpretability for business stakeholders.
2. **Decision Tree Classifier:** Cost-complexity pruning minimizing Gini impurity with depth constraints ($d=8$).
3. **Random Forest Classifier:** Bagging ensemble of 180 decision trees with feature sub-sampling ($\sqrt{p}$) and Out-of-Bag (OOB) error monitoring.
4. **HistGradientBoosting Classifier:** Histogram-based gradient tree boosting with automatic binning of continuous features.
5. **XGBoost Classifier:** Regularized gradient boosting incorporating `scale_pos_weight = 8.5` to penalize minority fraud misclassifications.
6. **LightGBM Classifier:** Highly efficient leaf-wise gradient boosting using Gradient-based One-Side Sampling (GOSS).
7. **Support Vector Machine (SVM):** Radial Basis Function (RBF) kernel with Platt probability calibration.
8. **K-Nearest Neighbors (KNN):** Distance-weighted Euclidean metric with $k=7$ neighbors.
9. **Gaussian Naive Bayes:** Probabilistic classifier modeling continuous Gaussian likelihoods.
10. **AdaBoost Classifier:** SAMME.R sequential boosting with decision stump base learners.
11. **Extra Trees Classifier:** Extremely randomized trees evaluating random split thresholds.
12. **Stacking Ensemble:** Layered architecture combining Random Forest, XGBoost, and LightGBM with Logistic Regression meta-learner.

---

## 5.2 Summary of Traditional ML Findings
- **Top ML Model:** XGBoost achieved the highest F2-Score among classical algorithms (**0.941**), with a test recall of **94.8%** and precision of **91.2%**.
- **Inference Speed:** LightGBM was the fastest gradient booster (0.8 ms latency), followed by Logistic Regression (0.3 ms).
- **Threshold Tuning:** Shifting the decision threshold from 0.50 to $\theta^* = 0.385$ boosted XGBoost F2-score by +2.9%, capturing 94.8% of all fraud cases.
