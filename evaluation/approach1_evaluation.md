# APPROACH 1: TRADITIONAL MACHINE LEARNING EVALUATION REPORT
**Institution:** IIIT Dharwad | B.Tech Data Science & AI  
**Faculty Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

## 1. Classical Supervised Machine Learning Benchmarking Table
| Algorithm Name | F2 Score | Recall | Precision | F1 Score | Accuracy | AUC-ROC | Cost (INR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AdaBoost** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Rs. 0 |
| **LightGBM** | 0.9950 | 1.0000 | 0.9756 | 0.9877 | 0.9985 | 0.9987 | Rs. 5,000 |
| **XGBoost** | 0.9901 | 1.0000 | 0.9524 | 0.9756 | 0.9970 | 0.9998 | Rs. 10,000 |
| **Decision_Tree** | 0.9750 | 0.9750 | 0.9750 | 0.9750 | 0.9970 | 0.9867 | Rs. 155,000 |
| **Random_Forest** | 0.9750 | 0.9750 | 0.9750 | 0.9750 | 0.9970 | 1.0000 | Rs. 155,000 |
| **Gradient_Boosting_Hist** | 0.9750 | 0.9750 | 0.9750 | 0.9750 | 0.9970 | 0.9999 | Rs. 155,000 |
| **ANN_MLP_Baseline** | 0.9559 | 0.9750 | 0.8864 | 0.9286 | 0.9911 | 0.9963 | Rs. 175,000 |
| **Support_Vector_Machine** | 0.9330 | 0.9750 | 0.7959 | 0.8764 | 0.9837 | 0.9959 | Rs. 200,000 |
| **Gaussian_Naive_Bayes** | 0.9112 | 0.9750 | 0.7222 | 0.8298 | 0.9763 | 0.9914 | Rs. 225,000 |
| **Logistic_Regression_L1_L2** | 0.9048 | 0.9500 | 0.7600 | 0.8444 | 0.9793 | 0.9937 | Rs. 360,000 |
| **K_Nearest_Neighbors** | 0.7800 | 0.9750 | 0.4333 | 0.6000 | 0.9230 | 0.9815 | Rs. 405,000 |
| **Quadratic_Discriminant_Analysis** | 0.7605 | 1.0000 | 0.3883 | 0.5594 | 0.9067 | 0.9922 | Rs. 315,000 |

## 2. Technical Commentary
All 12 classical algorithms were evaluated via StratifiedKFold CV targeting F2-Score.