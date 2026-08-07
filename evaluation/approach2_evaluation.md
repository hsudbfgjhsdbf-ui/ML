# APPROACH 2: DEEP LEARNING & XAI EVALUATION REPORT
**Institution:** IIIT Dharwad | B.Tech Data Science & AI  
**Faculty Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

## 1. Deep Tabular Neural Network Benchmarking Table
| Algorithm Name | F2 Score | Recall | Precision | F1 Score | Accuracy | AUC-ROC | Cost (INR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TabularTransformer** | 0.9799 | 0.9750 | 1.0000 | 0.9873 | 0.9985 | 0.9988 | Rs. 150,000 |
| **WideAndDeep** | 0.9653 | 0.9750 | 0.9286 | 0.9512 | 0.9941 | 0.9946 | Rs. 165,000 |
| **TabNetStyle** | 0.9653 | 0.9750 | 0.9286 | 0.9512 | 0.9941 | 0.9937 | Rs. 165,000 |
| **VariationalAutoencoder** | 0.9559 | 0.9750 | 0.8864 | 0.9286 | 0.9911 | 0.9969 | Rs. 175,000 |
| **AutoencoderAnomaly** | 0.9466 | 0.9750 | 0.8478 | 0.9070 | 0.9881 | 0.9970 | Rs. 185,000 |
| **LSTMSequential** | 0.9296 | 0.9250 | 0.9487 | 0.9367 | 0.9926 | 0.9957 | Rs. 460,000 |
| **DeepAndCrossNetwork** | 0.9223 | 0.9500 | 0.8261 | 0.8837 | 0.9852 | 0.9966 | Rs. 340,000 |
| **MLP** | 0.9155 | 0.9750 | 0.7358 | 0.8387 | 0.9778 | 0.9961 | Rs. 220,000 |
| **NODE** | 0.9048 | 0.9500 | 0.7600 | 0.8444 | 0.9793 | 0.9940 | Rs. 360,000 |
| **ResNetTabular** | 0.8911 | 0.9000 | 0.8571 | 0.8780 | 0.9852 | 0.9953 | Rs. 630,000 |

## 2. Technical Commentary
All 10 PyTorch deep tabular architectures were trained with Focal Loss and Cosine Annealing.