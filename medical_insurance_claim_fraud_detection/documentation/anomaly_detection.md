# Anomaly Detection Approach

## Methods
Isolation Forest, LOF, One-Class SVM, EllipticEnvelope, Autoencoder optional, Ensemble avg

## Training
Only non-fraud records used where configured. Contamination 0.06 based on fraud rate.

## Distinction
- Anomaly score: deviation measure
- Fraud probability: calibrated supervised probability
- Fraud label: ground truth

Anomaly score must NOT be presented as verified fraud probability unless calibrated.

## Evaluation
Precision@k, Recall@k, PR-AUC, ROC-AUC using fraud labels as reference.

## Results summary
                model    pr_auc   roc_auc  prec@10  prec@50  prec@100  prec@200    rec@10    rec@50   rec@100   rec@200
0     IsolationForest  0.071780  0.487238      0.2     0.06      0.03      0.05  0.037037  0.055556  0.055556  0.185185
1                 LOF  0.146853  0.737457      0.2     0.16      0.17      0.14  0.037037  0.148148  0.314815  0.518519
2         OneClassSVM  0.122815  0.679757      0.1     0.18      0.17      0.12  0.018519  0.166667  0.314815  0.444444
3  EnsembleAnomalyAvg  0.129360  0.656203      0.2     0.16      0.12      0.10  0.037037  0.148148  0.222222  0.370370

## Limitations
- High FP, cannot replace supervised model
- Needs human review
