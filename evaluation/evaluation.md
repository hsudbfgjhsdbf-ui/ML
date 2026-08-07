# Evaluation hub — Approach 1 traditional ML

**Run:** `run_20260807_151423`  
**Generated:** 07-08-2026 15:14:23 UTC  
**Positive class:** `Fraud = 1`; `Legitimate = 0`.  
**Selection:** validation F2, then validation PR-AUC, then training time.  

## 1. Dataset and protocol

The frozen supplied snapshot contains **4,500** rows, **19** source columns, and a fraud rate of **6.0%**. The split is stratified 70/15/15 with seed 42. Test rows were not used for model selection or threshold tuning.

## 2. Leaderboard

| Rank | Model | Family | Val F2 | Val PR-AUC | Val ROC-AUC | Val precision | Val recall | Test F2 | Test PR-AUC | Train sec | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | random_forest | bagging | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 | 0.9994 | 3.2208 | complete |
| 2 | gradient_boosting | boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9799 | 0.9994 | 6.8311 | complete |
| 3 | hist_gradient_boosting | boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9799 | 0.9994 | 1.1335 | complete |
| 4 | adaboost | boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9799 | 0.9885 | 4.5233 | complete |
| 5 | voting | ensemble | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9799 | 1.0000 | 0.4590 | complete |
| 6 | stacking | ensemble | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 1.2606 | complete |
| 7 | svm_rbf | margin | 0.9804 | 0.9820 | 0.9955 | 1.0000 | 0.9756 | 0.9406 | 0.9800 | 1.2949 | complete |
| 8 | decision_tree | tree | 0.9804 | 0.9771 | 0.9878 | 1.0000 | 0.9756 | 0.9799 | 0.9765 | 0.3682 | complete |
| 9 | extra_trees | bagging | 0.9709 | 0.9874 | 0.9986 | 0.9524 | 0.9756 | 0.9701 | 0.9947 | 2.5644 | complete |
| 10 | logistic_l1 | linear | 0.9662 | 0.9775 | 0.9950 | 0.9302 | 0.9756 | 0.9709 | 0.9902 | 1.9109 | complete |
| 11 | mlp | neural_baseline | 0.9615 | 0.9755 | 0.9945 | 0.9091 | 0.9756 | 0.9615 | 0.9736 | 1.4585 | complete |
| 12 | linear_discriminant | probabilistic | 0.9569 | 0.9133 | 0.9920 | 0.8889 | 0.9756 | 0.9569 | 0.9201 | 0.1893 | complete |
| 13 | logistic_l2 | linear | 0.9479 | 0.9747 | 0.9949 | 0.8511 | 0.9756 | 0.9615 | 0.9863 | 0.9946 | complete |
| 14 | passive_aggressive | online_linear | 0.9479 | 0.9524 | 0.9942 | 0.8511 | 0.9756 | 0.9198 | 0.9249 | 0.2935 | complete |
| 15 | ridge | linear | 0.9259 | 0.9501 | 0.9908 | 0.7692 | 0.9756 | 0.9242 | 0.9776 | 0.2685 | complete |
| 16 | qda | probabilistic | 0.8952 | 0.7000 | 0.9860 | 0.6308 | 1.0000 | 0.8547 | 0.6390 | 0.2172 | complete |
| 17 | knn | instance | 0.8520 | 0.8219 | 0.9767 | 0.6441 | 0.9268 | 0.8028 | 0.7271 | 0.3389 | complete |
| 18 | gaussian_nb | probabilistic | 0.8511 | 0.6629 | 0.9723 | 0.5634 | 0.9756 | 0.8368 | 0.5634 | 0.1658 | complete |
| 19 | bernoulli_nb | probabilistic | 0.8032 | 0.4821 | 0.9583 | 0.4706 | 0.9756 | 0.7937 | 0.4314 | 0.1752 | complete |
| 20 | majority | baseline | 0.2443 | 0.0607 | 0.5000 | 0.0607 | 1.0000 | 0.2395 | 0.0593 | 0.0006 | complete |

**Selected model:** `voting` — Soft voting ensemble. The selected model is refit on training plus validation rows only after this decision and then evaluated on the locked test set.

## 3. Metric definitions

Accuracy is the fraction of all correct rows; it is not the primary fraud metric. Precision is the fraction of flagged rows that are fraud. Recall is the fraction of fraud rows caught. F1 is the harmonic mean of precision and recall. F2 weights recall more heavily. ROC-AUC summarizes ranking over all thresholds. PR-AUC is emphasized because fraud is rare. MCC is a balanced correlation coefficient. Brier score and log loss evaluate probability quality. Specificity is the legitimate-claim true-negative rate.

## 4. Per-model audit records

### majority — Majority-class baseline

- **Family:** baseline
- **Status:** complete
- **Search:** none with 1 recorded trials; scoring `average_precision`.
- **Best parameters:** `{}`
- **Validation threshold:** 0.0500; selection metric `f2_fallback_no_precision_candidate`.
- **Validation accuracy / precision / recall:** 0.0607 / 0.0607 / 1.0000.
- **Validation F1 / F2:** 0.1145 / 0.2443.
- **Validation ROC-AUC / PR-AUC:** 0.5000 / 0.0607.
- **Validation MCC / Brier / log loss:** 0.0000 / 0.0571 / 0.2290.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.2395, PR-AUC=0.0593, ROC-AUC=0.5000; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 0.001; prediction milliseconds per sample: 0.0001; artifact KB: 0.6.
- **Caveat:** Sanity floor; predicts no fraud at the default threshold.

### logistic_l2 — Logistic regression (L2)

- **Family:** linear
- **Status:** complete
- **Search:** grid with 3 recorded trials; scoring `average_precision`.
- **Best parameters:** `{"C": 10.0}`
- **Validation threshold:** 0.5300; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 0.9881 / 0.8511 / 0.9756.
- **Validation F1 / F2:** 0.9091 / 0.9479.
- **Validation ROC-AUC / PR-AUC:** 0.9949 / 0.9747.
- **Validation MCC / Brier / log loss:** 0.9052 / 0.0084 / 0.0368.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.9615, PR-AUC=0.9863, ROC-AUC=0.9991; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 0.995; prediction milliseconds per sample: 0.0003; artifact KB: 1.3.
- **Caveat:** 

### logistic_l1 — Logistic regression (L1)

- **Family:** linear
- **Status:** complete
- **Search:** grid with 3 recorded trials; scoring `average_precision`.
- **Best parameters:** `{"C": 1.0}`
- **Validation threshold:** 0.6500; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 0.9941 / 0.9302 / 0.9756.
- **Validation F1 / F2:** 0.9524 / 0.9662.
- **Validation ROC-AUC / PR-AUC:** 0.9950 / 0.9775.
- **Validation MCC / Brier / log loss:** 0.9495 / 0.0080 / 0.0360.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.9709, PR-AUC=0.9902, ROC-AUC=0.9994; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 1.911; prediction milliseconds per sample: 0.0003; artifact KB: 1.3.
- **Caveat:** 

### decision_tree — Decision tree

- **Family:** tree
- **Status:** complete
- **Search:** grid with 12 recorded trials; scoring `average_precision`.
- **Best parameters:** `{"max_depth": 3, "min_samples_leaf": 3}`
- **Validation threshold:** 0.9900; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 0.9985 / 1.0000 / 0.9756.
- **Validation F1 / F2:** 0.9877 / 0.9804.
- **Validation ROC-AUC / PR-AUC:** 0.9878 / 0.9771.
- **Validation MCC / Brier / log loss:** 0.9870 / 0.0015 / 0.0239.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.9799, PR-AUC=0.9765, ROC-AUC=0.9875; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 0.368; prediction milliseconds per sample: 0.0003; artifact KB: 2.3.
- **Caveat:** 

### random_forest — Random forest

- **Family:** bagging
- **Status:** complete
- **Search:** random with 4 recorded trials; scoring `average_precision`.
- **Best parameters:** `{"max_depth": null, "min_samples_leaf": 1, "n_estimators": 220}`
- **Validation threshold:** 0.2000; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 1.0000 / 1.0000 / 1.0000.
- **Validation F1 / F2:** 1.0000 / 1.0000.
- **Validation ROC-AUC / PR-AUC:** 1.0000 / 1.0000.
- **Validation MCC / Brier / log loss:** 1.0000 / 0.0015 / 0.0077.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.9950, PR-AUC=0.9994, ROC-AUC=1.0000; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 3.221; prediction milliseconds per sample: 0.0807; artifact KB: 694.9.
- **Caveat:** Out-of-bag estimate is retained when bootstrap is enabled.

### extra_trees — Extra trees

- **Family:** bagging
- **Status:** complete
- **Search:** random with 4 recorded trials; scoring `average_precision`.
- **Best parameters:** `{"max_depth": null, "min_samples_leaf": 1, "n_estimators": 140}`
- **Validation threshold:** 0.4000; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 0.9956 / 0.9524 / 0.9756.
- **Validation F1 / F2:** 0.9639 / 0.9709.
- **Validation ROC-AUC / PR-AUC:** 0.9986 / 0.9874.
- **Validation MCC / Brier / log loss:** 0.9616 / 0.0079 / 0.0338.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.9701, PR-AUC=0.9947, ROC-AUC=0.9997; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 2.564; prediction milliseconds per sample: 0.0507; artifact KB: 2543.1.
- **Caveat:** 

### gradient_boosting — Gradient boosting

- **Family:** boosting
- **Status:** complete
- **Search:** random with 4 recorded trials; scoring `average_precision`.
- **Best parameters:** `{"learning_rate": 0.1, "max_depth": 2, "n_estimators": 140}`
- **Validation threshold:** 0.9900; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 1.0000 / 1.0000 / 1.0000.
- **Validation F1 / F2:** 1.0000 / 1.0000.
- **Validation ROC-AUC / PR-AUC:** 1.0000 / 1.0000.
- **Validation MCC / Brier / log loss:** 1.0000 / 0.0000 / 0.0000.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.9799, PR-AUC=0.9994, ROC-AUC=1.0000; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 6.831; prediction milliseconds per sample: 0.0013; artifact KB: 160.3.
- **Caveat:** 

### hist_gradient_boosting — Histogram gradient boosting

- **Family:** boosting
- **Status:** complete
- **Search:** random with 4 recorded trials; scoring `average_precision`.
- **Best parameters:** `{"l2_regularization": 1.0, "learning_rate": 0.1, "max_iter": 180, "max_leaf_nodes": 15}`
- **Validation threshold:** 0.9700; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 1.0000 / 1.0000 / 1.0000.
- **Validation F1 / F2:** 1.0000 / 1.0000.
- **Validation ROC-AUC / PR-AUC:** 1.0000 / 1.0000.
- **Validation MCC / Brier / log loss:** 1.0000 / 0.0000 / 0.0003.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.9799, PR-AUC=0.9994, ROC-AUC=1.0000; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 1.134; prediction milliseconds per sample: 0.0077; artifact KB: 149.3.
- **Caveat:** 

### adaboost — AdaBoost

- **Family:** boosting
- **Status:** complete
- **Search:** random with 4 recorded trials; scoring `average_precision`.
- **Best parameters:** `{"learning_rate": 0.05, "n_estimators": 80}`
- **Validation threshold:** 0.9900; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 1.0000 / 1.0000 / 1.0000.
- **Validation F1 / F2:** 1.0000 / 1.0000.
- **Validation ROC-AUC / PR-AUC:** 1.0000 / 1.0000.
- **Validation MCC / Brier / log loss:** 1.0000 / 0.0000 / 0.0001.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.9799, PR-AUC=0.9885, ROC-AUC=0.9993; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 4.523; prediction milliseconds per sample: 0.0194; artifact KB: 51.4.
- **Caveat:** 

### svm_rbf — Support vector machine (RBF)

- **Family:** margin
- **Status:** complete
- **Search:** random with 4 recorded trials; scoring `average_precision`.
- **Best parameters:** `{"C": 10.0, "gamma": 0.01}`
- **Validation threshold:** 0.4300; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 0.9985 / 1.0000 / 0.9756.
- **Validation F1 / F2:** 0.9877 / 0.9804.
- **Validation ROC-AUC / PR-AUC:** 0.9955 / 0.9820.
- **Validation MCC / Brier / log loss:** 0.9870 / 0.0046 / 0.0276.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.9406, PR-AUC=0.9800, ROC-AUC=0.9988; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 1.295; prediction milliseconds per sample: 0.0108; artifact KB: 79.6.
- **Caveat:** Probability fitting adds runtime; the run records it explicitly.

### knn — K-nearest neighbors

- **Family:** instance
- **Status:** complete
- **Search:** grid with 6 recorded trials; scoring `average_precision`.
- **Best parameters:** `{"n_neighbors": 5, "weights": "distance"}`
- **Validation threshold:** 0.3700; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 0.9644 / 0.6441 / 0.9268.
- **Validation F1 / F2:** 0.7600 / 0.8520.
- **Validation ROC-AUC / PR-AUC:** 0.9767 / 0.8219.
- **Validation MCC / Brier / log loss:** 0.7558 / 0.0214 / 0.0830.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.8028, PR-AUC=0.7271, ROC-AUC=0.9846; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 0.339; prediction milliseconds per sample: 0.0183; artifact KB: 1231.3.
- **Caveat:** 

### gaussian_nb — Gaussian naive Bayes

- **Family:** probabilistic
- **Status:** complete
- **Search:** grid with 3 recorded trials; scoring `average_precision`.
- **Best parameters:** `{"var_smoothing": 1e-07}`
- **Validation threshold:** 0.9900; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 0.9526 / 0.5634 / 0.9756.
- **Validation F1 / F2:** 0.7143 / 0.8511.
- **Validation ROC-AUC / PR-AUC:** 0.9723 / 0.6629.
- **Validation MCC / Brier / log loss:** 0.7215 / 0.0531 / 0.6720.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.8368, PR-AUC=0.5634, ROC-AUC=0.9756; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 0.166; prediction milliseconds per sample: 0.0009; artifact KB: 2.3.
- **Caveat:** 

### bernoulli_nb — Bernoulli naive Bayes

- **Family:** probabilistic
- **Status:** complete
- **Search:** grid with 3 recorded trials; scoring `average_precision`.
- **Best parameters:** `{"alpha": 5.0}`
- **Validation threshold:** 0.9700; selection metric `f2_fallback_no_precision_candidate`.
- **Validation accuracy / precision / recall:** 0.9319 / 0.4706 / 0.9756.
- **Validation F1 / F2:** 0.6349 / 0.8032.
- **Validation ROC-AUC / PR-AUC:** 0.9583 / 0.4821.
- **Validation MCC / Brier / log loss:** 0.6513 / 0.0811 / 0.4089.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.7937, PR-AUC=0.4314, ROC-AUC=0.9583; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 0.175; prediction milliseconds per sample: 0.0012; artifact KB: 2.3.
- **Caveat:** 

### qda — Quadratic discriminant analysis

- **Family:** probabilistic
- **Status:** complete
- **Search:** grid with 4 recorded trials; scoring `average_precision`.
- **Best parameters:** `{"reg_param": 0.0}`
- **Validation threshold:** 0.0800; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 0.9644 / 0.6308 / 1.0000.
- **Validation F1 / F2:** 0.7736 / 0.8952.
- **Validation ROC-AUC / PR-AUC:** 0.9860 / 0.7000.
- **Validation MCC / Brier / log loss:** 0.7790 / 0.0359 / 0.4942.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.8547, PR-AUC=0.6390, ROC-AUC=0.9822; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 0.217; prediction milliseconds per sample: 0.0014; artifact KB: 39.9.
- **Caveat:** 

### linear_discriminant — Linear discriminant analysis

- **Family:** probabilistic
- **Status:** complete
- **Search:** grid with 2 recorded trials; scoring `average_precision`.
- **Best parameters:** `{"solver": "lsqr"}`
- **Validation threshold:** 0.8700; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 0.9911 / 0.8889 / 0.9756.
- **Validation F1 / F2:** 0.9302 / 0.9569.
- **Validation ROC-AUC / PR-AUC:** 0.9920 / 0.9133.
- **Validation MCC / Brier / log loss:** 0.9266 / 0.0172 / 0.0867.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.9569, PR-AUC=0.9201, ROC-AUC=0.9961; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 0.189; prediction milliseconds per sample: 0.0002; artifact KB: 20.8.
- **Caveat:** 

### mlp — Multi-layer perceptron

- **Family:** neural_baseline
- **Status:** complete
- **Search:** random with 4 recorded trials; scoring `average_precision`.
- **Best parameters:** `{"alpha": 0.01, "hidden_layer_sizes": [64, 32], "learning_rate_init": 0.003}`
- **Validation threshold:** 0.2300; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 0.9926 / 0.9091 / 0.9756.
- **Validation F1 / F2:** 0.9412 / 0.9615.
- **Validation ROC-AUC / PR-AUC:** 0.9945 / 0.9755.
- **Validation MCC / Brier / log loss:** 0.9379 / 0.0066 / 0.0351.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.9615, PR-AUC=0.9736, ROC-AUC=0.9985; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 1.459; prediction milliseconds per sample: 0.0015; artifact KB: 132.9.
- **Caveat:** Classical MLP baseline; deep-learning approach remains separate.

### ridge — Calibrated ridge classifier

- **Family:** linear
- **Status:** complete
- **Search:** grid with 3 recorded trials; scoring `average_precision`.
- **Best parameters:** `{"estimator__alpha": 1.0}`
- **Validation threshold:** 0.2600; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 0.9807 / 0.7692 / 0.9756.
- **Validation F1 / F2:** 0.8602 / 0.9259.
- **Validation ROC-AUC / PR-AUC:** 0.9908 / 0.9501.
- **Validation MCC / Brier / log loss:** 0.8570 / 0.0114 / 0.0479.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.9242, PR-AUC=0.9776, ROC-AUC=0.9985; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 0.269; prediction milliseconds per sample: 0.0018; artifact KB: 3.4.
- **Caveat:** 

### passive_aggressive — Passive-aggressive classifier

- **Family:** online_linear
- **Status:** complete
- **Search:** grid with 3 recorded trials; scoring `average_precision`.
- **Best parameters:** `{"estimator__C": 0.1}`
- **Validation threshold:** 0.2700; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 0.9881 / 0.8511 / 0.9756.
- **Validation F1 / F2:** 0.9091 / 0.9479.
- **Validation ROC-AUC / PR-AUC:** 0.9942 / 0.9524.
- **Validation MCC / Brier / log loss:** 0.9052 / 0.0122 / 0.0445.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.9198, PR-AUC=0.9249, ROC-AUC=0.9957; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 0.294; prediction milliseconds per sample: 0.0015; artifact KB: 4.1.
- **Caveat:** 

### voting — Soft voting ensemble

- **Family:** ensemble
- **Status:** complete
- **Search:** none with 1 recorded trials; scoring `average_precision`.
- **Best parameters:** `{}`
- **Validation threshold:** 0.4900; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 1.0000 / 1.0000 / 1.0000.
- **Validation F1 / F2:** 1.0000 / 1.0000.
- **Validation ROC-AUC / PR-AUC:** 1.0000 / 1.0000.
- **Validation MCC / Brier / log loss:** 1.0000 / 0.0012 / 0.0100.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.9799, PR-AUC=1.0000, ROC-AUC=1.0000; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 0.459; prediction milliseconds per sample: 0.0756; artifact KB: 849.4.
- **Caveat:** Diversity demonstration; base estimators use frozen reference settings.

### stacking — Stacking ensemble

- **Family:** ensemble
- **Status:** complete
- **Search:** none with 1 recorded trials; scoring `average_precision`.
- **Best parameters:** `{}`
- **Validation threshold:** 0.5000; selection metric `f2_with_precision_floor`.
- **Validation accuracy / precision / recall:** 1.0000 / 1.0000 / 1.0000.
- **Validation F1 / F2:** 1.0000 / 1.0000.
- **Validation ROC-AUC / PR-AUC:** 1.0000 / 1.0000.
- **Validation MCC / Brier / log loss:** 1.0000 / 0.0011 / 0.0135.
- **Validation confusion matrix:** TN=None, FP=None, FN=None, TP=None.
- **Test metrics:** F2=0.9950, PR-AUC=1.0000, ROC-AUC=1.0000; omitted for failed rows and the pre-refit winner.
- **Training seconds:** 1.261; prediction milliseconds per sample: 0.0265; artifact KB: 721.9.
- **Caveat:** Out-of-fold meta-learning; higher complexity than a single booster.

## 5. Calibration and threshold

The validation-selected threshold is **0.4900**. Calibration artifacts and the reliability diagram are under `evaluation/calibration/` and `images/models/`. The threshold is not chosen from test labels.

## 6. Fairness audit

Slice metrics are exported under `evaluation/fairness/`. Small groups are marked unstable. A gap is a review trigger, not proof of discrimination, because the supplied workbook is not a representative population sample.

## 7. Statistical tests

McNemar comparisons and Wilcoxon signed-rank comparisons are written to `evaluation/statistical_tests.md`. P-values are reported with the limitation that one held-out split and a small number of CV folds limit inferential strength.

## 8. Artifact manifest

The run manifest is `evaluation/runs/run_20260807_151423/run_manifest.json`. All generated figures are listed in `documentation/figure_index.md`. The input workbook checksum is stored in `data/metadata/raw_manifest.json`.

## 9. Responsible-use conclusion

The model supports triage and investigator prioritization. It is not an autonomous claim-denial system. A production decision requires verified policy and clinical documents, calibrated monitoring, fairness governance, and a human appeal pathway.
