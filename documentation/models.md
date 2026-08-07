# Model zoo and tuning protocol

**Purpose:** explain the classical algorithms benchmarked on the same feature matrix.  
**Run:** `run_20260807_151423`.  
**Last updated:** 07-08-2026 15:14:23 UTC.

## Leaderboard snapshot

| Key | Family | Validation PR-AUC | Validation F2 | Test F2 | Notes |
| --- | --- | --- | --- | --- | --- |
| random_forest | bagging | 1.0000 | 1.0000 | 0.9950 | Out-of-bag estimate is retained when bootstrap is enabled. |
| gradient_boosting | boosting | 1.0000 | 1.0000 | 0.9799 |  |
| hist_gradient_boosting | boosting | 1.0000 | 1.0000 | 0.9799 |  |
| adaboost | boosting | 1.0000 | 1.0000 | 0.9799 |  |
| voting | ensemble | 1.0000 | 1.0000 | 0.9799 | Diversity demonstration; base estimators use frozen reference settings. |
| stacking | ensemble | 1.0000 | 1.0000 | 0.9950 | Out-of-fold meta-learning; higher complexity than a single booster. |
| svm_rbf | margin | 0.9820 | 0.9804 | 0.9406 | Probability fitting adds runtime; the run records it explicitly. |
| decision_tree | tree | 0.9771 | 0.9804 | 0.9799 |  |
| extra_trees | bagging | 0.9874 | 0.9709 | 0.9701 |  |
| logistic_l1 | linear | 0.9775 | 0.9662 | 0.9709 |  |

## Algorithm families

| Family | Models | Why included |
| --- | --- | --- |
| Baseline | majority | Establishes the prevalence/accuracy floor. |
| Linear | logistic L2, logistic L1, calibrated ridge, passive-aggressive | Interpretable and computationally efficient baselines. |
| Tree | decision tree | Direct if-then auditability. |
| Bagging | random forest, extra trees | Variance reduction and nonlinear interactions. |
| Boosting | gradient boosting, histogram gradient boosting, AdaBoost | Strong tabular learners under constrained compute. |
| Margin/instance | RBF SVM, KNN | Tests distance and margin assumptions after scaling. |
| Probabilistic | Gaussian NB, Bernoulli NB, LDA, QDA | Simple distributional references with different assumptions. |
| Neural baseline | scikit-learn MLP | Bridge to the later deep-learning approach without conflating scope. |
| Ensemble | soft voting, stacking | Tests whether diverse learners complement one another. |

## Search policy

The search metric is average precision on stratified three-fold training folds.
The public configuration contains small grids and random-search budgets so a
fresh environment can finish on a laptop. Search results are saved under
`evaluation/tuning/`; complete failures are retained with a status and error
message. The reported threshold is selected only from validation probabilities
and uses F2 because missing fraud is more costly than reviewing a false alarm.

## Model-specific notes

### Majority-class baseline

- Key: `majority`
- Family: `baseline`
- Status: `complete`
- Search method: `none`
- Search trials: 1
- Best parameters: `{}`
- Validation F2: 0.2443; validation PR-AUC: 0.0607
- Test evaluation: available once
- Caveat: Sanity floor; predicts no fraud at the default threshold.

### Logistic regression (L2)

- Key: `logistic_l2`
- Family: `linear`
- Status: `complete`
- Search method: `grid`
- Search trials: 3
- Best parameters: `{"C": 10.0}`
- Validation F2: 0.9479; validation PR-AUC: 0.9747
- Test evaluation: available once
- Caveat: 

### Logistic regression (L1)

- Key: `logistic_l1`
- Family: `linear`
- Status: `complete`
- Search method: `grid`
- Search trials: 3
- Best parameters: `{"C": 1.0}`
- Validation F2: 0.9662; validation PR-AUC: 0.9775
- Test evaluation: available once
- Caveat: 

### Decision tree

- Key: `decision_tree`
- Family: `tree`
- Status: `complete`
- Search method: `grid`
- Search trials: 12
- Best parameters: `{"max_depth": 3, "min_samples_leaf": 3}`
- Validation F2: 0.9804; validation PR-AUC: 0.9771
- Test evaluation: available once
- Caveat: 

### Random forest

- Key: `random_forest`
- Family: `bagging`
- Status: `complete`
- Search method: `random`
- Search trials: 4
- Best parameters: `{"max_depth": null, "min_samples_leaf": 1, "n_estimators": 220}`
- Validation F2: 1.0000; validation PR-AUC: 1.0000
- Test evaluation: available once
- Caveat: Out-of-bag estimate is retained when bootstrap is enabled.

### Extra trees

- Key: `extra_trees`
- Family: `bagging`
- Status: `complete`
- Search method: `random`
- Search trials: 4
- Best parameters: `{"max_depth": null, "min_samples_leaf": 1, "n_estimators": 140}`
- Validation F2: 0.9709; validation PR-AUC: 0.9874
- Test evaluation: available once
- Caveat: 

### Gradient boosting

- Key: `gradient_boosting`
- Family: `boosting`
- Status: `complete`
- Search method: `random`
- Search trials: 4
- Best parameters: `{"learning_rate": 0.1, "max_depth": 2, "n_estimators": 140}`
- Validation F2: 1.0000; validation PR-AUC: 1.0000
- Test evaluation: available once
- Caveat: 

### Histogram gradient boosting

- Key: `hist_gradient_boosting`
- Family: `boosting`
- Status: `complete`
- Search method: `random`
- Search trials: 4
- Best parameters: `{"l2_regularization": 1.0, "learning_rate": 0.1, "max_iter": 180, "max_leaf_nodes": 15}`
- Validation F2: 1.0000; validation PR-AUC: 1.0000
- Test evaluation: available once
- Caveat: 

### AdaBoost

- Key: `adaboost`
- Family: `boosting`
- Status: `complete`
- Search method: `random`
- Search trials: 4
- Best parameters: `{"learning_rate": 0.05, "n_estimators": 80}`
- Validation F2: 1.0000; validation PR-AUC: 1.0000
- Test evaluation: available once
- Caveat: 

### Support vector machine (RBF)

- Key: `svm_rbf`
- Family: `margin`
- Status: `complete`
- Search method: `random`
- Search trials: 4
- Best parameters: `{"C": 10.0, "gamma": 0.01}`
- Validation F2: 0.9804; validation PR-AUC: 0.9820
- Test evaluation: available once
- Caveat: Probability fitting adds runtime; the run records it explicitly.

### K-nearest neighbors

- Key: `knn`
- Family: `instance`
- Status: `complete`
- Search method: `grid`
- Search trials: 6
- Best parameters: `{"n_neighbors": 5, "weights": "distance"}`
- Validation F2: 0.8520; validation PR-AUC: 0.8219
- Test evaluation: available once
- Caveat: 

### Gaussian naive Bayes

- Key: `gaussian_nb`
- Family: `probabilistic`
- Status: `complete`
- Search method: `grid`
- Search trials: 3
- Best parameters: `{"var_smoothing": 1e-07}`
- Validation F2: 0.8511; validation PR-AUC: 0.6629
- Test evaluation: available once
- Caveat: 

### Bernoulli naive Bayes

- Key: `bernoulli_nb`
- Family: `probabilistic`
- Status: `complete`
- Search method: `grid`
- Search trials: 3
- Best parameters: `{"alpha": 5.0}`
- Validation F2: 0.8032; validation PR-AUC: 0.4821
- Test evaluation: available once
- Caveat: 

### Quadratic discriminant analysis

- Key: `qda`
- Family: `probabilistic`
- Status: `complete`
- Search method: `grid`
- Search trials: 4
- Best parameters: `{"reg_param": 0.0}`
- Validation F2: 0.8952; validation PR-AUC: 0.7000
- Test evaluation: available once
- Caveat: 

### Linear discriminant analysis

- Key: `linear_discriminant`
- Family: `probabilistic`
- Status: `complete`
- Search method: `grid`
- Search trials: 2
- Best parameters: `{"solver": "lsqr"}`
- Validation F2: 0.9569; validation PR-AUC: 0.9133
- Test evaluation: available once
- Caveat: 

### Multi-layer perceptron

- Key: `mlp`
- Family: `neural_baseline`
- Status: `complete`
- Search method: `random`
- Search trials: 4
- Best parameters: `{"alpha": 0.01, "hidden_layer_sizes": [64, 32], "learning_rate_init": 0.003}`
- Validation F2: 0.9615; validation PR-AUC: 0.9755
- Test evaluation: available once
- Caveat: Classical MLP baseline; deep-learning approach remains separate.

### Calibrated ridge classifier

- Key: `ridge`
- Family: `linear`
- Status: `complete`
- Search method: `grid`
- Search trials: 3
- Best parameters: `{"estimator__alpha": 1.0}`
- Validation F2: 0.9259; validation PR-AUC: 0.9501
- Test evaluation: available once
- Caveat: 

### Passive-aggressive classifier

- Key: `passive_aggressive`
- Family: `online_linear`
- Status: `complete`
- Search method: `grid`
- Search trials: 3
- Best parameters: `{"estimator__C": 0.1}`
- Validation F2: 0.9479; validation PR-AUC: 0.9524
- Test evaluation: available once
- Caveat: 

### Soft voting ensemble

- Key: `voting`
- Family: `ensemble`
- Status: `complete`
- Search method: `none`
- Search trials: 1
- Best parameters: `{}`
- Validation F2: 1.0000; validation PR-AUC: 1.0000
- Test evaluation: available once
- Caveat: Diversity demonstration; base estimators use frozen reference settings.

### Stacking ensemble

- Key: `stacking`
- Family: `ensemble`
- Status: `complete`
- Search method: `none`
- Search trials: 1
- Best parameters: `{}`
- Validation F2: 1.0000; validation PR-AUC: 1.0000
- Test evaluation: available once
- Caveat: Out-of-fold meta-learning; higher complexity than a single booster.
