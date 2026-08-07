"""Model registry and F2-first hyperparameter tuning for Approach 1.

Every registry entry exposes an auditable estimator, a deliberately bounded search
space, and the number of tuned hyperparameters. Optional XGBoost/LightGBM imports
are guarded so the rest of the project remains usable on minimal environments.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:  # Optional libraries are installed by the full requirements but kept graceful.
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - depends on environment
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - depends on environment
    LGBMClassifier = None


@dataclass(frozen=True)
class ModelSpec:
    """Static definition of one benchmarked classifier and its tuning protocol."""

    name: str
    factory: Callable[[float, int, int], Any]
    parameters: dict[str, list[Any]]
    search: str
    tuned_hyperparameters: int
    requires_nonnegative: bool = False


@dataclass
class TrainedModel:
    """Fitted model plus selection/timing information needed by evaluation and serving."""

    name: str
    estimator: Any
    threshold: float
    validation_f2: float
    best_params: dict[str, Any]
    cv_f2_mean: float
    cv_f2_std: float
    cv_f2_scores: list[float]
    training_seconds: float
    nonnegative_scaler: MinMaxScaler | None
    tuned_hyperparameters: int
    search_space: dict[str, list[Any]]


def _class_weight_ratio(y: np.ndarray) -> float:
    """Calculate negative-to-positive ratio for imbalance-aware estimators."""
    positives = max(int(np.sum(y == 1)), 1)
    negatives = max(int(np.sum(y == 0)), 1)
    return negatives / positives


def model_registry(y_train: np.ndarray, seed: int, n_jobs: int) -> list[ModelSpec]:
    """Build the complete model registry, including optional boosting libraries.

    Args:
        y_train: Binary train labels used solely to calculate balancing weight.
        seed: Reproducibility seed.
        n_jobs: Worker count for compatible estimators.

    Returns:
        List of at least twelve model specifications.
    """
    ratio = _class_weight_ratio(y_train)
    specs = [
        ModelSpec("Logistic Regression (L2)", lambda r, s, n: LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=s), {"C": [0.05, 0.2, 1.0, 4.0]}, "grid", 1),
        ModelSpec("Logistic Regression (L1)", lambda r, s, n: LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", penalty="l1", random_state=s), {"C": [0.05, 0.2, 1.0, 4.0]}, "grid", 1),
        ModelSpec("Decision Tree", lambda r, s, n: DecisionTreeClassifier(class_weight="balanced", random_state=s), {"max_depth": [4, 7, 11], "min_samples_leaf": [10, 30], "min_impurity_decrease": [0.0, 0.001]}, "grid", 3),
        ModelSpec("Random Forest", lambda r, s, n: RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample", oob_score=True, n_jobs=n, random_state=s), {"max_depth": [8, 14, None], "min_samples_leaf": [2, 8, 20], "max_features": ["sqrt", 0.6]}, "random", 3),
        ModelSpec("Histogram Gradient Boosting", lambda r, s, n: HistGradientBoostingClassifier(random_state=s, class_weight={0: 1.0, 1: r}), {"learning_rate": [0.04, 0.08, 0.15], "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0.0, 0.2]}, "random", 3),
        ModelSpec("SVM (RBF)", lambda r, s, n: SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=s), {"C": [0.4, 1.0, 3.0], "gamma": ["scale", 0.02, 0.08]}, "grid", 2),
        ModelSpec("SVM (linear)", lambda r, s, n: SVC(kernel="linear", class_weight="balanced", probability=True, random_state=s), {"C": [0.1, 0.5, 1.5]}, "grid", 1),
        ModelSpec("K-Nearest Neighbors", lambda r, s, n: KNeighborsClassifier(n_jobs=n), {"n_neighbors": [7, 15, 31], "weights": ["distance", "uniform"], "p": [1, 2]}, "grid", 3),
        ModelSpec("Gaussian Naive Bayes", lambda r, s, n: GaussianNB(), {"var_smoothing": [1e-11, 1e-9, 1e-7]}, "grid", 1),
        ModelSpec("Multinomial Naive Bayes", lambda r, s, n: MultinomialNB(), {"alpha": [0.05, 0.25, 0.8, 1.5]}, "grid", 1, requires_nonnegative=True),
        ModelSpec("Shallow Neural Network", lambda r, s, n: MLPClassifier(hidden_layer_sizes=(96, 48), activation="relu", solver="adam", alpha=0.0005, early_stopping=True, validation_fraction=.15, max_iter=250, random_state=s), {"hidden_layer_sizes": [(64, 32), (128, 64, 32)], "alpha": [0.0001, 0.001], "learning_rate_init": [0.0005, 0.001]}, "random", 3),
        ModelSpec("AdaBoost", lambda r, s, n: AdaBoostClassifier(random_state=s), {"n_estimators": [80, 160, 260], "learning_rate": [0.03, 0.1, 0.3]}, "grid", 2),
        ModelSpec("Quadratic Discriminant Analysis", lambda r, s, n: QuadraticDiscriminantAnalysis(), {"reg_param": [0.01, 0.1, 0.3, 0.6]}, "grid", 1),
    ]
    if XGBClassifier is not None:
        specs.append(ModelSpec("XGBoost", lambda r, s, n: XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=220, scale_pos_weight=r, learning_rate=.08, n_jobs=n, random_state=s, tree_method="hist"), {"max_depth": [3, 5, 7], "min_child_weight": [1, 5], "subsample": [.7, 1.0], "colsample_bytree": [.65, 1.0]}, "random", 4))
    if LGBMClassifier is not None:
        specs.append(ModelSpec("LightGBM", lambda r, s, n: LGBMClassifier(objective="binary", n_estimators=220, class_weight={0: 1.0, 1: r}, learning_rate=.08, n_jobs=n, random_state=s, verbosity=-1), {"num_leaves": [15, 31, 63], "min_child_samples": [15, 35], "subsample": [.7, 1.0], "colsample_bytree": [.65, 1.0]}, "random", 4))
    return specs


def optimal_f2_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Find a validation-only probability threshold maximizing F2.

    Args:
        y_true: Binary validation labels.
        probability: Predicted fraud probabilities.

    Returns:
        Tuple of selected threshold and corresponding F2 score.
    """
    candidates = np.linspace(.05, .95, 181)
    scores = [fbeta_score(y_true, probability >= threshold, beta=2, zero_division=0) for threshold in candidates]
    best_index = int(np.argmax(scores))
    return float(candidates[best_index]), float(scores[best_index])


def _fit_search(spec: ModelSpec, X: np.ndarray, y: np.ndarray, config: dict[str, Any], seed: int) -> tuple[Any, float, float, list[float]]:
    """Fit a stratified-CV F2 hyperparameter search for one model specification."""
    n_jobs = int(config["training"]["n_jobs"])
    estimator = spec.factory(_class_weight_ratio(y), seed, n_jobs)
    folds = int(config["training"]["cv_folds"])
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    common = {"scoring": scorer, "cv": cv, "n_jobs": n_jobs, "refit": True, "return_train_score": False}
    if spec.search == "grid":
        search = GridSearchCV(estimator, spec.parameters, **common)
    else:
        all_combinations = int(np.prod([len(v) for v in spec.parameters.values()]))
        search = RandomizedSearchCV(estimator, spec.parameters, n_iter=min(int(config["training"]["search_iterations"]), all_combinations), random_state=seed, **common)
    search.fit(X, y)
    fold_scores = [float(search.cv_results_[f"split{fold}_test_score"][search.best_index_]) for fold in range(folds)]
    return search, float(search.best_score_), float(search.cv_results_["std_test_score"][search.best_index_]), fold_scores


def fit_tuned_model(spec: ModelSpec, X_train: np.ndarray, y_train: np.ndarray, X_validation: np.ndarray, y_validation: np.ndarray, config: dict[str, Any], seed: int) -> TrainedModel:
    """Tune a model by CV F2, then set its operational threshold from validation F2.

    A small stratified subset is used for SVM fitting if configured, protecting a
    laptop-scale reproducible run from the quadratic RBF kernel cost. This constraint is
    recorded in model metadata and does not affect the common validation/test partitions.

    Args:
        spec: Registered classifier specification.
        X_train: Selected, transformed training features.
        y_train: Binary training labels.
        X_validation: Selected, transformed validation features.
        y_validation: Binary validation labels.
        config: Project configuration.
        seed: Per-model deterministic seed.

    Returns:
        TrainedModel including selected threshold and CV evidence.
    """
    start = time.perf_counter()
    scaler: MinMaxScaler | None = None
    train_x, validation_x, train_y = X_train, X_validation, y_train
    if spec.requires_nonnegative:
        scaler = MinMaxScaler()
        train_x = scaler.fit_transform(X_train)
        validation_x = scaler.transform(X_validation)
    if spec.name.startswith("SVM") and len(train_x) > int(config["training"]["max_training_rows_for_svm"]):
        # Deterministic stratified downsample only for costly SVM tuning/training.
        from sklearn.model_selection import train_test_split
        train_x, _, train_y, _ = train_test_split(train_x, train_y, train_size=int(config["training"]["max_training_rows_for_svm"]), stratify=train_y, random_state=seed)
    if spec.name == "Shallow Neural Network":
        # MLPClassifier in sklearn 1.5 has no class_weight; use train-only SMOTE.
        sampler = SMOTE(random_state=seed, k_neighbors=int(config["preprocessing"]["smote_k_neighbors"]))
        train_x, train_y = sampler.fit_resample(train_x, train_y)
    search, mean_score, std_score, fold_scores = _fit_search(spec, train_x, train_y, config, seed)
    probabilities = search.best_estimator_.predict_proba(validation_x)[:, 1]
    threshold, validation_f2 = optimal_f2_threshold(y_validation, probabilities)
    return TrainedModel(
        name=spec.name, estimator=search.best_estimator_, threshold=threshold, validation_f2=validation_f2,
        best_params={key: value.item() if isinstance(value, np.generic) else value for key, value in search.best_params_.items()},
        cv_f2_mean=mean_score, cv_f2_std=std_score, cv_f2_scores=fold_scores, training_seconds=time.perf_counter() - start,
        nonnegative_scaler=scaler, tuned_hyperparameters=spec.tuned_hyperparameters, search_space=spec.parameters,
    )
