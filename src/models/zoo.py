"""Model zoo definitions for the traditional fraud-classification benchmark.

The zoo deliberately favors algorithms available in scikit-learn so a fresh
laptop environment can reproduce the baseline without paid services or GPU
libraries. Optional XGBoost, LightGBM, and CatBoost adapters are documented in
``documentation/models.md`` and can be added without changing the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ModelSpec:
    """Declarative specification shared by search, evaluation, and reporting.

    Args:
        key: Stable artifact key.
        display_name: Human-readable model name.
        family: Algorithm family.
        builder: Callable constructing a fresh estimator.
        search_space: Hyperparameter grid or distribution dictionary.
        search_kind: ``grid``, ``random``, or ``none``.
        n_iter: Random-search budget when applicable.
        notes: Practical caveat displayed in the leaderboard.
    """

    key: str
    display_name: str
    family: str
    builder: Callable[[], Any]
    search_space: dict[str, list[Any]]
    search_kind: str = "grid"
    n_iter: int = 4
    notes: str = ""


def _calibrated_ridge() -> CalibratedClassifierCV:
    """Construct a probability-producing Ridge classifier."""
    return CalibratedClassifierCV(
        estimator=RidgeClassifier(class_weight="balanced"),
        method="sigmoid",
        cv=3,
    )


def _adaboost() -> AdaBoostClassifier:
    """Construct shallow-stump AdaBoost with an explicit random state."""
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        random_state=42,
    )


def _voting() -> VotingClassifier:
    """Construct a compact soft-voting ensemble of diverse classifiers."""
    return VotingClassifier(
        estimators=[
            (
                "logistic",
                LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42),
            ),
            (
                "forest",
                RandomForestClassifier(
                    n_estimators=140, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
                ),
            ),
            (
                "hist",
                HistGradientBoostingClassifier(max_iter=140, learning_rate=0.08, max_leaf_nodes=31, random_state=42),
            ),
        ],
        voting="soft",
        weights=[1, 2, 2],
        n_jobs=-1,
    )


def _stacking() -> StackingClassifier:
    """Construct an out-of-fold stacking model with a logistic meta learner."""
    return StackingClassifier(
        estimators=[
            (
                "logistic",
                LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42),
            ),
            (
                "forest",
                RandomForestClassifier(
                    n_estimators=120, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
                ),
            ),
            (
                "hist",
                HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_leaf_nodes=31, random_state=42),
            ),
        ],
        final_estimator=LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42
        ),
        stack_method="predict_proba",
        cv=3,
        n_jobs=-1,
    )


def build_model_specs(seed: int = 42) -> list[ModelSpec]:
    """Return the reproducible model suite used in the run.

    Args:
        seed: Seed propagated to stochastic estimators.
    Returns:
        List of 18 model specifications, including baselines and ensembles.
    """
    return [
        ModelSpec(
            "majority",
            "Majority-class baseline",
            "baseline",
            lambda: DummyClassifier(strategy="prior"),
            {},
            "none",
            notes="Sanity floor; predicts no fraud at the default threshold.",
        ),
        ModelSpec(
            "logistic_l2",
            "Logistic regression (L2)",
            "linear",
            lambda: LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear", random_state=seed),
            {"C": [0.1, 1.0, 10.0]},
        ),
        ModelSpec(
            "logistic_l1",
            "Logistic regression (L1)",
            "linear",
            lambda: LogisticRegression(
                max_iter=1500, class_weight="balanced", solver="liblinear", penalty="l1", random_state=seed
            ),
            {"C": [0.1, 1.0, 10.0]},
        ),
        ModelSpec(
            "decision_tree",
            "Decision tree",
            "tree",
            lambda: DecisionTreeClassifier(class_weight="balanced", random_state=seed),
            {"max_depth": [3, 6, 10, None], "min_samples_leaf": [1, 3, 8]},
        ),
        ModelSpec(
            "random_forest",
            "Random forest",
            "bagging",
            lambda: RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
            {"n_estimators": [140, 220], "max_depth": [None, 12], "min_samples_leaf": [1, 3]},
            "random",
            notes="Out-of-bag estimate is retained when bootstrap is enabled.",
        ),
        ModelSpec(
            "extra_trees",
            "Extra trees",
            "bagging",
            lambda: ExtraTreesClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
            {"n_estimators": [140, 220], "max_depth": [None, 12], "min_samples_leaf": [1, 3]},
            "random",
        ),
        ModelSpec(
            "gradient_boosting",
            "Gradient boosting",
            "boosting",
            lambda: GradientBoostingClassifier(random_state=seed),
            {"n_estimators": [80, 140], "learning_rate": [0.04, 0.10], "max_depth": [2, 3]},
            "random",
        ),
        ModelSpec(
            "hist_gradient_boosting",
            "Histogram gradient boosting",
            "boosting",
            lambda: HistGradientBoostingClassifier(random_state=seed),
            {
                "max_iter": [100, 180],
                "learning_rate": [0.04, 0.10],
                "max_leaf_nodes": [15, 31],
                "l2_regularization": [0.0, 1.0],
            },
            "random",
        ),
        ModelSpec(
            "adaboost",
            "AdaBoost",
            "boosting",
            lambda: _adaboost(),
            {"n_estimators": [80, 160], "learning_rate": [0.05, 0.20]},
            "random",
        ),
        ModelSpec(
            "svm_rbf",
            "Support vector machine (RBF)",
            "margin",
            lambda: SVC(probability=True, class_weight="balanced", random_state=seed),
            {"C": [0.5, 2.0, 10.0], "gamma": ["scale", 0.01]},
            "random",
            notes="Probability fitting adds runtime; the run records it explicitly.",
        ),
        ModelSpec(
            "knn",
            "K-nearest neighbors",
            "instance",
            lambda: KNeighborsClassifier(n_jobs=-1),
            {"n_neighbors": [5, 15, 31], "weights": ["uniform", "distance"]},
        ),
        ModelSpec(
            "gaussian_nb",
            "Gaussian naive Bayes",
            "probabilistic",
            lambda: GaussianNB(),
            {"var_smoothing": [1e-11, 1e-9, 1e-7]},
        ),
        ModelSpec(
            "bernoulli_nb", "Bernoulli naive Bayes", "probabilistic", lambda: BernoulliNB(), {"alpha": [0.1, 1.0, 5.0]}
        ),
        ModelSpec(
            "qda",
            "Quadratic discriminant analysis",
            "probabilistic",
            lambda: QuadraticDiscriminantAnalysis(reg_param=0.2),
            {"reg_param": [0.0, 0.2, 0.5, 0.8]},
        ),
        ModelSpec(
            "linear_discriminant",
            "Linear discriminant analysis",
            "probabilistic",
            lambda: LinearDiscriminantAnalysis(),
            {"solver": ["svd", "lsqr"]},
        ),
        ModelSpec(
            "mlp",
            "Multi-layer perceptron",
            "neural_baseline",
            lambda: MLPClassifier(max_iter=350, early_stopping=True, validation_fraction=0.15, random_state=seed),
            {"hidden_layer_sizes": [(32,), (64, 32)], "alpha": [1e-4, 1e-2], "learning_rate_init": [1e-3, 3e-3]},
            "random",
            notes="Classical MLP baseline; deep-learning approach remains separate.",
        ),
        ModelSpec(
            "ridge",
            "Calibrated ridge classifier",
            "linear",
            lambda: _calibrated_ridge(),
            {"estimator__alpha": [0.1, 1.0, 10.0]},
        ),
        ModelSpec(
            "passive_aggressive",
            "Passive-aggressive classifier",
            "online_linear",
            lambda: CalibratedClassifierCV(
                estimator=PassiveAggressiveClassifier(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed),
                method="sigmoid",
                cv=3,
            ),
            {"estimator__C": [0.1, 1.0, 10.0]},
        ),
        ModelSpec(
            "voting",
            "Soft voting ensemble",
            "ensemble",
            lambda: _voting(),
            {},
            "none",
            notes="Diversity demonstration; base estimators use frozen reference settings.",
        ),
        ModelSpec(
            "stacking",
            "Stacking ensemble",
            "ensemble",
            lambda: _stacking(),
            {},
            "none",
            notes="Out-of-fold meta-learning; higher complexity than a single booster.",
        ),
    ]
