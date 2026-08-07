"""Model definitions and hyperparameter search spaces.

Defines the twelve classification algorithms required for the Traditional ML
approach together with their hyperparameter search spaces. Each model is built
fresh inside a factory so tuning grids stay independent and reproducible.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import loguniform, uniform, randint
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils import setup_logging

logger = setup_logging()


# --------------------------------------------------------------------------
# Model factories - each returns a fresh unfitted estimator
# --------------------------------------------------------------------------
def logistic_regression(**kw):
    """Logistic Regression (Algorithm 1)."""
    return LogisticRegression(max_iter=2000, **kw)


def decision_tree(**kw):
    """Decision Tree Classifier (Algorithm 2)."""
    return DecisionTreeClassifier(**kw)


def random_forest(**kw):
    """Random Forest Classifier (Algorithm 3)."""
    return RandomForestClassifier(n_jobs=1, **kw)


def gradient_boosting(**kw):
    """Histogram-based Gradient Boosting (Algorithm 4)."""
    return HistGradientBoostingClassifier(**kw)


def xgboost(**kw):
    """XGBoost Classifier (Algorithm 5)."""
    return XGBClassifier(eval_metric="logloss", use_label_encoder=False, **kw)


def lightgbm(**kw):
    """LightGBM Classifier (Algorithm 6)."""
    return LGBMClassifier(verbose=-1, n_jobs=1, **kw)


def svm(**kw):
    """Support Vector Machine (Algorithm 7)."""
    return SVC(probability=True, **kw)


def knn(**kw):
    """K-Nearest Neighbours (Algorithm 8)."""
    return KNeighborsClassifier(**kw)


def naive_bayes(**kw):
    """Gaussian Naive Bayes (Algorithm 9)."""
    return GaussianNB(**kw)


def neural_network(**kw):
    """Artificial Neural Network (Algorithm 10, MLP bridge to DL)."""
    return MLPClassifier(max_iter=600, **kw)


def adaboost(**kw):
    """AdaBoost Classifier (Algorithm 11)."""
    return AdaBoostClassifier(**kw)


def qda(**kw):
    """Quadratic Discriminant Analysis (Algorithm 12)."""
    return QuadraticDiscriminantAnalysis(**kw)


# --------------------------------------------------------------------------
# Hyperparameter search spaces (grid or random distributions)
# --------------------------------------------------------------------------
MODEL_FACTORIES = {
    "Logistic Regression": logistic_regression,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "Gradient Boosting": gradient_boosting,
    "XGBoost": xgboost,
    "LightGBM": lightgbm,
    "SVM": svm,
    "KNN": knn,
    "Naive Bayes": naive_bayes,
    "Neural Network": neural_network,
    "AdaBoost": adaboost,
    "QDA": qda,
}

# Grid-style spaces (small search spaces -> exhaustive GridSearchCV)
GRID_SPACES = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100],
                             "penalty": ["l1", "l2"], "solver": ["liblinear"]},
    "Decision Tree": {"max_depth": [3, 5, 10, None],
                       "min_samples_leaf": [1, 5, 10]},
    "Naive Bayes": {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
    "KNN": {"n_neighbors": [3, 5, 7, 9, 11],
             "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]},
    "QDA": {"reg_param": [0.0, 0.1, 0.5, 1.0]},
    "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0, 1.5]},
}

# Random-style spaces (large search spaces -> RandomizedSearchCV)
RANDOM_SPACES = {
    "Random Forest": {
        "n_estimators": randint(100, 500),
        "max_depth": randint(5, 30),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
    },
    "Gradient Boosting": {
        "learning_rate": loguniform(0.01, 0.3),
        "max_iter": randint(100, 300),
        "max_depth": randint(3, 8),
        "max_leaf_nodes": randint(15, 60),
        "min_samples_leaf": randint(10, 60),
    },
    "XGBoost": {
        "n_estimators": randint(100, 400),
        "learning_rate": loguniform(0.01, 0.3),
        "max_depth": randint(3, 8),
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.7, 0.3),
        "min_child_weight": randint(1, 10),
        "scale_pos_weight": [1, 5, 10, 15],
    },
    "LightGBM": {
        "n_estimators": randint(100, 250),
        "learning_rate": loguniform(0.02, 0.2),
        "num_leaves": randint(15, 50),
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.7, 0.3),
        "min_child_samples": randint(10, 50),
    },
    "SVM": {
        "C": loguniform(0.01, 100),
        "gamma": loguniform(0.0001, 1),
        "kernel": ["rbf", "linear"],
    },
    "Neural Network": {
        "hidden_layer_sizes": [(64, 32), (128, 64), (256, 128), (128, 64, 32)],
        "alpha": loguniform(1e-5, 1e-1),
        "learning_rate_init": loguniform(1e-4, 1e-2),
    },
}


def get_search_space(name: str) -> dict:
    """Return the search space for a named model.

    Args:
        name: Model display name.

    Returns:
        dict: Hyperparameter search space.
    """
    if name in GRID_SPACES:
        return GRID_SPACES[name]
    if name in RANDOM_SPACES:
        return RANDOM_SPACES[name]
    return {}


def is_grid_space(name: str) -> bool:
    """Return True if the model should be tuned with grid search."""
    return name in GRID_SPACES


def build_model(name: str, **params):
    """Build a fresh estimator for a named model.

    Args:
        name: Model display name.
        **params: Hyperparameters to pass to the constructor.

    Returns:
        estimator: An unfitted scikit-learn estimator.
    """
    return MODEL_FACTORIES[name](**params)
