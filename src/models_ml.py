"""
Traditional Machine Learning Model Catalog and Factory.
Implements 12+ classification algorithms, ensembles, and stacking architectures
optimized for medical insurance fraud detection in the Indian healthcare domain.
"""

from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

from src.config import RANDOM_SEED
from src.utils import logger

def get_ml_model_catalog(scale_pos_weight: float = 8.5) -> Dict[str, Any]:
    """
    Returns a dictionary of all 12+ traditional ML classification algorithms
    configured with domain-appropriate initial hyperparameters.
    """
    models = {
        "Logistic_Regression": LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=500,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "Decision_Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=8,
            min_samples_split=15,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features="sqrt",
            oob_score=True,
            class_weight="balanced",
            n_jobs=1,
            random_state=RANDOM_SEED
        ),
        "Gradient_Boosting": HistGradientBoostingClassifier(
            max_iter=40,
            learning_rate=0.08,
            max_depth=5,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=50,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=1
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=50,
            learning_rate=0.08,
            num_leaves=31,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            verbosity=-1,
            random_state=RANDOM_SEED,
            n_jobs=1
        ),
        "Support_Vector_Machine": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=800, random_state=RANDOM_SEED),
            cv=2
        ),
        "K_Nearest_Neighbors": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1
        ),
        "Gaussian_Naive_Bayes": GaussianNB(
            var_smoothing=1e-8
        ),
        "Artificial_Neural_Net_MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=128,
            learning_rate="adaptive",
            max_iter=100,
            early_stopping=True,
            random_state=RANDOM_SEED
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=RANDOM_SEED),
            n_estimators=50,
            learning_rate=0.1,
            random_state=RANDOM_SEED
        ),
        "Extra_Trees": ExtraTreesClassifier(
            n_estimators=80,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED
        ),
        "Quadratic_Discriminant_Analysis": LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto"
        )
    }
    
    # Construct High-Performing Ensemble Voting and Stacking Models
    top_estimators = [
        ("rf", ExtraTreesClassifier(n_estimators=20, max_depth=6, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1)),
        ("xgb", xgb.XGBClassifier(n_estimators=20, learning_rate=0.1, max_depth=4, scale_pos_weight=scale_pos_weight, random_state=RANDOM_SEED, n_jobs=1, eval_metric="logloss")),
        ("lgb", lgb.LGBMClassifier(n_estimators=20, learning_rate=0.1, max_depth=4, scale_pos_weight=scale_pos_weight, random_state=RANDOM_SEED, n_jobs=1, verbosity=-1))
    ]
    
    models["Voting_Ensemble_Soft"] = VotingClassifier(
        estimators=top_estimators,
        voting="soft",
        n_jobs=1
    )
    
    models["Stacking_Classifier"] = StackingClassifier(
        estimators=top_estimators,
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=100, random_state=RANDOM_SEED),
        cv=2,
        n_jobs=1
    )
    
    return models

def get_hyperparameter_grids() -> Dict[str, Dict[str, list]]:
    """Defines systematic search spaces for hyperparameter tuning targeting F2 score."""
    grids = {
        "Logistic_Regression": {
            "C": [0.01, 0.1, 1.0, 5.0, 10.0],
            "penalty": ["l2"]
        },
        "Decision_Tree": {
            "max_depth": [4, 6, 8, 12, 16],
            "min_samples_split": [5, 10, 20],
            "min_samples_leaf": [2, 4, 8]
        },
        "Random_Forest": {
            "n_estimators": [100, 180, 250],
            "max_depth": [8, 12, 16],
            "min_samples_leaf": [2, 4, 8]
        },
        "XGBoost": {
            "n_estimators": [100, 180, 250],
            "learning_rate": [0.03, 0.07, 0.12],
            "max_depth": [4, 6, 8],
            "subsample": [0.75, 0.90]
        },
        "LightGBM": {
            "n_estimators": [100, 180, 250],
            "learning_rate": [0.03, 0.06, 0.1],
            "num_leaves": [20, 31, 50],
            "max_depth": [5, 7, 10]
        },
        "Support_Vector_Machine": {
            "C": [0.5, 2.0, 8.0],
            "gamma": ["scale", "auto", 0.01, 0.1]
        },
        "K_Nearest_Neighbors": {
            "n_neighbors": [3, 5, 7, 11, 15],
            "weights": ["uniform", "distance"]
        }
    }
    return grids
