"""Training pipeline: tuning, fitting and evaluating all ML algorithms.

Orchestrates the full Approach-1 training workflow: builds the feature matrix,
applies class-imbalance handling, performs stratified cross-validation
hyperparameter tuning (grid or random), trains the final model, and produces
full evaluation results on the held-out test set.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.metrics import fbeta_score, make_scorer

# Custom scorer prioritising recall (fraud detection business need).
F2_SCORER = make_scorer(fbeta_score, beta=2, zero_division=0)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from src import models as M
from src.evaluate import (
    EvalResult, business_impact, compute_metrics, optimal_threshold_f2,
)
from src.utils import setup_logging

logger = setup_logging()


def handle_imbalance(X, y, strategy: str = "smote", seed: int = 42):
    """Resample/balance the training data per the configured strategy.

    Args:
        X: Feature matrix.
        y: Target vector.
        strategy: smote | smoteenn | random_undersample | none.
        seed: Random seed.

    Returns:
        tuple: (X_resampled, y_resampled).
    """
    if strategy == "smote":
        return SMOTE(random_state=seed).fit_resample(X, y)
    if strategy == "smoteenn":
        return SMOTEENN(random_state=seed).fit_resample(X, y)
    if strategy == "random_undersample":
        from imblearn.under_sampling import RandomUnderSampler
        return RandomUnderSampler(random_state=seed).fit_resample(X, y)
    return X, y


def tune_model(name, X, y, cv_folds=5, scoring="f2", random_iter=40, seed=42,
               n_jobs=1):
    """Tune a model's hyperparameters with stratified cross-validation.

    Args:
        name: Model name.
        X: Feature matrix.
        y: Target.
        cv_folds: Number of CV folds.
        scoring: Scoring metric for selection (F2).
        random_iter: Iterations for randomized search.
        seed: Random seed.
        n_jobs: Parallel workers for the search (1 avoids nested-parallelism issues).

    Returns:
        tuple: (best_estimator, best_params, best_score).
    """
    space = M.get_search_space(name)
    model = M.build_model(name)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    if M.is_grid_space(name):
        search = GridSearchCV(model, space, cv=cv, scoring=F2_SCORER, n_jobs=n_jobs, refit=True)
    else:
        search = RandomizedSearchCV(
            model, space, n_iter=random_iter, cv=cv, scoring=F2_SCORER,
            n_jobs=n_jobs, random_state=seed, refit=True,
        )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def train_and_evaluate(name, X_train, y_train, X_test, y_test, cfg, extra_weights=None):
    """Tune, fit and evaluate one model end to end.

    Args:
        name: Model name.
        X_train, y_train: Training features/target.
        X_test, y_test: Held-out test features/target.
        cfg: Configuration dict.
        extra_weights: Optional class-weight override.

    Returns:
        EvalResult: Full evaluation result.
    """
    logger.info("=== Training %s ===", name)
    best, params, cv_score = tune_model(
        name, X_train, y_train,
        cv_folds=cfg["training"]["cv_folds"],
        scoring=cfg["training"]["cv_scoring"],
        random_iter=cfg["training"]["random_search_iter"],
        seed=cfg["training"]["random_state"],
        n_jobs=cfg["training"].get("n_jobs", 1),
    )

    # Override class weight for imbalance-sensitive estimators if requested
    if extra_weights is not None and hasattr(best, "set_params"):
        try:
            best.set_params(class_weight=extra_weights)
        except Exception:
            pass
        best.fit(X_train, y_train)

    t0 = time.time()
    best.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_prob = best.predict_proba(X_test)[:, 1]
    pred_time = (time.time() - t0) / len(X_test) * 1000.0

    threshold, _ = optimal_threshold_f2(y_test, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_prob)

    res = EvalResult(
        name=name, metrics=metrics,
        confusion_matrix=np.asarray(pd.crosstab(
            y_test, y_pred, rownames=["true"], colnames=["pred"]
        ).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values),
        optimal_threshold=threshold,
        probabilities=y_prob, y_true=np.asarray(y_test),
        train_time=train_time, pred_time_ms=pred_time,
        n_hyperparams=len(params),
    )
    res.metrics["cv_best_score"] = cv_score
    res.metrics["best_params"] = params
    res.model = best
    return res


def save_model(name, model, path: Path) -> None:
    """Persist a fitted model to disk.

    Args:
        name: Model name.
        model: Fitted estimator.
        path: Output directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    fname = path / (name.replace(" ", "_").lower() + ".pkl")
    with open(fname, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model %s -> %s", name, fname)
