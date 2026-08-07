"""Paired statistical comparisons used in the evaluation appendix."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import wilcoxon


def mcnemar_p_value(
    y_true: Iterable[int], pred_a: Iterable[int], pred_b: Iterable[int]
) -> dict[str, float | int | str]:
    """Compute a continuity-corrected McNemar comparison without statsmodels.

    Args:
        y_true: Shared binary test labels.
        pred_a: Predictions from model A.
        pred_b: Predictions from model B.
    Returns:
        Discordant-cell counts and an asymptotic chi-square p-value.
    """
    actual = np.asarray(list(y_true), dtype=int)
    a = np.asarray(list(pred_a), dtype=int)
    b = np.asarray(list(pred_b), dtype=int)
    if not (len(actual) == len(a) == len(b)):
        raise ValueError("McNemar inputs must have equal length")
    a_correct = a == actual
    b_correct = b == actual
    b01 = int(np.sum(a_correct & ~b_correct))
    b10 = int(np.sum(~a_correct & b_correct))
    discordant = b01 + b10
    statistic = ((abs(b01 - b10) - 1) ** 2 / discordant) if discordant else 0.0
    # Survival function for chi-square with one degree of freedom.
    from scipy.stats import chi2

    p_value = float(chi2.sf(statistic, 1))
    return {
        "a_only_errors": b10,
        "b_only_errors": b01,
        "discordant_pairs": discordant,
        "chi_square": float(statistic),
        "p_value": p_value,
        "method": "continuity-corrected McNemar asymptotic test",
    }


def wilcoxon_p_value(scores_a: Iterable[float], scores_b: Iterable[float]) -> dict[str, float | str]:
    """Compare paired cross-validation scores using Wilcoxon signed-rank test.

    Args:
        scores_a: Fold scores for model A.
        scores_b: Fold scores for model B.
    Returns:
        Test statistic and two-sided p-value.
    """
    a = np.asarray(list(scores_a), dtype=float)
    b = np.asarray(list(scores_b), dtype=float)
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("Wilcoxon comparison needs at least three paired scores")
    try:
        result = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return {
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "method": "Wilcoxon signed-rank test",
        }
    except ValueError:
        return {"statistic": 0.0, "p_value": 1.0, "method": "Wilcoxon signed-rank test; all paired differences zero"}
