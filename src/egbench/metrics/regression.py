"""Spearman correlation between predicted scores and the gene effect strength."""
from __future__ import annotations

import math

import numpy as np

from egbench.metrics._input import UNDEFINED, check_continuous_pair
from egbench.metrics.discrimination import average_ranks


def _dot(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sum(first * second))


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.min() == first.max() or second.min() == second.max():
        return UNDEFINED
    centred_first = first - first.mean()
    centred_second = second - second.mean()
    denominator = math.sqrt(
        _dot(centred_first, centred_first) * _dot(centred_second, centred_second)
    )
    if denominator == 0.0:
        return UNDEFINED
    return float(min(max(_dot(centred_first, centred_second) / denominator, -1.0), 1.0))


def spearman(y_true: object, y_score: object) -> float:
    target, scores = check_continuous_pair(y_true, y_score)
    return _correlation(average_ranks(scores), average_ranks(-target))
