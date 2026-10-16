"""Input checks shared by the metrics."""
from __future__ import annotations

import numpy as np

UNDEFINED = float("nan")

_NUMERIC_KINDS = "biuf"
_EXAMPLE_LIMIT = 5


def _check_numbers(array: np.ndarray, name: str) -> None:
    if array.dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(f"{name} must hold numbers, its dtype is {array.dtype}")


def numeric_array(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a one dimensional array, its shape is {array.shape}"
        )
    _check_numbers(array, name)
    return array


def _check_pair_shape(y_true: object, y_score: object) -> tuple[np.ndarray, np.ndarray]:
    target = numeric_array(y_true, "y_true")
    scores = numeric_array(y_score, "y_score").astype("float64", copy=False)
    if target.size != scores.size:
        raise ValueError(
            f"y_true and y_score must be the same length, y_true holds "
            f"{target.size} values and y_score holds {scores.size}"
        )
    if target.size == 0:
        raise ValueError(
            "y_true and y_score are empty; a slice with no gene in it is an "
            "empty selection upstream, not a metric with no value"
        )
    return target, scores


def _check_finite_scores(scores: np.ndarray) -> None:
    nan_count = int(np.isnan(scores).sum())
    infinite_count = int(np.isinf(scores).sum())
    if nan_count or infinite_count:
        raise ValueError(
            f"y_score must be finite, NaN or missing: {nan_count}, infinite: "
            f"{infinite_count}"
        )


def check_pair(y_true: object, y_score: object) -> tuple[np.ndarray, np.ndarray]:
    labels, scores = _check_pair_shape(y_true, y_score)
    positive = labels == 1
    unknown = labels[~(positive | (labels == 0))]
    if unknown.size:
        distinct = np.unique(unknown)
        shown = ", ".join(repr(value.item()) for value in distinct[:_EXAMPLE_LIMIT])
        if distinct.size > _EXAMPLE_LIMIT:
            shown += ", ..."
        raise ValueError(
            f"y_true holds the labels 0 and 1, it also holds {unknown.size} other "
            f"values ({shown}); a missing label is one of them, and a gene whose "
            f"label is unknown is not a gene that is negative"
        )
    _check_finite_scores(scores)
    return positive, scores


def check_continuous_pair(
    y_true: object, y_score: object
) -> tuple[np.ndarray, np.ndarray]:
    target, scores = _check_pair_shape(y_true, y_score)
    target = target.astype("float64", copy=False)
    nan_count = int(np.isnan(target).sum())
    infinite_count = int(np.isinf(target).sum())
    if nan_count or infinite_count:
        raise ValueError(
            f"y_true must be finite, NaN or missing: {nan_count}, infinite: "
            f"{infinite_count}; a gene whose target was not measured is not a "
            f"gene whose target is zero"
        )
    _check_finite_scores(scores)
    return target, scores
