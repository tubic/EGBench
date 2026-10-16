"""AUROC, AUPRC, average ranks and hits in the top k."""
from __future__ import annotations

import numpy as np

from egbench.metrics._input import UNDEFINED, check_pair

TOP_K = 100

def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    count = ordered.size
    starts_group = np.empty(count, dtype=bool)
    starts_group[0] = True
    starts_group[1:] = ordered[1:] != ordered[:-1]
    group_start = np.flatnonzero(starts_group)
    group_stop = np.append(group_start[1:], count)
    group_rank = 0.5 * (group_start + group_stop - 1) + 1.0
    ranks = np.empty(count, dtype="float64")
    ranks[order] = group_rank[np.cumsum(starts_group) - 1]
    return ranks


def auroc(y_true: object, y_score: object) -> float:
    positive, scores = check_pair(y_true, y_score)
    positive_count = int(np.count_nonzero(positive))
    negative_count = positive.size - positive_count
    if positive_count == 0 or negative_count == 0:
        return UNDEFINED
    ranks = average_ranks(scores)
    positive_rank_sum = float(ranks[positive].sum())
    concordant = positive_rank_sum - positive_count * (positive_count + 1) / 2
    return concordant / (positive_count * negative_count)


def auprc(y_true: object, y_score: object) -> float:
    positive, scores = check_pair(y_true, y_score)
    positive_count = int(np.count_nonzero(positive))
    if positive_count == 0:
        return UNDEFINED
    order = np.argsort(-scores, kind="stable")
    descending = scores[order]
    ends_group = np.empty(descending.size, dtype=bool)
    ends_group[:-1] = descending[:-1] != descending[1:]
    ends_group[-1] = True
    operating_point = np.flatnonzero(ends_group)
    found = np.cumsum(positive[order], dtype="int64")[operating_point]
    selected = operating_point + 1
    precision = found / selected
    recall = found / positive_count
    return float(np.sum(np.diff(recall, prepend=0.0) * precision))


def _top_k_boundary(
    values: np.ndarray, k: int = TOP_K
) -> tuple[np.ndarray, np.ndarray, int, int]:
    boundary = float(np.partition(values, -k)[-k])
    above = values > boundary
    tied = values == boundary
    return above, tied, k - int(np.count_nonzero(above)), int(np.count_nonzero(tied))


def _expected_hits_in_top_k(
    positive: np.ndarray, scores: np.ndarray, k: int = TOP_K
) -> float:
    above, tied, places_left, tied_count = _top_k_boundary(scores, k)
    return float(
        np.count_nonzero(above & positive)
        + np.count_nonzero(tied & positive) * places_left / tied_count
    )


def hits_at_k(y_true: object, y_score: object, *, k: int) -> float:
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"hits_at_k: k must be a positive integer, got {k!r}")
    positive, scores = check_pair(y_true, y_score)
    if positive.size < k:
        return UNDEFINED
    return _expected_hits_in_top_k(positive, scores, k)
