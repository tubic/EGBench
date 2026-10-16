"""Train-role feature ranking and recursive feature elimination for EssTFNet.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from egbench.contracts import SEED

CANDIDATE_COUNT = 8263
SELECTED_COUNT = 4250
RFE_STEP = 512


def point_biserial_from_sufficient_statistics(
    *,
    row_count: int,
    label_sum: float,
    feature_sum: np.ndarray,
    feature_square_sum: np.ndarray,
    feature_label_sum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    if row_count < 2 or not (0.0 < label_sum < float(row_count)):
        raise ValueError("outer-train labels must contain both binary classes")
    sum_x = np.asarray(feature_sum, dtype=np.float64)
    sum_x2 = np.asarray(feature_square_sum, dtype=np.float64)
    sum_xy = np.asarray(feature_label_sum, dtype=np.float64)
    if sum_x.ndim != 1 or sum_x2.shape != sum_x.shape or sum_xy.shape != sum_x.shape:
        raise ValueError("streamed feature statistics must be aligned one-dimensional arrays")
    mean_square_term = np.square(sum_x) / float(row_count)
    centered_x_square_sum = sum_x2 - mean_square_term
    cancellation_tolerance = (
        np.finfo(np.float64).eps
        * float(row_count)
        * 8.0
        * np.maximum(
            np.maximum(np.abs(sum_x2), np.abs(mean_square_term)),
            np.finfo(np.float64).tiny,
        )
    )
    centered_x_square_sum = np.where(
        centered_x_square_sum > cancellation_tolerance,
        centered_x_square_sum,
        0.0,
    )
    centered_y_square_sum = label_sum - label_sum * label_sum / float(row_count)
    numerator = sum_xy - sum_x * label_sum / float(row_count)
    denominator = np.sqrt(centered_x_square_sum * centered_y_square_sum)
    correlation = np.divide(
        numerator,
        denominator,
        out=np.zeros(sum_x.shape, dtype=np.float64),
        where=denominator > 0.0,
    )
    indices = np.arange(sum_x.size, dtype=np.int64)
    ranking = np.lexsort((indices, -np.abs(correlation)))
    return ranking, correlation


def select_ranked_outer_train_features(
    candidate_features: np.ndarray,
    train_labels: np.ndarray,
    candidate_indices_ranked: np.ndarray,
    candidate_correlations: np.ndarray,
    *,
    selected_count: int = SELECTED_COUNT,
    step: int = RFE_STEP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:

    features = np.asarray(candidate_features, dtype=np.float32)
    labels = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    candidates = np.asarray(candidate_indices_ranked, dtype=np.int64).reshape(-1)
    correlations = np.asarray(candidate_correlations, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or features.shape != (labels.size, candidates.size):
        raise ValueError("candidate matrix, labels and ranked indices are not aligned")
    if correlations.shape != candidates.shape:
        raise ValueError("candidate correlations are not aligned with ranked indices")
    if len(set(candidates.tolist())) != candidates.size or np.any(candidates < 0):
        raise ValueError("candidate indices must be unique non-negative original indices")
    if not (0 < selected_count <= candidates.size) or step <= 0:
        raise ValueError("selected_count and RFE step must be positive and in range")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("outer-train labels must contain both binary classes")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    estimator = LogisticRegression(
        solver="liblinear",
        C=1.0,
        class_weight="balanced",
        random_state=SEED,
        max_iter=2000,
        tol=1e-4,
    )
    selector = RFE(estimator, n_features_to_select=selected_count, step=step)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        selector.fit(scaled, labels)
    support_positions = np.flatnonzero(selector.support_)
    selected = candidates[support_positions]
    order = np.argsort(selected, kind="stable")
    selected = selected[order]
    selected_candidate_positions = support_positions[order]
    metadata = {
        "candidate_indices_ranked": candidates.tolist(),
        "selected_indices": selected.tolist(),
        "selected_candidate_positions": selected_candidate_positions.tolist(),
        "correlation": correlations.tolist(),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
    }
    return selected, scaler.mean_, scaler.scale_, metadata


def standardize_selected_matrix(
    raw_selected_features: np.ndarray,
    selected_candidate_positions: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    raw = np.asarray(raw_selected_features, dtype=np.float64)
    positions = np.asarray(selected_candidate_positions, dtype=np.int64).reshape(-1)
    mean = np.asarray(candidate_mean, dtype=np.float64).reshape(-1)
    scale = np.asarray(candidate_scale, dtype=np.float64).reshape(-1)
    if raw.ndim != 2 or raw.shape[1] != positions.size:
        raise ValueError("selected raw matrix and candidate positions are not aligned")
    if np.any(positions < 0) or np.any(positions >= mean.size) or scale.shape != mean.shape:
        raise ValueError("candidate scaler state and selected positions are not aligned")
    selected_mean = mean[positions]
    selected_scale = scale[positions]
    if not np.isfinite(raw).all() or not np.isfinite(selected_mean).all() or not np.isfinite(selected_scale).all():
        raise ValueError("selected matrix or scaler state contains non-finite values")
    if np.any(selected_scale <= 0.0):
        raise ValueError("selected scaler values must be positive")
    standardized = raw.copy()
    standardized -= selected_mean
    standardized /= selected_scale
    standardized = standardized.astype(np.float32)
    return standardized, selected_mean, selected_scale
