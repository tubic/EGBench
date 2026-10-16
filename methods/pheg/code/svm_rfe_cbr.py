"""Linear SVM-RFE with correlation bias reduction, as used by Pheg.
Imported by run.py; not run directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.svm import SVC


@dataclass(frozen=True)
class SVMRFECBRConfig:

    c: float = 1.0
    n_stop_chunk: int = 60
    remove_ratio: float = 0.5
    use_cbr: bool = True
    n_out_corr_max: int = 1
    correlation_threshold: float = 0.9
    constant_std_threshold: float = 1e-5
    svm_tolerance: float = 1e-3
    svm_cache_mb: float = 4096.0

    def validate(self) -> None:
        if self.c <= 0:
            raise ValueError("c must be positive")
        if self.n_stop_chunk < 1:
            raise ValueError("n_stop_chunk must be at least one")
        if not 0 < self.remove_ratio < 1:
            raise ValueError("remove_ratio must be in (0, 1)")
        if self.n_out_corr_max < 0:
            raise ValueError("n_out_corr_max must be non-negative")
        if not 0 <= self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be in [0, 1]")


@dataclass(frozen=True)
class RankingResult:

    ranking: NDArray[np.int64]
    removal_batches: tuple[tuple[int, ...], ...]
    constant_features: tuple[int, ...]


def _as_valid_arrays(
    features: ArrayLike, labels: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels).reshape(-1)
    if x.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("labels must contain one value for each sample")
    if x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("at least two samples and one feature are required")
    if not np.isfinite(x).all():
        raise ValueError("features contain NaN or infinite values")
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("the published linear implementation requires two classes")
    return np.ascontiguousarray(x), y.astype(np.int64, copy=False)


def _linear_svm_weights_squared(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    config: SVMRFECBRConfig,
) -> NDArray[np.float64]:
    model = SVC(
        C=config.c,
        kernel="linear",
        shrinking=True,
        tol=config.svm_tolerance,
        cache_size=config.svm_cache_mb,
    )
    model.fit(x, y)
    weights = np.asarray(model.coef_, dtype=np.float64)
    if weights.shape != (1, x.shape[1]):
        raise RuntimeError(f"unexpected binary LIBSVM coefficient shape: {weights.shape}")
    return np.square(weights[0])


def rank_features(
    features: ArrayLike,
    labels: ArrayLike,
    config: SVMRFECBRConfig | None = None,
) -> RankingResult:

    config = config or SVMRFECBRConfig()
    config.validate()
    x, y = _as_valid_arrays(features, labels)
    n_features = x.shape[1]

    std = np.std(x, axis=0, ddof=1)
    constant = np.flatnonzero(std <= config.constant_std_threshold).astype(np.int64)
    active = np.setdiff1d(
        np.arange(n_features, dtype=np.int64), constant, assume_unique=True
    )
    removed: list[int] = constant.tolist()
    batches: list[tuple[int, ...]] = []

    correlations: NDArray[np.float64] | None = None
    if config.use_cbr:
        with np.errstate(divide="ignore", invalid="ignore"):
            correlations = np.abs(np.corrcoef(x, rowvar=False))

    while active.size:
        n_active = int(active.size)
        criteria = _linear_svm_weights_squared(x[:, active], y, config)
        order = np.argsort(criteria, kind="stable")

        if n_active > config.n_stop_chunk:
            n_remove = int(np.floor(n_active * config.remove_ratio))
            if n_active - n_remove < config.n_stop_chunk:
                n_remove = n_active - config.n_stop_chunk
        else:
            n_remove = 1

        candidate_positions = order[:n_remove]
        candidate_features = active[candidate_positions]

        if (
            config.use_cbr
            and config.correlation_threshold < 1
            and n_remove > 1
        ):
            assert correlations is not None
            keep_mask = np.ones(candidate_positions.size, dtype=bool)
            survivor_positions = np.ones(active.size, dtype=bool)
            survivor_positions[candidate_positions] = False
            temporary_active = active[survivor_positions].tolist()

            for candidate_offset in range(candidate_features.size - 1, -1, -1):
                feature_index = int(candidate_features[candidate_offset])
                in_similar = int(
                    np.count_nonzero(
                        correlations[feature_index, temporary_active]
                        > config.correlation_threshold
                    )
                )
                out_similar = int(
                    np.count_nonzero(
                        correlations[feature_index, candidate_features]
                        > config.correlation_threshold
                    )
                )
                if in_similar < 1 and out_similar > config.n_out_corr_max:
                    keep_mask[candidate_offset] = False
                    temporary_active.append(feature_index)

            candidate_positions = candidate_positions[keep_mask]
            candidate_features = active[candidate_positions]

        if candidate_positions.size == 0:
            raise RuntimeError("CBR rescued every candidate; ranking cannot progress")

        batch = tuple(int(index) for index in candidate_features)
        batches.append(batch)
        removed.extend(batch)
        active = np.delete(active, candidate_positions)

    ranking = np.asarray(removed[::-1], dtype=np.int64)
    if ranking.size != n_features or np.unique(ranking).size != n_features:
        raise RuntimeError("feature ranking is not a permutation of all input columns")
    return RankingResult(
        ranking=ranking,
        removal_batches=tuple(batches),
        constant_features=tuple(int(index) for index in constant),
    )
