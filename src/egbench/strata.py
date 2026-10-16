"""Per-gene strata table, quartile bands and the subset matched on literature count and PPI degree."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from egbench.baselines import (
    COMPLEX_INDICATOR_COLUMN,
    LITERATURE_COLUMN,
    PROCESS_INDICATOR_COLUMN,
    literature_counts,
)
from egbench.contracts import SEED, TargetKind
from egbench.data import (
    EXPRESSION_COLUMN,
    FAMILY_SIZE_COLUMN,
    LABEL_CONFIDENCE_COLUMNS,
    PPI_DEGREE_COLUMN,
    cohort_tables_directory,
    cohort_target,
    read_cohort_labels,
    read_strata_inputs,
)
from egbench.metrics import discrimination
from egbench.modules import COMPLEX, PROCESS, annotation_indicator, cohort_modules
from egbench.stats import Interval, confidence_intervals

GENE_KEY = "gene_key"
LABEL = "label"
CLUSTER = "cluster_id"

QUARTILE_BANDS: tuple[str, ...] = ("q1", "q2", "q3", "q4")
MATCHED = "matched"

MATCH_COVARIATES: tuple[str, ...] = (LITERATURE_COLUMN, PPI_DEGREE_COLUMN)
MATCH_CALIPER = 0.25
MATCH_RATIO = 2

SLICE_METRICS: dict[str, Callable[..., float]] = {
    "auprc": discrimination.auprc,
    "auroc": discrimination.auroc,
}


class SliceScore(NamedTuple):
    gene_count: int
    positive_count: int
    intervals: dict[str, Interval]


def strata_table(cohort_id: str, *, root: str | Path | None = None) -> pd.DataFrame:
    frame = read_strata_inputs(cohort_id, root=root)
    target = cohort_target(cohort_id)
    labels = read_cohort_labels(cohort_id, root=root)
    projection = pd.read_parquet(
        cohort_tables_directory(cohort_id, root=root) / "cluster_projection.parquet",
        columns=[GENE_KEY, CLUSTER],
    )
    frame = frame.merge(labels, on=GENE_KEY, how="inner", validate="one_to_one")
    frame = frame.merge(projection, on=GENE_KEY, how="left", validate="one_to_one")
    if len(frame) != len(labels) or frame[CLUSTER].isna().any():
        raise ValueError(
            f"strata_table: the strata inputs, the labels and the cluster projection "
            f"of {cohort_id} do not cover the same genes"
        )
    keys = pd.Index(frame[GENE_KEY].astype("object"))
    frame[LITERATURE_COLUMN] = (
        literature_counts(root).reindex(keys).fillna(0).astype("float64").to_numpy()
    )
    frame[COMPLEX_INDICATOR_COLUMN] = annotation_indicator(
        cohort_modules(cohort_id, COMPLEX, root=root), keys
    )
    frame[PROCESS_INDICATOR_COLUMN] = annotation_indicator(
        cohort_modules(cohort_id, PROCESS, root=root), keys
    )
    columns = [
        GENE_KEY,
        target.column,
        CLUSTER,
        FAMILY_SIZE_COLUMN,
        PPI_DEGREE_COLUMN,
        EXPRESSION_COLUMN,
        *LABEL_CONFIDENCE_COLUMNS[cohort_id],
        LITERATURE_COLUMN,
        COMPLEX_INDICATOR_COLUMN,
        PROCESS_INDICATOR_COLUMN,
    ]
    return frame[columns].reset_index(drop=True)


def _series(values: object, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        return values
    return pd.Series(np.asarray(values), name=name)


def quartile_edges(values: object) -> np.ndarray:
    present = _series(values, "values").astype("float64").dropna()
    if present.empty:
        raise ValueError("quartile_edges: no value to take quartiles of")
    return np.quantile(present.to_numpy(), [0.25, 0.5, 0.75])


def quartile_bands(values: object) -> pd.Series:
    series = _series(values, "values").astype("float64")
    edges = quartile_edges(series)
    bands = pd.Series(np.full(len(series), None, dtype="object"), index=series.index)
    present = series.notna().to_numpy()
    position = np.searchsorted(edges, series.to_numpy()[present], side="left")
    bands[present] = np.asarray(QUARTILE_BANDS, dtype="object")[position]
    return bands


def matched_subset(
    table: pd.DataFrame,
    *,
    caliper: float = MATCH_CALIPER,
    ratio: int = MATCH_RATIO,
    covariates: tuple[str, ...] = MATCH_COVARIATES,
) -> pd.Series:
    labels = table[LABEL].astype("float64")
    if not labels.isin([0.0, 1.0]).all():
        raise ValueError("matched_subset: labels must be 0 or 1")
    if ratio < 1 or caliper <= 0.0 or not covariates:
        raise ValueError("matched_subset: the ratio is at least 1, the caliper positive")
    values = table[list(covariates)].astype("float64")
    if (values < 0).any().any():
        raise ValueError("matched_subset: the matching covariates are non-negative")
    present = values.notna().all(axis=1).to_numpy()
    logged = np.log1p(values.to_numpy()[present])
    standardised = (logged - logged.mean(axis=0)) / logged.std(axis=0, ddof=1)
    keys = table[GENE_KEY].to_numpy("object")[present]
    kept_labels = labels.to_numpy()[present]
    positive_rows = np.flatnonzero(kept_labels == 1.0)
    negative_rows = np.flatnonzero(kept_labels == 0.0)
    positive_rows = positive_rows[np.argsort(keys[positive_rows], kind="stable")]
    negative_rows = negative_rows[np.argsort(keys[negative_rows], kind="stable")]
    order = np.random.default_rng(SEED).permutation(positive_rows.size)
    negatives = standardised[negative_rows]
    used = np.zeros(negative_rows.size, dtype=bool)
    matched_positive = np.zeros(positive_rows.size, dtype=bool)
    for round_index in range(ratio):
        for index in order:
            distance = np.linalg.norm(negatives - standardised[positive_rows[index]], axis=1)
            distance[used] = np.inf
            nearest = int(np.argmin(distance))
            if distance[nearest] <= caliper:
                used[nearest] = True
                if round_index == 0:
                    matched_positive[index] = True
    mask = np.zeros(len(table), dtype=bool)
    rows = np.flatnonzero(present)
    mask[rows[positive_rows[matched_positive]]] = True
    mask[rows[negative_rows[used]]] = True
    return pd.Series(mask, index=table.index)


def matched_mask(table: pd.DataFrame) -> pd.Series:
    keys = pd.Index(table[GENE_KEY].astype("object"), name=GENE_KEY)
    return matched_subset(table.set_index(keys)).set_axis(keys).astype(bool)


def score_slice(
    y_true: object,
    y_score: object,
    mask: object,
    *,
    fold: object = None,
    target_kind: TargetKind = TargetKind.BINARY,
) -> SliceScore:
    keep = np.asarray(mask, dtype=bool)
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype="float64")
    if keep.shape != labels.shape or scores.shape != labels.shape:
        raise ValueError("score_slice: y_true, y_score and mask must be one value per gene")
    folds = None if fold is None else np.asarray(fold)[keep]
    kept_labels = labels[keep].astype("float64")
    count = int(keep.sum())
    positives = int(kept_labels.sum()) if count and target_kind is TargetKind.BINARY else 0
    if count == 0:
        undefined = Interval(float("nan"), float("nan"), float("nan"))
        return SliceScore(0, 0, {name: undefined for name in SLICE_METRICS})
    intervals = confidence_intervals(
        kept_labels,
        scores[keep],
        dict(SLICE_METRICS),
        fold=folds,
        target_kind=target_kind,
    )
    return SliceScore(count, positives, intervals)
