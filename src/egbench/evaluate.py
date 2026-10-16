"""Scores prediction tables against the test labels of a fold: AUPRC and AUROC for binary tasks, Spearman for GeneEffect."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from egbench import __version__
from egbench.contracts import (
    PREDICTION_FILES,
    RUN_IDENTITY_KEYS,
    ContractError,
    TargetKind,
    check_prediction_identity,
    read_fold_products,
    validate_evaluation_manifest,
)
from egbench.data import (
    TEST_ROLE,
    cohort_target,
    model_directory,
    prediction_cohort,
    primary_scope,
    read_evaluation_truth,
    read_split_assignments,
    scope_gene_keys,
)
from egbench.metrics import discrimination, regression
from egbench.stats import Interval, confidence_intervals

FOLD_COUNT = 5
GENE_KEY = "gene_key"
SCORE = "y_score"

METRICS: dict[str, Callable[..., float]] = {
    "auprc": discrimination.auprc,
    "auroc": discrimination.auroc,
}

REGRESSION_METRICS: dict[str, Callable[..., float]] = {
    "spearman": regression.spearman,
}

PRIMARY_METRICS: dict[TargetKind, str] = {
    TargetKind.BINARY: "auprc",
    TargetKind.CONTINUOUS: "spearman",
}


def metric_table(cohort_id: str) -> dict[str, Callable[..., float]]:
    if cohort_target(cohort_id).kind is TargetKind.CONTINUOUS:
        return REGRESSION_METRICS
    return METRICS


class Scored(NamedTuple):
    method_id: str
    cohort_id: str
    split_strategy: str
    scope: str
    values: dict[str, float]
    intervals: dict[str, Interval]


def _excluded_genes(
    cohort_id: str, split_strategy: str, root: str | Path | None
) -> set[str]:
    source = prediction_cohort(cohort_id)
    if source == cohort_id:
        return set()
    held = set(
        read_split_assignments(source, split_strategy=split_strategy, root=root)[GENE_KEY]
    )
    kept = set(
        read_split_assignments(cohort_id, split_strategy=split_strategy, root=root)[GENE_KEY]
    )
    return held - kept


def join_targets(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    cohort_id: str,
    split_strategy: str,
    source: str,
    root: str | Path | None = None,
) -> pd.DataFrame:
    merged = predictions.merge(
        targets, on=GENE_KEY, how="inner", validate="one_to_one"
    )
    scored = set(predictions[GENE_KEY])
    labeled = set(targets[GENE_KEY])
    unscored = labeled - scored
    unlabeled = scored - labeled
    stray = unlabeled - _excluded_genes(cohort_id, split_strategy, root)
    if unscored or stray:
        raise ValueError(
            f"{source}: {len(unscored)} genes of {cohort_id} carry no score and "
            f"{len(stray)} scored genes carry no target without being genes "
            f"{cohort_id} excludes; the prediction table and the fold table "
            f"disagree on which genes are in this role of this fold"
        )
    return merged


def _fold_frame(
    method_root: Path, cohort_id: str, split_strategy: str, outer_fold: int
) -> pd.DataFrame:
    target = cohort_target(cohort_id)
    directory = model_directory(
        method_root,
        prediction_cohort(cohort_id),
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )
    predictions = pd.read_parquet(directory / PREDICTION_FILES["blind"])
    truth = read_evaluation_truth(
        cohort_id, outer_fold, split_strategy=split_strategy
    )
    return join_targets(
        predictions,
        truth[[GENE_KEY, target.column]],
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        source=str(directory / PREDICTION_FILES["blind"]),
    )


def _mean_of_defined(values: list[float]) -> float:
    defined = [value for value in values if not np.isnan(value)]
    if not defined:
        return float("nan")
    return float(np.mean(defined))


def scope_frame(
    frame: pd.DataFrame, *, cohort_id: str, scope: str, root: str | Path | None = None
) -> pd.DataFrame:
    kept = scope_gene_keys(cohort_id, scope, root=root)
    if kept is None:
        return frame
    target = cohort_target(cohort_id).column
    selected = frame.loc[frame[GENE_KEY].isin(kept)]
    dropped = frame.loc[~frame[GENE_KEY].isin(kept), target]
    if (dropped.astype("float64") != 1.0).any():
        raise ValueError(
            f"scope {scope!r} of {cohort_id} dropped a gene that is not a "
            f"positive; the frame and the scope belong to different cohorts"
        )
    if selected.empty:
        raise ValueError(f"scope {scope!r} of {cohort_id} keeps no gene of this fold")
    return selected.reset_index(drop=True)


def score_frames(
    folds: Sequence[pd.DataFrame],
    *,
    method_id: str,
    cohort_id: str,
    split_strategy: str,
    scope: str | None = None,
) -> Scored:
    if not folds:
        raise ValueError("score_frames: no fold to score")
    resolved_scope = primary_scope(cohort_id) if scope is None else scope
    target = cohort_target(cohort_id)
    table = metric_table(cohort_id)
    frames = [
        scope_frame(frame, cohort_id=cohort_id, scope=resolved_scope)
        for frame in folds
    ]
    pooled = pd.concat(
        [frame.assign(fold=index) for index, frame in enumerate(frames)],
        ignore_index=True,
    )
    targets, scores = pooled[target.column].to_numpy(), pooled[SCORE].to_numpy()
    fold_column = pooled["fold"].to_numpy()
    values = {
        name: _mean_of_defined(
            [function(frame[target.column].to_numpy(), frame[SCORE].to_numpy()) for frame in frames]
        )
        for name, function in table.items()
    }
    intervals = confidence_intervals(
        targets, scores, dict(table), fold=fold_column, target_kind=target.kind
    )
    return Scored(
        method_id=method_id,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        scope=resolved_scope,
        values=values,
        intervals=intervals,
    )


def score_folds(
    method_root: str | Path,
    cohort_id: str,
    *,
    split_strategy: str,
    scope: str | None = None,
) -> Scored:
    root = Path(method_root)
    folds = [
        _fold_frame(root, cohort_id, split_strategy, outer_fold)
        for outer_fold in range(FOLD_COUNT)
    ]
    return score_frames(
        folds,
        method_id=root.name,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        scope=scope,
    )


def _json_value(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def evaluate_fold(directory: str | Path) -> dict[str, object]:
    products = read_fold_products(directory)
    manifest = products.manifest
    identity = {key: manifest[key] for key in RUN_IDENTITY_KEYS}
    cohort_id = str(identity["cohort_id"])
    split_strategy = str(identity["split_strategy"])
    outer_fold = int(identity["outer_fold"])

    assignments = read_split_assignments(cohort_id, split_strategy=split_strategy)
    check_prediction_identity(
        products.blind,
        assignments,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        role=TEST_ROLE,
    )
    truth = read_evaluation_truth(cohort_id, outer_fold, split_strategy=split_strategy)
    truth_hash = str(truth["split_build_hash"].iat[0])
    if truth_hash != identity["split_build_hash"]:
        raise ContractError(
            f"{products.directory}: the run manifest declares split_build_hash "
            f"{identity['split_build_hash']!r} and the fold table of {cohort_id} "
            f"{split_strategy} carries {truth_hash!r}; these predictions were "
            f"made for another build of the split and are not scored against "
            f"this one"
        )
    target = cohort_target(cohort_id)
    merged = products.blind.merge(
        truth[[GENE_KEY, target.column]],
        on=GENE_KEY,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(products.blind):
        raise ContractError(
            f"{products.directory}: {len(products.blind) - len(merged)} scored "
            f"genes carry no test label in evaluation_truth.parquet"
        )
    intervals = confidence_intervals(
        merged[target.column].to_numpy(),
        merged[SCORE].to_numpy(),
        dict(metric_table(cohort_id)),
        target_kind=target.kind,
    )
    record: dict[str, object] = {
        **identity,
        "evaluator_version": __version__,
        "decision_threshold": float(products.model_card["decision_threshold"]),
        "metrics": {
            name: {
                "point": _json_value(interval.point),
                "half_width": _json_value(interval.half_width),
                "undefined_fraction": _json_value(interval.undefined_fraction),
            }
            for name, interval in intervals.items()
        },
    }
    validate_evaluation_manifest(record, source=f"{products.directory}: evaluation")
    return record
