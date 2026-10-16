"""Top-5 fusion of the methods' predictions and pooled out-of-fold predictions of any predictor."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from egbench.baselines import (
    BASELINE_IDS,
    baseline_predictions,
    logistic_fit,
    logistic_probability,
    prediction_table,
    rank_normal_scores,
    score_predictions,
)
from egbench.contracts import METHOD_IDS, PREDICTION_FILES, TargetKind
from egbench.data import (
    PROJECT_ROOT,
    TEST_ROLE,
    VALIDATION_ROLE,
    cohort_target,
    model_directory,
    prediction_cohort,
    primary_scope,
    read_evaluation_truth,
    read_predictions,
    read_training_labels,
)
from egbench.evaluate import FOLD_COUNT, Scored, join_targets, scope_frame
from egbench.metrics import discrimination, regression
from egbench.stats import Comparison, Interval, confidence_intervals, paired_difference

GENE_KEY = "gene_key"
SCORE = "y_score"
FOLD = "fold"

FUSION_TOP = "fusion_top5"
FUSION_IDS: tuple[str, ...] = (FUSION_TOP,)
TOP_COUNT = 5

SELECTION_METRICS: dict[TargetKind, Callable[..., float]] = {
    TargetKind.BINARY: discrimination.auprc,
    TargetKind.CONTINUOUS: regression.spearman,
}
COMPARISON_METRICS: dict[TargetKind, dict[str, Callable[..., float]]] = {
    TargetKind.BINARY: {"auprc": discrimination.auprc, "auroc": discrimination.auroc},
    TargetKind.CONTINUOUS: {"spearman": regression.spearman},
}


def fusion_size(fusion_id: str, member_count: int) -> int:
    if fusion_id == FUSION_TOP:
        return min(TOP_COUNT, member_count)
    raise ValueError(f"unknown fusion {fusion_id!r}, the fusions are {list(FUSION_IDS)}")


class MemberScores(NamedTuple):

    member_ids: tuple[str, ...]
    fit_keys: np.ndarray
    fit_targets: np.ndarray
    fit_scores: np.ndarray
    test_keys: np.ndarray
    test_targets: np.ndarray
    test_scores: np.ndarray


def _aligned(
    predictions: pd.DataFrame, labels: pd.DataFrame, *, cohort_id: str, split_strategy: str, source: str, root: str | Path | None
) -> pd.Series:
    target = cohort_target(cohort_id).column
    merged = join_targets(
        predictions[[GENE_KEY, SCORE]],
        labels[[GENE_KEY, target]],
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        source=source,
        root=root,
    )
    return merged.set_index(GENE_KEY)[SCORE].reindex(labels[GENE_KEY].to_numpy("object"))


def member_scores(
    cohort_id: str,
    *,
    split_strategy: str,
    outer_fold: int,
    member_ids: Sequence[str],
    methods_directory: str | Path | None = None,
    root: str | Path | None = None,
) -> MemberScores:
    members = tuple(member_ids)
    if not members:
        raise ValueError("member_scores: no member to fuse")
    if len(set(members)) != len(members):
        raise ValueError(f"member_scores: a member is listed twice in {list(members)}")
    source = prediction_cohort(cohort_id)
    target = cohort_target(cohort_id).column
    directory = PROJECT_ROOT / "methods" if methods_directory is None else Path(methods_directory)
    validation = read_training_labels(
        cohort_id, outer_fold, VALIDATION_ROLE, split_strategy=split_strategy, root=root
    )
    truth = read_evaluation_truth(cohort_id, outer_fold, split_strategy=split_strategy, root=root)
    fit_columns, test_columns = [], []
    for member in members:
        fold_directory = model_directory(
            directory / member, source, split_strategy=split_strategy, outer_fold=outer_fold
        )
        for name, role, labels, columns in (
            ("validation", VALIDATION_ROLE, validation, fit_columns),
            ("blind", TEST_ROLE, truth, test_columns),
        ):
            path = fold_directory / PREDICTION_FILES[name]
            predictions = read_predictions(
                path,
                cohort_id=source,
                split_strategy=split_strategy,
                outer_fold=outer_fold,
                role=role,
                root=root,
            )
            columns.append(
                _aligned(
                    predictions,
                    labels,
                    cohort_id=cohort_id,
                    split_strategy=split_strategy,
                    source=str(path),
                    root=root,
                ).to_numpy("float64")
            )
    return MemberScores(
        members,
        validation[GENE_KEY].to_numpy("object"),
        validation[target].to_numpy("float64"),
        np.column_stack(fit_columns),
        truth[GENE_KEY].to_numpy("object"),
        truth[target].to_numpy("float64"),
        np.column_stack(test_columns),
    )


def validation_order(fit_scores: object, fit_targets: object, target_kind: TargetKind) -> np.ndarray:
    scores = np.asarray(fit_scores, dtype="float64")
    targets = np.asarray(fit_targets, dtype="float64")
    if scores.ndim != 2 or scores.shape[0] != targets.size:
        raise ValueError(
            f"validation_order: {scores.shape} scores for {targets.size} validation genes"
        )
    metric = SELECTION_METRICS[target_kind]
    values = np.array(
        [metric(targets, scores[:, column]) for column in range(scores.shape[1])], dtype="float64"
    )
    defined = ~np.isnan(values)
    keyed = np.where(defined, -values, np.inf)
    return np.lexsort((np.arange(values.size), keyed))


class Fit(NamedTuple):

    y_score: np.ndarray
    coefficients: np.ndarray


def fuse(
    fit_scores: object, fit_targets: object, test_scores: object, target_kind: TargetKind
) -> Fit:
    fitted = np.asarray(fit_scores, dtype="float64")
    targets = np.asarray(fit_targets, dtype="float64")
    scored = np.asarray(test_scores, dtype="float64")
    if fitted.ndim != 2 or scored.ndim != 2 or fitted.shape[1] != scored.shape[1]:
        raise ValueError(
            f"fuse: the validation scores are {fitted.shape} and the test scores {scored.shape}; "
            f"both are one row per gene and one column per member"
        )
    if fitted.shape[0] != targets.size:
        raise ValueError(f"fuse: {fitted.shape[0]} validation genes and {targets.size} targets")
    if fitted.shape[0] == 0:
        raise ValueError("fuse: no validation gene to fit on")
    placed_fit = np.empty_like(fitted)
    placed_test = np.empty_like(scored)
    for column in range(fitted.shape[1]):
        placed_fit[:, column], placed_test[:, column] = rank_normal_scores(
            fitted[:, column], scored[:, column]
        )
    varying = placed_fit.max(axis=0) != placed_fit.min(axis=0)
    coefficients = np.zeros(fitted.shape[1] + 1, dtype="float64")
    if not varying.any():
        if target_kind is TargetKind.BINARY:
            coefficients[0] = float(targets.mean())
            return Fit(np.full(scored.shape[0], float(targets.mean())), coefficients)
        coefficients[0] = float(targets.mean())
        return Fit(np.full(scored.shape[0], -float(targets.mean())), coefficients)
    kept_fit = placed_fit[:, varying]
    center = kept_fit.mean(axis=0)
    spread = kept_fit.std(axis=0)
    design_fit = (kept_fit - center) / spread
    design_test = (placed_test[:, varying] - center) / spread
    if target_kind is TargetKind.BINARY:
        beta = logistic_fit(design_fit, targets)
        y_score = logistic_probability(design_test, beta)
    else:
        design = np.column_stack([np.ones(design_fit.shape[0]), design_fit])
        beta, *_ = np.linalg.lstsq(design, targets, rcond=None)
        y_score = -(beta[0] + design_test @ beta[1:])
    coefficients[0] = beta[0]
    coefficients[1:][varying] = beta[1:]
    return Fit(y_score, coefficients)


class FusedFold(NamedTuple):

    outer_fold: int
    members: tuple[str, ...]
    best_member: str
    test_keys: np.ndarray
    test_targets: np.ndarray
    y_score: np.ndarray
    best_scores: np.ndarray
    coefficients: np.ndarray


def fusion_fold(
    fusion_id: str,
    cohort_id: str,
    *,
    split_strategy: str,
    outer_fold: int,
    member_ids: Sequence[str],
    methods_directory: str | Path | None = None,
    root: str | Path | None = None,
) -> FusedFold:
    source = prediction_cohort(cohort_id)
    scores = member_scores(
        source,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        member_ids=member_ids,
        methods_directory=methods_directory,
        root=root,
    )
    kind = cohort_target(source).kind
    order = validation_order(scores.fit_scores, scores.fit_targets, kind)
    chosen = order[: fusion_size(fusion_id, len(scores.member_ids))]
    fit = fuse(
        scores.fit_scores[:, chosen], scores.fit_targets, scores.test_scores[:, chosen], kind
    )
    test_keys, test_targets = scores.test_keys, scores.test_targets
    rows = np.arange(test_keys.size)
    if source != cohort_id:
        truth = read_evaluation_truth(cohort_id, outer_fold, split_strategy=split_strategy, root=root)
        rows = pd.Index(test_keys).get_indexer(truth[GENE_KEY])
        if (rows < 0).any():
            raise ValueError(
                f"fusion_fold: {int((rows < 0).sum())} test genes of {cohort_id} "
                f"{split_strategy} fold {outer_fold} are not test genes of {source}"
            )
        test_keys = truth[GENE_KEY].to_numpy("object")
        test_targets = truth[cohort_target(cohort_id).column].to_numpy("float64")
    return FusedFold(
        outer_fold,
        tuple(scores.member_ids[index] for index in chosen),
        scores.member_ids[order[0]],
        test_keys,
        test_targets,
        fit.y_score[rows],
        scores.test_scores[rows, order[0]],
        fit.coefficients,
    )


def fusion_predictions(
    fusion_id: str,
    cohort_id: str,
    *,
    split_strategy: str,
    outer_fold: int,
    member_ids: Sequence[str],
    role: str = TEST_ROLE,
    methods_directory: str | Path | None = None,
    root: str | Path | None = None,
) -> pd.DataFrame:
    if role != TEST_ROLE:
        raise ValueError(
            f"fusion_predictions: the fusion scores the {TEST_ROLE!r} role only, its "
            f"{role!r} scores would be in sample"
        )
    fused = fusion_fold(
        fusion_id,
        cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        member_ids=member_ids,
        methods_directory=methods_directory,
        root=root,
    )
    return prediction_table(
        pd.Index(fused.test_keys, name=GENE_KEY),
        fused.y_score,
        source=f"fusion {fusion_id!r} on {cohort_id} {split_strategy} fold {outer_fold}",
    )


def score_fusion(
    fusion_id: str,
    cohort_id: str,
    *,
    split_strategy: str,
    member_ids: Sequence[str],
    scope: str | None = None,
    methods_directory: str | Path | None = None,
    root: str | Path | None = None,
) -> Scored:
    fusion_size(fusion_id, len(member_ids))

    def predict(source: str, outer_fold: int, role: str) -> pd.DataFrame | None:
        if role == VALIDATION_ROLE:
            return None
        return fusion_predictions(
            fusion_id,
            cohort_id,
            split_strategy=split_strategy,
            outer_fold=outer_fold,
            member_ids=member_ids,
            role=role,
            methods_directory=methods_directory,
            root=root,
        )

    return score_predictions(
        predict,
        method_id=fusion_id,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        scope=scope,
        root=root,
    )


def pooled_predictions(
    fusion_id: str,
    cohort_id: str,
    *,
    split_strategy: str,
    member_ids: Sequence[str],
    methods_directory: str | Path | None = None,
    root: str | Path | None = None,
    scope: str | None = None,
) -> pd.DataFrame:
    target = cohort_target(cohort_id).column
    cut = primary_scope(cohort_id) if scope is None else scope
    frames = []
    for outer_fold in range(FOLD_COUNT):
        fused = fusion_fold(
            fusion_id,
            cohort_id,
            split_strategy=split_strategy,
            outer_fold=outer_fold,
            member_ids=member_ids,
            methods_directory=methods_directory,
            root=root,
        )
        frame = pd.DataFrame(
            {
                GENE_KEY: pd.Series(list(fused.test_keys), dtype="object"),
                target: fused.test_targets,
                SCORE: fused.y_score,
            }
        )
        frame = scope_frame(frame, cohort_id=cohort_id, scope=cut, root=root)
        frames.append(frame.assign(**{FOLD: outer_fold}))
    return pd.concat(frames, ignore_index=True)[[GENE_KEY, target, SCORE, FOLD]]


class FusionComparison(NamedTuple):

    fusion_id: str
    cohort_id: str
    split_strategy: str
    members: tuple[tuple[str, ...], ...]
    best_members: tuple[str, ...]
    gene_count: int
    intervals: dict[str, Interval]
    best_intervals: dict[str, Interval]
    comparisons: dict[str, Comparison]


def compare_fusion(
    fusion_id: str,
    cohort_id: str,
    *,
    split_strategy: str,
    member_ids: Sequence[str],
    methods_directory: str | Path | None = None,
    root: str | Path | None = None,
) -> FusionComparison:
    target = cohort_target(cohort_id)
    metrics = COMPARISON_METRICS[target.kind]
    folds = [
        fusion_fold(
            fusion_id,
            cohort_id,
            split_strategy=split_strategy,
            outer_fold=outer_fold,
            member_ids=member_ids,
            methods_directory=methods_directory,
            root=root,
        )
        for outer_fold in range(FOLD_COUNT)
    ]
    frames = []
    for fused in folds:
        frame = pd.DataFrame(
            {
                GENE_KEY: pd.Series(list(fused.test_keys), dtype="object"),
                target.column: fused.test_targets,
                SCORE: fused.y_score,
                "best": fused.best_scores,
            }
        )
        frame = scope_frame(frame, cohort_id=cohort_id, scope=primary_scope(cohort_id), root=root)
        frames.append(frame.assign(**{FOLD: fused.outer_fold}))
    pooled = pd.concat(frames, ignore_index=True)
    y_true = pooled[target.column].to_numpy("float64")
    fold_of = pooled[FOLD].to_numpy("int64")
    fused_scores = pooled[SCORE].to_numpy("float64")
    best_scores = pooled["best"].to_numpy("float64")
    return FusionComparison(
        fusion_id,
        cohort_id,
        split_strategy,
        tuple(fused.members for fused in folds),
        tuple(fused.best_member for fused in folds),
        int(y_true.size),
        confidence_intervals(y_true, fused_scores, dict(metrics), fold=fold_of, target_kind=target.kind),
        confidence_intervals(y_true, best_scores, dict(metrics), fold=fold_of, target_kind=target.kind),
        {
            name: paired_difference(
                y_true, fused_scores, best_scores, function, fold=fold_of, target_kind=target.kind
            )
            for name, function in metrics.items()
        },
    )


FUSION_TABLE_COLUMNS: tuple[str, ...] = (
    "predictor",
    "cohort_id",
    "split_strategy",
    "metric",
    "gene_count",
    "point",
    "half_width",
    "best_members",
    "best_point",
    "best_half_width",
    "difference",
    "difference_half_width",
    "p_value",
    "members",
)


def fusion_table(comparisons: Sequence[FusionComparison]) -> pd.DataFrame:
    rows = []
    for item in comparisons:
        for metric, interval in item.intervals.items():
            comparison = item.comparisons[metric]
            best = item.best_intervals[metric]
            rows.append(
                {
                    "predictor": item.fusion_id,
                    "cohort_id": item.cohort_id,
                    "split_strategy": item.split_strategy,
                    "metric": metric,
                    "gene_count": item.gene_count,
                    "point": interval.point,
                    "half_width": interval.half_width,
                    "best_members": ";".join(item.best_members),
                    "best_point": best.point,
                    "best_half_width": best.half_width,
                    "difference": comparison.difference,
                    "difference_half_width": comparison.half_width,
                    "p_value": comparison.p_value,
                    "members": "|".join(";".join(members) for members in item.members),
                }
            )
    if not rows:
        raise ValueError("fusion_table: no comparison to tabulate")
    table = pd.DataFrame(rows, columns=list(FUSION_TABLE_COLUMNS))
    duplicated = table.duplicated(["predictor", "cohort_id", "split_strategy", "metric"])
    if duplicated.any():
        first = table.loc[duplicated.idxmax()]
        raise ValueError(
            f"fusion_table: {first['predictor']!r} on {first['cohort_id']} "
            f"{first['split_strategy']} was compared more than once"
        )
    return table


def pooled(
    predictor_id: str,
    cohort_id: str,
    *,
    split_strategy: str = "family_5fold",
    methods_directory: str | Path | None = None,
    member_ids: Sequence[str] = METHOD_IDS,
    root: str | Path | None = None,
    scope: str | None = None,
) -> pd.DataFrame:
    cut = primary_scope(cohort_id) if scope is None else scope
    if predictor_id in FUSION_IDS:
        return pooled_predictions(
            predictor_id,
            cohort_id,
            split_strategy=split_strategy,
            member_ids=member_ids,
            methods_directory=methods_directory,
            root=root,
            scope=cut,
        )
    source = prediction_cohort(cohort_id)
    target = cohort_target(cohort_id)
    directory = (
        PROJECT_ROOT / "methods" if methods_directory is None else Path(methods_directory)
    )
    frames = []
    for outer_fold in range(FOLD_COUNT):
        if predictor_id in BASELINE_IDS:
            predictions = baseline_predictions(
                predictor_id,
                source,
                split_strategy=split_strategy,
                outer_fold=outer_fold,
                root=root,
            )
            label = f"baseline {predictor_id!r} on {cohort_id} {split_strategy} fold {outer_fold}"
        else:
            path = (
                model_directory(
                    directory / predictor_id,
                    source,
                    split_strategy=split_strategy,
                    outer_fold=outer_fold,
                )
                / PREDICTION_FILES["blind"]
            )
            predictions = read_predictions(
                path,
                cohort_id=source,
                split_strategy=split_strategy,
                outer_fold=outer_fold,
                role=TEST_ROLE,
                root=root,
            )
            label = str(path)
        truth = read_evaluation_truth(
            cohort_id, outer_fold, split_strategy=split_strategy, root=root
        )
        merged = join_targets(
            predictions[[GENE_KEY, SCORE]],
            truth[[GENE_KEY, target.column]],
            cohort_id=cohort_id,
            split_strategy=split_strategy,
            source=label,
            root=root,
        )
        merged = scope_frame(merged, cohort_id=cohort_id, scope=cut, root=root)
        frames.append(merged.assign(**{FOLD: outer_fold}))
    return pd.concat(frames, ignore_index=True)[[GENE_KEY, target.column, SCORE, FOLD]]
