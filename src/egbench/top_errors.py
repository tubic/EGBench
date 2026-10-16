"""Independent evidence for the non-essential genes a predictor ranks among its top k."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from egbench.baselines import FIT_ROLE, partner_essential_fraction
from egbench.data import (
    PROJECT_ROOT,
    cohort_target,
    primary_scope,
    read_evaluation_truth,
    read_training_labels,
)
from egbench.evaluate import FOLD_COUNT, scope_frame
from egbench.fusion import pooled
from egbench.modules import cohort_complexes
from egbench.reference import (
    GENE_EFFECT_COLUMN,
    MAXIMUM_MISSING_FRACTION,
    MISSING_COLUMN,
    deg_vote_counts,
    gene_effect_means,
)
from egbench.stats import fisher_exact_p

GENE_KEY = "gene_key"
SCORE = "y_score"
FOLD = "fold"
LABEL = "label"

TOP_ERROR_COHORTS: tuple[str, ...] = ("deg", "depmap")
DEPLETED_EFFECT = -0.5
MAJORITY_SHARE = 0.5

VOTE_COLUMN = "deg_vote_count"
PARTNER_COLUMN = "complex_partner_essential"
DEG_VOTE = "deg_vote"
COMPLEX_PARTNER = "complex_partner_majority"
GENE_EFFECT_DEPLETED = "gene_effect_depleted"
GENE_EFFECT_MEAN = "gene_effect_mean"
EVIDENCE_ITEMS: tuple[str, ...] = (DEG_VOTE, COMPLEX_PARTNER, GENE_EFFECT_DEPLETED, GENE_EFFECT_MEAN)
LABEL_SOURCE_ITEMS: dict[str, frozenset[str]] = {
    "deg": frozenset({DEG_VOTE}),
    "depmap": frozenset({GENE_EFFECT_DEPLETED, GENE_EFFECT_MEAN}),
}

_EVIDENCE: dict[tuple[str, str, str], pd.DataFrame] = {}


def _resolve_root(root: str | Path | None) -> Path:
    return PROJECT_ROOT if root is None else Path(root)


def independent(cohort_id: str, item: str) -> bool:
    if cohort_id not in LABEL_SOURCE_ITEMS:
        raise ValueError(f"independent: {cohort_id!r} is not one of {list(TOP_ERROR_COHORTS)}")
    if item not in EVIDENCE_ITEMS:
        raise ValueError(f"independent: unknown evidence item {item!r}, the items are {list(EVIDENCE_ITEMS)}")
    return item not in LABEL_SOURCE_ITEMS[cohort_id]


def evidence_frame(
    cohort_id: str, *, split_strategy: str, root: str | Path | None = None
) -> pd.DataFrame:
    if cohort_id not in TOP_ERROR_COHORTS:
        raise ValueError(f"evidence_frame: {cohort_id!r} is not one of {list(TOP_ERROR_COHORTS)}")
    key = (str(_resolve_root(root)), cohort_id, split_strategy)
    cached = _EVIDENCE.get(key)
    if cached is not None:
        return cached
    target = cohort_target(cohort_id).column
    pairs = cohort_complexes(cohort_id, root=root)
    frames = []
    for outer_fold in range(FOLD_COUNT):
        truth = read_evaluation_truth(cohort_id, outer_fold, split_strategy=split_strategy, root=root)
        frame = scope_frame(
            truth[[GENE_KEY, target]].copy(), cohort_id=cohort_id, scope=primary_scope(cohort_id), root=root
        )
        fitted = read_training_labels(cohort_id, outer_fold, FIT_ROLE, split_strategy=split_strategy, root=root)
        frame[PARTNER_COLUMN] = partner_essential_fraction(
            pairs, fitted[GENE_KEY], fitted[target].to_numpy(), frame[GENE_KEY]
        )
        frames.append(frame.assign(**{FOLD: outer_fold}))
    joined = pd.concat(frames, ignore_index=True).rename(columns={target: LABEL})
    keys = joined[GENE_KEY].to_numpy("object")
    joined[VOTE_COLUMN] = deg_vote_counts(root).reindex(keys).to_numpy("float64")
    effects = gene_effect_means(root)
    kept = effects.loc[effects[MISSING_COLUMN] <= MAXIMUM_MISSING_FRACTION, GENE_EFFECT_COLUMN]
    joined[GENE_EFFECT_COLUMN] = kept.reindex(keys).to_numpy("float64")
    if joined[GENE_KEY].duplicated().any():
        raise ValueError(f"evidence_frame: a gene of {cohort_id} {split_strategy} is tested in two folds")
    result = joined[[GENE_KEY, LABEL, FOLD, VOTE_COLUMN, PARTNER_COLUMN, GENE_EFFECT_COLUMN]]
    _EVIDENCE[key] = result
    return result


def evidence_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    votes = frame[VOTE_COLUMN].to_numpy("float64")
    partners = frame[PARTNER_COLUMN].to_numpy("float64")
    effect = frame[GENE_EFFECT_COLUMN].to_numpy("float64")
    with np.errstate(invalid="ignore"):
        return pd.DataFrame(
            {
                DEG_VOTE: np.where(np.isnan(votes), np.nan, (votes >= 1.0).astype("float64")),
                COMPLEX_PARTNER: np.where(np.isnan(partners), 0.0, (partners > MAJORITY_SHARE).astype("float64")),
                GENE_EFFECT_DEPLETED: np.where(np.isnan(effect), np.nan, (effect < DEPLETED_EFFECT).astype("float64")),
                GENE_EFFECT_MEAN: effect,
            },
            index=frame.index,
        )


def top_positive_count_mask(scores: object, labels: object, fold: object, keys: object) -> np.ndarray:
    values = np.asarray(scores, dtype="float64")
    targets = np.asarray(labels, dtype="float64")
    folds = np.asarray(fold)
    names = np.asarray(keys, dtype="object").astype(str)
    if not (values.size == targets.size == folds.size == names.size):
        raise ValueError("top_positive_count_mask: the four arrays must be as long as each other")
    if not np.isfinite(values).all():
        raise ValueError("top_positive_count_mask: a score is not finite")
    if not np.isin(targets, (0.0, 1.0)).all():
        raise ValueError("top_positive_count_mask: labels must be 0 or 1")
    mask = np.zeros(values.size, dtype=bool)
    for name in np.unique(folds):
        inside = np.flatnonzero(folds == name)
        count = int(targets[inside].sum())
        order = np.lexsort((names[inside], -values[inside]))
        mask[inside[order[:count]]] = True
    return mask


class EvidenceShare(NamedTuple):
    item: str
    independent: bool
    false_positive_count: int
    false_positive_value: float
    negative_count: int
    negative_value: float
    positive_count: int
    positive_value: float
    p_value: float


class FalsePositiveProfile(NamedTuple):
    predictor_id: str
    cohort_id: str
    split_strategy: str
    positive_count: int
    false_positive_count: int
    negative_count: int
    shares: list[EvidenceShare]


def _mean(values: np.ndarray) -> tuple[int, float]:
    defined = values[~np.isnan(values)]
    return int(defined.size), (float("nan") if defined.size == 0 else float(defined.mean()))


def _fisher(first: np.ndarray, second: np.ndarray) -> float:
    a = first[~np.isnan(first)]
    b = second[~np.isnan(second)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return fisher_exact_p([[int(a.sum()), int(a.size - a.sum())], [int(b.sum()), int(b.size - b.sum())]])


def evidence_shares(
    cohort_id: str, indicators: pd.DataFrame, labels: np.ndarray, false_positive: np.ndarray
) -> list[EvidenceShare]:
    negative = labels == 0.0
    positive = labels == 1.0
    other = negative & ~false_positive
    shares = []
    for item in EVIDENCE_ITEMS:
        values = indicators[item].to_numpy("float64")
        fp_count, fp_value = _mean(values[false_positive])
        negative_count, negative_value = _mean(values[negative])
        positive_count, positive_value = _mean(values[positive])
        p_value = float("nan") if item == GENE_EFFECT_MEAN else _fisher(values[false_positive], values[other])
        shares.append(
            EvidenceShare(
                item,
                independent(cohort_id, item),
                fp_count,
                fp_value,
                negative_count,
                negative_value,
                positive_count,
                positive_value,
                p_value,
            )
        )
    return shares


def false_positive_profile(
    predictor_id: str,
    cohort_id: str,
    *,
    split_strategy: str,
    member_ids: Sequence[str],
    methods_directory: str | Path | None = None,
    root: str | Path | None = None,
) -> FalsePositiveProfile:
    predictions = pooled(
        predictor_id,
        cohort_id,
        split_strategy=split_strategy,
        methods_directory=methods_directory,
        member_ids=member_ids,
        root=root,
    )
    evidence = evidence_frame(cohort_id, split_strategy=split_strategy, root=root)
    merged = evidence.merge(
        predictions[[GENE_KEY, FOLD, SCORE]], on=[GENE_KEY, FOLD], how="left", validate="one_to_one"
    )
    if merged[SCORE].isna().any() or len(merged) != len(predictions):
        raise ValueError(
            f"top_errors: {predictor_id!r} on {cohort_id} {split_strategy} scores "
            f"{len(predictions)} genes and the evidence frame holds {len(evidence)}"
        )
    labels = merged[LABEL].to_numpy("float64")
    top = top_positive_count_mask(
        merged[SCORE].to_numpy("float64"), labels, merged[FOLD].to_numpy("int64"), merged[GENE_KEY].to_numpy("object")
    )
    false_positive = top & (labels == 0.0)
    return FalsePositiveProfile(
        predictor_id,
        cohort_id,
        split_strategy,
        int(top.sum()),
        int(false_positive.sum()),
        int((labels == 0.0).sum()),
        evidence_shares(cohort_id, evidence_indicators(merged), labels, false_positive),
    )


FALSE_POSITIVE_COLUMNS: tuple[str, ...] = (
    "predictor",
    "cohort_id",
    "split_strategy",
    "positive_count",
    "false_positive_count",
    "negative_count",
    "evidence",
    "independent",
    "false_positive_with_value",
    "false_positive_value",
    "negative_with_value",
    "negative_value",
    "positive_with_value",
    "positive_value",
    "p_value",
)


def false_positive_table(profiles: Sequence[FalsePositiveProfile]) -> pd.DataFrame:
    rows = []
    for item in profiles:
        for share in item.shares:
            rows.append(
                {
                    "predictor": item.predictor_id,
                    "cohort_id": item.cohort_id,
                    "split_strategy": item.split_strategy,
                    "positive_count": item.positive_count,
                    "false_positive_count": item.false_positive_count,
                    "negative_count": item.negative_count,
                    "evidence": share.item,
                    "independent": share.independent,
                    "false_positive_with_value": share.false_positive_count,
                    "false_positive_value": share.false_positive_value,
                    "negative_with_value": share.negative_count,
                    "negative_value": share.negative_value,
                    "positive_with_value": share.positive_count,
                    "positive_value": share.positive_value,
                    "p_value": share.p_value,
                }
            )
    if not rows:
        raise ValueError("false_positive_table: no profile to tabulate")
    return pd.DataFrame(rows, columns=list(FALSE_POSITIVE_COLUMNS))
