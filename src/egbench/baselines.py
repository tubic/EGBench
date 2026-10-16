"""Literature-count and PPI-degree baselines, per-gene covariates and the logistic and rank-normal helpers used by the fusion."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from egbench.contracts import PREDICTION_COLUMNS, validate_prediction_table
from egbench.data import (
    EXPRESSION_COLUMN,
    FAMILY_SIZE_COLUMN,
    PPI_DEGREE_COLUMN,
    PROJECT_ROOT,
    TEST_ROLE,
    cohort_target,
    fold_gene_keys,
    prediction_cohort,
    read_evaluation_truth,
    read_split_assignments,
    read_strata_inputs,
    read_training_labels,
)
from egbench.evaluate import FOLD_COUNT, Scored, join_targets, score_frames

GENE_KEY = "gene_key"
SCORE = "y_score"
PREDICTION = "y_pred"
TARGET = "label"
CLUSTER = "cluster_id"

FIT_ROLE = "train"

DECISION_THRESHOLD = 0.5

LENGTH_COLUMN = "gene_length_bp"
LITERATURE_COLUMN = "pubmed_count"

_GENE_TABLE = "datasets/DEG/data/ensembl97_protein_coding_genes.parquet"

_GENE2PUBMED_FILE = "data/source/annotations/gene2pubmed.gz"
_GENE2PUBMED_COLUMNS = ("tax_id", "gene_id", "pubmed_id")
_HUMAN_TAX_ID = 9606
_HGNC_FILE = "data/hgnc/hgnc_complete_set_2026-07-07.txt"

_LITERATURE_COUNTS: dict[str, pd.Series] = {}

_MAX_ITERATIONS = 100
_STEP_TOLERANCE = 1e-12

COVARIATE_COLUMNS: tuple[str, ...] = (
    FAMILY_SIZE_COLUMN,
    LENGTH_COLUMN,
    EXPRESSION_COLUMN,
    PPI_DEGREE_COLUMN,
    LITERATURE_COLUMN,
)

COMPLEX_INDICATOR_COLUMN = "has_complex_annotation"
PROCESS_INDICATOR_COLUMN = "has_process_annotation"

BASELINES: dict[str, tuple[str, ...]] = {
    "ppi_degree": (PPI_DEGREE_COLUMN,),
    "literature": (LITERATURE_COLUMN,),
}
BASELINE_IDS: tuple[str, ...] = tuple(BASELINES)

def _resolve_root(root: str | Path | None) -> Path:
    return PROJECT_ROOT if root is None else Path(root)


def _require_baseline(baseline_id: str) -> tuple[str, ...]:
    if baseline_id not in BASELINES:
        raise ValueError(
            f"unknown baseline {baseline_id!r}, the baselines are "
            f"{list(BASELINE_IDS)}"
        )
    return BASELINES[baseline_id]


def positive_rate(y_true: object) -> float:
    labels = np.asarray(y_true)
    if labels.size == 0:
        raise ValueError("positive_rate: no labels to take the rate of")
    return float(np.mean(labels.astype("float64")))


def _design(x: object, name: str) -> np.ndarray:
    feature = np.asarray(x, dtype="float64")
    if feature.ndim == 1:
        return feature.reshape(-1, 1)
    if feature.ndim != 2:
        raise ValueError(
            f"logistic regression: {name!r} must be one value per gene or one "
            f"row per gene, its shape is {feature.shape}"
        )
    return feature


def logistic_fit(x: object, y: object) -> np.ndarray:
    feature = _design(x, "x")
    labels = np.asarray(y, dtype="float64")
    if feature.shape[0] != labels.size:
        raise ValueError(
            f"logistic_fit: {feature.shape[0]} covariate rows and {labels.size} "
            f"labels"
        )
    if feature.shape[0] == 0:
        raise ValueError("logistic_fit: no genes to fit on")
    design = np.column_stack([np.ones(feature.shape[0]), feature])
    beta = np.zeros(design.shape[1], dtype="float64")
    for _ in range(_MAX_ITERATIONS):
        probability = _sigmoid(design @ beta)
        weight = probability * (1.0 - probability)
        hessian = design.T @ (design * weight[:, None])
        gradient = design.T @ (labels - probability)
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < _STEP_TOLERANCE:
            break
    return beta


def logistic_probability(x: object, beta: object) -> np.ndarray:
    feature = _design(x, "x")
    coefficients = np.asarray(beta, dtype="float64")
    if coefficients.shape != (feature.shape[1] + 1,):
        raise ValueError(
            f"logistic_probability: 'beta' is an intercept and "
            f"{feature.shape[1]} slopes, its shape is {coefficients.shape}"
        )
    return _sigmoid(coefficients[0] + feature @ coefficients[1:])


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0.0
    result = np.empty_like(value, dtype="float64")
    exponent = np.exp(-np.abs(value))
    result[positive] = 1.0 / (1.0 + exponent[positive])
    result[~positive] = exponent[~positive] / (1.0 + exponent[~positive])
    return result


def rank_normal_scores(score_fit: object, score_test: object) -> tuple[np.ndarray, np.ndarray]:
    fitted = np.asarray(score_fit, dtype="float64").reshape(-1)
    scored = np.asarray(score_test, dtype="float64").reshape(-1)
    if not (np.isfinite(fitted).all() and np.isfinite(scored).all()):
        raise ValueError("rank_normal_scores: a score is not finite")
    if fitted.size == 0:
        raise ValueError("rank_normal_scores: no fitted score")
    values, counts = np.unique(fitted, return_counts=True)
    if values.size == 1:
        return np.zeros(fitted.size), np.zeros(scored.size)
    upper = np.cumsum(counts)
    mid_rank = (upper - counts / 2.0) / fitted.size
    inverse = NormalDist().inv_cdf
    quantiles = np.array([inverse(float(rank)) for rank in mid_rank], dtype="float64")

    def place(scores: np.ndarray) -> np.ndarray:
        placed = np.interp(scores, values, quantiles)
        low = scores < values[0]
        high = scores > values[-1]
        if low.any():
            slope = (quantiles[1] - quantiles[0]) / (values[1] - values[0])
            placed[low] = quantiles[0] + (scores[low] - values[0]) * slope
        if high.any():
            slope = (quantiles[-1] - quantiles[-2]) / (values[-1] - values[-2])
            placed[high] = quantiles[-1] + (scores[high] - values[-1]) * slope
        return placed

    return place(fitted), place(scored)


def _measured(
    x_fit: object, y_fit: object, x_test: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fitted = _design(x_fit, "x_fit")
    labels = np.asarray(y_fit).astype("float64")
    wanted = _design(x_test, "x_test")
    if fitted.shape[0] != labels.size:
        raise ValueError(
            f"covariate_scores: {fitted.shape[0]} covariate rows and "
            f"{labels.size} labels"
        )
    if fitted.shape[1] != wanted.shape[1]:
        raise ValueError(
            f"covariate_scores: 'x_fit' holds {fitted.shape[1]} covariates and "
            f"'x_test' holds {wanted.shape[1]}"
        )
    for name, values in (("x_fit", fitted), ("x_test", wanted)):
        negative = values[~np.isnan(values)] < 0.0
        if negative.any():
            raise ValueError(
                f"covariate_scores: {name!r} holds {int(negative.sum())} "
                f"negative values, which log1p has no value at; the five "
                f"covariates of this module are counts and concentrations"
            )
    present = ~np.isnan(fitted).any(axis=1)
    if not present.any():
        raise ValueError("covariate_scores: no fitted gene carries every covariate")
    covered = ~np.isnan(wanted).any(axis=1)
    return np.log1p(fitted), labels, np.log1p(wanted), present, covered


def _regression_scores(
    design_fit: np.ndarray,
    labels: np.ndarray,
    design_test: np.ndarray,
    present: np.ndarray,
    covered: np.ndarray,
) -> np.ndarray:
    observed = design_fit[present]
    center = observed.mean(axis=0)
    spread = observed.std(axis=0)
    if (spread == 0.0).any():
        raise ValueError(
            "covariate_scores: a covariate takes one value on every fitted "
            "gene and carries no information to standardise or to fit"
        )
    beta = logistic_fit((observed - center) / spread, labels[present])
    scores = np.full(design_test.shape[0], positive_rate(labels), dtype="float64")
    scores[covered] = logistic_probability(
        (design_test[covered] - center) / spread, beta
    )
    return scores


def covariate_scores(x_fit: object, y_fit: object, x_test: object) -> np.ndarray:
    fitted, labels, wanted, present, covered = _measured(x_fit, y_fit, x_test)
    return _regression_scores(fitted, labels, wanted, present, covered)


def partner_essential_fraction(
    pairs: pd.DataFrame,
    fit_keys: object,
    fit_labels: object,
    wanted_keys: object,
) -> np.ndarray:
    keys = pd.Index(np.asarray(fit_keys, dtype="object"))
    labels = np.asarray(fit_labels).astype("float64")
    if keys.size != labels.size:
        raise ValueError(
            f"partner_essential_fraction: {keys.size} fitted keys and "
            f"{labels.size} fitted labels"
        )
    if keys.has_duplicates:
        raise ValueError("partner_essential_fraction: a fitted key appears more than once")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("partner_essential_fraction: fitted labels must be 0 or 1")
    wanted = pd.Index(np.asarray(wanted_keys, dtype="object"))
    fitted = pd.Series(labels, index=keys)
    members = pairs.loc[pairs[GENE_KEY].isin(keys)]
    per_term = (
        pd.DataFrame(
            {
                "essential": fitted.reindex(members[GENE_KEY]).to_numpy(),
                "fitted": 1.0,
            },
            index=members["term_id"].to_numpy(),
        )
        .groupby(level=0)
        .sum()
    )
    own = pairs.loc[pairs[GENE_KEY].isin(wanted)]
    counts = pd.DataFrame(
        {
            "essential": per_term["essential"].reindex(own["term_id"]).fillna(0.0).to_numpy(),
            "fitted": per_term["fitted"].reindex(own["term_id"]).fillna(0.0).to_numpy(),
            "terms": 1.0,
        },
        index=own[GENE_KEY].to_numpy(),
    ).groupby(level=0).sum()
    self_label = fitted.reindex(counts.index)
    is_fitted = self_label.notna().to_numpy()
    counts.loc[is_fitted, "essential"] -= (
        self_label[is_fitted].to_numpy() * counts.loc[is_fitted, "terms"].to_numpy()
    )
    counts.loc[is_fitted, "fitted"] -= counts.loc[is_fitted, "terms"].to_numpy()
    fraction = counts["essential"] / counts["fitted"].where(counts["fitted"] > 0.0)
    return fraction.reindex(wanted).to_numpy("float64")


def literature_counts(root: str | Path | None = None) -> pd.Series:
    resolved = _resolve_root(root)
    key = str(resolved)
    cached = _LITERATURE_COUNTS.get(key)
    if cached is not None:
        return cached
    records = pd.read_csv(
        resolved / _GENE2PUBMED_FILE,
        sep="\t",
        header=0,
        names=list(_GENE2PUBMED_COLUMNS),
        usecols=[0, 1, 2],
        dtype="int64",
    )
    human = records.loc[records["tax_id"] == _HUMAN_TAX_ID]
    counts = (
        human.drop_duplicates(["gene_id", "pubmed_id"])
        .groupby("gene_id")
        .size()
        .astype("int64")
    )
    identifiers = pd.read_csv(
        resolved / _HGNC_FILE,
        sep="\t",
        usecols=["hgnc_id", "entrez_id"],
        dtype={"hgnc_id": "string"},
        low_memory=False,
    ).dropna(subset=["entrez_id"])
    mapped = pd.Series(
        counts.reindex(identifiers["entrez_id"].astype("int64")).to_numpy(),
        index=pd.Index(identifiers["hgnc_id"].astype("object"), name=GENE_KEY),
        name=LITERATURE_COLUMN,
    )
    _LITERATURE_COUNTS[key] = mapped
    return mapped


def _gene_lengths(root: Path) -> pd.Series:
    table = pd.read_parquet(root / _GENE_TABLE, columns=[GENE_KEY, "start", "end"])
    keys = table[GENE_KEY].astype("string").fillna("")
    table = table.loc[(keys != "").to_numpy()]
    span = (table["end"] - table["start"] + 1).astype("float64")
    return (
        pd.Series(span.to_numpy(), index=pd.Index(table[GENE_KEY].astype("object"), name=GENE_KEY))
        .groupby(level=0)
        .max()
        .rename(LENGTH_COLUMN)
    )


def family_sizes(
    cohort_id: str, *, split_strategy: str, root: str | Path | None = None
) -> pd.Series:
    assignments = read_split_assignments(
        cohort_id, split_strategy=split_strategy, root=root
    ).drop_duplicates(GENE_KEY)
    sizes = assignments[CLUSTER].map(assignments[CLUSTER].value_counts())
    return pd.Series(
        sizes.to_numpy("float64"),
        index=pd.Index(assignments[GENE_KEY].astype("object"), name=GENE_KEY),
        name=FAMILY_SIZE_COLUMN,
    )


def gene_covariates(
    cohort_id: str, *, split_strategy: str, root: str | Path | None = None
) -> pd.DataFrame:
    resolved = _resolve_root(root)
    frame = read_strata_inputs(cohort_id, root=root)[
        [GENE_KEY, PPI_DEGREE_COLUMN, EXPRESSION_COLUMN]
    ].copy()
    keys = pd.Index(frame[GENE_KEY])
    sizes = family_sizes(cohort_id, split_strategy=split_strategy, root=root)
    frame[FAMILY_SIZE_COLUMN] = sizes.reindex(keys).astype("float64").to_numpy()
    if frame[FAMILY_SIZE_COLUMN].isna().any():
        raise ValueError(
            f"gene_covariates: {int(frame[FAMILY_SIZE_COLUMN].isna().sum())} genes "
            f"of {cohort_id} are not in the fold table of {split_strategy}"
        )
    lengths = _gene_lengths(resolved)
    frame[LENGTH_COLUMN] = lengths.reindex(keys).astype("float64").to_numpy()
    counts = literature_counts(resolved)
    frame[LITERATURE_COLUMN] = (
        counts.reindex(keys).fillna(0).astype("float64").to_numpy()
    )
    frame[PPI_DEGREE_COLUMN] = frame[PPI_DEGREE_COLUMN].astype("Float64").astype(
        "float64"
    )
    return frame[[GENE_KEY, *COVARIATE_COLUMNS]]


def prediction_table(
    scored_keys: pd.Index, scores: np.ndarray, *, source: str
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            GENE_KEY: pd.Series(list(scored_keys), dtype="object"),
            SCORE: scores,
            PREDICTION: (scores >= DECISION_THRESHOLD).astype("int8"),
        }
    )[list(PREDICTION_COLUMNS)]
    validate_prediction_table(frame, table=source, decision_threshold=DECISION_THRESHOLD)
    return frame


def regression_predictions(
    covariates: Sequence[str],
    cohort_id: str,
    *,
    split_strategy: str,
    outer_fold: int,
    role: str = TEST_ROLE,
    root: str | Path | None = None,
) -> pd.DataFrame:
    if not covariates:
        raise ValueError("regression_predictions: no covariate to regress on")
    fitted = read_training_labels(
        cohort_id, outer_fold, FIT_ROLE, split_strategy=split_strategy, root=root
    )
    scored_keys = fold_gene_keys(
        cohort_id, outer_fold, role, split_strategy=split_strategy, root=root
    )
    fit_keys = pd.Index(fitted[GENE_KEY])
    labels = fitted[TARGET].to_numpy()
    values = gene_covariates(
        cohort_id, split_strategy=split_strategy, root=root
    ).set_index(GENE_KEY)[list(covariates)]
    x_fit = values.reindex(fit_keys).to_numpy("float64")
    x_test = values.reindex(scored_keys).to_numpy("float64")
    scores = covariate_scores(x_fit, labels, x_test)
    return prediction_table(
        scored_keys,
        scores,
        source=(
            f"regression on {list(covariates)} on {cohort_id} "
            f"{split_strategy} fold {outer_fold} {role}"
        ),
    )


def baseline_predictions(
    baseline_id: str,
    cohort_id: str,
    *,
    split_strategy: str,
    outer_fold: int,
    role: str = TEST_ROLE,
    root: str | Path | None = None,
) -> pd.DataFrame:
    return regression_predictions(
        _require_baseline(baseline_id),
        cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        role=role,
        root=root,
    )


def score_predictions(
    predict: Callable[..., pd.DataFrame],
    *,
    method_id: str,
    cohort_id: str,
    split_strategy: str,
    scope: str | None = None,
    root: str | Path | None = None,
) -> Scored:
    source = prediction_cohort(cohort_id)
    target = cohort_target(cohort_id)
    folds = []
    for outer_fold in range(FOLD_COUNT):
        label = f"{method_id!r} on {cohort_id} {split_strategy} fold {outer_fold}"
        predictions = predict(source, outer_fold, TEST_ROLE)
        truth = read_evaluation_truth(
            cohort_id, outer_fold, split_strategy=split_strategy, root=root
        )
        folds.append(
            join_targets(
                predictions,
                truth[[GENE_KEY, target.column]],
                cohort_id=cohort_id,
                split_strategy=split_strategy,
                source=label,
                root=root,
            )
        )
    return score_frames(
        folds,
        method_id=method_id,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        scope=scope,
    )


def score_baseline(
    baseline_id: str,
    cohort_id: str,
    *,
    split_strategy: str,
    scope: str | None = None,
    root: str | Path | None = None,
) -> Scored:
    _require_baseline(baseline_id)

    def predict(source: str, outer_fold: int, role: str) -> pd.DataFrame:
        return baseline_predictions(
            baseline_id,
            source,
            split_strategy=split_strategy,
            outer_fold=outer_fold,
            role=role,
            root=root,
        )

    return score_predictions(
        predict,
        method_id=baseline_id,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        scope=scope,
        root=root,
    )
