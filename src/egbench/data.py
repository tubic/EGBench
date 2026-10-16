"""Paths and readers for the benchmark datasets: labels, fold tables, method directories and per-gene covariates."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from egbench.contracts import (
    BINARY_TARGET,
    COHORT_IDS,
    SPLIT_STRATEGIES,
    ContractError,
    TargetKind,
    TargetSpec,
    check_prediction_identity,
    read_fold_table,
    read_prediction_table,
)

_SPLIT_DIRECTORY_NAMES: dict[str, str] = {"family_5fold": "folds"}

_COHORT_TABLE_DIRECTORY = "folds"

TRAIN_ROLE = "train"
VALIDATION_ROLE = "validation"
TEST_ROLE = "test"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FAMILY_SIZE_COLUMN = "cluster_size"
PPI_DEGREE_COLUMN = "ppi_degree"
EXPRESSION_COLUMN = "expression_median"


@dataclass(frozen=True)
class _Cohort:

    directory: str
    label_confidence_file: str | None
    label_confidence: Mapping[str, str]
    target: TargetSpec = BINARY_TARGET
    predictions_from: str | None = None
    primary_scope: str | None = None


_COHORTS: dict[str, _Cohort] = {
    "deg": _Cohort(
        directory="DEG",
        label_confidence_file="data/label_core.parquet",
        label_confidence={"deg_vote_count": "Int64"},
    ),
    "depmap": _Cohort(
        directory="DepMap",
        label_confidence_file="data/depmap26q1_gene_universe.parquet",
        label_confidence={
            "official_common_essential": "Int64",
            "strongly_selective": "Int64",
        },
    ),
    "context": _Cohort(
        directory="Context",
        label_confidence_file="data/gene_universe.parquet",
        label_confidence={
            "deg_vote_count": "Int64",
            "official_common_essential": "Int64",
            "strongly_selective": "Int64",
        },
        primary_scope="crispr_screen_supported",
    ),
    "gene_effect": _Cohort(
        directory="GeneEffect",
        label_confidence_file=None,
        label_confidence={},
        target=TargetSpec(column="gene_effect", kind=TargetKind.CONTINUOUS),
        predictions_from="depmap",
    ),
    "blood": _Cohort(
        directory="Blood",
        label_confidence_file="data/gene_universe.parquet",
        label_confidence={"dependent_fraction": "Float64"},
    ),
    "mouse": _Cohort(
        directory="Mouse",
        label_confidence_file="data/gene_universe.parquet",
        label_confidence={"impc_viability_call": "string"},
    ),
    "depmap_23q4": _Cohort(
        directory="DepMap23Q4",
        label_confidence_file="data/gene_universe.parquet",
        label_confidence={"official_common_essential": "Int64"},
    ),
}

LABEL_CONFIDENCE_COLUMNS: dict[str, tuple[str, ...]] = {
    cohort_id: tuple(cohort.label_confidence)
    for cohort_id, cohort in _COHORTS.items()
}

FULL_SCOPE = "all"

_COHORT_LABEL_FILES: dict[TargetKind, str] = {
    TargetKind.BINARY: "common_binary.parquet",
    TargetKind.CONTINUOUS: "common_continuous.parquet",
}

_PPI_EDGE_FILE = "methods/epgat/feature/source/network/biogrid_physical_edges.parquet"
_PPI_ENDPOINT_COLUMNS = ("gene_key_a", "gene_key_b")

_EXPRESSION_FILE = "data/source/expression/tcga_gene_by_sample.parquet"
_GENE_KEY = "gene_key"

_EXPRESSION_COLUMN_BLOCK = 256
_EXPRESSION_MEDIANS: dict[str, pd.Series] = {}


def _resolve_root(root: str | Path | None) -> Path:
    return PROJECT_ROOT if root is None else Path(root)


def _cohort(cohort_id: str) -> _Cohort:
    try:
        return _COHORTS[cohort_id]
    except KeyError:
        raise ValueError(
            f"unknown cohort_id {cohort_id!r}, the cohorts are {list(COHORT_IDS)}"
        ) from None


def cohort_target(cohort_id: str) -> TargetSpec:
    return _cohort(cohort_id).target


def prediction_cohort(cohort_id: str) -> str:
    cohort = _cohort(cohort_id)
    if cohort.predictions_from is None:
        return cohort_id
    _cohort(cohort.predictions_from)
    return cohort.predictions_from


def evaluation_scopes(cohort_id: str) -> tuple[str, ...]:
    cohort = _cohort(cohort_id)
    if cohort.primary_scope is None:
        return (FULL_SCOPE,)
    return (cohort.primary_scope, FULL_SCOPE)


def primary_scope(cohort_id: str) -> str:
    return evaluation_scopes(cohort_id)[0]


def scope_gene_keys(
    cohort_id: str, scope: str, *, root: str | Path | None = None
) -> pd.Index | None:
    cohort = _cohort(cohort_id)
    if scope not in evaluation_scopes(cohort_id):
        raise ValueError(
            f"unknown scope {scope!r} for cohort {cohort_id!r}, its scopes are "
            f"{list(evaluation_scopes(cohort_id))}"
        )
    if scope == FULL_SCOPE:
        return None
    if cohort.label_confidence_file is None:
        raise ValueError(f"cohort {cohort_id!r} declares no label confidence file to read {scope!r} from")
    column = cohort.target.column
    path = _resolve_root(root) / "datasets" / cohort.directory / cohort.label_confidence_file
    frame = pd.read_parquet(path, columns=[_GENE_KEY, column, scope])
    labeled = frame.loc[frame[column].notna()]
    flags = labeled[scope]
    if flags.isna().any():
        raise ContractError(
            f"{path}: column {scope!r} is missing on {int(flags.isna().sum())} "
            f"labeled genes; a subset scope must be recorded for every one"
        )
    kept = labeled[column].astype("int64").eq(0) | flags.astype("int64").eq(1)
    return pd.Index(labeled.loc[kept, _GENE_KEY].astype("object"), name=_GENE_KEY)


def read_cohort_labels(
    cohort_id: str, *, root: str | Path | None = None
) -> pd.DataFrame:
    cohort = _cohort(cohort_id)
    path = cohort_tables_directory(cohort_id, root=root) / _COHORT_LABEL_FILES[cohort.target.kind]
    frame = pd.read_parquet(path, columns=[_GENE_KEY, cohort.target.column])
    if frame[_GENE_KEY].duplicated().any():
        raise ContractError(f"{path}: a gene_key appears more than once")
    if frame[cohort.target.column].isna().any():
        raise ContractError(f"{path}: the target column {cohort.target.column!r} has missing values")
    return frame


def _require_split_strategy(split_strategy: str) -> str:
    if split_strategy not in SPLIT_STRATEGIES:
        raise ValueError(
            f"unknown split_strategy {split_strategy!r}, the strategies are "
            f"{list(SPLIT_STRATEGIES)}"
        )
    return split_strategy


def folds_directory(
    cohort_id: str, *, split_strategy: str, root: str | Path | None = None
) -> Path:
    _require_split_strategy(split_strategy)
    return (
        _resolve_root(root)
        / "datasets"
        / _cohort(cohort_id).directory
        / _SPLIT_DIRECTORY_NAMES[split_strategy]
    )


def model_directory(
    method_root: str | Path,
    cohort_id: str,
    *,
    split_strategy: str,
    outer_fold: int,
) -> Path:
    _require_split_strategy(split_strategy)
    _cohort(cohort_id)
    return (
        Path(method_root)
        / "model"
        / cohort_id
        / split_strategy
        / f"fold_{outer_fold}"
    )


def feature_directory(
    method_root: str | Path,
    cohort_id: str,
    *,
    split_strategy: str,
    outer_fold: int,
) -> Path:
    _require_split_strategy(split_strategy)
    return (
        Path(method_root)
        / "feature"
        / _cohort(cohort_id).directory
        / split_strategy
        / f"fold_{outer_fold}"
    )


def cohort_tables_directory(
    cohort_id: str, *, root: str | Path | None = None
) -> Path:
    return (
        _resolve_root(root)
        / "datasets"
        / _cohort(cohort_id).directory
        / _COHORT_TABLE_DIRECTORY
    )


def _read_strategy_table(
    path: Path, split_strategy: str, target: TargetSpec | None = None
) -> pd.DataFrame:
    frame = read_fold_table(path, target=target)
    observed = sorted(set(frame["split_strategy"]))
    if observed != [split_strategy]:
        raise ContractError(
            f"{path}: read as split_strategy {split_strategy!r} because of the "
            f"directory it sits in, but its 'split_strategy' column holds "
            f"{observed}"
        )
    return frame


def _select_fold(
    frame: pd.DataFrame,
    *,
    outer_fold: int,
    role: str | None,
    source: str,
) -> pd.DataFrame:
    selection = frame["outer_fold"].eq(outer_fold)
    wanted = f"outer_fold={outer_fold}"
    if role is not None:
        selection &= frame["role"].eq(role)
        wanted += f", role={role!r}"
    selected = frame.loc[selection]
    if selected.empty:
        raise ValueError(
            f"{source}: no rows for {wanted}; the table holds outer_fold "
            f"{sorted(int(value) for value in frame['outer_fold'].unique())} and "
            f"role {sorted(set(frame['role']))}"
        )
    return selected.reset_index(drop=True)


def read_split_assignments(
    cohort_id: str, *, split_strategy: str, root: str | Path | None = None
) -> pd.DataFrame:
    path = (
        folds_directory(cohort_id, split_strategy=split_strategy, root=root)
        / "split_assignments.parquet"
    )
    return _read_strategy_table(path, split_strategy)


def fold_gene_keys(
    cohort_id: str,
    outer_fold: int,
    role: str,
    *,
    split_strategy: str,
    root: str | Path | None = None,
) -> pd.Index:
    assignments = read_split_assignments(
        cohort_id, split_strategy=split_strategy, root=root
    )
    selected = _select_fold(
        assignments,
        outer_fold=outer_fold,
        role=role,
        source=(
            f"split assignments of cohort {cohort_id!r} under {split_strategy!r}"
        ),
    )
    return pd.Index(selected[_GENE_KEY], name=_GENE_KEY)


def read_training_labels(
    cohort_id: str,
    outer_fold: int,
    role: str,
    *,
    split_strategy: str,
    root: str | Path | None = None,
) -> pd.DataFrame:
    path = (
        folds_directory(cohort_id, split_strategy=split_strategy, root=root)
        / "training_labels"
        / f"fold_{outer_fold}.parquet"
    )
    frame = _read_strategy_table(path, split_strategy, _cohort(cohort_id).target)
    return _select_fold(frame, outer_fold=outer_fold, role=role, source=str(path))


def read_evaluation_truth(
    cohort_id: str,
    outer_fold: int,
    *,
    split_strategy: str,
    root: str | Path | None = None,
) -> pd.DataFrame:
    path = (
        folds_directory(cohort_id, split_strategy=split_strategy, root=root)
        / "evaluation_truth.parquet"
    )
    frame = _read_strategy_table(path, split_strategy, _cohort(cohort_id).target)
    return _select_fold(frame, outer_fold=outer_fold, role=TEST_ROLE, source=str(path))


def read_predictions(
    path: str | Path,
    *,
    cohort_id: str,
    split_strategy: str,
    outer_fold: int,
    role: str,
    root: str | Path | None = None,
) -> pd.DataFrame:
    predictions = read_prediction_table(path)
    assignments = read_split_assignments(
        cohort_id, split_strategy=split_strategy, root=root
    )
    check_prediction_identity(
        predictions,
        assignments,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        role=role,
    )
    return predictions


def _expression_medians(path: Path) -> pd.Series:
    key = str(path)
    cached = _EXPRESSION_MEDIANS.get(key)
    if cached is not None:
        return cached
    parquet = pq.ParquetFile(path)
    names = list(parquet.schema_arrow.names)
    if _GENE_KEY not in names:
        raise ContractError(
            f"{path}: the expression matrix must carry a '{_GENE_KEY}' column, "
            f"its first columns are {names[:5]}"
        )
    sample_columns = [name for name in names if name != _GENE_KEY]
    if not sample_columns:
        raise ContractError(f"{path}: the expression matrix carries no sample column")
    gene_keys = parquet.read(columns=[_GENE_KEY]).column(_GENE_KEY).to_pylist()
    matrix = np.empty((len(gene_keys), len(sample_columns)), dtype="float32")
    filled = 0
    for start in range(0, len(sample_columns), _EXPRESSION_COLUMN_BLOCK):
        block = sample_columns[start : start + _EXPRESSION_COLUMN_BLOCK]
        table = parquet.read(columns=block)
        for offset, name in enumerate(block):
            matrix[:, filled + offset] = table.column(name).to_numpy(
                zero_copy_only=False
            )
        filled += len(block)
        del table
    medians = pd.Series(
        np.median(matrix, axis=1, overwrite_input=True).astype("float64"),
        index=pd.Index(gene_keys, name=_GENE_KEY),
        name=EXPRESSION_COLUMN,
    )
    _EXPRESSION_MEDIANS[key] = medians
    return medians


def _ppi_degrees(path: Path, keys: pd.Index) -> pd.Series:
    edges = pd.read_parquet(path, columns=list(_PPI_ENDPOINT_COLUMNS))
    inside = edges[list(_PPI_ENDPOINT_COLUMNS)].isin(keys).all(axis=1)
    endpoints = pd.concat(
        [edges.loc[inside, column] for column in _PPI_ENDPOINT_COLUMNS], ignore_index=True
    )
    degrees = endpoints.value_counts().reindex(keys, fill_value=0)
    degrees.index.name = _GENE_KEY
    return degrees.rename(PPI_DEGREE_COLUMN)


def read_strata_inputs(
    cohort_id: str, *, root: str | Path | None = None
) -> pd.DataFrame:
    cohort = _cohort(cohort_id)
    resolved = _resolve_root(root)
    frame = pd.read_parquet(
        cohort_tables_directory(cohort_id, root=root) / "cluster_projection.parquet",
        columns=[_GENE_KEY, FAMILY_SIZE_COLUMN],
    )
    keys = pd.Index(frame[_GENE_KEY])
    degrees = _ppi_degrees(resolved / _PPI_EDGE_FILE, keys)
    frame[PPI_DEGREE_COLUMN] = degrees.astype("Int64").array
    medians = _expression_medians(resolved / _EXPRESSION_FILE)
    frame[EXPRESSION_COLUMN] = medians.reindex(keys).astype("float64").to_numpy()
    if cohort.label_confidence_file is None:
        return frame
    confidence = read_label_confidence(cohort_id, root=root)
    return frame.merge(confidence, on=_GENE_KEY, how="left", validate="one_to_one")


def read_label_confidence(
    cohort_id: str, *, root: str | Path | None = None
) -> pd.DataFrame:
    cohort = _cohort(cohort_id)
    if cohort.label_confidence_file is None:
        raise ValueError(f"cohort {cohort_id!r} declares no label confidence file")
    frame = pd.read_parquet(
        _resolve_root(root) / "datasets" / cohort.directory / cohort.label_confidence_file,
        columns=[_GENE_KEY, *cohort.label_confidence],
    )
    if frame[_GENE_KEY].duplicated().any():
        raise ContractError(
            f"the label confidence file of {cohort_id!r} describes a gene twice"
        )
    for column, dtype in cohort.label_confidence.items():
        frame[column] = frame[column].astype(dtype)
    return frame
