"""File contracts shared by the methods and the evaluation: identifiers, table schemas, run manifests and model cards."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd
from pandas.api import types as pandas_types

SEED = 2026

PREDICTION_COLUMNS: tuple[str, ...] = ("gene_key", "y_score", "y_pred")

COHORT_IDS: tuple[str, ...] = (
    "deg",
    "depmap",
    "context",
    "gene_effect",
    "blood",
    "mouse",
    "depmap_23q4",
)
LEADERBOARD_COHORT_IDS: tuple[str, ...] = COHORT_IDS[:6]
SPLIT_STRATEGIES: tuple[str, ...] = ("family_5fold",)
OUTER_FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)
METHOD_IDS: tuple[str, ...] = (
    "bingo",
    "deepcelless",
    "deephe",
    "deeplyessential",
    "deepsnap",
    "ednn_eg",
    "epgat",
    "esssubgraph",
    "esstfnet",
    "gcnn_sfm",
    "help",
    "mahi",
    "pheg",
    "proteomelm",
    "xgep",
)

FOLD_COLUMNS: tuple[str, ...] = (
    "gene_key",
    "cohort_id",
    "split_strategy",
    "outer_fold",
    "role",
    "cluster_id",
    "split_build_hash",
)

ROLES = frozenset({"train", "validation", "test"})
BINARY_LABELS = frozenset({0, 1})

RUN_MANIFEST_KEYS: tuple[str, ...] = (
    "method_id",
    "cohort_id",
    "split_strategy",
    "outer_fold",
    "split_build_hash",
    "software_versions",
    "parameters",
    "started_at",
    "finished_at",
    "sample_counts",
)

RESOURCE_USAGE_KEYS: tuple[str, ...] = (
    "cpu_seconds",
    "peak_rss_bytes",
    "device",
    "gpu_name",
    "peak_gpu_memory_bytes",
)

MODEL_CARD_KEYS: tuple[str, ...] = (
    "method_id",
    "cohort_id",
    "split_strategy",
    "outer_fold",
    "split_build_hash",
    "decision_threshold",
    "artifacts",
    "required_source_assets",
    "software_versions",
)

PREDICTION_NAMES: tuple[str, ...] = ("blind", "validation")
PREDICTION_FILES: dict[str, str] = {
    name: f"{name}_predictions.parquet" for name in PREDICTION_NAMES
}
RUN_MANIFEST_FILE = "run_manifest.json"
RESOURCE_USAGE_FILE = "resource_usage.json"
MODEL_CARD_FILE = "model_card.json"
PREDICT_MANIFEST_FILE = "predict_manifest.json"

_FOLD_STRING_COLUMNS = (
    "gene_key",
    "cohort_id",
    "split_strategy",
    "role",
    "cluster_id",
    "split_build_hash",
)
_EXAMPLE_LIMIT = 5


class ContractError(ValueError):
    pass


class PredictionIdentityError(ContractError):
    pass


class TargetKind(Enum):

    BINARY = "binary"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class TargetSpec:

    column: str
    kind: TargetKind


BINARY_TARGET = TargetSpec(column="label", kind=TargetKind.BINARY)


def _examples(values: list[str]) -> str:
    shown = ", ".join(repr(value) for value in values[:_EXAMPLE_LIMIT])
    if len(values) > _EXAMPLE_LIMIT:
        shown += ", ..."
    return shown


def _require_exact_columns(
    frame: pd.DataFrame, expected: tuple[str, ...], table: str
) -> None:
    observed = list(frame.columns)
    missing = [name for name in expected if name not in observed]
    unexpected = sorted(set(observed) - set(expected))
    duplicated = sorted({name for name in observed if observed.count(name) > 1})
    if not (missing or unexpected or duplicated):
        return
    problems = []
    if missing:
        problems.append(f"missing {missing}")
    if unexpected:
        problems.append(f"unexpected {unexpected}")
    if duplicated:
        problems.append(f"duplicated {duplicated}")
    raise ContractError(
        f"{table}: expected exactly the columns {list(expected)}, "
        f"found {observed} ({'; '.join(problems)})"
    )


def _require_string_column(frame: pd.DataFrame, column: str, table: str) -> None:
    series = frame[column]
    dtype = series.dtype
    if not (isinstance(dtype, pd.StringDtype) or pandas_types.is_object_dtype(dtype)):
        raise ContractError(
            f"{table}: column '{column}' must hold strings, its dtype is {dtype}"
        )
    missing = int(series.isna().sum())
    if missing:
        raise ContractError(
            f"{table}: column '{column}' missing values: {missing}"
        )
    if pandas_types.is_object_dtype(dtype):
        offending = [
            f"{type(value).__name__}({value!r})"
            for value in series
            if not isinstance(value, str)
        ]
        if offending:
            raise ContractError(
                f"{table}: column '{column}' must hold strings, non-string values: "
                f"{len(offending)}, for example {_examples(offending)}"
            )


def _require_integer_column(frame: pd.DataFrame, column: str, table: str) -> None:
    series = frame[column]
    if not pandas_types.is_integer_dtype(series.dtype):
        raise ContractError(
            f"{table}: column '{column}' must be an integer column, "
            f"its dtype is {series.dtype}"
        )
    missing = int(series.isna().sum())
    if missing:
        raise ContractError(
            f"{table}: column '{column}' missing values: {missing}"
        )


def _require_float_column(frame: pd.DataFrame, column: str, table: str) -> None:
    series = frame[column]
    if not pandas_types.is_float_dtype(series.dtype):
        raise ContractError(
            f"{table}: column '{column}' must be a floating point column, "
            f"its dtype is {series.dtype}"
        )


def _require_float64_column(frame: pd.DataFrame, column: str, table: str) -> None:
    dtype = frame[column].dtype
    if dtype != np.dtype("float64"):
        raise ContractError(
            f"{table}: column '{column}' must be float64, its dtype is {dtype}. "
            f"A narrower float ties genes the method scored apart, which moves "
            f"Precision@k and the expected calibration error"
        )


def _float_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].to_numpy(dtype="float64", na_value=np.nan)


def validate_prediction_table(
    frame: pd.DataFrame,
    *,
    table: str = "prediction table",
    decision_threshold: float | None = None,
) -> None:
    _require_exact_columns(frame, PREDICTION_COLUMNS, table)
    _require_string_column(frame, "gene_key", table)
    keys = frame["gene_key"]
    duplicated = sorted(set(keys[keys.duplicated()]))
    if duplicated:
        raise ContractError(
            f"{table}: column 'gene_key' repeated values: {len(duplicated)}, "
            f"for example {_examples(duplicated)}"
        )
    _require_float64_column(frame, "y_score", table)
    scores = _float_values(frame, "y_score")
    nan_count = int(np.isnan(scores).sum())
    infinite_count = int(np.isinf(scores).sum())
    if nan_count or infinite_count:
        raise ContractError(
            f"{table}: column 'y_score' must be finite, NaN or missing: "
            f"{nan_count}, infinite: {infinite_count}"
        )
    _require_integer_column(frame, "y_pred", table)
    unknown = sorted(set(frame["y_pred"]) - BINARY_LABELS)
    if unknown:
        raise ContractError(
            f"{table}: column 'y_pred' takes values in {sorted(BINARY_LABELS)}, "
            f"found {unknown}"
        )
    if decision_threshold is None:
        return
    if not isinstance(decision_threshold, float) or not np.isfinite(
        decision_threshold
    ):
        raise ContractError(
            f"{table}: 'decision_threshold' must be a finite float, got "
            f"{decision_threshold!r}"
        )
    expected = (scores >= decision_threshold).astype("int64")
    disagreeing = np.flatnonzero(frame["y_pred"].to_numpy(dtype="int64") != expected)
    if disagreeing.size:
        rows = frame.iloc[disagreeing[:_EXAMPLE_LIMIT]]
        shown = [
            f"{key}: y_score {score!r}, y_pred {int(call)}, expected "
            f"{int(score >= decision_threshold)}"
            for key, score, call in rows[["gene_key", "y_score", "y_pred"]].itertuples(
                index=False
            )
        ]
        raise ContractError(
            f"{table}: column 'y_pred' must equal y_score >= "
            f"{decision_threshold!r}, disagreeing rows: {disagreeing.size} of "
            f"{len(frame)}, for example {_examples(shown)}"
        )


def validate_fold_table(
    frame: pd.DataFrame,
    *,
    target: TargetSpec | None = None,
    table: str = "fold table",
) -> None:
    if target is None:
        expected = FOLD_COLUMNS
    else:
        expected = (FOLD_COLUMNS[0], target.column, *FOLD_COLUMNS[1:])
    _require_exact_columns(frame, expected, table)
    for column in _FOLD_STRING_COLUMNS:
        _require_string_column(frame, column, table)
    _require_integer_column(frame, "outer_fold", table)
    unknown_roles = sorted(set(frame["role"]) - ROLES)
    if unknown_roles:
        raise ContractError(
            f"{table}: column 'role' takes values in {sorted(ROLES)}, "
            f"found {unknown_roles}"
        )
    build_hashes = sorted(set(frame["split_build_hash"]))
    if len(build_hashes) > 1:
        raise ContractError(
            f"{table}: column 'split_build_hash' identifies the build the split "
            f"came from, so one table carries one value, found {len(build_hashes)}: "
            f"{_examples(build_hashes)}. Rows from two builds in one table make "
            f"the fold membership of a gene untraceable to a single split"
        )
    repeated = frame.duplicated(subset=["gene_key", "outer_fold"])
    if repeated.any():
        offending = sorted(
            f"{key} in fold {fold}"
            for key, fold in frame.loc[repeated, ["gene_key", "outer_fold"]].itertuples(
                index=False
            )
        )
        raise ContractError(
            f"{table}: a gene_key appears at most once per outer_fold, repeated "
            f"combinations: {len(offending)}, for example "
            f"{_examples(offending)}. A repeat is either a duplicated row or one "
            f"gene holding two roles in one fold, which puts training genes into "
            f"the test set of that fold"
        )
    if target is None:
        return
    if target.kind is TargetKind.BINARY:
        _require_integer_column(frame, target.column, table)
        unknown_labels = sorted(set(frame[target.column]) - BINARY_LABELS)
        if unknown_labels:
            raise ContractError(
                f"{table}: binary target column '{target.column}' takes values in "
                f"{sorted(BINARY_LABELS)}, found {unknown_labels}"
            )
    else:
        _require_float_column(frame, target.column, table)
        infinite = int(np.isinf(_float_values(frame, target.column)).sum())
        if infinite:
            raise ContractError(
                f"{table}: continuous target column '{target.column}' must be "
                f"finite, infinite values: {infinite}. Missing values are not "
                f"checked here, how a task treats them belongs to the definition "
                f"of that task"
            )


def check_prediction_identity(
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    cohort_id: str,
    split_strategy: str,
    outer_fold: int,
    role: str,
) -> None:
    selectors = {
        "cohort_id": cohort_id,
        "split_strategy": split_strategy,
        "outer_fold": outer_fold,
        "role": role,
    }
    label = ", ".join(f"{name}={value!r}" for name, value in selectors.items())
    for column in ("gene_key", *selectors):
        if column not in folds.columns:
            raise ContractError(
                f"fold table: column '{column}' is needed to select a fold, "
                f"its columns are {list(folds.columns)}"
            )
    if "gene_key" not in predictions.columns:
        raise ContractError(
            f"prediction table: column 'gene_key' is missing, "
            f"its columns are {list(predictions.columns)}"
        )
    selection = pd.Series(True, index=folds.index)
    for column, value in selectors.items():
        selection &= folds[column].eq(value)
    expected = set(folds.loc[selection, "gene_key"])
    if not expected:
        raise PredictionIdentityError(
            f"fold table has no rows for {label}, so the prediction table cannot "
            f"be matched against it"
        )
    observed = set(predictions["gene_key"])
    surplus = sorted(observed - expected)
    absent = sorted(expected - observed)
    if not (surplus or absent):
        return
    raise PredictionIdentityError(
        f"prediction table does not match {label}: "
        f"{len(surplus)} gene_key not in that fold and role"
        f"{f' ({_examples(surplus)})' if surplus else ''}, "
        f"{len(absent)} gene_key of that fold and role missing from the "
        f"predictions{f' ({_examples(absent)})' if absent else ''}; "
        f"the fold holds {len(expected)} gene_key, the prediction table holds "
        f"{len(observed)}"
    )


def _parse_timestamp(value: object, key: str, source: str) -> datetime:
    if not isinstance(value, str):
        raise ContractError(
            f"{source}: '{key}' must be an ISO 8601 timestamp string, "
            f"got {type(value).__name__}"
        )
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as error:
        raise ContractError(
            f"{source}: '{key}' is not an ISO 8601 timestamp: {value!r}"
        ) from error
    if moment.tzinfo is None:
        raise ContractError(
            f"{source}: '{key}' must carry a time zone offset, got {value!r}"
        )
    return moment


def _require_object(
    payload: object, *, source: str, keys: tuple[str, ...], closed: bool
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ContractError(
            f"{source}: expected a JSON object, got {type(payload).__name__}"
        )
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ContractError(f"{source}: missing required keys {missing}")
    if closed:
        unexpected = sorted(set(payload) - set(keys))
        if unexpected:
            raise ContractError(
                f"{source}: expected exactly the keys {list(keys)}, "
                f"unexpected {unexpected}"
            )
    return payload


def _require_mapping(
    payload: Mapping[str, object], key: str, source: str
) -> Mapping[str, object]:
    value = payload[key]
    if not isinstance(value, Mapping):
        raise ContractError(
            f"{source}: '{key}' must be a JSON object, got {type(value).__name__}"
        )
    return value


def _require_text(payload: Mapping[str, object], key: str, source: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise ContractError(
            f"{source}: '{key}' must be a non-empty string, got {value!r}"
        )
    return value


def _require_choice(
    payload: Mapping[str, object],
    key: str,
    source: str,
    allowed: tuple[str, ...] | tuple[int, ...],
) -> object:
    value = payload[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, type(allowed[0]))
        or value not in allowed
    ):
        raise ContractError(
            f"{source}: '{key}' takes values in {list(allowed)}, got {value!r}"
        )
    return value


def _require_positive_int(payload: Mapping[str, object], key: str, source: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(
            f"{source}: '{key}' must be a positive integer, got {value!r}"
        )
    return value


def _require_finite_float(
    payload: Mapping[str, object], key: str, source: str
) -> float:
    value = payload[key]
    if not isinstance(value, float) or not np.isfinite(value):
        raise ContractError(
            f"{source}: '{key}' must be a finite float, got {value!r}"
        )
    return value


def _require_versions(payload: Mapping[str, object], source: str) -> None:
    versions = _require_mapping(payload, "software_versions", source)
    for name, version in versions.items():
        if not isinstance(name, str) or not isinstance(version, str):
            raise ContractError(
                f"{source}: 'software_versions' maps package names to version "
                f"strings, found {name!r}: {version!r}"
            )
    if "python" not in versions:
        raise ContractError(
            f"{source}: 'software_versions' must record the python version under "
            f"the key 'python'"
        )


RUN_IDENTITY_KEYS: tuple[str, ...] = (
    "method_id",
    "cohort_id",
    "split_strategy",
    "outer_fold",
    "split_build_hash",
)


def _require_run_identity(payload: Mapping[str, object], source: str) -> None:
    _require_text(payload, "method_id", source)
    _require_choice(payload, "cohort_id", source, COHORT_IDS)
    _require_choice(payload, "split_strategy", source, SPLIT_STRATEGIES)
    _require_choice(payload, "outer_fold", source, OUTER_FOLDS)
    _require_text(payload, "split_build_hash", source)


def validate_run_manifest(
    manifest: Mapping[str, object], *, source: str = "run manifest"
) -> None:
    _require_object(manifest, source=source, keys=RUN_MANIFEST_KEYS, closed=False)
    _require_run_identity(manifest, source)
    _require_versions(manifest, source)
    _require_mapping(manifest, "parameters", source)
    counts = _require_object(
        _require_mapping(manifest, "sample_counts", source),
        source=f"{source}: 'sample_counts'",
        keys=tuple(sorted(ROLES)),
        closed=True,
    )
    for role in sorted(ROLES):
        _require_positive_int(counts, role, f"{source}: 'sample_counts'")
    started = _parse_timestamp(manifest["started_at"], "started_at", source)
    finished = _parse_timestamp(manifest["finished_at"], "finished_at", source)
    if finished < started:
        raise ContractError(
            f"{source}: 'finished_at' ({manifest['finished_at']!r}) precedes "
            f"'started_at' ({manifest['started_at']!r})"
        )


EVALUATION_MANIFEST_KEYS: tuple[str, ...] = (
    "method_id",
    "cohort_id",
    "split_strategy",
    "outer_fold",
    "split_build_hash",
    "evaluator_version",
    "decision_threshold",
    "metrics",
)

INTERVAL_KEYS: tuple[str, ...] = ("point", "half_width", "undefined_fraction")


def validate_evaluation_manifest(
    manifest: Mapping[str, object], *, source: str = "evaluation manifest"
) -> None:
    _require_object(
        manifest, source=source, keys=EVALUATION_MANIFEST_KEYS, closed=True
    )
    _require_run_identity(manifest, source)
    _require_text(manifest, "evaluator_version", source)
    _require_finite_float(manifest, "decision_threshold", source)
    metrics = _require_mapping(manifest, "metrics", source)
    if not metrics:
        raise ContractError(f"{source}: 'metrics' holds no metric")
    for name, entry in metrics.items():
        if not isinstance(name, str) or not name:
            raise ContractError(
                f"{source}: 'metrics' maps metric names to intervals, found the "
                f"key {name!r}"
            )
        where = f"{source}: 'metrics' -> {name!r}"
        _require_object(entry, source=where, keys=INTERVAL_KEYS, closed=True)
        for key in INTERVAL_KEYS:
            value = entry[key]
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractError(
                    f"{where}: '{key}' must be a finite float or null, got "
                    f"{value!r}"
                )
            if not np.isfinite(value):
                raise ContractError(
                    f"{where}: '{key}' must be a finite float or null, got "
                    f"{value!r}; an undefined value is written as null"
                )
        width = entry["half_width"]
        if width is not None and width < 0.0:
            raise ContractError(
                f"{where}: 'half_width' is added to and subtracted from the "
                f"point and cannot be negative, got {width!r}"
            )
        share = entry["undefined_fraction"]
        if share is not None and not 0.0 <= share <= 1.0:
            raise ContractError(
                f"{where}: 'undefined_fraction' is a share of the resamples and "
                f"lies in [0, 1], got {share!r}"
            )


def validate_resource_usage(
    usage: Mapping[str, object], *, source: str = "resource usage"
) -> None:
    _require_object(usage, source=source, keys=RESOURCE_USAGE_KEYS, closed=True)
    seconds = _require_finite_float(usage, "cpu_seconds", source)
    if seconds < 0:
        raise ContractError(
            f"{source}: 'cpu_seconds' is a duration and cannot be negative, "
            f"got {seconds!r}"
        )
    _require_positive_int(usage, "peak_rss_bytes", source)
    _require_text(usage, "device", source)
    name = usage["gpu_name"]
    if name is not None and not (isinstance(name, str) and name):
        raise ContractError(
            f"{source}: 'gpu_name' must be a non-empty string, or null on a "
            f"method that ran on the CPU, got {name!r}"
        )
    memory = usage["peak_gpu_memory_bytes"]
    if not isinstance(memory, int) or isinstance(memory, bool) or memory < 0:
        raise ContractError(
            f"{source}: 'peak_gpu_memory_bytes' must be a non-negative integer, "
            f"zero on a method that ran on the CPU, got {memory!r}"
        )


def _require_relative_path(value: object, key: str, source: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(
            f"{source}: '{key}' must be a non-empty path string, got {value!r}"
        )
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ContractError(
            f"{source}: '{key}' must be relative, got {value!r}. An absolute "
            f"path names a file on the machine that trained the model and not a "
            f"file in what is published"
        )
    return value


def validate_model_card(
    card: Mapping[str, object], *, source: str = "model card"
) -> None:
    _require_object(card, source=source, keys=MODEL_CARD_KEYS, closed=False)
    _require_run_identity(card, source)
    _require_finite_float(card, "decision_threshold", source)
    _require_versions(card, source)
    artifacts = _require_object(
        _require_mapping(card, "artifacts", source),
        source=f"{source}: 'artifacts'",
        keys=("weights", "transforms"),
        closed=True,
    )
    _require_relative_path(artifacts["weights"], "artifacts.weights", source)
    transforms = artifacts["transforms"]
    if not isinstance(transforms, list):
        raise ContractError(
            f"{source}: 'artifacts.transforms' must be a list of paths, empty "
            f"when the method fitted nothing inside the fold, got "
            f"{type(transforms).__name__}"
        )
    for position, entry in enumerate(transforms):
        _require_relative_path(entry, f"artifacts.transforms[{position}]", source)
    assets = card["required_source_assets"]
    if not isinstance(assets, list):
        raise ContractError(
            f"{source}: 'required_source_assets' must be a list, empty when "
            f"inference reads nothing outside this directory, got "
            f"{type(assets).__name__}"
        )
    for position, entry in enumerate(assets):
        where = f"{source}: 'required_source_assets[{position}]'"
        asset = _require_object(
            entry, source=where, keys=("path", "purpose"), closed=True
        )
        _require_relative_path(asset["path"], "path", where)
        _require_text(asset, "purpose", where)


def read_prediction_table(
    path: str | Path, *, decision_threshold: float | None = None
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    validate_prediction_table(
        frame, table=str(path), decision_threshold=decision_threshold
    )
    return frame


def read_fold_table(
    path: str | Path, *, target: TargetSpec | None = None
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    validate_fold_table(frame, target=target, table=str(path))
    return frame


def _read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_run_manifest(path: str | Path) -> dict[str, object]:
    payload = _read_json(path)
    validate_run_manifest(payload, source=str(path))
    return payload


def read_resource_usage(path: str | Path) -> dict[str, object]:
    payload = _read_json(path)
    validate_resource_usage(payload, source=str(path))
    return payload


def read_model_card(path: str | Path) -> dict[str, object]:
    payload = _read_json(path)
    validate_model_card(payload, source=str(path))
    return payload


@dataclass(frozen=True)
class FoldProducts:

    directory: Path
    blind: pd.DataFrame
    validation: pd.DataFrame
    manifest: dict[str, object]
    resource_usage: dict[str, object]
    model_card: dict[str, object]

    @property
    def decision_threshold(self) -> float:
        return float(self.model_card["decision_threshold"])

    @property
    def identity(self) -> dict[str, object]:
        return {
            key: self.manifest[key]
            for key in ("cohort_id", "split_strategy", "outer_fold")
        }


def read_fold_products(directory: str | Path) -> FoldProducts:
    path = Path(directory)
    files = {
        "blind": PREDICTION_FILES["blind"],
        "validation": PREDICTION_FILES["validation"],
        "manifest": RUN_MANIFEST_FILE,
        "resource_usage": RESOURCE_USAGE_FILE,
        "model_card": MODEL_CARD_FILE,
    }
    absent = [name for name in files.values() if not (path / name).is_file()]
    if absent:
        raise ContractError(
            f"{path}: a fold of a method writes {sorted(files.values())}, "
            f"missing {absent}"
        )
    card = read_model_card(path / MODEL_CARD_FILE)
    manifest = read_run_manifest(path / RUN_MANIFEST_FILE)
    threshold = float(card["decision_threshold"])
    products = FoldProducts(
        directory=path,
        blind=read_prediction_table(
            path / PREDICTION_FILES["blind"], decision_threshold=threshold
        ),
        validation=read_prediction_table(
            path / PREDICTION_FILES["validation"], decision_threshold=threshold
        ),
        manifest=manifest,
        resource_usage=read_resource_usage(path / RESOURCE_USAGE_FILE),
        model_card=card,
    )
    disagreeing = [
        f"{key}: {RUN_MANIFEST_FILE} {manifest[key]!r}, "
        f"{MODEL_CARD_FILE} {card[key]!r}"
        for key in RUN_IDENTITY_KEYS
        if manifest[key] != card[key]
    ]
    if disagreeing:
        raise ContractError(
            f"{path}: the run manifest and the model card describe different "
            f"runs, {len(disagreeing)} of {len(RUN_IDENTITY_KEYS)} identifying "
            f"keys differ: {_examples(disagreeing)}"
        )
    counts: Mapping[str, object] = manifest["sample_counts"]
    scored = (
        ("test", "blind", products.blind),
        ("validation", "validation", products.validation),
    )
    miscounted = [
        f"{role}: {RUN_MANIFEST_FILE} says {counts[role]!r}, "
        f"{PREDICTION_FILES[name]} holds {len(frame)}"
        for role, name, frame in scored
        if counts[role] != len(frame)
    ]
    if miscounted:
        raise ContractError(
            f"{path}: the fold sizes the run manifest reports are not the "
            f"number of genes it scored: {_examples(miscounted)}"
        )
    shared = sorted(
        set(products.blind["gene_key"]) & set(products.validation["gene_key"])
    )
    if shared:
        raise ContractError(
            f"{path}: {len(shared)} gene_key is scored in both "
            f"{PREDICTION_FILES['blind']} and {PREDICTION_FILES['validation']}, "
            f"for example {_examples(shared)}. One gene holds one role in one "
            f"fold, so the two tables come from one run only if they are "
            f"disjoint"
        )
    return products
