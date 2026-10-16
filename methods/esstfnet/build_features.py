#!/usr/bin/env python3
"""Write the EssTFNet fold view: train-role feature selection on the physicochemical candidates and the Evo2 embeddings.
Imported by run.py; not run directly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from local_dataset import METHOD_ROOT

REPO_ROOT = METHOD_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(METHOD_ROOT / "code"))
from egbench.contracts import SPLIT_STRATEGIES
from egbench.data import feature_directory, folds_directory
from physicochemical import RAW_CANDIDATE_DIMENSION
from fold_selection import (
    CANDIDATE_COUNT,
    RFE_STEP,
    SELECTED_COUNT,
    point_biserial_from_sufficient_statistics,
    select_ranked_outer_train_features,
    standardize_selected_matrix,
)


RAW_DIMENSION = RAW_CANDIDATE_DIMENSION

SOURCE_ROOT = METHOD_ROOT / "feature" / "source"
RAW_CANDIDATE_REBUILD = "python build_raw_candidates.py --dataset {dataset}"
EVO2_REBUILD = (
    "CUDA_VISIBLE_DEVICES=<physical_gpu> python code/embed_dna.py "
    "--dataset {dataset} --gene-list support_gene_list.csv --device cuda:0"
)


def _support_path() -> Path:
    return METHOD_ROOT / "support_gene_list.csv"

def _read_support() -> pd.DataFrame:
    path = _support_path()
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    if list(frame.columns) != ["dataset", "gene_key", "gene_symbol"]:
        raise ValueError(f"{path} must have dataset,gene_key,gene_symbol columns")
    selected = (
        frame.loc[:, ["gene_key", "gene_symbol"]]
        .drop_duplicates(subset="gene_key")
        .copy()
    )
    if selected.empty:
        raise ValueError(f"{path} holds no support genes")
    return selected.sort_values("gene_key", kind="mergesort").reset_index(drop=True)

def _fold_frame(
    split_strategy: str, cohort_id: str, outer_fold: int
) -> tuple[pd.DataFrame, str]:
    folds = folds_directory(
        cohort_id, split_strategy=split_strategy, root=REPO_ROOT
    )
    path = folds / "fold_genes" / f"fold_{outer_fold}_genes.parquet"
    frame = pd.read_parquet(path, columns=["gene_key", "gene_symbol", "role"])
    if frame["gene_key"].duplicated().any():
        raise ValueError(f"{path} is not the expected {cohort_id} fold")
    if set(frame["role"].astype(str)) != {"train", "validation", "test"}:
        raise ValueError(f"{path} lacks train/validation/test roles")
    assignments = pd.read_parquet(
        folds / "split_assignments.parquet",
        columns=["gene_key", "outer_fold", "split_strategy", "split_build_hash"],
    )
    scope = assignments.loc[
        assignments["outer_fold"].eq(outer_fold)
        & assignments["split_strategy"].astype(str).eq(split_strategy)
    ]
    hashes = set(scope["split_build_hash"].astype(str))
    if len(hashes) != 1 or set(scope["gene_key"].astype(str)) != set(frame["gene_key"].astype(str)):
        raise ValueError(f"split assignments disagree with {path}")
    return frame.sort_values("gene_key", kind="mergesort").reset_index(drop=True), hashes.pop()

def _write_json_atomic(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def _raw_path(root: Path, cohort_id: str) -> Path:
    return root / cohort_id / "raw_candidates_86151.parquet"

def _evo2_path(root: Path, cohort_id: str) -> Path:
    return root / cohort_id / "evo2_7b_base_embeddings.parquet"

def _require_fixed_matrix_source(
    path: Path,
    *,
    field: str,
    value_type: pa.DataType,
    dimensions: int,
    required_keys: set[str],
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = pa.schema(
        [
            pa.field("gene_key", pa.string()),
            pa.field(field, pa.list_(value_type, dimensions)),
        ]
    )
    schema = pq.ParquetFile(path).schema_arrow.remove_metadata()
    if schema != expected:
        raise ValueError(f"{path} has schema {schema}; expected {expected}")
    keys = [str(value) for value in pq.read_table(path, columns=["gene_key"])["gene_key"].to_pylist()]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path} contains duplicate gene_key values")
    missing = required_keys.difference(keys)
    if missing:
        raise ValueError(f"{path} lacks {len(missing)} required family gene_key values")

def _validate_family_assets(
    *,
    raw_path: Path,
    evo2_path: Path,
    contract_root: Path,
    required_keys: set[str],
) -> dict[int, str]:

    names = _feature_names(contract_root)
    _require_fixed_matrix_source(
        raw_path,
        field="features",
        value_type=pa.float32(),
        dimensions=RAW_DIMENSION,
        required_keys=required_keys,
    )
    _require_fixed_matrix_source(
        evo2_path,
        field="embedding",
        value_type=pa.float16(),
        dimensions=4096,
        required_keys=required_keys,
    )
    _validate_evo2_rows(evo2_path, required_keys)
    return names

def _stream_raw_point_biserial(
    path: Path,
    train_key_to_position: dict[str, int],
    train_labels: np.ndarray,
    *,
    required_keys: set[str] | None = None,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:

    labels = np.asarray(train_labels, dtype=np.float64).reshape(-1)
    if labels.size != len(train_key_to_position) or set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError("outer-train keys and binary labels are not aligned")
    feature_sum = np.zeros(RAW_DIMENSION, dtype=np.float64)
    feature_square_sum = np.zeros(RAW_DIMENSION, dtype=np.float64)
    feature_label_sum = np.zeros(RAW_DIMENSION, dtype=np.float64)
    found = np.zeros(labels.size, dtype=bool)
    required = set(train_key_to_position) if required_keys is None else set(required_keys)
    if not set(train_key_to_position).issubset(required):
        raise ValueError("outer-train keys must be included in required family keys")
    required_found: set[str] = set()
    if batch_size <= 0:
        raise ValueError("streaming batch size must be positive")
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["gene_key", "features"]):
        keys = [str(value) for value in batch.column(0).to_pylist()]
        required_positions = [index for index, key in enumerate(keys) if key in required]
        if not required_positions:
            continue
        values = np.asarray(batch.column(1).values.to_numpy(zero_copy_only=False), dtype=np.float32)
        values = values.reshape(len(keys), RAW_DIMENSION)
        if not np.isfinite(values[required_positions]).all():
            raise ValueError(f"{path} contains non-finite required family values")
        for index in required_positions:
            key = keys[index]
            if key in required_found:
                raise ValueError(f"{path} contains duplicate required gene_key values")
            required_found.add(key)
        positions = [index for index in required_positions if keys[index] in train_key_to_position]
        if not positions:
            continue
        selected = np.asarray(values[positions], dtype=np.float64)
        destinations = np.asarray([train_key_to_position[keys[index]] for index in positions], dtype=np.int64)
        if found[destinations].any():
            raise ValueError(f"{path} contains duplicate requested gene_key values")
        selected_labels = labels[destinations]
        feature_sum += selected.sum(axis=0, dtype=np.float64)
        feature_square_sum += np.einsum("ij,ij->j", selected, selected, dtype=np.float64)
        feature_label_sum += np.einsum(
            "ij,i->j", selected, selected_labels, dtype=np.float64
        )
        found[destinations] = True
    if not found.all():
        raise ValueError(f"{path} lacks {int((~found).sum())} outer-train gene_key values")
    if required_found != required:
        raise ValueError(f"{path} lacks {len(required.difference(required_found))} required family gene_key values")
    return point_biserial_from_sufficient_statistics(
        row_count=labels.size,
        label_sum=float(labels.sum()),
        feature_sum=feature_sum,
        feature_square_sum=feature_square_sum,
        feature_label_sum=feature_label_sum,
    )

def _validate_evo2_rows(path: Path, required_keys: set[str]) -> None:
    found: set[str] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=128, columns=["gene_key", "embedding"]):
        keys = [str(value) for value in batch.column(0).to_pylist()]
        positions = [index for index, key in enumerate(keys) if key in required_keys]
        if not positions:
            continue
        values = np.asarray(batch.column(1).values.to_numpy(zero_copy_only=False), dtype=np.float16)
        values = values.reshape(len(keys), 4096)
        if not np.isfinite(values[positions]).all():
            raise ValueError(f"{path} contains non-finite required family values")
        for index in positions:
            key = keys[index]
            if key in found:
                raise ValueError(f"{path} contains duplicate required gene_key values")
            found.add(key)
    if found != required_keys:
        raise ValueError(f"{path} lacks {len(required_keys.difference(found))} required family gene_key values")

def _scan_raw_rows(path: Path, wanted: dict[str, int], columns: np.ndarray | None = None) -> np.ndarray:
    width = RAW_DIMENSION if columns is None else int(columns.size)
    output = np.empty((len(wanted), width), dtype=np.float32)
    found = np.zeros(len(wanted), dtype=bool)
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=64, columns=["gene_key", "features"]):
        keys = [str(value) for value in batch.column(0).to_pylist()]
        positions = [index for index, key in enumerate(keys) if key in wanted]
        if not positions:
            continue
        values = np.asarray(batch.column(1).values.to_numpy(zero_copy_only=False), dtype=np.float32)
        values = values.reshape(len(keys), RAW_DIMENSION)
        for index in positions:
            destination = wanted[keys[index]]
            if found[destination]:
                raise ValueError(f"{path} contains duplicate requested gene_key values")
            output[destination] = values[index] if columns is None else values[index, columns]
            found[destination] = True
    if not found.all():
        raise ValueError(f"{path} lacks {int((~found).sum())} requested gene_key values")
    return output

def _scan_evo2_rows(path: Path, wanted: dict[str, int]) -> np.ndarray:
    output = np.empty((len(wanted), 4096), dtype=np.float16)
    found = np.zeros(len(wanted), dtype=bool)
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=128, columns=["gene_key", "embedding"]):
        keys = [str(value) for value in batch.column(0).to_pylist()]
        positions = [index for index, key in enumerate(keys) if key in wanted]
        if not positions:
            continue
        values = np.asarray(batch.column(1).values.to_numpy(zero_copy_only=False), dtype=np.float16)
        values = values.reshape(len(keys), 4096)
        for index in positions:
            destination = wanted[keys[index]]
            if found[destination]:
                raise ValueError(f"{path} contains duplicate requested gene_key values")
            output[destination] = values[index]
            found[destination] = True
    if not found.all():
        raise ValueError(f"{path} lacks {int((~found).sum())} requested gene_key values")
    return output

def _write_arrow_table_atomic(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def _write_fixed_matrix(path: Path, keys: list[str], field: str, matrix: np.ndarray, value_type: pa.DataType) -> None:
    if matrix.ndim != 2 or len(keys) != matrix.shape[0] or not np.isfinite(matrix).all():
        raise ValueError("output feature matrix is malformed")
    array = pa.FixedSizeListArray.from_arrays(
        pa.array(matrix.reshape(-1), type=value_type), matrix.shape[1]
    )
    _write_arrow_table_atomic(
        pa.Table.from_arrays([pa.array(keys, type=pa.string()), array], names=["gene_key", field]), path
    )

def _write_scaler_state_atomic(
    path: Path,
    *,
    selected_indices: np.ndarray,
    selected_mean: np.ndarray,
    selected_scale: np.ndarray,
) -> None:
    if (
        selected_indices.ndim != 1
        or selected_mean.shape != selected_indices.shape
        or selected_scale.shape != selected_indices.shape
        or not np.isfinite(selected_mean).all()
        or not np.isfinite(selected_scale).all()
        or np.any(selected_scale <= 0.0)
    ):
        raise ValueError("selected scaler state is malformed")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                selected_raw_indices=np.asarray(selected_indices, dtype=np.int64),
                mean=np.asarray(selected_mean, dtype=np.float64),
                scale=np.asarray(selected_scale, dtype=np.float64),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def _feature_names(contract_root: Path) -> dict[int, str]:
    path = contract_root / "feature_contract" / "raw_candidate_columns.parquet"
    table = pd.read_parquet(path, columns=["feature_index", "feature_name"])
    indices = table["feature_index"].astype(int)
    if (
        len(table) != RAW_DIMENSION
        or indices.duplicated().any()
        or indices.tolist() != list(range(RAW_DIMENSION))
    ):
        raise ValueError(f"{path} does not define {RAW_DIMENSION} unique raw columns")
    return dict(zip(indices, table["feature_name"].astype(str), strict=True))

def build_fold(
    cohort_id: str, outer_fold: int, *, split_strategy: str
) -> dict[str, object]:

    source_root = SOURCE_ROOT
    contract_root = SOURCE_ROOT
    fold, split_build_hash = _fold_frame(split_strategy, cohort_id, outer_fold)
    support = _read_support()
    keys = fold["gene_key"].astype(str).tolist()
    if not set(keys).issubset(set(support["gene_key"].astype(str))):
        raise ValueError("the fold is outside the method support list")
    labels_path = (
        folds_directory(
            cohort_id,
            split_strategy=split_strategy,
            root=REPO_ROOT,
        )
        / "training_labels"
        / f"fold_{outer_fold}.parquet"
    )
    labels = pd.read_parquet(
        labels_path,
        columns=["gene_key", "outer_fold", "split_strategy", "role", "label"],
        filters=[
            ("outer_fold", "=", outer_fold),
            ("split_strategy", "=", split_strategy),
            ("role", "=", "train"),
        ],
    )
    if not (
        labels["outer_fold"].eq(outer_fold).all()
        and labels["split_strategy"].astype(str).eq(split_strategy).all()
        and labels["role"].astype(str).eq("train").all()
    ):
        raise ValueError("training-label predicate returned rows outside the requested outer-train split")
    if labels["gene_key"].duplicated().any():
        raise ValueError("outer-train labels contain duplicate gene_key values")
    train_keys = sorted(labels["gene_key"].astype(str).tolist())
    expected_train_keys = set(fold.loc[fold["role"].astype(str).eq("train"), "gene_key"].astype(str))
    if set(train_keys) != expected_train_keys:
        raise ValueError("outer-train labels disagree with authority fold roles")
    train_labels = labels.set_index(labels["gene_key"].astype(str)).loc[train_keys, "label"].to_numpy(dtype=np.int64)
    if set(train_labels) != {0, 1}:
        raise ValueError("outer-train labels must contain both classes")
    raw = _raw_path(source_root, cohort_id)
    evo2 = _evo2_path(source_root, cohort_id)
    absent = [path for path in (raw, evo2) if not path.is_file()]
    if absent:
        raise FileNotFoundError(
            "the whole-genome EssTFNet assets this fold reads are absent: "
            + ", ".join(str(path) for path in absent)
            + ". They are built once by the rebuild_command entries of "
            + f"method.yaml: '{RAW_CANDIDATE_REBUILD.format(dataset=cohort_id)}' "
            + f"and '{EVO2_REBUILD.format(dataset=cohort_id)}'."
        )
    names = _validate_family_assets(
        raw_path=raw,
        evo2_path=evo2,
        contract_root=contract_root,
        required_keys=set(keys),
    )
    selection_keys = [str(key) for key in train_keys]
    selection_labels = np.asarray(train_labels, dtype="int64").reshape(-1)
    selection_lookup = {key: index for index, key in enumerate(selection_keys)}
    ranking, correlations = _stream_raw_point_biserial(
        raw,
        selection_lookup,
        selection_labels,
        required_keys=set(keys),
    )
    candidates = ranking[:CANDIDATE_COUNT]
    candidate_matrix = _scan_raw_rows(raw, selection_lookup, candidates)
    selected, candidate_mean, candidate_scale, selector = select_ranked_outer_train_features(
        candidate_matrix,
        selection_labels,
        candidates,
        correlations[candidates],
        selected_count=SELECTED_COUNT,
        step=RFE_STEP,
    )
    del candidate_matrix
    selected_positions = np.asarray(selector["selected_candidate_positions"], dtype=np.int64)
    fold_raw = _scan_raw_rows(raw, {key: index for index, key in enumerate(keys)}, selected)
    fold_matrix, selected_mean, selected_scale = standardize_selected_matrix(
        fold_raw,
        selected_positions,
        candidate_mean,
        candidate_scale,
    )
    output_dir = feature_directory(
        METHOD_ROOT,
        cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    physicochemical_path = output_dir / "physicochemical_features.parquet"
    scaler_state_path = output_dir / "physicochemical_scaler_state.npz"
    evo2_output = output_dir / "evo2_embeddings.parquet"
    manifest_path = output_dir / "feature_selection_manifest.json"
    _write_fixed_matrix(physicochemical_path, keys, "features", fold_matrix, pa.float32())
    _write_scaler_state_atomic(
        scaler_state_path,
        selected_indices=selected,
        selected_mean=selected_mean,
        selected_scale=selected_scale,
    )
    canonical_evo2 = (
        feature_directory(
            METHOD_ROOT,
            cohort_id,
            split_strategy=SPLIT_STRATEGIES[0],
            outer_fold=0,
        )
        / "evo2_embeddings.parquet"
    )
    if evo2_output != canonical_evo2 and canonical_evo2.is_file():
        os.link(canonical_evo2, evo2_output)
        evo2_linked = True
    else:
        evo2_matrix = _scan_evo2_rows(evo2, {key: index for index, key in enumerate(keys)})
        _write_fixed_matrix(evo2_output, keys, "embedding", evo2_matrix, pa.float16())
        evo2_linked = False
    manifest = {
        "schema_version": "1.0",
        "adaptation": "source-release information is insufficient for exact published grouped RFE; this is an adapted outer-train-safe selector",
        "dataset": cohort_id,
        "outer_fold": outer_fold,
        "split_strategy": split_strategy,
        "split_build_hash": split_build_hash,
        "train_gene_count": len(train_keys),
        "train_class_counts": {"nonessential": int((train_labels == 0).sum()), "essential": int((train_labels == 1).sum())},
        "raw_feature_dimension": RAW_DIMENSION,
        "candidate_selection": {"criterion": "descending absolute point-biserial correlation", "tie_break": "ascending feature_index", "count": CANDIDATE_COUNT},
        "scaler": {
            "estimator": "StandardScaler",
            "fit_role": "train",
            "fit_scope": "outer-train candidate columns",
            "output_transform": "selected physicochemical columns are standardized with outer-train mean and scale",
            "state": scaler_state_path.relative_to(METHOD_ROOT).as_posix(),
        },
        "rfe": {"estimator": "LogisticRegression(liblinear,l2,C=1,class_weight=balanced)", "n_features_to_select": SELECTED_COUNT, "step": RFE_STEP, "max_iter": 2000, "tol": 0.0001, "convergence_warning": "error"},
        "candidate_raw_indices_ranked": selector["candidate_indices_ranked"],
        "candidate_raw_feature_names_ranked": [names[int(index)] for index in selector["candidate_indices_ranked"]],
        "selected_raw_indices_ordered": selected.tolist(),
        "selected_raw_feature_names_ordered": [names[int(index)] for index in selected],
        "physicochemical_output": physicochemical_path.relative_to(METHOD_ROOT).as_posix(),
        "physicochemical_output_schema": "gene_key:string,features:fixed_size_list<float32>[4250]",
        "evo2_output": evo2_output.relative_to(METHOD_ROOT).as_posix(),
        "evo2_output_schema": "gene_key:string,embedding:fixed_size_list<float16>[4096]",
        "evo2_hardlinked": evo2_linked,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest
