#!/usr/bin/env python3
"""Write the Mahi fold view of the static embedding matrix.
Imported by run.py; not run directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from local_dataset import METHOD_ROOT, method_path

REPO_ROOT = METHOD_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from egbench.contracts import SPLIT_STRATEGIES
from egbench.data import feature_directory, folds_directory


METHOD_ID = "mahi"
FORMAL_EMBEDDING_ROWS = 19_073
FORMAL_EMBEDDING_DIMENSIONS = 3_666
EMBEDDING_COLUMNS = ("entrez_id", "tissue", "embedding")


def _support_path() -> Path:
    return METHOD_ROOT / "support_gene_list.csv"


def _support_keys() -> set[str]:
    support = pd.read_csv(_support_path(), dtype="string", keep_default_na=False)
    if list(support.columns) != ["dataset", "gene_key", "gene_symbol"]:
        raise ValueError("support_gene_list.csv must have dataset,gene_key,gene_symbol")
    keys = support["gene_key"].astype(str)
    if keys.empty:
        raise ValueError("support_gene_list.csv is empty")
    return set(keys)


def _read_fold_keys(
    cohort_id: str, outer_fold: int, *, split_strategy: str
) -> tuple[list[str], str]:
    folds = folds_directory(
        cohort_id, split_strategy=split_strategy, root=REPO_ROOT
    )
    path = folds / "fold_genes" / f"fold_{outer_fold}_genes.parquet"
    frame = pd.read_parquet(path, columns=["gene_key", "role"])
    keys = frame["gene_key"].astype(str).tolist()
    if (
        len(set(keys)) != len(keys)
        or set(frame["role"].astype(str)) != {"train", "validation", "test"}
    ):
        raise ValueError(f"{path} is not a complete fold")
    assignments = pd.read_parquet(
        folds / "split_assignments.parquet",
        columns=["gene_key", "outer_fold", "split_strategy", "split_build_hash"],
    )
    scope = assignments.loc[
        assignments["outer_fold"].eq(outer_fold)
        & assignments["split_strategy"].astype(str).eq(split_strategy)
    ]
    hashes = set(scope["split_build_hash"].astype(str))
    if len(hashes) != 1 or set(scope["gene_key"].astype(str)) != set(keys):
        raise ValueError(f"split assignments disagree with {path}")
    return sorted(keys), hashes.pop()


def _write_fixed_embedding(path: Path, keys: list[str], vectors: np.ndarray) -> None:
    if vectors.shape != (len(keys), FORMAL_EMBEDDING_DIMENSIONS) or not np.isfinite(vectors).all():
        raise ValueError("Mahi fold matrix has invalid dimensions or non-finite values")
    table = pa.Table.from_arrays(
        [
            pa.array(keys, type=pa.string()),
            pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1), type=pa.float16()), FORMAL_EMBEDDING_DIMENSIONS
            ),
        ],
        names=["gene_key", "embedding"],
    )
    pq.write_table(table, path)


def _read_static_source(path: Path) -> tuple[list[str], np.ndarray]:
    table = pq.read_table(path, columns=["gene_key", "embedding"])
    if table.schema.field("gene_key").type != pa.string():
        raise ValueError(f"{path} requires string gene_key values")
    embeddings = table["embedding"].combine_chunks()
    embedding_type = embeddings.type
    if pa.types.is_fixed_size_list(embedding_type):
        valid_type = (
            embedding_type.list_size == FORMAL_EMBEDDING_DIMENSIONS
            and embedding_type.value_type == pa.float16()
        )
    elif pa.types.is_list(embedding_type):
        lengths = np.asarray(pc.list_value_length(embeddings).to_numpy(zero_copy_only=False))
        valid_type = embedding_type.value_type == pa.float16() and bool(
            np.all(lengths == FORMAL_EMBEDDING_DIMENSIONS)
        )
    else:
        valid_type = False
    if not valid_type or embeddings.null_count or embeddings.values.null_count:
        raise ValueError(f"{path} requires non-null float16 vectors of length {FORMAL_EMBEDDING_DIMENSIONS}")
    keys = [str(value) for value in table["gene_key"].to_pylist()]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path} contains duplicate gene_key values")
    values = np.asarray(embeddings.values.to_numpy(zero_copy_only=False), dtype=np.float16)
    values = values.reshape(len(keys), FORMAL_EMBEDDING_DIMENSIONS)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite embeddings")
    return keys, values


def _normalise_entrez(value: Any) -> str:
    if value is None or pd.isna(value):
        raise ValueError("embedding source contains a missing entrez_id")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"invalid numeric entrez_id: {value!r}")
        return str(int(value))
    text = str(value).strip()
    if not text.isdigit():
        raise ValueError(f"invalid entrez_id: {value!r}")
    return text


def _normalise_embedding(value: Any, row_number: int) -> np.ndarray:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError(f"embedding source row {row_number} has a missing embedding")
    try:
        vector = np.asarray(value, dtype=np.float16)
    except (TypeError, ValueError) as error:
        raise ValueError(f"embedding source row {row_number} has a non-numeric embedding") from error
    if vector.ndim != 1 or vector.shape[0] != FORMAL_EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"embedding source row {row_number} has dimension {vector.shape}, "
            f"expected ({FORMAL_EMBEDDING_DIMENSIONS},)"
        )
    if not np.isfinite(vector).all():
        raise ValueError(f"embedding source row {row_number} contains a non-finite value")
    return vector


def validate_embedding_source(
    source: Path,
    *,
    expected_rows: int = FORMAL_EMBEDDING_ROWS,
    expected_dimensions: int = FORMAL_EMBEDDING_DIMENSIONS,
) -> pd.DataFrame:
    source = method_path(source, label="embedding source")
    if not source.is_file():
        raise FileNotFoundError(
            f"Mahi embedding source is absent: {source}. "
            "Run acquire_assets.py to obtain the official asset."
        )
    frame = pd.read_parquet(source)
    if list(frame.columns).count("entrez_id") != 1 or frame.columns.duplicated().any():
        raise ValueError(f"{source} has duplicate or malformed column names")
    if set(frame.columns) != set(EMBEDDING_COLUMNS):
        raise ValueError(
            f"{source} must have exactly columns {list(EMBEDDING_COLUMNS)}, "
            f"found {list(frame.columns)}"
        )
    if len(frame) != expected_rows:
        raise ValueError(
            f"{source} has {len(frame)} rows; official Mahi global source requires "
            f"{expected_rows}"
        )
    tissue_values = frame["tissue"].astype("string")
    if tissue_values.isna().any() or tissue_values.eq("").any():
        raise ValueError(f"{source} contains missing or empty tissue values")
    if tissue_values.ne("global").any():
        bad_tissues = sorted(frame.loc[tissue_values.ne("global"), "tissue"].astype(str).unique())
        raise ValueError(f"{source} contains tissue values other than global: {bad_tissues}")

    entrez_ids = [_normalise_entrez(value) for value in frame["entrez_id"].tolist()]
    if len(set(entrez_ids)) != len(entrez_ids):
        raise ValueError(f"{source} contains duplicate Entrez IDs")
    embeddings = [
        _normalise_embedding(value, row_number)
        for row_number, value in enumerate(frame["embedding"].tolist(), start=1)
    ]
    return pd.DataFrame(
        {
            "entrez_id": pd.Series(entrez_ids, dtype="string"),
            "tissue": tissue_values.reset_index(drop=True),
            "embedding": embeddings,
        }
    )


def write_embedding_source(path: Path, frame: pd.DataFrame) -> None:
    path = method_path(path, label="embedding source")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "entrez_id": pa.array(frame["entrez_id"].tolist(), type=pa.string()),
            "tissue": pa.array(frame["tissue"].tolist(), type=pa.string()),
            "embedding": pa.array(
                [vector.tolist() for vector in frame["embedding"]],
                type=pa.list_(pa.float16()),
            ),
        }
    )
    pq.write_table(table, path)


def build_family_fold(
    cohort_id: str, outer_fold: int, *, split_strategy: str
) -> dict[str, object]:
    directory = feature_directory(
        METHOD_ROOT,
        cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )
    output = directory / "features.parquet"
    if output.is_file():
        return {
            "status": "already materialized",
            "output": output.relative_to(METHOD_ROOT).as_posix(),
        }
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"{directory} holds something other than features.parquet; remove it "
            f"to build this fold again"
        )

    source = METHOD_ROOT / "feature" / "source" / cohort_id / "features.parquet"
    if not source.is_file():
        raise FileNotFoundError(
            f"the Mahi static matrix is absent: {source}. It is a source "
            f"asset of this method under feature/source"
        )
    source_keys, values = _read_static_source(source)
    lookup = {key: index for index, key in enumerate(source_keys)}
    support = _support_keys()
    keys, split_build_hash = _read_fold_keys(
        cohort_id, outer_fold, split_strategy=split_strategy
    )
    if not set(keys).issubset(support):
        raise ValueError("family fold is outside Mahi support")
    missing = set(keys).difference(lookup)
    if missing:
        raise ValueError(f"{source} lacks {len(missing)} family genes")

    directory.mkdir(parents=True, exist_ok=True)
    canonical = (
        feature_directory(
            METHOD_ROOT,
            cohort_id,
            split_strategy=SPLIT_STRATEGIES[0],
            outer_fold=0,
        )
        / "features.parquet"
    )
    if output != canonical and canonical.is_file():
        os.link(canonical, output)
        hardlinked = True
    else:
        _write_fixed_embedding(output, keys, values[[lookup[key] for key in keys]])
        hardlinked = False
    return {
        "dataset": cohort_id,
        "outer_fold": outer_fold,
        "split_strategy": split_strategy,
        "rows": len(keys),
        "embedding_dimensions": FORMAL_EMBEDDING_DIMENSIONS,
        "split_build_hash": split_build_hash,
        "construction": "static official Mahi global embedding subset by gene_key; no labels used",
        "source": source.relative_to(METHOD_ROOT).as_posix(),
        "output": output.relative_to(METHOD_ROOT).as_posix(),
        "hardlinked_from_fold_0": hardlinked,
    }
