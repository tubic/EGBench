#!/usr/bin/env python3
"""Build the 86,151 raw physicochemical candidate features of one task.
Run from methods/esstfnet: python build_raw_candidates.py --dataset <task>
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from build_features import (
    COHORT_IDS,
    SOURCE_ROOT,
    _raw_path,
    _read_support,
    _write_arrow_table_atomic,
)
from local_dataset import METHOD_ROOT, read_fasta

sys.path.insert(0, str(METHOD_ROOT / "code"))
from physicochemical import (
    RAW_CANDIDATE_DIMENSION,
    RawCandidateColumn,
    feature_vector,
    generate_raw_candidate_columns,
)


WORKERS = max(1, (os.cpu_count() or 1) - 4)
CHUNK_SIZE = 8
CHUNK_DIRECTORY_NAME = ".raw_candidate_chunks"
COLUMN_CONTRACT = SOURCE_ROOT / "feature_contract" / "raw_candidate_columns.parquet"


def _support_sequences(cohort_id: str) -> list[tuple[str, str]]:

    fasta = SOURCE_ROOT / cohort_id / "cds_sequences.fa.gz"
    if not fasta.is_file():
        raise FileNotFoundError(f"the coding sequences of {cohort_id} are absent: {fasta}")
    sequences = read_fasta(fasta)
    keys = sorted(sequences)
    support = set(_read_support()["gene_key"].astype(str))
    outside = sorted(set(keys).difference(support))
    if outside:
        raise ValueError(
            f"{fasta} holds {len(outside)} genes the support list does not: "
            f"{outside[:5]}"
        )
    return [(gene_key, sequences[gene_key]) for gene_key in keys]


def _chunk_path(chunk_directory: Path, chunk_index: int) -> Path:
    return chunk_directory / f"chunk-{chunk_index:08d}.parquet"

def _raw_output_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("gene_key", pa.string()),
            pa.field(
                "features",
                pa.list_(pa.float32(), list_size=RAW_CANDIDATE_DIMENSION),
            ),
        ]
    )

def _raw_contract_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("feature_index", pa.int32()),
            pa.field("feature_name", pa.string()),
            pa.field("feature_family", pa.string()),
            pa.field("w", pa.int8()),
            pa.field("k", pa.int8()),
        ]
    )

def _write_raw_table_atomic(
    gene_keys: list[str], vectors: list[np.ndarray], path: Path
) -> None:
    if len(gene_keys) != len(vectors) or not gene_keys:
        raise ValueError("raw feature table requires equally sized, non-empty gene and vector lists")
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (len(gene_keys), RAW_CANDIDATE_DIMENSION):
        raise ValueError(f"raw feature matrix has unexpected shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("raw feature matrix contains NaN or infinite values")
    feature_array = pa.FixedSizeListArray.from_arrays(
        pa.array(matrix.reshape(-1), type=pa.float32()), RAW_CANDIDATE_DIMENSION
    )
    table = pa.Table.from_arrays(
        [pa.array(gene_keys, type=pa.string()), feature_array],
        schema=_raw_output_schema(),
    )
    _write_arrow_table_atomic(table, path)

def _write_raw_candidate_contract(path: Path) -> None:
    columns: tuple[RawCandidateColumn, ...] = generate_raw_candidate_columns()
    table = pa.Table.from_arrays(
        [
            pa.array([column.feature_index for column in columns], type=pa.int32()),
            pa.array([column.feature_name for column in columns], type=pa.string()),
            pa.array([column.feature_family for column in columns], type=pa.string()),
            pa.array([column.w for column in columns], type=pa.int8()),
            pa.array([column.k for column in columns], type=pa.int8()),
        ],
        schema=_raw_contract_schema(),
    )
    _write_arrow_table_atomic(table, path)

def _write_raw_chunk_task(
    task: tuple[int, str, tuple[tuple[str, str], ...]]
) -> tuple[int, int]:
    chunk_index, path_text, rows = task
    gene_keys: list[str] = []
    vectors: list[np.ndarray] = []
    for gene_key, sequence in rows:
        try:
            vector = feature_vector(sequence)
        except Exception as error:
            raise ValueError(f"cannot extract raw features for gene_key={gene_key}: {error}") from error
        gene_keys.append(gene_key)
        vectors.append(vector)
    _write_raw_table_atomic(gene_keys, vectors, Path(path_text))
    return chunk_index, len(rows)

def _raw_chunk_is_complete(
    path: Path, expected_keys: list[str]
) -> bool:
    if not path.is_file():
        return False
    try:
        table = pq.read_table(path)
        if not table.schema.equals(_raw_output_schema(), check_metadata=False):
            return False
        if table.num_rows != len(expected_keys):
            return False
        keys = [str(value) for value in table.column("gene_key").to_pylist()]
        if keys != expected_keys or len(set(keys)) != len(keys):
            return False
        features = table.column("features").combine_chunks()
        if features.null_count or features.values.null_count:
            return False
        feature_values = np.asarray(features.values.to_numpy(zero_copy_only=False))
        if feature_values.size != len(expected_keys) * RAW_CANDIDATE_DIMENSION:
            return False
        if not np.isfinite(feature_values).all():
            return False
    except Exception:
        return False
    return True

def _merge_raw_chunks(
    chunk_paths: Iterable[Path], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    succeeded = False
    try:
        expected_schema = _raw_output_schema()
        for path in chunk_paths:
            table = pq.read_table(path)
            if not table.schema.equals(expected_schema, check_metadata=False):
                raise ValueError(f"chunk {path} has an unexpected raw-candidate schema")
            if writer is None:
                writer = pq.ParquetWriter(temporary, expected_schema)
            writer.write_table(table)
        if writer is None:
            raise ValueError("cannot merge an empty raw feature set")
        writer.close()
        writer = None
        succeeded = True
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            if not succeeded:
                temporary.unlink(missing_ok=True)
    os.replace(temporary, output_path)

def materialize_raw_candidates(
    cohort_id: str, *, workers: int = WORKERS, chunk_size: int = CHUNK_SIZE
) -> dict[str, object]:

    if workers <= 0 or chunk_size <= 0:
        raise ValueError("workers and chunk_size must be positive")
    rows = _support_sequences(cohort_id)
    slug = cohort_id
    output_path = _raw_path(SOURCE_ROOT, cohort_id)
    contract_path = COLUMN_CONTRACT
    _write_raw_candidate_contract(contract_path)
    chunk_directory = SOURCE_ROOT / slug / CHUNK_DIRECTORY_NAME
    chunk_directory.mkdir(parents=True, exist_ok=True)

    chunks: list[tuple[int, list[tuple[str, str]], Path]] = []
    for chunk_index, start in enumerate(range(0, len(rows), chunk_size)):
        chunk_rows = rows[start : start + chunk_size]
        chunks.append((chunk_index, chunk_rows, _chunk_path(chunk_directory, chunk_index)))

    tasks: list[tuple[int, str, tuple[tuple[str, str], ...]]] = []
    for chunk_index, chunk_rows, path in chunks:
        expected_keys = [gene_key for gene_key, _sequence in chunk_rows]
        if not _raw_chunk_is_complete(path, expected_keys):
            tasks.append((chunk_index, str(path), tuple(chunk_rows)))

    if tasks:
        if workers == 1:
            for task in tasks:
                _write_raw_chunk_task(task)
        else:
            try:
                executor = ProcessPoolExecutor(max_workers=workers)
            except PermissionError:
                print(
                    "multiprocessing is unavailable in this runtime; falling back to one CPU process",
                    file=sys.stderr,
                )
                for task in tasks:
                    _write_raw_chunk_task(task)
            else:
                with executor:
                    for _chunk_index, _row_count in executor.map(_write_raw_chunk_task, tasks):
                        pass

    chunk_paths = [path for _index, _rows, path in chunks]
    for chunk_index, chunk_rows, path in chunks:
        expected_keys = [gene_key for gene_key, _sequence in chunk_rows]
        if not _raw_chunk_is_complete(path, expected_keys):
            raise RuntimeError(f"raw feature chunk {chunk_index} was not completed: {path}")
    _merge_raw_chunks(chunk_paths, output_path)
    return {
        "dataset": cohort_id,
        "rows": len(rows),
        "dimensions": RAW_CANDIDATE_DIMENSION,
        "path": output_path.relative_to(METHOD_ROOT).as_posix(),
        "columns": contract_path.relative_to(METHOD_ROOT).as_posix(),
        "chunks": len(chunks),
        "resumed_chunks": len(chunks) - len(tasks),
        "resume_dir": chunk_directory.relative_to(METHOD_ROOT).as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=COHORT_IDS, required=True)
    arguments = parser.parse_args()
    print(materialize_raw_candidates(arguments.dataset))


if __name__ == "__main__":
    main()
