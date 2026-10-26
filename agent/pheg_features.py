"""Pheg's FV4,5 k-interval Z-curve features of a coding sequence, 4,545 per gene.

The variables are those of section 2.2 of Guo et al. (2017), doi:10.1093/bioinformatics/btx055;
feature selection and the classifier are part of the method, not of this module. The paper writes
``w`` for the longest adjacent context and ``lambda`` for the number of intervening bases, here ``k``.
For ``k > 0`` a pattern is a context of ``w`` bases, ``k`` skipped bases and a terminal base; for
``k = 0`` the context has ``w - 1`` bases and the terminal base completes the adjacent w-mer. Pattern
starts are three bases apart within each of the three coding phases.

In every group and phase the count of each pattern is divided by the number of valid patterns, those
whose context and terminal base are A, C, G or T; a pattern with N in them is left out of both, an
empty denominator gives zeros, and the skipped bases do not affect validity. The 19 groups are
(w, k) for w = 1..3 and k = 0..5, then (4, 0); the columns run over group, phase, context in A, C, G,
T order and the x, y, z axes of the Z curve.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DIMENSIONS = 4545
BASES = "ACGT"
AXES = ("x", "y", "z")
LEGAL_BASES = frozenset("ACGTN")
CODE = {base: index for index, base in enumerate(BASES)}
# x = (A + G) - (C + T), y = (A + C) - (G + T), z = (A + T) - (G + C)
Z_COEFFICIENTS = np.asarray(
    ((1.0, -1.0, 1.0, -1.0), (1.0, 1.0, -1.0, -1.0), (1.0, -1.0, -1.0, 1.0)), dtype=np.float32
)
ROW_GROUP_SIZE = 4096
QA_COLUMNS = (
    "gene_key",
    "cds_length_nt",
    "n_count",
    "candidate_windows_total",
    "valid_windows_total",
    "skipped_n_windows_total",
    "zero_denominator_phase_groups",
    "feature_count",
    "min_feature_value",
    "max_feature_value",
    "finite",
)


class Group(NamedTuple):
    w: int
    k: int
    context_length: int


GROUPS = tuple(
    Group(w, k, w - 1 if k == 0 else w) for w, k in ((w, k) for w in range(1, 4) for k in range(6))
) + (Group(4, 0, 3),)


def _contexts(length: int) -> tuple[str, ...]:
    return ("".join(bases) for bases in product(BASES, repeat=length)) if length else ("empty",)


FEATURE_NAMES = tuple(
    f"fv{group.w}_{group.k}.phase{phase}.context_{context}.{axis}"
    for group in GROUPS
    for phase in range(1, 4)
    for context in _contexts(group.context_length)
    for axis in AXES
)


def _phase_block(encoded: np.ndarray, group: Group, phase: int) -> tuple[np.ndarray, int, int]:
    """Z-curve values of one group and phase, with the numbers of candidate and valid patterns."""
    contexts = 4**group.context_length
    terminal = group.w - 1 if group.k == 0 else group.w + group.k
    stop = encoded.size - terminal
    if stop <= phase:
        return np.zeros(contexts * 3, dtype=np.float32), 0, 0
    starts = np.arange(phase, stop, 3, dtype=np.int64)
    terminal_codes = encoded[starts + terminal]
    valid = terminal_codes >= 0
    if group.context_length:
        context_codes = encoded[starts[:, None] + np.arange(group.context_length, dtype=np.int64)]
        valid &= np.all(context_codes >= 0, axis=1)
        weights = 4 ** np.arange(group.context_length - 1, -1, -1, dtype=np.int64)
        context_index = context_codes.astype(np.int64) @ weights
    else:
        context_index = np.zeros(starts.size, dtype=np.int64)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return np.zeros(contexts * 3, dtype=np.float32), int(starts.size), 0
    counts = np.bincount(context_index[valid] * 4 + terminal_codes[valid], minlength=contexts * 4)
    frequencies = counts.reshape(contexts, 4).astype(np.float32) / np.float32(valid_count)
    return _z_curve(frequencies).ravel(), int(starts.size), valid_count


def _z_curve(frequencies: np.ndarray) -> np.ndarray:
    """The product ``frequencies @ Z_COEFFICIENTS.T`` in float32, summed in the order of Apple Accelerate.

    The published matrix was computed with Accelerate, which sums the four terms of a one-row
    product as (A + G) + (C + T) and those of larger products from left to right; OpenBLAS sums
    them otherwise and changes the last bit of most genes. The terms are exact, so the order alone
    fixes the result on every platform.
    """
    terms = frequencies[:, None, :] * Z_COEFFICIENTS[None, :, :]
    if len(frequencies) == 1:
        return (terms[..., 0] + terms[..., 2]) + (terms[..., 1] + terms[..., 3])
    return ((terms[..., 0] + terms[..., 1]) + terms[..., 2]) + terms[..., 3]


def features(gene_key: str, cds: str) -> tuple[np.ndarray, dict]:
    """The 4,545 float32 values of one coding sequence and its QA record."""
    cds = "".join(cds.split()).upper().replace("U", "T")
    if not cds or not set(cds) <= LEGAL_BASES:
        raise ValueError(f"{gene_key}: the CDS must be non-empty and hold only A, C, G, T and N")
    encoded = np.fromiter((CODE.get(base, -1) for base in cds), dtype=np.int8, count=len(cds))
    blocks, candidates, valid, empty = [], 0, 0, 0
    for group in GROUPS:
        for phase in range(3):
            block, candidate_count, valid_count = _phase_block(encoded, group, phase)
            blocks.append(block)
            candidates += candidate_count
            valid += valid_count
            empty += valid_count == 0
    values = np.concatenate(blocks)
    finite = bool(np.isfinite(values).all())
    if not finite:
        raise ValueError(f"{gene_key}: non-finite Pheg features")
    qa = {
        "gene_key": gene_key,
        "cds_length_nt": len(cds),
        "n_count": cds.count("N"),
        "candidate_windows_total": candidates,
        "valid_windows_total": valid,
        "skipped_n_windows_total": candidates - valid,
        "zero_denominator_phase_groups": empty,
        "feature_count": DIMENSIONS,
        "min_feature_value": float(values.min()),
        "max_feature_value": float(values.max()),
        "finite": finite,
    }
    return values, qa


def _write(table: pa.Table, path: Path, **options) -> None:
    pq.write_table(
        table.replace_schema_metadata(None),
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
        version="2.6",
        **options,
    )


def write(cds_by_gene: dict[str, str], matrix_path: Path, qa_path: Path) -> None:
    """Write the gene by 4,545 float32 matrix and the per-gene QA table, rows sorted by gene key."""
    keys = sorted(cds_by_gene)
    rows, records = zip(*(features(key, cds_by_gene[key]) for key in keys))
    matrix = np.stack(rows)
    arrays = [pa.array(keys, type=pa.string())]
    arrays += [pa.array(np.ascontiguousarray(matrix[:, column]), type=pa.float32()) for column in range(DIMENSIONS)]
    _write(pa.Table.from_arrays(arrays, names=("gene_key", *FEATURE_NAMES)), matrix_path, row_group_size=ROW_GROUP_SIZE)
    _write(pa.Table.from_pylist([{column: record[column] for column in QA_COLUMNS} for record in records]), qa_path)
