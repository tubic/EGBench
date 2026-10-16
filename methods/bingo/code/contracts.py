"""Fold tables, partitions and file checks used by BiNGO training.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from egbench.contracts import COHORT_IDS


METHOD_ID = "bingo"
SPLIT_ASSIGNMENTS_RELATIVE = Path("split_assignments.parquet")


def training_labels_relative(outer_fold: int) -> Path:
    return Path(f"training_labels/fold_{outer_fold}.parquet")


_CONTEXT_COLUMNS = ("cohort_id", "split_strategy", "outer_fold")
_KEY_COLUMNS = (*_CONTEXT_COLUMNS, "role", "gene_key")


@dataclass(frozen=True)
class Partition:
    gene_keys: tuple[str, ...]
    cluster_ids: tuple[str, ...]
    labels: np.ndarray | None


@dataclass(frozen=True)
class PreparedSplit:
    train: Partition
    validation: Partition
    test: Partition
    split_build_hash: str
    label_build_hash: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_method_path(path: Path, label: str) -> Path:

    resolved = path.resolve()
    method_root = Path(__file__).resolve().parents[1]
    if not resolved.is_relative_to(method_root):
        raise ValueError(f"{label} must be located under {method_root}")
    return resolved


def read_fasta(path: Path) -> dict[str, str]:

    if not path.is_file():
        raise FileNotFoundError(path)
    records: dict[str, list[str]] = {}
    current: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                gene_key = line[1:].split("|", 1)[0].split()[0]
                if not gene_key or gene_key in records:
                    raise ValueError(
                        f"invalid or duplicate FASTA key at line {line_number}"
                    )
                records[gene_key] = []
                current = gene_key
            elif current is None:
                raise ValueError(
                    f"FASTA sequence precedes a header at line {line_number}"
                )
            else:
                records[current].append(line.upper())
    sequences = {key: "".join(parts) for key, parts in records.items()}
    if not sequences or any(not sequence for sequence in sequences.values()):
        raise ValueError("FASTA contains no records or an empty sequence")
    allowed = set("ACDEFGHIKLMNPQRSTVWYBXZUOJ")
    invalid = sorted(
        {aa for sequence in sequences.values() for aa in sequence if aa not in allowed}
    )
    if invalid:
        raise ValueError(f"FASTA contains unsupported residues: {invalid}")
    return sequences


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], table: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{table} lacks columns: {missing}")
    if not frame.columns.is_unique:
        raise ValueError(f"{table} has duplicate columns")


def _one_value(frame: pd.DataFrame, column: str, table: str) -> Any:
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"{table} requires one {column} value, found {len(values)}")
    return values[0]


def _context(
    frame: pd.DataFrame,
    cohort: str,
    split_strategy: str,
    outer_fold: int,
) -> pd.DataFrame:
    return frame.loc[
        frame["cohort_id"].astype(str).eq(cohort)
        & frame["split_strategy"].astype(str).eq(split_strategy)
        & pd.to_numeric(frame["outer_fold"], errors="coerce").eq(outer_fold)
    ].copy()


def _partition(frame: pd.DataFrame, labels: bool) -> Partition:
    ordered = frame.sort_values("gene_key", kind="mergesort")
    values = ordered["label"].to_numpy(dtype=np.int64, copy=True) if labels else None
    return Partition(
        gene_keys=tuple(ordered["gene_key"].astype(str)),
        cluster_ids=tuple(ordered["cluster_id"].astype(str)),
        labels=values,
    )


def validate_split_tables(
    assignments: pd.DataFrame,
    training_labels: pd.DataFrame,
    *,
    cohort: str,
    split_strategy: str,
    outer_fold: int,
) -> PreparedSplit:

    if cohort not in COHORT_IDS:
        raise ValueError(f"Bingo formal cohort must be one of {list(COHORT_IDS)}")
    _require_columns(
        assignments,
        (*_KEY_COLUMNS, "cluster_id", "split_build_hash"),
        "split assignments",
    )
    if "label" in assignments:
        raise ValueError("split assignments must be label-free")
    _require_columns(
        training_labels,
        (*_KEY_COLUMNS, "cluster_id", "label", "split_build_hash"),
        "training labels",
    )
    if not set(training_labels["role"].astype(str)).issubset({"train", "validation"}):
        raise ValueError("training labels may contain only train and validation roles")
    if assignments.duplicated(list(_KEY_COLUMNS)).any():
        raise ValueError("split assignments contain duplicate keys")
    if training_labels.duplicated(list(_KEY_COLUMNS)).any():
        raise ValueError("training labels contain duplicate keys")

    assigned = _context(assignments, cohort, split_strategy, outer_fold)
    labelled = _context(training_labels, cohort, split_strategy, outer_fold)
    if set(assigned["role"].astype(str)) != {"train", "validation", "test"}:
        raise ValueError("split context requires train, validation, and test roles")
    if set(labelled["role"].astype(str)) != {"train", "validation"}:
        raise ValueError("label context requires train and validation roles")

    compare = [*_KEY_COLUMNS, "cluster_id", "split_build_hash"]
    expected = assigned.loc[assigned["role"].ne("test"), compare]
    observed = labelled.loc[:, compare]
    coverage = expected.merge(
        observed,
        on=compare,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not coverage["_merge"].eq("both").all():
        raise ValueError(
            "training labels do not exactly cover assigned train and validation rows"
        )
    numeric = pd.to_numeric(labelled["label"], errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise ValueError("training labels must be binary 0/1")
    labelled["label"] = numeric.astype(np.int64)

    joined: dict[str, pd.DataFrame] = {}
    for role in ("train", "validation"):
        role_assigned = assigned.loc[
            assigned["role"].eq(role), ["gene_key", "cluster_id"]
        ]
        role_labels = labelled.loc[labelled["role"].eq(role), ["gene_key", "label"]]
        role_frame = role_assigned.merge(
            role_labels, on="gene_key", validate="one_to_one"
        )
        if set(role_frame["label"].unique()) != {0, 1}:
            raise ValueError(f"{role} partition must contain both binary classes")
        joined[role] = role_frame
    joined["test"] = assigned.loc[
        assigned["role"].eq("test"), ["gene_key", "cluster_id"]
    ].copy()
    if any(frame.empty for frame in joined.values()):
        raise ValueError("one or more split partitions are empty")

    gene_sets = {
        role: set(frame["gene_key"].astype(str)) for role, frame in joined.items()
    }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = gene_sets[left] & gene_sets[right]
        if overlap:
            raise ValueError(f"gene leakage between {left} and {right}: {len(overlap)}")
    if split_strategy == "family_5fold":
        clusters = {
            role: set(frame["cluster_id"].astype(str)) for role, frame in joined.items()
        }
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            overlap = clusters[left] & clusters[right]
            if overlap:
                raise ValueError(
                    f"family cluster leakage between {left} and {right}: {len(overlap)}"
                )

    split_hash = str(_one_value(assigned, "split_build_hash", "split context"))
    if str(_one_value(labelled, "split_build_hash", "label context")) != split_hash:
        raise ValueError("split build hashes disagree")
    label_hash = (
        str(_one_value(labelled, "label_build_hash", "label context"))
        if "label_build_hash" in labelled
        else None
    )
    return PreparedSplit(
        train=_partition(joined["train"], True),
        validation=_partition(joined["validation"], True),
        test=_partition(joined["test"], False),
        split_build_hash=split_hash,
        label_build_hash=label_hash,
    )


def load_prepared_split(
    folds_dir: Path,
    *,
    cohort: str,
    split_strategy: str,
    outer_fold: int,
) -> PreparedSplit:
    root = folds_dir.resolve()
    return validate_split_tables(
        pd.read_parquet(root / SPLIT_ASSIGNMENTS_RELATIVE),
        pd.read_parquet(root / training_labels_relative(outer_fold)),
        cohort=cohort,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )
