"""Load one eDNN-EG fold: sequences, labels and roles."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .errors import ContractViolation
from .io import read_fasta


EXPECTED_SEQUENCE_CORE_ROWS = 5_363
_CONTEXT_COLUMNS = {"gene_key", "cohort_id", "split_strategy", "outer_fold", "role"}
_ASSIGNMENT_COLUMNS = _CONTEXT_COLUMNS | {"cluster_id"}
_TRAINING_LABEL_COLUMNS = _ASSIGNMENT_COLUMNS | {"label"}


@dataclass(frozen=True)
class FrozenPartition:
    gene_keys: tuple[str, ...]
    cluster_ids: tuple[str, ...]
    sequences: tuple[str, ...]
    labels: tuple[int, ...] | None

    def sequence_map(self) -> dict[str, str]:
        return dict(zip(self.gene_keys, self.sequences, strict=True))

    def label_map(self) -> dict[str, int]:
        if self.labels is None:
            raise ContractViolation("labels are unavailable for the sealed test partition")
        return dict(zip(self.gene_keys, self.labels, strict=True))

    def group_map(self) -> dict[str, str]:
        return dict(zip(self.gene_keys, self.cluster_ids, strict=True))


@dataclass(frozen=True)
class FrozenSplit:
    cohort_id: str
    split_strategy: str
    outer_fold: int
    split_build_hash: str | None
    train: FrozenPartition
    validation: FrozenPartition
    test: FrozenPartition
    outer_audit: dict[str, object]


def _expected_sequence_core_rows(split_assignments_path: str | Path) -> int:
    source = Path(split_assignments_path).resolve()
    dataset_root = source.parents[2]
    if not (dataset_root / "config" / "dataset_profile.yaml").is_file():
        return EXPECTED_SEQUENCE_CORE_ROWS
    audit_path = dataset_root / "processed" / "cohorts" / "cohort_audit.parquet"
    if not audit_path.is_file():
        raise ContractViolation("profile cohort audit is missing")
    audit = pd.read_parquet(audit_path)
    selected = audit.loc[
        audit["cohort_id"].astype(str).eq("sequence_core"), "n_genes"
    ]
    if len(selected) != 1 or int(selected.iloc[0]) < 1:
        raise ContractViolation("profile sequence_core cohort audit is invalid")
    return int(selected.iloc[0])


def _require_columns(frame: pd.DataFrame, required: set[str], table: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ContractViolation(f"{table} is missing columns: {sorted(missing)}")


def _select_context(
    frame: pd.DataFrame,
    *,
    cohort_id: str,
    split_strategy: str,
    outer_fold: int,
    table: str,
) -> pd.DataFrame:
    numeric_fold = pd.to_numeric(frame["outer_fold"], errors="coerce")
    selected = frame.loc[
        frame["cohort_id"].astype(str).eq(cohort_id)
        & frame["split_strategy"].astype(str).eq(split_strategy)
        & numeric_fold.eq(outer_fold)
    ].copy()
    if selected.empty:
        raise ContractViolation(
            f"{table} has no rows for {cohort_id}/{split_strategy}/fold_{outer_fold}"
        )
    selected["gene_key"] = selected["gene_key"].astype(str)
    selected["cluster_id"] = selected["cluster_id"].astype(str)
    selected["role"] = selected["role"].astype(str)
    if selected["gene_key"].duplicated().any():
        raise ContractViolation(f"{table} contains duplicate context gene_key rows")
    if selected["gene_key"].eq("").any() or selected["cluster_id"].eq("").any():
        raise ContractViolation(f"{table} contains empty gene or cluster identifiers")
    return selected


def _single_optional_value(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame:
        return None
    values = sorted(set(frame[column].dropna().astype(str)))
    if len(values) != 1:
        raise ContractViolation(f"split context must contain exactly one {column}")
    return values[0]


def _assert_outer_isolation(
    partitions: dict[str, pd.DataFrame], split_strategy: str
) -> dict[str, object]:
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    gene_overlap: dict[str, int] = {}
    cluster_overlap: dict[str, int] = {}
    for left, right in pairs:
        name = f"{left}__{right}"
        genes = set(partitions[left]["gene_key"]) & set(partitions[right]["gene_key"])
        gene_overlap[name] = len(genes)
        if genes:
            raise ContractViolation(f"outer gene leakage between {left} and {right}")
        clusters = set(partitions[left]["cluster_id"]) & set(
            partitions[right]["cluster_id"]
        )
        cluster_overlap[name] = len(clusters)
        if split_strategy == "family_5fold" and clusters:
            raise ContractViolation(
                f"outer family cluster leakage between {left} and {right}"
            )
    return {
        "gene_overlap_counts": gene_overlap,
        "cluster_overlap_counts": cluster_overlap,
        "family_cluster_isolation_enforced": split_strategy == "family_5fold",
    }


def _materialize_partition(
    frame: pd.DataFrame,
    sequences: dict[str, str],
    *,
    include_labels: bool,
) -> FrozenPartition:
    ordered = frame.sort_values("gene_key", kind="mergesort").reset_index(drop=True)
    missing = sorted(set(ordered["gene_key"]) - set(sequences))
    if missing:
        raise ContractViolation(
            f"frozen partition contains {len(missing)} genes without a selected CDS: {missing[:5]}"
        )
    labels: tuple[int, ...] | None = None
    if include_labels:
        numeric = pd.to_numeric(ordered["label"], errors="coerce")
        if numeric.isna().any() or not numeric.isin([0, 1]).all():
            raise ContractViolation("training labels must be complete binary 0/1 values")
        labels = tuple(int(value) for value in numeric)
        if set(labels) != {0, 1}:
            raise ContractViolation("each outer labelled partition must contain both classes")
    return FrozenPartition(
        gene_keys=tuple(ordered["gene_key"].astype(str)),
        cluster_ids=tuple(ordered["cluster_id"].astype(str)),
        sequences=tuple(sequences[gene_id] for gene_id in ordered["gene_key"]),
        labels=labels,
    )


def load_frozen_split(
    cds_fasta_path: str | Path,
    training_labels_path: str | Path,
    split_assignments_path: str | Path,
    *,
    cohort_id: str,
    split_strategy: str,
    outer_fold: int,
) -> FrozenSplit:
    if split_strategy != "family_5fold":
        raise ContractViolation("split_strategy must be family_5fold")
    if outer_fold not in range(5):
        raise ContractViolation("outer_fold must be one of 0, 1, 2, 3, 4")

    context_filters = [
        ("cohort_id", "==", cohort_id),
        ("split_strategy", "==", split_strategy),
        ("outer_fold", "==", outer_fold),
    ]
    assignments = pd.read_parquet(
        split_assignments_path, filters=context_filters
    )
    labels = pd.read_parquet(training_labels_path, filters=context_filters)
    _require_columns(assignments, _ASSIGNMENT_COLUMNS, "split_assignments")
    _require_columns(labels, _TRAINING_LABEL_COLUMNS, "training_labels")
    if "label" in assignments.columns:
        raise ContractViolation("split_assignments must remain label-free")

    selected_assignments = _select_context(
        assignments,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        table="split_assignments",
    )
    roles = set(selected_assignments["role"])
    if roles != {"train", "validation", "test"}:
        raise ContractViolation(f"split context has unexpected role set: {sorted(roles)}")

    selected_labels = _select_context(
        labels,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        table="training_labels",
    )
    label_roles = set(selected_labels["role"])
    if not label_roles.issubset({"train", "validation"}):
        raise ContractViolation("training_labels exposes a sealed test role")

    expected_labelled = selected_assignments.loc[
        selected_assignments["role"].isin(["train", "validation"]),
        ["gene_key", "role", "cluster_id"],
    ].set_index("gene_key")
    observed_labelled = selected_labels.loc[
        :, ["gene_key", "role", "cluster_id", "label"]
    ].set_index("gene_key")
    if set(expected_labelled.index) != set(observed_labelled.index):
        raise ContractViolation(
            "training_labels gene coverage differs from train and validation assignments"
        )
    aligned = observed_labelled.loc[expected_labelled.index]
    if not aligned["role"].eq(expected_labelled["role"]).all():
        raise ContractViolation("training_labels roles differ from split_assignments")
    if not aligned["cluster_id"].eq(expected_labelled["cluster_id"]).all():
        raise ContractViolation("training_labels clusters differ from split_assignments")

    labelled_by_role = {
        role: selected_labels.loc[selected_labels["role"].eq(role)].copy()
        for role in ("train", "validation")
    }
    assignment_by_role = {
        role: selected_assignments.loc[selected_assignments["role"].eq(role)].copy()
        for role in ("train", "validation", "test")
    }
    partitions_for_audit = {
        "train": labelled_by_role["train"],
        "validation": labelled_by_role["validation"],
        "test": assignment_by_role["test"],
    }
    outer_audit = _assert_outer_isolation(partitions_for_audit, split_strategy)

    context_rows = sum(len(frame) for frame in assignment_by_role.values())
    expected_sequence_core_rows = _expected_sequence_core_rows(split_assignments_path)
    if cohort_id == "sequence_core" and context_rows != expected_sequence_core_rows:
        raise ContractViolation(
            f"sequence_core must contain {expected_sequence_core_rows} genes; found {context_rows}"
        )

    sequences = read_fasta(cds_fasta_path, canonical_gene_keys=True)
    train = _materialize_partition(
        labelled_by_role["train"], sequences, include_labels=True
    )
    validation = _materialize_partition(
        labelled_by_role["validation"], sequences, include_labels=True
    )
    test = _materialize_partition(
        assignment_by_role["test"], sequences, include_labels=False
    )
    outer_audit.update(
        {
            "sample_counts": {
                "train": len(train.gene_keys),
                "validation": len(validation.gene_keys),
                "test": len(test.gene_keys),
            },
        }
    )
    return FrozenSplit(
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        split_build_hash=_single_optional_value(
            selected_assignments, "split_build_hash"
        ),
        train=train,
        validation=validation,
        test=test,
        outer_audit=outer_audit,
    )
