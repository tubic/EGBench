"""Load and standardise one DeeplyEssential fold.
Imported by run.py; not run directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Partition:
    gene_keys: tuple[str, ...]
    cluster_ids: tuple[str, ...]
    features: np.ndarray
    labels: np.ndarray | None


@dataclass(frozen=True)
class PreparedData:
    train: Partition
    validation: Partition
    test: Partition
    feature_names: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


def _select_partition(
    table: pd.DataFrame,
    cohort: str,
    split_strategy: str,
    outer_fold: int,
    role: str,
    *,
    require_label: bool,
) -> pd.DataFrame:
    required = {
        "gene_key",
        "cohort_id",
        "split_strategy",
        "outer_fold",
        "role",
        "cluster_id",
    }
    if require_label:
        required.add("label")
    missing_columns = required - set(table.columns)
    if missing_columns:
        raise ValueError(f"split table lacks columns: {sorted(missing_columns)}")
    columns = ["gene_key", "cluster_id"]
    if require_label:
        columns.insert(1, "label")
    selected = table.loc[
        (table["cohort_id"] == cohort)
        & (table["split_strategy"] == split_strategy)
        & (table["outer_fold"] == outer_fold)
        & (table["role"] == role),
        columns,
    ].copy()
    if selected.empty:
        raise ValueError(f"empty partition: {cohort}/{split_strategy}/{outer_fold}/{role}")
    if selected["gene_key"].duplicated().any():
        raise ValueError(f"duplicate gene_key values in {role}")
    if require_label and set(selected["label"].unique()) != {0, 1}:
        raise ValueError(f"{role} partition must contain both binary classes")
    return selected


def _materialize_partition(
    labels: pd.DataFrame,
    indexed_features: pd.DataFrame,
    feature_names: list[str],
) -> Partition:
    missing = sorted(set(labels["gene_key"]) - set(indexed_features.index))
    if missing:
        raise ValueError(f"{len(missing)} labelled genes lack 89-dimensional features")
    matrix = indexed_features.loc[labels["gene_key"], feature_names].to_numpy(
        dtype=np.float64, copy=True
    )
    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix contains NaN or infinite values")
    return Partition(
        gene_keys=tuple(labels["gene_key"].astype(str)),
        cluster_ids=tuple(labels["cluster_id"].astype(str)),
        features=matrix,
        labels=(
            labels["label"].to_numpy(dtype=np.int64)
            if "label" in labels
            else None
        ),
    )


def load_prepared_data(
    feature_path: Path,
    training_labels_path: Path,
    split_assignments_path: Path,
    cohort: str,
    split_strategy: str,
    outer_fold: int,
) -> PreparedData:
    features = pd.read_parquet(feature_path)
    if "gene_key" not in features:
        raise ValueError("feature table lacks gene_key")
    if features["gene_key"].duplicated().any():
        raise ValueError("feature table has duplicate gene_key values")
    feature_names = [column for column in features.columns if column != "gene_key"]

    training_labels = pd.read_parquet(training_labels_path)
    split_assignments = pd.read_parquet(split_assignments_path)
    selected = {
        "train": _select_partition(
            training_labels,
            cohort,
            split_strategy,
            outer_fold,
            "train",
            require_label=True,
        ),
        "validation": _select_partition(
            training_labels,
            cohort,
            split_strategy,
            outer_fold,
            "validation",
            require_label=True,
        ),
        "test": _select_partition(
            split_assignments,
            cohort,
            split_strategy,
            outer_fold,
            "test",
            require_label=False,
        ),
    }
    gene_sets = {role: set(frame["gene_key"]) for role, frame in selected.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = gene_sets[left] & gene_sets[right]
        if overlap:
            raise ValueError(f"gene leakage between {left} and {right}: {len(overlap)}")

    if split_strategy == "family_5fold":
        cluster_sets = {
            role: set(frame["cluster_id"].astype(str)) for role, frame in selected.items()
        }
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            overlap = cluster_sets[left] & cluster_sets[right]
            if overlap:
                raise ValueError(
                    f"family cluster leakage between {left} and {right}: {len(overlap)}"
                )

    indexed = features.set_index("gene_key")
    raw = {
        role: _materialize_partition(frame, indexed, feature_names)
        for role, frame in selected.items()
    }
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(raw["train"].features)

    def scaled(partition: Partition, matrix: np.ndarray) -> Partition:
        return Partition(
            gene_keys=partition.gene_keys,
            cluster_ids=partition.cluster_ids,
            features=matrix.astype(np.float32, copy=False),
            labels=partition.labels,
        )

    return PreparedData(
        train=scaled(raw["train"], train_scaled),
        validation=scaled(
            raw["validation"], scaler.transform(raw["validation"].features)
        ),
        test=scaled(raw["test"], scaler.transform(raw["test"].features)),
        feature_names=tuple(feature_names),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
    )
