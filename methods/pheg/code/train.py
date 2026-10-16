"""Fit the Pheg linear SVM and Platt calibration on one fold.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from egbench.contracts import SEED


SELECTED_FEATURE_COUNT = 800
C_GRID = tuple(float(2**power) for power in range(-18, 11))
SVM_TOLERANCE = 1e-4
SVM_MAX_ITERATIONS = 200_000
CALIBRATOR_C = 1e10


def _load_partition(
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
    part = table.loc[
        (table["cohort_id"] == cohort)
        & (table["split_strategy"] == split_strategy)
        & (table["outer_fold"] == outer_fold)
        & (table["role"] == role),
        columns,
    ].copy()
    if part.empty:
        raise ValueError(f"empty partition: {cohort}/{split_strategy}/{outer_fold}/{role}")
    if part["gene_key"].duplicated().any():
        raise ValueError(f"duplicate gene_key values in {role} partition")
    if require_label and set(part["label"].unique()) != {0, 1}:
        raise ValueError(f"{role} partition must contain both binary classes")
    return part


def load_data(
    feature_path: Path,
    training_labels_path: Path,
    split_assignments_path: Path,
    cohort: str,
    split_strategy: str,
    outer_fold: int,
    *,
    expected_feature_count: int = SELECTED_FEATURE_COUNT,
) -> tuple[dict[str, np.ndarray], list[str], dict[str, list[str]]]:
    features = pd.read_parquet(feature_path)
    if "gene_key" not in features or features["gene_key"].duplicated().any():
        raise ValueError("feature table requires a unique gene_key column")
    feature_names = [name for name in features.columns if name != "gene_key"]
    if len(feature_names) != expected_feature_count:
        raise ValueError(
            f"Pheg requires {expected_feature_count} features, found {len(feature_names)}"
        )

    training = pd.read_parquet(training_labels_path)
    assignments = pd.read_parquet(split_assignments_path)
    partitions = {
        "train": _load_partition(
            training, cohort, split_strategy, outer_fold, "train", require_label=True
        ),
        "validation": _load_partition(
            training,
            cohort,
            split_strategy,
            outer_fold,
            "validation",
            require_label=True,
        ),
        "test": _load_partition(
            assignments,
            cohort,
            split_strategy,
            outer_fold,
            "test",
            require_label=False,
        ),
    }
    gene_sets = {
        role: set(frame["gene_key"].astype(str)) for role, frame in partitions.items()
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
        cluster_sets = {
            role: set(frame["cluster_id"].astype(str))
            for role, frame in partitions.items()
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
    arrays: dict[str, np.ndarray] = {}
    keys: dict[str, list[str]] = {}
    for role, labels in partitions.items():
        missing = sorted(set(labels["gene_key"]) - set(indexed.index))
        if missing:
            raise ValueError(f"{role} has {len(missing)} genes without Pheg features")
        ordered = indexed.loc[labels["gene_key"], feature_names]
        x = ordered.to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(x).all():
            raise ValueError(f"{role} features contain NaN or infinite values")
        arrays[f"x_{role}"] = x
        if "label" in labels:
            arrays[f"y_{role}"] = labels["label"].to_numpy(dtype=np.int64)
        keys[role] = labels["gene_key"].astype(str).tolist()
    return arrays, feature_names, keys


def fit_linear_svm(x: np.ndarray, y: np.ndarray, c: float) -> LinearSVC:
    model = LinearSVC(
        C=c,
        penalty="l2",
        loss="squared_hinge",
        dual=False,
        tol=SVM_TOLERANCE,
        max_iter=SVM_MAX_ITERATIONS,
        random_state=SEED,
    )
    model.fit(x, y)
    if model.n_iter_ >= model.max_iter:
        raise RuntimeError(f"LIBLINEAR did not converge for C={c}")
    return model


def fit_and_score(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    x_train = arrays["x_train"]
    x_validation = arrays["x_validation"]
    y_validation = arrays["y_validation"]

    best: tuple[float, float, LinearSVC] | None = None
    for c in C_GRID:
        model = fit_linear_svm(x_train, arrays["y_train"], c)
        auc = float(
            roc_auc_score(y_validation, model.decision_function(x_validation))
        )
        candidate = (auc, -c, model)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("the C grid is empty")
    model = best[2]
    selected_c = -best[1]

    validation_margins = model.decision_function(x_validation)
    calibrator = LogisticRegression(C=CALIBRATOR_C, solver="lbfgs")
    calibrator.fit(validation_margins.reshape(-1, 1), y_validation)
    return {
        "model": model,
        "calibrator": calibrator,
        "selected_c": selected_c,
        "validation_scores": probability(calibrator, validation_margins),
        "test_scores": probability(
            calibrator, model.decision_function(arrays["x_test"])
        ),
    }


def probability(calibrator: LogisticRegression, margins: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(np.asarray(margins).reshape(-1, 1))[:, 1].astype(
        np.float64
    )
