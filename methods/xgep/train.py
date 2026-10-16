#!/usr/bin/env python3
"""Train one XGEP-XGB fold and score genes with a trained fold.
Imported by run.py; not run directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from egbench.artifacts import (
    RunRecord,
    software_versions,
    write_model_card,
    write_predict_manifest,
    write_predictions,
    write_resource_usage,
    write_run_manifest,
)
from egbench.contracts import SEED, ContractError, read_model_card
from egbench.data import folds_directory

from build_features import SOURCE_MATRIX, fold_feature_directory

METHOD_ID = "xgep"
THREADS = 8
N_ESTIMATORS = 500
CANDIDATE_MAX_DEPTHS = (3, 6)
CANDIDATE_LEARNING_RATES = (0.03, 0.1)
MIN_CHILD_WEIGHT = 1
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
REG_LAMBDA = 1.0
EARLY_STOPPING_ROUNDS = 30
DECISION_THRESHOLD = 0.5
WEIGHTS_DIRECTORY = "weights"
MODEL_FILE = "model.json"
_EXAMPLE_LIMIT = 10


def _fold_context(
    cohort_id: str, split_strategy: str, outer_fold: int
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    folds = folds_directory(cohort_id, split_strategy=split_strategy, root=REPO_ROOT)
    assignments = pd.read_parquet(folds / "split_assignments.parquet")
    labels = pd.read_parquet(folds / "training_labels" / f"fold_{outer_fold}.parquet")
    context = assignments.loc[
        assignments["cohort_id"].eq(cohort_id)
        & assignments["split_strategy"].eq(split_strategy)
        & assignments["outer_fold"].eq(outer_fold)
    ]
    label_context = labels.loc[
        labels["cohort_id"].eq(cohort_id)
        & labels["split_strategy"].eq(split_strategy)
        & labels["outer_fold"].eq(outer_fold)
    ]
    if context.empty or label_context.empty:
        raise ContractError(
            f"no {split_strategy} rows for {cohort_id}/fold_{outer_fold} in {folds}"
        )
    hashes = set(context["split_build_hash"].astype(str))
    if len(hashes) != 1:
        raise ContractError(
            f"{folds}: split_build_hash is not unique for this fold: {sorted(hashes)}"
        )
    return context, label_context, hashes.pop()


def load_fold(cohort_id: str, split_strategy: str, outer_fold: int):
    directory = fold_feature_directory(cohort_id, split_strategy, outer_fold)
    features_path = directory / "features.parquet"
    if not features_path.is_file():
        raise ContractError(
            f"{features_path} is absent. Run 'python run.py features --dataset "
            f"{cohort_id} --split-strategy {split_strategy} --outer-fold "
            f"{outer_fold}' first."
        )
    features = pd.read_parquet(features_path).set_index("gene_key")
    context, label_context, split_build_hash = _fold_context(
        cohort_id, split_strategy, outer_fold
    )
    label_lookup = dict(
        zip(label_context["gene_key"], label_context["label"].astype(int), strict=True)
    )
    partitions = {}
    for role in ("train", "validation", "test"):
        keys = context.loc[context["role"].eq(role), "gene_key"].astype(str).tolist()
        missing = sorted(set(keys) - set(features.index))
        if missing:
            raise ContractError(f"{len(missing)} {role} genes lack XGEP features")
        matrix = features.loc[keys].to_numpy(dtype=np.float32, copy=True)
        y = None if role == "test" else np.asarray(
            [label_lookup[key] for key in keys], dtype=np.int64
        )
        partitions[role] = (keys, matrix, y)
    return cohort_id, partitions, split_build_hash


def preflight(cohort_id: str, split_strategy: str, outer_fold: int) -> None:
    cohort_id, partitions, split_build_hash = load_fold(
        cohort_id, split_strategy, outer_fold
    )
    print(
        json.dumps(
            {
                "method_id": METHOD_ID,
                "dataset": cohort_id,
                "cohort_id": cohort_id,
                "split_strategy": split_strategy,
                "outer_fold": outer_fold,
                "split_build_hash": split_build_hash,
                "feature_dimensions": partitions["train"][1].shape[1],
                "role_counts": {
                    role: len(partitions[role][0]) for role in partitions
                },
            },
            indent=2,
        )
    )


def _required_source_assets() -> list[dict[str, str]]:
    return [
        {
            "path": "methods/xgep/feature/source/xgep_150.parquet",
            "purpose": (
                "the author-published 150-dimensional pan-cancer embedding "
                "table; a gene absent here cannot be scored."
            ),
        },
    ]


def train_fold(
    cohort_id: str, split_strategy: str, outer_fold: int, model_dir: Path
) -> Path:
    import xgboost
    from xgboost import XGBClassifier

    cohort_id, partitions, split_build_hash = load_fold(
        cohort_id, split_strategy, outer_fold
    )
    train_keys, train_x, train_y = partitions["train"]
    validation_keys, validation_x, validation_y = partitions["validation"]
    test_keys, test_x, _ = partitions["test"]

    with RunRecord() as record:
        candidates = []
        for max_depth in CANDIDATE_MAX_DEPTHS:
            for learning_rate in CANDIDATE_LEARNING_RATES:
                model = XGBClassifier(
                    n_estimators=N_ESTIMATORS,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    min_child_weight=MIN_CHILD_WEIGHT,
                    subsample=SUBSAMPLE,
                    colsample_bytree=COLSAMPLE_BYTREE,
                    reg_lambda=REG_LAMBDA,
                    tree_method="hist",
                    eval_metric="aucpr",
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    random_state=SEED,
                    n_jobs=THREADS,
                )
                model.fit(
                    train_x,
                    train_y,
                    eval_set=[(validation_x, validation_y)],
                    verbose=False,
                )
                validation_score = model.predict_proba(validation_x)[:, 1].astype(
                    np.float64
                )
                candidates.append(
                    (
                        float(average_precision_score(validation_y, validation_score)),
                        -max_depth,
                        -learning_rate,
                        model,
                        validation_score,
                    )
                )
        best_ap, _, _, model, validation_scores = max(
            candidates, key=lambda item: item[:3]
        )
        test_scores = model.predict_proba(test_x)[:, 1].astype(np.float64)

    weights_dir = model_dir / WEIGHTS_DIRECTORY
    weights_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(weights_dir / MODEL_FILE)

    write_predictions(
        model_dir, "blind", test_keys, test_scores, decision_threshold=DECISION_THRESHOLD
    )
    write_predictions(
        model_dir,
        "validation",
        validation_keys,
        validation_scores,
        decision_threshold=DECISION_THRESHOLD,
    )

    versions = software_versions(np, pd, xgboost)
    write_run_manifest(
        model_dir,
        method_id=METHOD_ID,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        split_build_hash=split_build_hash,
        software_versions=versions,
        parameters={
            "threads": THREADS,
            "feature_dimensions": int(train_x.shape[1]),
            "n_estimators": N_ESTIMATORS,
            "candidate_max_depths": list(CANDIDATE_MAX_DEPTHS),
            "candidate_learning_rates": list(CANDIDATE_LEARNING_RATES),
            "min_child_weight": MIN_CHILD_WEIGHT,
            "subsample": SUBSAMPLE,
            "colsample_bytree": COLSAMPLE_BYTREE,
            "reg_lambda": REG_LAMBDA,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "selected_max_depth": model.get_params()["max_depth"],
            "selected_learning_rate": model.get_params()["learning_rate"],
            "validation_average_precision": best_ap,
        },
        sample_counts={role: len(partitions[role][0]) for role in partitions},
        **record.timestamps,
    )
    write_resource_usage(
        model_dir,
        record,
        device="cpu",
        gpu_name=None,
        peak_gpu_memory_bytes=0,
    )
    return write_model_card(
        model_dir,
        method_id=METHOD_ID,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        split_build_hash=split_build_hash,
        decision_threshold=DECISION_THRESHOLD,
        artifacts={
            "weights": f"{WEIGHTS_DIRECTORY}/{MODEL_FILE}",
            "transforms": [],
        },
        required_source_assets=_required_source_assets(),
        software_versions=versions,
    )


def _read_gene_keys(path: Path) -> list[str]:
    frame = pd.read_parquet(path)
    if list(frame.columns) != ["gene_key"]:
        raise ContractError(
            f"{path}: a predict gene list carries the single column 'gene_key', "
            f"this one carries {list(frame.columns)}"
        )
    keys = frame["gene_key"].astype(str).tolist()
    if not keys:
        raise ContractError(f"{path} names no gene")
    if len(set(keys)) != len(keys):
        raise ContractError(f"{path} names a gene twice")
    return keys


def predict(model_dir: Path, genes_path: Path, output_dir: Path) -> Path:
    from xgboost import XGBClassifier

    card = read_model_card(model_dir / "model_card.json")
    keys = _read_gene_keys(genes_path)
    if not SOURCE_MATRIX.is_file():
        raise ContractError(f"the XGEP source matrix is absent: {SOURCE_MATRIX}")
    matrix = pd.read_parquet(SOURCE_MATRIX).set_index("gene_key", drop=False)
    missing = sorted(set(keys) - set(matrix.index))
    if missing:
        raise ContractError(
            f"XGEP inference_scope is 'author_embedding_table': {len(missing)} "
            f"of {len(keys)} requested genes are absent from the author "
            f"pan-cancer embedding table, for example {missing[:_EXAMPLE_LIMIT]}"
        )
    values = matrix.loc[keys].drop(columns="gene_key").to_numpy(
        dtype=np.float32, copy=True
    )
    model = XGBClassifier()
    model.load_model(model_dir / WEIGHTS_DIRECTORY / MODEL_FILE)

    with RunRecord() as record:
        scores = model.predict_proba(values)[:, 1].astype(np.float64)

    threshold = float(card["decision_threshold"])
    write_predictions(output_dir, "blind", keys, scores, decision_threshold=threshold)
    return write_predict_manifest(output_dir, card, record, gene_count=len(keys))
