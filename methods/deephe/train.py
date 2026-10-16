#!/usr/bin/env python3
"""Train one DeepHE fold and score genes with a trained fold.
Imported by run.py; not run directly.
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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

from build_features import (
    fold_feature_directory,
    read_source_matrix,
)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

METHOD_ID = "deephe"
DEVICE = "cuda"
MAX_EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10
LEARNING_RATE = 1e-3
DROPOUT = 0.2
HIDDEN_WIDTHS = (128, 256, 512)
DECISION_THRESHOLD = 0.5
WEIGHTS_DIRECTORY = "weights"
MODEL_FILE = "best_model.pt"
SCALER_FILE = "scaler.npz"
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
    ].copy()
    label_context = labels.loc[
        labels["cohort_id"].eq(cohort_id)
        & labels["split_strategy"].eq(split_strategy)
        & labels["outer_fold"].eq(outer_fold)
    ].copy()
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
    features_path = (
        fold_feature_directory(cohort_id, split_strategy, outer_fold) / "features.parquet"
    )
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
            raise ContractError(f"{len(missing)} {role} genes lack DeepHE features")
        values = features.loc[keys].to_numpy(dtype=np.float32, copy=True)
        role_labels = None
        if role != "test":
            role_labels = np.asarray(
                [label_lookup[key] for key in keys], dtype=np.float32
            )
            if set(role_labels.tolist()) != {0.0, 1.0}:
                raise ContractError(f"{role} requires both binary classes")
        partitions[role] = (keys, values, role_labels)
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


def _build_model(input_dim: int):
    from torch import nn

    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN_WIDTHS[0]),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_WIDTHS[0], HIDDEN_WIDTHS[1]),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_WIDTHS[1], HIDDEN_WIDTHS[2]),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_WIDTHS[2], 1),
    )


def _set_determinism(torch, device) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _required_source_assets() -> list[dict[str, str]]:
    return [
        {
            "path": "methods/deephe/feature/source/deephe_sequence_89.parquet",
            "purpose": (
                "89-dimensional sequence-derived component of the 153-"
                "dimensional feature matrix."
            ),
        },
        {
            "path": "methods/deephe/feature/source/node2vec_64.parquet",
            "purpose": (
                "64-dimensional BioGRID node2vec component; a gene absent "
                "here cannot be scored."
            ),
        },
    ]


def train_fold(
    cohort_id: str, split_strategy: str, outer_fold: int, model_dir: Path
) -> Path:
    import torch
    from torch import nn

    cohort_id, partitions, split_build_hash = load_fold(
        cohort_id, split_strategy, outer_fold
    )
    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeepHE training but is unavailable")
    _set_determinism(torch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_keys, train_x, train_y = partitions["train"]
    validation_keys, validation_x, validation_y = partitions["validation"]
    test_keys, test_x, _ = partitions["test"]
    means = train_x.mean(axis=0)
    scales = train_x.std(axis=0)
    scales[scales == 0] = 1.0

    def tensor(values):
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    x_train = tensor((train_x - means) / scales)
    y_train = tensor(train_y)
    x_validation = tensor((validation_x - means) / scales)
    y_validation = tensor(validation_y)
    x_test = tensor((test_x - means) / scales)

    model = _build_model(train_x.shape[1]).to(device)
    positive = float(train_y.sum())
    negative = float(len(train_y) - positive)
    criterion = nn.BCEWithLogitsLoss(pos_weight=tensor([negative / positive]))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator(device="cpu").manual_seed(SEED)

    with RunRecord() as record:
        best_state = None
        best_loss = float("inf")
        best_epoch = 0
        stale = 0
        for epoch in range(MAX_EPOCHS):
            model.train()
            order = torch.randperm(len(train_y), generator=generator)
            for start in range(0, len(train_y), BATCH_SIZE):
                index = order[start : start + BATCH_SIZE].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x_train[index]).reshape(-1), y_train[index])
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    criterion(model(x_validation).reshape(-1), y_validation).item()
                )
            if validation_loss < best_loss - 1e-12:
                best_loss = validation_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        if best_state is None:
            raise RuntimeError("DeepHE training did not produce a checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_scores = (
                torch.sigmoid(model(x_test).reshape(-1)).cpu().numpy().astype(np.float64)
            )
            validation_scores = (
                torch.sigmoid(model(x_validation).reshape(-1))
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    weights_dir = model_dir / WEIGHTS_DIRECTORY
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_dir / MODEL_FILE)
    np.savez_compressed(weights_dir / SCALER_FILE, mean=means, scale=scales)

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

    versions = software_versions(np, pd, torch)
    write_run_manifest(
        model_dir,
        method_id=METHOD_ID,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        split_build_hash=split_build_hash,
        software_versions=versions,
        parameters={
            "device": str(device),
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "dropout": DROPOUT,
            "architecture": [train_x.shape[1], *HIDDEN_WIDTHS, 1],
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
        },
        sample_counts={role: len(partitions[role][0]) for role in partitions},
        **record.timestamps,
    )
    write_resource_usage(
        model_dir,
        record,
        device=str(device),
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        peak_gpu_memory_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
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
            "transforms": [f"{WEIGHTS_DIRECTORY}/{SCALER_FILE}"],
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
    import torch

    card = read_model_card(model_dir / "model_card.json")
    keys = _read_gene_keys(genes_path)
    matrix = read_source_matrix().set_index("gene_key", drop=False)
    missing = sorted(set(keys) - set(matrix.index))
    if missing:
        raise ContractError(
            f"DeepHE inference_scope is 'network': {len(missing)} of "
            f"{len(keys)} requested genes are absent from the source feature "
            f"table, for example {missing[:_EXAMPLE_LIMIT]}"
        )
    values = matrix.loc[keys].drop(columns="gene_key").to_numpy(
        dtype=np.float32, copy=True
    )
    weights_dir = model_dir / WEIGHTS_DIRECTORY
    with np.load(weights_dir / SCALER_FILE) as scaler:
        means = scaler["mean"]
        scales = scaler["scale"]

    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeepHE inference but is unavailable")
    _set_determinism(torch, device)
    model = _build_model(values.shape[1]).to(device)
    state = torch.load(weights_dir / MODEL_FILE, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with RunRecord() as record:
        with torch.no_grad():
            x = torch.as_tensor((values - means) / scales, dtype=torch.float32, device=device)
            scores = torch.sigmoid(model(x).reshape(-1)).cpu().numpy().astype(np.float64)

    threshold = float(card["decision_threshold"])
    write_predictions(output_dir, "blind", keys, scores, decision_threshold=threshold)
    return write_predict_manifest(output_dir, card, record, gene_count=len(keys))
