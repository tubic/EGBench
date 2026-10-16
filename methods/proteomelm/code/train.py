"""Train the ProteomeLM-Ess classification head on one fold.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from egbench.artifacts import (
    RunRecord,
    software_versions,
    write_model_card,
    write_predictions,
    write_resource_usage,
    write_run_manifest,
)
from egbench.contracts import SEED

METHOD_ID = "proteomelm"
WEIGHTS_FILE = "weights/mlp_head.pt"
SCALER_FILE = "weights/scaler_state.npz"
DECISION_THRESHOLD = 0.5

HIDDEN = 512
DROPOUT = 0.1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class PreparedData:
    train_keys: list[str]
    train_x: np.ndarray
    train_y: np.ndarray
    validation_keys: list[str]
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_keys: list[str]
    test_x: np.ndarray
    feature_columns: list[str]
    split_build_hash: str


def _select(frame: pd.DataFrame, cohort: str, split: str, fold: int) -> pd.DataFrame:
    out = frame[
        frame["cohort_id"].eq(cohort)
        & frame["split_strategy"].eq(split)
        & frame["outer_fold"].eq(fold)
    ]
    if out.empty:
        raise ValueError(f"no rows for {cohort}/{split}/fold{fold}")
    return out


def load_prepared_data(
    embeddings_path: Path,
    training_labels_path: Path,
    split_assignments_path: Path,
    cohort: str,
    split_strategy: str,
    outer_fold: int,
) -> PreparedData:
    feats = pd.read_parquet(embeddings_path)
    feats["gene_key"] = feats["gene_key"].astype(str)
    feats = feats.set_index("gene_key")
    columns = list(feats.columns)

    labels = pd.read_parquet(training_labels_path)
    labels["gene_key"] = labels["gene_key"].astype(str)
    fold_labels = _select(labels, cohort, split_strategy, outer_fold)
    train_rows = fold_labels[fold_labels["role"].eq("train")]
    val_rows = fold_labels[fold_labels["role"].eq("validation")]

    assignments = pd.read_parquet(split_assignments_path)
    assignments["gene_key"] = assignments["gene_key"].astype(str)
    fold_assign = _select(assignments, cohort, split_strategy, outer_fold)
    test_keys = fold_assign.loc[fold_assign["role"].eq("test"), "gene_key"].tolist()

    train_keys = train_rows["gene_key"].tolist()
    val_keys = val_rows["gene_key"].tolist()
    if not val_keys:
        raise ValueError(
            f"no validation genes for {cohort}/{split_strategy}/fold{outer_fold}; "
            f"early stopping has no held-out split"
        )

    for name, keys in (("train", train_keys), ("validation", val_keys), ("test", test_keys)):
        missing = [key for key in keys if key not in feats.index]
        if missing:
            raise ValueError(
                f"{len(missing)} {name} genes missing from embeddings, e.g. {missing[:5]}"
            )

    split_hash_values = set(fold_assign["split_build_hash"].astype(str))
    if len(split_hash_values) != 1:
        raise ValueError(
            f"fold {outer_fold} of {cohort}/{split_strategy} carries "
            f"{len(split_hash_values)} split_build_hash values"
        )

    def matrix(keys: list[str]) -> np.ndarray:
        return feats.loc[keys, columns].to_numpy(dtype=np.float32)

    return PreparedData(
        train_keys=train_keys,
        train_x=matrix(train_keys),
        train_y=train_rows["label"].to_numpy(dtype=np.float32),
        validation_keys=val_keys,
        validation_x=matrix(val_keys),
        validation_y=val_rows["label"].to_numpy(dtype=np.float32),
        test_keys=test_keys,
        test_x=matrix(test_keys),
        feature_columns=columns,
        split_build_hash=split_hash_values.pop(),
    )


def preflight_summary(data: PreparedData) -> dict[str, Any]:
    return {
        "method_id": METHOD_ID,
        "sample_counts": {
            "train": len(data.train_keys),
            "validation": len(data.validation_keys),
            "test": len(data.test_keys),
        },
        "split_build_hash": data.split_build_hash,
        "feature_dimensions": len(data.feature_columns),
    }


def _formal_cuda(device_name: str) -> torch.device:
    if not device_name.startswith("cuda"):
        raise RuntimeError("formal ProteomeLM-Ess training requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("formal ProteomeLM-Ess training requires available CUDA")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    return device


def inference_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_formal_training(
    data: PreparedData,
    *,
    output_dir: Path,
    cohort: str,
    split_strategy: str,
    outer_fold: int,
    device_name: str,
    max_epochs: int,
    patience: int,
    hidden: int,
    learning_rate: float,
    required_source_assets: list[dict[str, str]],
) -> dict[str, Any]:
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    device = _formal_cuda(device_name)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    with RunRecord() as record:
        mean = data.train_x.mean(axis=0, keepdims=True)
        std = data.train_x.std(axis=0, keepdims=True) + 1e-6
        x_train = (data.train_x - mean) / std
        x_val = (data.validation_x - mean) / std
        x_test = (data.test_x - mean) / std

        xt = torch.tensor(x_train, device=device)
        yt = torch.tensor(data.train_y, device=device)
        xv = torch.tensor(x_val, device=device)
        yv = torch.tensor(data.validation_y, device=device)
        xs = torch.tensor(x_test, device=device)

        positive = float(data.train_y.sum())
        negative = float(len(data.train_y) - positive)
        pos_weight = torch.tensor([negative / max(positive, 1.0)], device=device)

        model = TwoLayerMLP(len(data.feature_columns), hidden=hidden).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_validation_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = -1
        waited = 0
        history: list[dict[str, float]] = []
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(xt), yt)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float(loss_fn(model(xv), yv))
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(loss.detach()),
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_validation_loss - 1e-5:
                best_validation_loss = validation_loss
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.state_dict().items()
                }
                best_epoch = epoch
                waited = 0
            else:
                waited += 1
                if waited >= patience:
                    break
        if best_state is None:
            raise RuntimeError("training did not produce a validation checkpoint")
        model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            test_scores = torch.sigmoid(model(xs)).float().cpu().numpy()
            validation_scores = torch.sigmoid(model(xv)).float().cpu().numpy()
        torch.cuda.synchronize(device)

    test_scores = np.clip(test_scores, 0.0, 1.0).astype(np.float64)
    validation_scores = np.clip(validation_scores, 0.0, 1.0).astype(np.float64)

    weights_path = output_dir / WEIGHTS_FILE
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "in_dim": len(data.feature_columns),
            "hidden": hidden,
            "dropout": DROPOUT,
        },
        weights_path,
    )
    scaler_path = output_dir / SCALER_FILE
    np.savez(scaler_path, mean=mean, std=std)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    write_predictions(
        output_dir,
        "blind",
        data.test_keys,
        test_scores,
        decision_threshold=DECISION_THRESHOLD,
    )
    write_predictions(
        output_dir,
        "validation",
        data.validation_keys,
        validation_scores,
        decision_threshold=DECISION_THRESHOLD,
    )

    versions = software_versions(np, pd, torch)
    manifest = {
        "method_id": METHOD_ID,
        "cohort_id": cohort,
        "split_strategy": split_strategy,
        "outer_fold": outer_fold,
        "split_build_hash": data.split_build_hash,
        "software_versions": versions,
        "parameters": {
            "max_epochs": max_epochs,
            "patience": patience,
            "hidden": hidden,
            "learning_rate": learning_rate,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "optimizer": "Adam",
            "imbalance": "bce_pos_weight",
            "scaling": "standard_scaler_fit_on_train_only",
            "selection": "best_validation_loss_checkpoint",
        },
        "sample_counts": {
            "train": len(data.train_keys),
            "validation": len(data.validation_keys),
            "test": len(data.test_keys),
        },
        "selected_checkpoint_epoch": best_epoch,
        **record.timestamps,
    }
    write_run_manifest(output_dir, **manifest)
    write_resource_usage(
        output_dir,
        record,
        device=str(device),
        gpu_name=torch.cuda.get_device_name(device),
        peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(device)),
    )
    write_model_card(
        output_dir,
        method_id=METHOD_ID,
        cohort_id=cohort,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        split_build_hash=data.split_build_hash,
        decision_threshold=DECISION_THRESHOLD,
        artifacts={"weights": WEIGHTS_FILE, "transforms": [SCALER_FILE]},
        required_source_assets=required_source_assets,
        software_versions=versions,
    )
    return manifest
