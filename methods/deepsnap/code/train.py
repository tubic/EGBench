"""Train the DeEPsnap snapshot ensemble on one fold and score genes.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from egbench.contracts import SEED

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise RuntimeError("DeEPsnap training requires PyTorch") from exc

try:
    from .contracts import (
        FORMAL_EPOCHS,
        SNAPSHOT_CYCLE,
        ContractError,
        formal_coverage_for_assignments,
        read_frame,
        require_columns,
        select_context,
        train_labels_only,
    )
    from .model import DeEPsnapMLP, architecture_contract, snapshot_learning_rate
except ImportError:
    from contracts import (
        FORMAL_EPOCHS,
        SNAPSHOT_CYCLE,
        ContractError,
        formal_coverage_for_assignments,
        read_frame,
        require_columns,
        select_context,
        train_labels_only,
    )
    from model import DeEPsnapMLP, architecture_contract, snapshot_learning_rate


BATCH_SIZE = 32
LEARNING_RATE = 0.001
ROLES = ("train", "validation", "test")


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def inference_device() -> Any:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def snapshot_probabilities(
    model: Any, states: list[dict[str, Any]], values: Any
) -> np.ndarray:
    predictions = []
    model.eval()
    with torch.no_grad():
        for state in states:
            model.load_state_dict(state)
            predictions.append(torch.sigmoid(model(values)).cpu().numpy())
    return np.mean(np.stack(predictions, axis=0), axis=0).astype(np.float64)


def run_training(
    *,
    features_path: str | Path,
    assignments_path: str | Path,
    labels_path: str | Path,
    cohort_id: str,
    split_strategy: str,
    outer_fold: int,
    epochs: int = FORMAL_EPOCHS,
    batch_size: int = BATCH_SIZE,
    device: str = "cuda",
) -> dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; the CPU is not a fallback "
            "for a run of this benchmark"
        )
    if epochs <= 0 or epochs % SNAPSHOT_CYCLE != 0:
        raise ContractError(
            f"the epoch count is a whole number of {SNAPSHOT_CYCLE}-epoch "
            f"snapshot cycles, got {epochs}"
        )
    context = select_context(
        read_frame(assignments_path),
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )
    expected_coverage = formal_coverage_for_assignments(assignments_path, cohort_id)
    if len(context) != expected_coverage:
        raise ContractError(
            f"DeEPsnap requires {expected_coverage} authoritative cohort genes"
        )
    features = read_frame(features_path)
    require_columns(features, ("gene_key",), "DeEPsnap features")
    feature_columns = [column for column in features if column != "gene_key"]
    if features["gene_key"].duplicated().any():
        raise ContractError("DeEPsnap features contain duplicate genes")
    ordered = context.merge(features, on="gene_key", how="left", validate="one_to_one")
    raw = ordered[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ContractError("DeEPsnap features are incomplete or non-finite")

    masks = {role: ordered["role"].eq(role).to_numpy() for role in ROLES}
    means = raw[masks["train"]].mean(axis=0)
    scales = raw[masks["train"]].std(axis=0)
    scales[scales == 0] = 1.0
    standardized = ((raw - means) / scales).astype(np.float32)

    train_labels = train_labels_only(
        read_frame(labels_path),
        context,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )
    label_lookup = dict(train_labels.itertuples(index=False, name=None))
    train_keys = ordered.loc[masks["train"], "gene_key"].tolist()
    y_train = np.asarray([label_lookup[key] for key in train_keys], dtype=np.float32)
    positive_count = int(y_train.sum())
    positive_weight = (len(y_train) - positive_count) / positive_count

    _seed_everything(SEED)
    target_device = torch.device("cuda:0" if device == "cuda" else device)
    if target_device.type == "cuda":
        torch.cuda.set_device(target_device)
        torch.cuda.reset_peak_memory_stats(target_device)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(standardized[masks["train"]], dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )
    model = DeEPsnapMLP(input_dimensions=len(feature_columns)).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pos_weight_tensor = torch.tensor(
        positive_weight, dtype=torch.float32, device=target_device
    )
    snapshots: list[dict[str, Any]] = []
    for epoch in range(epochs):
        learning_rate = snapshot_learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(target_device)
            batch_y = batch_y.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model(batch_x), batch_y, pos_weight=pos_weight_tensor
            )
            loss.backward()
            optimizer.step()
        if (epoch + 1) % SNAPSHOT_CYCLE == 0:
            snapshots.append(
                {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            )

    probabilities = snapshot_probabilities(
        model,
        snapshots,
        torch.as_tensor(standardized, dtype=torch.float32, device=target_device),
    )
    return {
        "snapshots": snapshots,
        "means": means,
        "scales": scales,
        "feature_columns": feature_columns,
        "gene_keys": {
            role: ordered.loc[masks[role], "gene_key"].astype(str).tolist()
            for role in ROLES
        },
        "scores": {role: probabilities[masks[role]] for role in ROLES},
        "sample_counts": {role: int(masks[role].sum()) for role in ROLES},
        "positive_weight": float(positive_weight),
        "device": str(target_device),
        "gpu_name": (
            torch.cuda.get_device_name(target_device)
            if target_device.type == "cuda"
            else None
        ),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(target_device))
            if target_device.type == "cuda"
            else 0
        ),
        "parameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "initial_learning_rate": LEARNING_RATE,
            "learning_rate_schedule": "snapshot_cosine",
            "snapshot_cycle": SNAPSHOT_CYCLE,
            "snapshot_count": len(snapshots),
            "optimizer": "Adam",
            "positive_weight_contract": "train-role negative_count / positive_count",
            "standardization": "train-role mean and population standard deviation",
            "ensemble": "arithmetic mean of every snapshot probability",
            "fitting_roles": ["train"],
            **architecture_contract(),
        },
    }
