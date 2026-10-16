"""Train EPGAT on one fold's interaction graph and score genes.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import copy
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from egbench.contracts import SEED
from egbench.perturbation import perturb_edges

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    raise RuntimeError("EPGAT training requires PyTorch") from exc

try:
    from .contracts import (
        EARLY_STOPPING_PATIENCE,
        MAX_EPOCHS,
        ContractError,
        formal_coverage_for_assignments,
        read_frame,
        require_columns,
        select_context,
        train_labels_only,
        validation_labels_only,
    )
    from .gat import EPGAT, architecture_contract
except ImportError:
    from contracts import (
        EARLY_STOPPING_PATIENCE,
        MAX_EPOCHS,
        ContractError,
        formal_coverage_for_assignments,
        read_frame,
        require_columns,
        select_context,
        train_labels_only,
        validation_labels_only,
    )
    from gat import EPGAT, architecture_contract


LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4
MINIMUM_IMPROVEMENT = 1e-12
ROLES = ("train", "validation", "test")


def deterministic_kernels() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic_kernels()


def build_edge_index(
    edges: pd.DataFrame,
    gene_keys: list[str],
    *,
    training_nodes: list[str] | None = None,
) -> np.ndarray:
    require_columns(edges, ("gene_key_a", "gene_key_b"), "PPI edges")
    edges = perturb_edges(edges, training_nodes=training_nodes)
    index = {gene_key: position for position, gene_key in enumerate(gene_keys)}
    directed: set[tuple[int, int]] = {
        (position, position) for position in range(len(gene_keys))
    }
    for left, right in edges[["gene_key_a", "gene_key_b"]].itertuples(
        index=False, name=None
    ):
        left_index = index.get(str(left))
        right_index = index.get(str(right))
        if left_index is None or right_index is None or left_index == right_index:
            continue
        directed.add((left_index, right_index))
        directed.add((right_index, left_index))
    return np.asarray(sorted(directed), dtype=np.int64).T


def class_balanced_loss(logits: Any, labels: Any, index: Any) -> Any:
    selected_logits = logits.index_select(0, index)
    positive = labels == 1
    negative = labels == 0
    if not bool(positive.any()) or not bool(negative.any()):
        raise ContractError("both classes are required in the rows the loss reads")
    positive_loss = F.binary_cross_entropy_with_logits(
        selected_logits[positive], labels[positive]
    )
    negative_loss = F.binary_cross_entropy_with_logits(
        selected_logits[negative], labels[negative]
    )
    return positive_loss + negative_loss


def run_training(
    *,
    features_path: str | Path,
    edges_path: str | Path,
    assignments_path: str | Path,
    labels_path: str | Path,
    cohort_id: str,
    split_strategy: str,
    outer_fold: int,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    device: str = "cuda",
) -> dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; the CPU is not a fallback "
            "for a run of this benchmark"
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
            f"EPGAT context must contain {expected_coverage} genes for {cohort_id}"
        )
    features = read_frame(features_path)
    require_columns(features, ("gene_key",), "EPGAT fold features")
    feature_columns = [column for column in features if column != "gene_key"]
    if features["gene_key"].duplicated().any():
        raise ContractError("EPGAT features contain duplicate genes")
    ordered = context.loc[:, ["gene_key", "role"]].merge(
        features, on="gene_key", how="left", validate="one_to_one"
    )
    values = ordered[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ContractError("EPGAT fold features are incomplete or non-finite")

    label_frames = {
        "train": train_labels_only(
            read_frame(labels_path),
            context,
            cohort_id=cohort_id,
            split_strategy=split_strategy,
            outer_fold=outer_fold,
        ),
        "validation": validation_labels_only(
            read_frame(labels_path),
            context,
            cohort_id=cohort_id,
            split_strategy=split_strategy,
            outer_fold=outer_fold,
        ),
    }
    positions = {
        role: np.flatnonzero(ordered["role"].eq(role).to_numpy()) for role in ROLES
    }
    targets: dict[str, np.ndarray] = {}
    for role, frame in label_frames.items():
        lookup = dict(frame.itertuples(index=False, name=None))
        keys = ordered.iloc[positions[role]]["gene_key"].tolist()
        targets[role] = np.asarray([lookup[key] for key in keys], dtype=np.float32)
    edge_index = build_edge_index(
        read_frame(edges_path),
        ordered["gene_key"].tolist(),
        training_nodes=ordered.loc[
            ordered["role"].isin(("train", "validation")), "gene_key"
        ].tolist(),
    )

    _seed_everything(SEED)
    target_device = torch.device("cuda:0" if device == "cuda" else device)
    if target_device.type == "cuda":
        torch.cuda.set_device(target_device)
        torch.cuda.reset_peak_memory_stats(target_device)
    x_tensor = torch.as_tensor(values, dtype=torch.float32, device=target_device)
    edge_tensor = torch.as_tensor(edge_index, dtype=torch.long, device=target_device)
    index_tensors = {
        role: torch.as_tensor(positions[role], dtype=torch.long, device=target_device)
        for role in label_frames
    }
    y_tensors = {
        role: torch.as_tensor(targets[role], dtype=torch.float32, device=target_device)
        for role in label_frames
    }
    model = EPGAT(input_dimensions=len(feature_columns)).to(target_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    best_epoch = 0
    epochs_run = 0
    stale = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = class_balanced_loss(
            model(x_tensor, edge_tensor), y_tensors["train"], index_tensors["train"]
        )
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                class_balanced_loss(
                    model(x_tensor, edge_tensor),
                    y_tensors["validation"],
                    index_tensors["validation"],
                )
            )
        epochs_run = epoch + 1
        if validation_loss < best_loss - MINIMUM_IMPROVEMENT:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise ContractError("EPGAT training produced no checkpoint")
    model.load_state_dict(best_state)

    return {
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "input_dimensions": len(feature_columns),
        "gene_keys": {
            role: ordered.iloc[positions[role]]["gene_key"].astype(str).tolist()
            for role in ROLES
        },
        "sample_counts": {role: int(len(positions[role])) for role in ROLES},
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_run": epochs_run,
        "device": str(target_device),
        "gpu_name": (
            torch.cuda.get_device_name(target_device)
            if target_device.type == "cuda"
            else None
        ),
        "parameters": {
            "feature_count": len(feature_columns),
            "directed_edges_with_self_loops": int(edge_index.shape[1]),
            "early_stopping_patience": patience,
            "early_stopping_criterion": "validation class-balanced loss",
            "fitting_roles": ["train"],
            **architecture_contract(),
        },
    }


def _inference_device() -> Any:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def score_graph(
    state_dict: dict[str, Any],
    input_dimensions: int,
    values: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    device = _inference_device()
    deterministic_kernels()
    model = EPGAT(input_dimensions=input_dimensions).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.as_tensor(values, dtype=torch.float32, device=device),
            torch.as_tensor(edge_index, dtype=torch.long, device=device),
        )
    return torch.sigmoid(logits).cpu().numpy().astype(np.float64)
