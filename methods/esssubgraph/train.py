#!/usr/bin/env python3
"""Train the EssSubgraph GraphSAGE model on one fold and score genes.
Imported by run.py; not run directly.
"""

from __future__ import annotations

import copy
import json
import os
import random
import shutil
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
from egbench.data import feature_directory, folds_directory
from egbench.perturbation import perturb_edges

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

METHOD_ID = "esssubgraph"
DEVICE = "cuda"
LEARNING_RATE = 0.001
MAX_EPOCHS = 200
PATIENCE = 10
WEIGHT_DECAY = 5e-7
BATCH_SIZE = 1024
POSITIVE_WEIGHT = 4.0
NEIGHBOR_FANOUT = (30, 25, 10)
HIDDEN_DIMENSIONS = (128, 128)
NODE_FEATURE_DIMENSIONS = 50
DECISION_THRESHOLD = 0.5
WEIGHTS_DIRECTORY = "weights"
MODEL_FILE = "best_model.pt"
PCA_STATE_FILE = "expression_pca_50_state.npz"
GRAPH_NODES_FILE = "graph_nodes.parquet"
EXPRESSION_SOURCE = (
    ROOT / "feature" / "source" / "expression" / "tcga_gene_by_sample.parquet"
)
BIOGRID_EDGES = ROOT / "feature" / "source" / "network" / "biogrid_physical_edges.parquet"
_EXAMPLE_LIMIT = 10


def set_random_seed(torch) -> None:

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def feature_matrix_path(cohort_id: str, split_strategy: str, outer_fold: int) -> Path:
    return (
        feature_directory(
            ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
        )
        / "features.parquet"
    )


def load_context(cohort_id: str, split_strategy: str, outer_fold: int):
    folds = folds_directory(cohort_id, split_strategy=split_strategy, root=REPO_ROOT)
    context = pd.read_parquet(
        folds / "fold_genes" / f"fold_{outer_fold}_genes.parquet",
        columns=["gene_key", "role"],
    )
    labels = pd.read_parquet(folds / "training_labels" / f"fold_{outer_fold}.parquet")
    label_context = labels.loc[
        labels["split_strategy"].eq(split_strategy) & labels["outer_fold"].eq(outer_fold)
    ].copy()
    if context.empty or label_context.empty:
        raise ContractError(
            f"no {split_strategy} rows for {cohort_id}/fold_{outer_fold} in {folds}"
        )
    hashes = set(label_context["split_build_hash"].astype(str))
    if len(hashes) != 1:
        raise ContractError(
            f"{folds}: split_build_hash is not unique for this fold: {sorted(hashes)}"
        )
    return context, label_context, hashes.pop()


def _build_graph(feature_path: Path, cohort_id: str, split_strategy: str, outer_fold: int):

    import torch
    from torch_geometric.data import Data

    features = pd.read_parquet(feature_path).sort_values("gene_key", kind="mergesort")
    if features["gene_key"].duplicated().any():
        raise ContractError("EssSubgraph fold features contain duplicate gene_key values")
    keys = features["gene_key"].astype(str).tolist()
    index = {key: position for position, key in enumerate(keys)}
    context, label_context, split_build_hash = load_context(
        cohort_id, split_strategy, outer_fold
    )
    if set(context["gene_key"].astype(str)) != set(keys):
        raise ContractError("fold feature keys differ from split assignments")
    labels = dict(
        zip(label_context["gene_key"], label_context["label"].astype(int), strict=True)
    )
    y = torch.full((len(keys),), -1.0, dtype=torch.float32)
    masks = {}
    for role in ("train", "validation", "test"):
        mask = torch.zeros(len(keys), dtype=torch.bool)
        for key in context.loc[context["role"].eq(role), "gene_key"].astype(str):
            mask[index[key]] = True
            if role != "test":
                y[index[key]] = labels[key]
        masks[role] = mask
    data = Data(
        x=torch.tensor(
            features.drop(columns="gene_key").to_numpy(),
            dtype=torch.float32,
        ),
        edge_index=_edge_index(
            index,
            training_nodes=context.loc[
                context["role"].isin(("train", "validation")), "gene_key"
            ]
            .astype(str)
            .tolist(),
        ),
        y=y,
        train_mask=masks["train"],
        validation_mask=masks["validation"],
        test_mask=masks["test"],
    )
    return data, keys, index, split_build_hash


def _edge_index(index: dict[str, int], *, training_nodes: list[str] | None = None):

    import torch

    if not BIOGRID_EDGES.is_file():
        raise ContractError(f"BioGRID physical edges are absent: {BIOGRID_EDGES}")
    edges = perturb_edges(
        pd.read_parquet(BIOGRID_EDGES, columns=["gene_key_a", "gene_key_b"]),
        training_nodes=training_nodes,
    )
    pairs = []
    for left, right in edges.itertuples(index=False, name=None):
        if left in index and right in index and left != right:
            pairs.extend(((index[left], index[right]), (index[right], index[left])))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def fold_view(model_dir: Path) -> tuple[list[str], np.ndarray]:
    weights = model_dir / WEIGHTS_DIRECTORY
    nodes = pd.read_parquet(weights / GRAPH_NODES_FILE)
    keys = nodes["gene_key"].astype(str).tolist()
    if len(set(keys)) != len(keys):
        raise ContractError(f"{weights / GRAPH_NODES_FILE} names a node twice")
    with np.load(weights / PCA_STATE_FILE, allow_pickle=False) as state:
        medians = np.asarray(state["medians"], dtype=np.float64)
        means = np.asarray(state["means"], dtype=np.float64)
        scales = np.asarray(state["scales"], dtype=np.float64)
        components = np.asarray(state["components"], dtype=np.float64)
        offset = np.asarray(state["pca_mean"], dtype=np.float64) @ components.T
    if not EXPRESSION_SOURCE.is_file():
        raise ContractError(
            f"the TCGA expression matrix is absent: {EXPRESSION_SOURCE}"
        )
    expression = pd.read_parquet(EXPRESSION_SOURCE)
    expression = expression.set_index(expression["gene_key"].astype(str)).drop(
        columns="gene_key"
    )
    missing = [key for key in keys if key not in expression.index]
    if missing:
        raise ContractError(
            f"{len(missing)} of {len(keys)} nodes lack a TCGA expression row, "
            f"the first ten are {missing[:_EXAMPLE_LIMIT]}"
        )
    raw = expression.loc[keys].to_numpy(dtype=np.float64, copy=True)
    logged = np.log2(raw + 1.0)
    imputed = np.where(np.isnan(logged), medians, logged)
    return keys, ((imputed - means) / scales) @ components.T - offset


def preflight(cohort_id: str, split_strategy: str, outer_fold: int) -> None:
    features_path = feature_matrix_path(cohort_id, split_strategy, outer_fold)
    if not features_path.is_file():
        raise ContractError(
            f"{features_path} is absent. Run 'python run.py features --dataset "
            f"{cohort_id} --split-strategy {split_strategy} --outer-fold "
            f"{outer_fold}' first."
        )
    context, _, split_build_hash = load_context(cohort_id, split_strategy, outer_fold)
    features = pd.read_parquet(features_path)
    print(
        json.dumps(
            {
                "method_id": METHOD_ID,
                "dataset": cohort_id,
                "cohort_id": cohort_id,
                "split_strategy": split_strategy,
                "outer_fold": outer_fold,
                "split_build_hash": split_build_hash,
                "feature_dimensions": features.shape[1] - 1,
                "role_counts": context["role"].value_counts().to_dict(),
            },
            indent=2,
        )
    )


def _build_model(input_dimensions: int = NODE_FEATURE_DIMENSIONS):
    from torch import nn
    from torch_geometric.nn import SAGEConv
    import torch.nn.functional as functional

    class GraphSAGE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = SAGEConv(input_dimensions, HIDDEN_DIMENSIONS[0])
            self.conv2 = SAGEConv(HIDDEN_DIMENSIONS[0], HIDDEN_DIMENSIONS[1])
            self.conv3 = SAGEConv(HIDDEN_DIMENSIONS[1], 1)

        def forward(self, x, graph_edges):
            x = functional.relu(self.conv1(x, graph_edges))
            x = functional.dropout(x, p=0.5, training=self.training)
            x = functional.relu(self.conv2(x, graph_edges))
            x = functional.dropout(x, p=0.5, training=self.training)
            return self.conv3(x, graph_edges).reshape(-1)

    return GraphSAGE()


def _required_source_assets() -> list[dict[str, str]]:
    return [
        {
            "path": EXPRESSION_SOURCE.relative_to(REPO_ROOT).as_posix(),
            "purpose": "The TCGA expression matrix the stored components are applied to.",
        },
        {
            "path": BIOGRID_EDGES.relative_to(REPO_ROOT).as_posix(),
            "purpose": "BioGRID physical interaction edges; predict rebuilds this fold's graph from them.",
        },
    ]


def train_fold(
    cohort_id: str, split_strategy: str, outer_fold: int, model_dir: Path
) -> Path:
    import torch
    from torch import nn
    import torch_geometric
    from torch_geometric.loader import NeighborLoader

    feature_path = feature_matrix_path(cohort_id, split_strategy, outer_fold)
    if not feature_path.is_file():
        raise ContractError(
            f"{feature_path} is absent. Run 'python run.py features --dataset "
            f"{cohort_id} --split-strategy {split_strategy} --outer-fold "
            f"{outer_fold}' first."
        )
    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EssSubgraph training but is unavailable")
    set_random_seed(torch)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data, keys, index, split_build_hash = _build_graph(
        feature_path, cohort_id, split_strategy, outer_fold
    )
    node_feature_dimensions = int(data.x.shape[1])
    test_mask = data.test_mask.tolist()
    validation_mask = data.validation_mask.tolist()
    sample_counts = {
        "train": int(data.train_mask.sum()),
        "validation": int(data.validation_mask.sum()),
        "test": int(data.test_mask.sum()),
    }
    model = _build_model(node_feature_dimensions).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POSITIVE_WEIGHT], dtype=torch.float32, device=device)
    )
    loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,
        num_neighbors=list(NEIGHBOR_FANOUT),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    with RunRecord() as record:
        best_state = None
        best_loss = float("inf")
        best_epoch = 0
        stale = 0
        full = data.to(device)
        for epoch in range(MAX_EPOCHS):
            model.train()
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch.x, batch.edge_index)[: batch.batch_size]
                loss = criterion(logits, batch.y[: batch.batch_size])
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                logits = model(full.x, full.edge_index)
                validation_loss = float(
                    criterion(
                        logits[full.validation_mask], full.y[full.validation_mask]
                    ).item()
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
            raise RuntimeError("EssSubgraph training did not produce a checkpoint")
        model.load_state_dict(best_state)

        weights_dir = model_dir / WEIGHTS_DIRECTORY
        weights_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights_dir / MODEL_FILE)
        fold_feature_dir = feature_directory(
            ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
        )
        shutil.copyfile(fold_feature_dir / PCA_STATE_FILE, weights_dir / PCA_STATE_FILE)
        pd.DataFrame({"gene_key": keys}).to_parquet(
            weights_dir / GRAPH_NODES_FILE, index=False
        )

        del loader, full, data, optimizer, model, best_state, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
        fold_keys, probabilities = _score_fold(
            model_dir, cohort_id, split_strategy, outer_fold
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    position = {key: number for number, key in enumerate(fold_keys)}
    test_keys = [key for key, keep in zip(keys, test_mask, strict=True) if keep]
    validation_keys = [
        key for key, keep in zip(keys, validation_mask, strict=True) if keep
    ]
    test_scores = probabilities[[position[key] for key in test_keys]]
    validation_scores = probabilities[[position[key] for key in validation_keys]]

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

    versions = software_versions(np, pd, torch, torch_geometric)
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
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "positive_weight": POSITIVE_WEIGHT,
            "node_feature_dimensions": node_feature_dimensions,
            "neighbor_fanout": list(NEIGHBOR_FANOUT),
            "hidden_dimensions": list(HIDDEN_DIMENSIONS),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
        },
        sample_counts=sample_counts,
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
            "transforms": [
                f"{WEIGHTS_DIRECTORY}/{PCA_STATE_FILE}",
                f"{WEIGHTS_DIRECTORY}/{GRAPH_NODES_FILE}",
            ],
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


def _score_fold(
    model_dir: Path, cohort_id: str, split_strategy: str, outer_fold: int
) -> tuple[list[str], np.ndarray]:
    import torch
    from torch_geometric.data import Data

    fold_keys, values = fold_view(model_dir)
    index = {key: position for position, key in enumerate(fold_keys)}
    data = Data(
        x=torch.tensor(values, dtype=torch.float32), edge_index=_edge_index(index)
    )

    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EssSubgraph inference but is unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = _build_model(int(values.shape[1])).to(device)
    state = torch.load(model_dir / WEIGHTS_DIRECTORY / MODEL_FILE, map_location=device)
    model.load_state_dict(state)
    model.eval()
    full = data.to(device)
    with torch.no_grad():
        logits_all = model(full.x, full.edge_index)
        probabilities = torch.sigmoid(logits_all).cpu().numpy().astype(np.float64)
    return fold_keys, probabilities


def predict(model_dir: Path, genes_path: Path, output_dir: Path) -> Path:
    card = read_model_card(model_dir / "model_card.json")
    keys = _read_gene_keys(genes_path)

    with RunRecord() as record:
        fold_keys, probabilities = _score_fold(
            model_dir,
            str(card["cohort_id"]),
            str(card["split_strategy"]),
            int(card["outer_fold"]),
        )
    index = {key: position for position, key in enumerate(fold_keys)}
    missing = sorted(set(keys) - set(fold_keys))
    if missing:
        raise ContractError(
            f"EssSubgraph inference_scope is 'network': {len(missing)} of "
            f"{len(keys)} requested genes are not nodes of this fold's graph, "
            f"for example {missing[:_EXAMPLE_LIMIT]}"
        )
    scores = probabilities[[index[key] for key in keys]]
    threshold = float(card["decision_threshold"])
    write_predictions(output_dir, "blind", keys, scores, decision_threshold=threshold)
    return write_predict_manifest(output_dir, card, record, gene_count=len(keys))
