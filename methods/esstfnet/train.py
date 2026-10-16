"""Train one EssTFNet fold and score genes with a trained fold.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import dataclasses
import os
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(METHOD_ROOT / "code"))

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

METHOD_ID = "esstfnet"

DEVICE = "cuda"

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

WEIGHTS_DIRECTORY = "weights"
MODEL_FILE = "best_model.pt"
PHYSICOCHEMICAL_SCALER_FILE = "physicochemical_scaler_state.npz"
SELECTION_MANIFEST_FILE = "feature_selection_manifest.json"
EVO2_SCALER_FILE = "evo2_scaler_state.npz"


def set_random_seed(torch) -> None:

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:

    best_threshold = 0.5
    best_accuracy = 0.0
    for step in range(1, 100):
        threshold = step / 100
        accuracy = float(np.mean((scores >= threshold).astype(np.int64) == labels))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold


def _versions(torch) -> dict[str, str]:
    import pyarrow

    return software_versions(np, pd, pyarrow, sklearn, torch)


def _required_source_assets(cohort_id: str) -> list[dict[str, str]]:
    return [
        {
            "path": (
                f"methods/esstfnet/feature/source/{cohort_id}/"
                f"raw_candidates_86151.parquet"
            ),
            "purpose": (
                "the 86,151 raw physicochemical candidate columns; inference "
                "takes this fold's selected columns out of them"
            ),
        },
        {
            "path": (
                f"methods/esstfnet/feature/source/{cohort_id}/"
                f"evo2_7b_base_embeddings.parquet"
            ),
            "purpose": "the pooled Evo2 7B-base embedding of each coding sequence",
        },
        {
            "path": "methods/esstfnet/feature/source/sequences",
            "purpose": (
                "the paired Ensembl 97 and GENCODE 50 coding and peptide "
                "sequences the two one-hot branches read"
            ),
        },
    ]


def _fold_transforms(feature_directory: Path, weights: Path) -> list[str]:

    import shutil

    weights.mkdir(parents=True, exist_ok=True)
    copied = (PHYSICOCHEMICAL_SCALER_FILE, SELECTION_MANIFEST_FILE)
    absent = [name for name in copied if not (feature_directory / name).is_file()]
    if absent:
        raise ContractError(
            "this fold's fitted transforms are absent from "
            f"{feature_directory}: {', '.join(absent)}"
        )
    for name in copied:
        shutil.copy2(feature_directory / name, weights / name)
    return [
        f"{WEIGHTS_DIRECTORY}/{name}"
        for name in (*copied, EVO2_SCALER_FILE)
    ]


def train_fold(
    *, cohort_id: str, split_strategy: str, outer_fold: int, model_dir: Path
) -> Path:

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from adapter import EssTFNet, FocalLoss, Hyperparameters, initial_model
    from data import load_fold

    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ContractError("EssTFNet requires CUDA and torch reports none available")
    set_random_seed(torch)
    parameters = Hyperparameters()
    partitions, evo2_statistics = load_fold(
        cohort_id,
        split_strategy,
        outer_fold,
        parameters.max_len,
        parameters.protein_max_len,
    )
    train, validation, test = (
        partitions["train"],
        partitions["validation"],
        partitions["test"],
    )
    split_build_hash = _split_build_hash(cohort_id, split_strategy, outer_fold)
    positives = int((train.labels == 1).sum())
    negatives = int((train.labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ContractError("the train role must carry both classes")
    model = EssTFNet(
        parameters,
        feature_dimensions=train.features.shape[1],
        language_model_dimensions=train.language_model_embedding.shape[1],
    ).to(device)
    initial_model(model)
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=parameters.learning_rate,
        weight_decay=parameters.weight_decay,
    )
    criterion = FocalLoss(
        gamma=0, pos_weight=negatives / positives, logits=False, reduction="sum"
    )

    def tensors(partition, with_labels: bool):
        stack = [
            partition.dna_tokens,
            partition.protein_tokens,
            partition.features,
            partition.language_model_embedding,
        ]
        if with_labels:
            stack.append(torch.from_numpy(partition.labels).to(torch.float32))
        return TensorDataset(*stack)

    def score(partition) -> np.ndarray:
        loader = DataLoader(
            tensors(partition, False), batch_size=parameters.batch_size, shuffle=False
        )
        model.eval()
        outputs = []
        with torch.no_grad():
            for batch in loader:
                outputs.append(
                    model(*[item.to(device) for item in batch]).detach().cpu().numpy()
                )
        return np.concatenate(outputs).astype(np.float64)

    train_loader = DataLoader(
        tensors(train, True),
        batch_size=parameters.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    with RunRecord() as record:
        torch.cuda.reset_peak_memory_stats(device)
        best_state = None
        best_auc = 0.0
        best_epoch = 0
        stale = 0
        for epoch in range(parameters.epoch_num):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                *inputs, y = [item.to(device) for item in batch]
                optimizer.zero_grad()
                loss = criterion(model(*inputs), y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            validation_auc = float(
                roc_auc_score(validation.labels, score(validation))
            )
            print(
                f"epoch {epoch + 1} "
                f"train_loss {total_loss / len(train.gene_keys):.5f} "
                f"validation_auc {validation_auc:.5f}",
                flush=True,
            )
            if validation_auc > best_auc:
                best_auc = validation_auc
                best_epoch = epoch + 1
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= parameters.patience:
                    break
        if best_state is None:
            raise ContractError("EssTFNet training did not produce a checkpoint")
        model.load_state_dict(best_state)
        model.to(device)
        validation_scores = score(validation)
        test_scores = score(test)
        threshold = select_threshold(validation.labels, validation_scores)

    weights = model_dir / WEIGHTS_DIRECTORY
    weights.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "feature_dimensions": int(train.features.shape[1]),
            "language_model_dimensions": int(
                train.language_model_embedding.shape[1]
            ),
        },
        weights / MODEL_FILE,
    )
    np.savez_compressed(
        weights / EVO2_SCALER_FILE,
        mean=evo2_statistics["mean"],
        scale=evo2_statistics["scale"],
    )
    transforms = _fold_transforms(
        feature_directory(
            METHOD_ROOT,
            cohort_id,
            split_strategy=split_strategy,
            outer_fold=outer_fold,
        ),
        weights,
    )
    write_predictions(
        model_dir,
        "validation",
        validation.gene_keys,
        validation_scores,
        decision_threshold=threshold,
    )
    write_predictions(
        model_dir,
        "blind",
        test.gene_keys,
        test_scores,
        decision_threshold=threshold,
    )
    versions = _versions(torch)
    write_run_manifest(
        model_dir,
        method_id=METHOD_ID,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=int(outer_fold),
        split_build_hash=split_build_hash,
        software_versions=versions,
        parameters={
            **dataclasses.asdict(parameters),
            "device": str(device),
            "positive_weight": negatives / positives,
        },
        sample_counts={
            role: len(partitions[role].gene_keys) for role in partitions
        },
        best_epoch=best_epoch,
        **record.timestamps,
    )
    write_resource_usage(
        model_dir,
        record,
        device=str(device),
        gpu_name=torch.cuda.get_device_name(device),
        peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(device)),
    )
    return write_model_card(
        model_dir,
        method_id=METHOD_ID,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=int(outer_fold),
        split_build_hash=split_build_hash,
        decision_threshold=float(threshold),
        artifacts={
            "weights": f"{WEIGHTS_DIRECTORY}/{MODEL_FILE}",
            "transforms": transforms,
        },
        required_source_assets=_required_source_assets(cohort_id),
        software_versions=versions,
    )


def _split_build_hash(cohort_id: str, split_strategy: str, outer_fold: int) -> str:

    path = (
        folds_directory(cohort_id, split_strategy=split_strategy, root=REPO_ROOT)
        / "split_assignments.parquet"
    )
    frame = pd.read_parquet(path, columns=["outer_fold", "split_strategy", "split_build_hash"])
    selected = frame.loc[
        frame["outer_fold"].eq(outer_fold)
        & frame["split_strategy"].astype(str).eq(split_strategy),
        "split_build_hash",
    ]
    values = sorted(set(selected.astype(str)))
    if len(values) != 1:
        raise ContractError(
            f"{path}: fold {outer_fold} of {split_strategy} carries "
            f"{len(values)} split_build_hash values"
        )
    return values[0]


def score_genes(
    *, model_dir: Path, gene_keys: Sequence[str], output_dir: Path
) -> Path:

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from adapter import EssTFNet, Hyperparameters
    from data import inference_partition

    card = read_model_card(model_dir / "model_card.json")
    cohort_id = str(card["cohort_id"])
    device = torch.device(DEVICE)
    if not torch.cuda.is_available():
        raise ContractError("EssTFNet requires CUDA and torch reports none available")
    set_random_seed(torch)
    parameters = Hyperparameters()
    weights = model_dir / WEIGHTS_DIRECTORY
    stored = torch.load(weights / MODEL_FILE, map_location="cpu", weights_only=False)

    with RunRecord() as record:
        torch.cuda.reset_peak_memory_stats(device)
        partition = inference_partition(
            cohort_id,
            [str(key) for key in gene_keys],
            parameters.max_len,
            parameters.protein_max_len,
            physicochemical_state=weights / PHYSICOCHEMICAL_SCALER_FILE,
            evo2_state=weights / EVO2_SCALER_FILE,
        )
        model = EssTFNet(
            parameters,
            feature_dimensions=int(stored["feature_dimensions"]),
            language_model_dimensions=int(stored["language_model_dimensions"]),
        )
        model.load_state_dict(stored["state_dict"])
        model = model.to(device)
        model.eval()
        loader = DataLoader(
            TensorDataset(
                partition.dna_tokens,
                partition.protein_tokens,
                partition.features,
                partition.language_model_embedding,
            ),
            batch_size=parameters.batch_size,
            shuffle=False,
        )
        outputs = []
        with torch.no_grad():
            for batch in loader:
                outputs.append(
                    model(*[item.to(device) for item in batch]).detach().cpu().numpy()
                )
        scores = np.concatenate(outputs).astype(np.float64)

    write_predictions(
        output_dir,
        "blind",
        partition.gene_keys,
        scores,
        decision_threshold=float(card["decision_threshold"]),
    )
    return write_predict_manifest(
        output_dir, card, record, gene_count=len(partition.gene_keys)
    )
