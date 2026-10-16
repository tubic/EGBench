"""Train one DeeplyEssential fold and score genes with a trained fold.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from data import load_prepared_data

ROOT = Path(__file__).resolve().parents[1]
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
from egbench.contracts import ContractError, read_model_card
from egbench.data import feature_directory, folds_directory

METHOD_ID = "deeplyessential"
UPSTREAM_COMMIT = "9c9d35938db6f1cd516902a5b6d834e6c18e4882"
DEVICE = "cuda"
DECISION_THRESHOLD = 0.5
WEIGHTS_DIRECTORY = "weights"
MODEL_FILE = "best_model.pt"
PREPROCESSING_FILE = "preprocessing.json"
_EXAMPLE_LIMIT = 10


def _fold_paths(cohort_id: str, split_strategy: str, outer_fold: int):
    feature_path = (
        feature_directory(
            ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
        )
        / "features.parquet"
    )
    folds = folds_directory(cohort_id, split_strategy=split_strategy, root=REPO_ROOT)
    return (
        cohort_id,
        feature_path,
        folds / "training_labels" / f"fold_{outer_fold}.parquet",
        folds / "split_assignments.parquet",
    )


def _split_build_hash(
    split_assignments_path: Path, cohort_id: str, split_strategy: str, outer_fold: int
) -> str:
    assignments = pd.read_parquet(
        split_assignments_path,
        columns=["cohort_id", "split_strategy", "outer_fold", "split_build_hash"],
    )
    scope = assignments.loc[
        assignments["cohort_id"].eq(cohort_id)
        & assignments["split_strategy"].eq(split_strategy)
        & assignments["outer_fold"].eq(outer_fold)
    ]
    hashes = set(scope["split_build_hash"].astype(str))
    if len(hashes) != 1:
        raise ContractError(
            f"{split_assignments_path}: split_build_hash is not unique for "
            f"this fold: {sorted(hashes)}"
        )
    return hashes.pop()


def preflight(cohort_id: str, split_strategy: str, outer_fold: int) -> None:
    cohort_id, feature_path, training_labels_path, split_assignments_path = _fold_paths(
        cohort_id, split_strategy, outer_fold
    )
    if not feature_path.is_file():
        raise ContractError(
            f"{feature_path} is absent. Run 'python run.py features --dataset "
            f"{cohort_id} --split-strategy {split_strategy} --outer-fold "
            f"{outer_fold}' first."
        )
    data = load_prepared_data(
        feature_path,
        training_labels_path,
        split_assignments_path,
        cohort_id,
        split_strategy,
        outer_fold,
    )
    split_build_hash = _split_build_hash(
        split_assignments_path, cohort_id, split_strategy, outer_fold
    )
    print(
        json.dumps(
            {
                "method_id": METHOD_ID,
                "cohort_id": cohort_id,
                "split_strategy": split_strategy,
                "outer_fold": outer_fold,
                "split_build_hash": split_build_hash,
                "feature_dimensions": len(data.feature_names),
                "role_counts": {
                    "train": len(data.train.gene_keys),
                    "validation": len(data.validation.gene_keys),
                    "test": len(data.test.gene_keys),
                },
            },
            indent=2,
        )
    )


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= DECISION_THRESHOLD).astype(np.int64)
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "mcc_at_0_5": float(matthews_corrcoef(labels, predictions)),
    }


def _required_source_assets() -> list[dict[str, str]]:
    return [
        {
            "path": (
                "methods/deeplyessential/feature/source/"
                "deeplyessential_sequence_89.parquet"
            ),
            "purpose": (
                "89-dimensional sequence-derived feature matrix; a gene "
                "absent here cannot be scored."
            ),
        },
    ]


def train_fold(
    cohort_id: str, split_strategy: str, outer_fold: int, model_dir: Path
) -> Path:
    import sklearn
    import torch

    from model import TrainConfig, resolve_device
    from training import predict_probabilities, train_model

    cohort_id, feature_path, training_labels_path, split_assignments_path = _fold_paths(
        cohort_id, split_strategy, outer_fold
    )
    if not feature_path.is_file():
        raise ContractError(
            f"{feature_path} is absent. Run 'python run.py features --dataset "
            f"{cohort_id} --split-strategy {split_strategy} --outer-fold "
            f"{outer_fold}' first."
        )
    device = resolve_device(DEVICE, allow_cpu=False)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    data = load_prepared_data(
        feature_path,
        training_labels_path,
        split_assignments_path,
        cohort_id,
        split_strategy,
        outer_fold,
    )
    split_build_hash = _split_build_hash(
        split_assignments_path, cohort_id, split_strategy, outer_fold
    )
    config = TrainConfig()

    with RunRecord() as record:
        result = train_model(
            data.train.features,
            data.train.labels,
            data.validation.features,
            data.validation.labels,
            device,
            config,
        )
        validation_probabilities = predict_probabilities(
            result.model, data.validation.features, device
        )
        test_probabilities = predict_probabilities(result.model, data.test.features, device)
        torch.cuda.synchronize(device)

    metrics = {
        "validation": _metrics(data.validation.labels, validation_probabilities),
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "positive_weight": result.positive_weight,
    }

    weights_dir = model_dir / WEIGHTS_DIRECTORY
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu()
                for name, value in result.model.state_dict().items()
            },
            "config": result.config,
            "upstream_commit": UPSTREAM_COMMIT,
        },
        weights_dir / MODEL_FILE,
    )
    (weights_dir / PREPROCESSING_FILE).write_text(
        json.dumps(
            {
                "feature_names": data.feature_names,
                "standard_scaler_mean": data.scaler_mean.tolist(),
                "standard_scaler_scale": data.scaler_scale.tolist(),
                "fit_partition": "train_only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_predictions(
        model_dir,
        "blind",
        data.test.gene_keys,
        test_probabilities,
        decision_threshold=DECISION_THRESHOLD,
    )
    write_predictions(
        model_dir,
        "validation",
        data.validation.gene_keys,
        validation_probabilities,
        decision_threshold=DECISION_THRESHOLD,
    )

    versions = software_versions(np, pd, sklearn, torch)
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
            "upstream_commit": UPSTREAM_COMMIT,
            **result.config,
            "adapter_decisions": {
                "output": "single_logit_bce_with_logits",
                "imbalance": "all_train_rows_with_train_negative_to_positive_loss_weight",
                "scaling": "standard_scaler_fit_on_train_only",
                "selection": "best_validation_loss_checkpoint",
                "formal_device": "cuda_required",
            },
            "metrics": metrics,
        },
        sample_counts={
            "train": len(data.train.gene_keys),
            "validation": len(data.validation.gene_keys),
            "test": len(data.test.gene_keys),
        },
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
        outer_fold=outer_fold,
        split_build_hash=split_build_hash,
        decision_threshold=DECISION_THRESHOLD,
        artifacts={
            "weights": f"{WEIGHTS_DIRECTORY}/{MODEL_FILE}",
            "transforms": [f"{WEIGHTS_DIRECTORY}/{PREPROCESSING_FILE}"],
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

    from model import DeeplyEssentialDNN, resolve_device
    from training import predict_probabilities

    card = read_model_card(model_dir / "model_card.json")
    keys = _read_gene_keys(genes_path)
    source = (
        ROOT / "feature" / "source" / "deeplyessential_sequence_89.parquet"
    )
    if not source.is_file():
        raise ContractError(f"the source feature table is absent: {source}")
    matrix = pd.read_parquet(source).set_index("gene_key", drop=False)
    missing = sorted(set(keys) - set(matrix.index))
    if missing:
        raise ContractError(
            f"DeeplyEssential inference_scope is 'sequence': {len(missing)} "
            f"of {len(keys)} requested genes are absent from the source "
            f"feature table, for example {missing[:_EXAMPLE_LIMIT]}"
        )
    feature_names = [column for column in matrix.columns if column != "gene_key"]
    values = matrix.loc[keys, feature_names].to_numpy(dtype=np.float64, copy=True)

    weights_dir = model_dir / WEIGHTS_DIRECTORY
    preprocessing = json.loads((weights_dir / PREPROCESSING_FILE).read_text(encoding="utf-8"))
    if list(preprocessing["feature_names"]) != feature_names:
        raise ContractError(
            f"{source} column order does not match the fitted scaler's "
            f"feature_names in {PREPROCESSING_FILE}"
        )
    mean = np.asarray(preprocessing["standard_scaler_mean"], dtype=np.float64)
    scale = np.asarray(preprocessing["standard_scaler_scale"], dtype=np.float64)
    scaled = ((values - mean) / scale).astype(np.float32)

    device = resolve_device(DEVICE, allow_cpu=False)
    checkpoint = torch.load(weights_dir / MODEL_FILE, map_location=device)
    model = DeeplyEssentialDNN(dropout=checkpoint["config"]["dropout"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    with RunRecord() as record:
        probabilities = predict_probabilities(model, scaled, device)

    threshold = float(card["decision_threshold"])
    write_predictions(output_dir, "blind", keys, probabilities, decision_threshold=threshold)
    return write_predict_manifest(output_dir, card, record, gene_count=len(keys))
