#!/usr/bin/env python3
"""Entry point of DeepCellEss: preflight, features, train and predict for one fold of one task.
Run: python methods/deepcelless/run.py {preflight,features,train} --dataset <task> --split-strategy family_5fold --outer-fold <k>
     python methods/deepcelless/run.py predict --model-dir <fold model dir> --genes <gene_key parquet> --output <new dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "deepcelless_build_features", ROOT / "build_features.py"
)
build_features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_features)

import pandas as pd
import torch

from egbench.artifacts import (
    RunRecord,
    write_predict_manifest,
    write_predictions,
)
from egbench.contracts import (
    COHORT_IDS,
    OUTER_FOLDS,
    ContractError,
    SPLIT_STRATEGIES,
    read_model_card,
)
from egbench.data import (
    feature_directory,
    folds_directory,
    model_directory,
)

import adapter
import train as code_train

METHOD_ID = "deepcelless"

DEVICE = "cuda:0"
MAX_EPOCHS = 500
PATIENCE = 30


def _feature_root(cohort_id: str, split_strategy: str, outer_fold: int) -> Path:
    return feature_directory(
        ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
    )


def _required_source_assets(cohort_id: str) -> list[dict[str, str]]:
    path = ROOT / "feature" / "source" / "selected_proteins" / f"{cohort_id}.fa.gz"
    return [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "purpose": (
                "The selected protein FASTA of this cohort; inference reads a "
                "gene's sequence from it and cannot score a gene it omits."
            ),
        }
    ]


def _load(cohort_id: str, split_strategy: str, outer_fold: int) -> tuple[code_train.PreparedData, dict[str, Path]]:
    feature_root = _feature_root(cohort_id, split_strategy, outer_fold)
    folds = folds_directory(cohort_id, split_strategy=split_strategy, root=REPO_ROOT)
    paths = {
        "assignments": folds / "split_assignments.parquet",
        "labels": folds / "training_labels" / f"fold_{outer_fold}.parquet",
        "sequence": feature_root / "selected_proteins.fa.gz",
    }
    data = code_train.load_prepared_data(
        paths["assignments"],
        paths["labels"],
        paths["sequence"],
        cohort_id,
        split_strategy,
        outer_fold,
    )
    return data, paths


def _split_context(cohort_id: str, split_strategy: str, outer_fold: int) -> pd.DataFrame:
    folds = folds_directory(cohort_id, split_strategy=split_strategy, root=REPO_ROOT)
    assignments = pd.read_parquet(folds / "split_assignments.parquet")
    context = assignments.loc[
        assignments["cohort_id"].astype(str).eq(cohort_id)
        & assignments["split_strategy"].astype(str).eq(split_strategy)
        & assignments["outer_fold"].eq(outer_fold)
    ]
    if context.empty:
        raise ContractError(
            f"no split assignments for {cohort_id}/{split_strategy}/fold {outer_fold}"
        )
    return context


def preflight(cohort_id: str, split_strategy: str, outer_fold: int) -> dict[str, object]:
    output = model_directory(
        ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
    )
    if output.exists():
        raise ContractError(
            f"the products of this fold are already there: {output}. A run "
            f"writes a directory that does not yet exist; remove it to run again"
        )
    context = _split_context(cohort_id, split_strategy, outer_fold)
    split_build_hash = sorted(set(context["split_build_hash"].astype(str)))
    if len(split_build_hash) != 1:
        raise ContractError(
            f"fold {outer_fold} of {cohort_id}/{split_strategy} carries "
            f"{len(split_build_hash)} split_build_hash values"
        )
    sample_counts = {
        role: int(context["role"].eq(role).sum())
        for role in ("train", "validation", "test")
    }
    sequence_path = (
        _feature_root(cohort_id, split_strategy, outer_fold) / "selected_proteins.fa.gz"
    )
    return {
        "method_id": METHOD_ID,
        "sample_counts": sample_counts,
        "split_build_hash": split_build_hash[0],
        "batch_size": adapter.FORMAL_BATCH_SIZE,
        "fold_features_present": sequence_path.is_file(),
        "model_directory": str(output.relative_to(ROOT)),
    }


def features(cohort_id: str, split_strategy: str, outer_fold: int) -> dict[str, object]:
    directory = _feature_root(cohort_id, split_strategy, outer_fold)
    target = directory / "selected_proteins.fa.gz"
    if target.is_file():
        return {
            "status": "already materialized",
            "features": str(target.relative_to(ROOT)),
        }
    if directory.exists() and any(directory.iterdir()):
        raise ContractError(
            f"{directory} holds something other than selected_proteins.fa.gz; "
            f"remove it to build this fold again"
        )
    build_features.build(cohort_id)
    return {"status": "materialized", "features": str(target.relative_to(ROOT))}


def train(cohort_id: str, split_strategy: str, outer_fold: int) -> dict[str, object]:
    output = model_directory(
        ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
    )
    data, paths = _load(cohort_id, split_strategy, outer_fold)
    return code_train.run_formal_training(
        data,
        assignments_path=paths["assignments"],
        training_labels_path=paths["labels"],
        sequence_path=paths["sequence"],
        output_dir=output,
        cohort=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        device_name=DEVICE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        required_source_assets=_required_source_assets(cohort_id),
    )


def predict(model_dir: Path, genes: Path, output: Path) -> dict[str, object]:
    card = read_model_card(model_dir / "model_card.json")
    cohort_id = str(card["cohort_id"])

    requested = pd.read_parquet(genes)
    if list(requested.columns) != ["gene_key"]:
        raise ContractError(
            f"{genes}: the gene list holds exactly one column, 'gene_key', "
            f"found {list(requested.columns)}"
        )
    gene_keys = requested["gene_key"].astype(str).tolist()
    if len(set(gene_keys)) != len(gene_keys):
        raise ContractError(f"{genes}: duplicate gene_key values")

    fasta_path = ROOT / "feature" / "source" / "selected_proteins" / f"{cohort_id}.fa.gz"
    sequence_by_key = code_train.read_fasta(fasta_path)
    missing = sorted(set(gene_keys).difference(sequence_by_key))
    if missing:
        raise ContractError(
            f"{len(missing)} of {len(gene_keys)} requested genes are absent "
            f"from {fasta_path}, the first ten are {missing[:10]}. DeepCellEss "
            f"scores a gene from its selected protein sequence and cannot "
            f"score a gene it omits"
        )
    sequences = [sequence_by_key[key] for key in gene_keys]
    device = code_train.inference_device()
    adapter.deterministic_kernels()
    payload = torch.load(
        model_dir / str(card["artifacts"]["weights"]), map_location=device
    )
    model = adapter.DeepCellEss()
    model.load_state_dict(payload["state_dict"])
    model = model.to(device)
    model.eval()

    with RunRecord() as record:
        features_matrix, _ = adapter.encode_proteins(sequences)
        scores = adapter.predict_scores(
            model,
            features_matrix,
            device=str(device),
            batch_size=adapter.FORMAL_BATCH_SIZE,
        )
    threshold = float(card["decision_threshold"])
    write_predictions(output, "blind", gene_keys, scores, decision_threshold=threshold)
    write_predict_manifest(output, card, record, gene_count=len(gene_keys))
    return {"output": str(output), "genes": len(gene_keys)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "features", "train"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--dataset", choices=COHORT_IDS, required=True)
        sub.add_argument("--split-strategy", choices=SPLIT_STRATEGIES, required=True)
        sub.add_argument("--outer-fold", type=int, choices=OUTER_FOLDS, required=True)
    inference = subparsers.add_parser("predict")
    inference.add_argument("--model-dir", type=Path, required=True)
    inference.add_argument("--genes", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            report = preflight(args.dataset, args.split_strategy, args.outer_fold)
        elif args.command == "features":
            report = features(args.dataset, args.split_strategy, args.outer_fold)
        elif args.command == "train":
            report = train(args.dataset, args.split_strategy, args.outer_fold)
        else:
            report = predict(args.model_dir, args.genes, args.output)
    except ContractError as error:
        print(f"{METHOD_ID}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
