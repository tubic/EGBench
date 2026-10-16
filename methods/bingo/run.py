#!/usr/bin/env python3
"""Entry point of BiNGO: preflight, features, train and predict for one fold of one task.
Run: python methods/bingo/run.py {preflight,features,train} --dataset <task> --split-strategy family_5fold --outer-fold <k>
     python methods/bingo/run.py predict --model-dir <fold model dir> --genes <gene_key parquet> --output <new dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]

sys.path[:] = [
    entry for entry in sys.path if entry and Path(entry).resolve() != METHOD_ROOT
]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(METHOD_ROOT / "code"))

from egbench.contracts import (
    COHORT_IDS,
    OUTER_FOLDS,
    SPLIT_STRATEGIES,
    ContractError,
    read_model_card,
)
from egbench.data import (
    feature_directory,
    folds_directory,
    model_directory,
)


def _beside(name: str):

    specification = importlib.util.spec_from_file_location(
        name, METHOD_ROOT / f"{name}.py"
    )
    if specification is None or specification.loader is None:
        raise ContractError(f"{METHOD_ROOT / (name + '.py')} is not importable")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


SHARED_CACHE = METHOD_ROOT / "feature" / "source" / "esm2_graph_cache"
REBUILD_COMMAND = "python cache_esm2_graphs.py --device cuda:0"

FEATURE_PRODUCTS = (
    "cache_index.parquet",
    "cache_manifest.json",
    "cache_provenance.json",
    "selected_proteins.fa.gz",
)


def _feature_state(products: list[Path]) -> str:

    present = [path for path in products if path.exists()]
    if not present:
        return "absent"
    return "complete" if len(present) == len(products) else "partial"


def _partial_features(directory: Path, products: list[Path]) -> str:
    absent = [
        str(path.relative_to(directory))
        if path.is_relative_to(directory)
        else str(path)
        for path in products
        if not path.exists()
    ]
    return (
        f"{directory} holds only part of these features, absent: "
        f"{', '.join(absent)}. Delete the directory and run features again; "
        f"this entry point never fills a half built fold in."
    )


def _cohort(cohort_id: str) -> str:
    if cohort_id not in COHORT_IDS:
        raise ContractError(
            f"unknown cohort {cohort_id!r}, the cohorts are {list(COHORT_IDS)}"
        )
    return cohort_id


def _fold_view(cohort_id: str, split_strategy: str, outer_fold: int) -> Path:
    return feature_directory(
        METHOD_ROOT,
        _cohort(cohort_id),
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )


def _dataset_feature_root(cohort_id: str) -> Path:

    return _fold_view(cohort_id, SPLIT_STRATEGIES[0], OUTER_FOLDS[0]).parents[1]


def _require_absent(directory: Path, what: str) -> Path:
    if directory.exists():
        raise ContractError(
            f"{what} already exists: {directory}. Remove it before running this "
            f"fold again; this entry point never overwrites."
        )
    return directory


def _require_shared_cache() -> None:

    absent = [
        path
        for path in (
            SHARED_CACHE / "graphs",
            SHARED_CACHE / "cache_index.parquet",
            SHARED_CACHE / "cache_manifest.json",
        )
        if not path.exists()
    ]
    if absent:
        raise ContractError(
            "the shared ESM-2 graph cache is incomplete, absent: "
            + ", ".join(str(path) for path in absent)
            + f". It is built once by '{REBUILD_COMMAND}' from this directory."
        )


def preflight(cohort_id: str, split_strategy: str, outer_fold: int) -> None:

    cohort_id = _cohort(cohort_id)
    _require_shared_cache()
    folds = folds_directory(cohort_id, split_strategy=split_strategy, root=REPO_ROOT)
    required = [
        folds / "split_assignments.parquet",
        folds / "training_labels" / f"fold_{outer_fold}.parquet",
        METHOD_ROOT / "feature" / "source" / f"{cohort_id}_protein_sequences.fa.gz",
    ]
    absent = [path for path in required if not path.is_file()]
    if absent:
        raise ContractError(
            "BiNGO cannot run this fold, the following inputs are absent: "
            + ", ".join(str(path) for path in absent)
        )
    import torch
    import torch_geometric

    from train import DEVICE

    if not torch.cuda.is_available():
        raise ContractError("BiNGO requires CUDA and torch reports none available")
    device = torch.device(DEVICE)
    view = _fold_view(cohort_id, split_strategy, outer_fold)
    model = model_directory(
        METHOD_ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
    )
    print("method_id bingo")
    print(f"cohort_id {cohort_id} split_strategy {split_strategy} outer_fold {outer_fold}")
    print(f"device {device} {torch.cuda.get_device_name(device)}")
    print(f"graph cache {SHARED_CACHE / 'graphs'}")
    print(f"features {view} {'present' if view.is_dir() else 'absent'}")
    print(f"model {model} {'present' if model.is_dir() else 'absent'}")


def features(cohort_id: str, split_strategy: str, outer_fold: int) -> None:

    _cohort(cohort_id)
    _require_shared_cache()
    products = [
        _fold_view(cohort_id, strategy, fold) / name
        for strategy in SPLIT_STRATEGIES
        for fold in OUTER_FOLDS
        for name in FEATURE_PRODUCTS
    ]
    state = _feature_state(products)
    if state == "complete":
        print(f"already materialized {_dataset_feature_root(cohort_id)}")
        return
    if state == "partial":
        raise ContractError(_partial_features(_dataset_feature_root(cohort_id), products))
    build_features = _beside("build_features")
    for index in build_features.build(cohort_id):
        print(index)


def train(cohort_id: str, split_strategy: str, outer_fold: int) -> None:

    from train import train_fold

    cohort_id = _cohort(cohort_id)
    view = _fold_view(cohort_id, split_strategy, outer_fold)
    if not (view / "cache_index.parquet").is_file():
        raise ContractError(
            f"the fold view is absent: {view / 'cache_index.parquet'}. Run "
            f"'features' for this dataset first."
        )
    model_dir = _require_absent(
        model_directory(
            METHOD_ROOT,
            cohort_id,
            split_strategy=split_strategy,
            outer_fold=outer_fold,
        ),
        "the model directory of this fold",
    )
    print(
        train_fold(
            folds_dir=folds_directory(
                cohort_id, split_strategy=split_strategy, root=REPO_ROOT
            ),
            cache_dir=view,
            model_dir=model_dir,
            cohort=cohort_id,
            split_strategy=split_strategy,
            outer_fold=outer_fold,
        )
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


def predict(model_dir: Path, genes: Path, output: Path) -> None:

    from train import score_genes

    card_path = model_dir / "model_card.json"
    if not card_path.is_file():
        raise ContractError(f"{model_dir} holds no model_card.json")
    read_model_card(card_path)
    _require_shared_cache()
    print(
        score_genes(
            model_dir=model_dir,
            gene_keys=_read_gene_keys(genes),
            shared_cache=SHARED_CACHE,
            output_dir=_require_absent(output, "the predict output directory"),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("preflight", "check that this fold has every input it needs"),
        ("features", "cut this fold's view of the shared graph cache"),
        ("train", "fit this fold and write its products"),
    ):
        subcommand = subcommands.add_parser(name, help=description)
        subcommand.add_argument("--dataset", choices=COHORT_IDS, required=True)
        subcommand.add_argument(
            "--split-strategy", choices=SPLIT_STRATEGIES, required=True
        )
        subcommand.add_argument(
            "--outer-fold", type=int, choices=OUTER_FOLDS, required=True
        )
    scoring = subcommands.add_parser(
        "predict", help="score a gene list with a trained fold"
    )
    scoring.add_argument("--model-dir", type=Path, required=True)
    scoring.add_argument("--genes", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "predict":
            predict(arguments.model_dir, arguments.genes, arguments.output)
        else:
            {"preflight": preflight, "features": features, "train": train}[
                arguments.command
            ](arguments.dataset, arguments.split_strategy, arguments.outer_fold)
    except ContractError as error:
        print(f"bingo {arguments.command}: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
