#!/usr/bin/env python3
"""Entry point of EssTFNet: preflight, features, train and predict for one fold of one task.
Run: python methods/esstfnet/run.py {preflight,features,train} --dataset <task> --split-strategy family_5fold --outer-fold <k>
     python methods/esstfnet/run.py predict --model-dir <fold model dir> --genes <gene_key parquet> --output <new dir>
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


def _method_modules():

    for name in ("local_dataset", "build_features"):
        if name not in sys.modules:
            _beside(name)
    return sys.modules["build_features"]


FEATURE_PRODUCTS = (
    "physicochemical_features.parquet",
    "physicochemical_scaler_state.npz",
    "evo2_embeddings.parquet",
    "feature_selection_manifest.json",
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


def _require_absent(directory: Path, what: str) -> Path:
    if directory.exists():
        raise ContractError(
            f"{what} already exists: {directory}. Remove it before running this "
            f"fold again; this entry point never overwrites."
        )
    return directory


def _source_assets(cohort_id: str) -> list[Path]:
    SOURCE_ROOT = _method_modules().SOURCE_ROOT

    return [
        SOURCE_ROOT / cohort_id / "raw_candidates_86151.parquet",
        SOURCE_ROOT / cohort_id / "evo2_7b_base_embeddings.parquet",
        SOURCE_ROOT / "feature_contract" / "raw_candidate_columns.parquet",
        SOURCE_ROOT / "sequences" / "ensembl97.cds.fa.gz",
        SOURCE_ROOT / "sequences" / "ensembl97.pep.fa.gz",
    ]


def _require_source_assets(cohort_id: str) -> None:
    build_features = _method_modules()
    EVO2_REBUILD = build_features.EVO2_REBUILD
    RAW_CANDIDATE_REBUILD = build_features.RAW_CANDIDATE_REBUILD

    absent = [path for path in _source_assets(cohort_id) if not path.exists()]
    if absent:
        raise ContractError(
            "the whole-genome EssTFNet assets are incomplete, absent: "
            + ", ".join(str(path) for path in absent)
            + ". They are built once by the rebuild_command entries of "
            + f"method.yaml: '{RAW_CANDIDATE_REBUILD.format(dataset=cohort_id)}' "
            + f"and '{EVO2_REBUILD.format(dataset=cohort_id)}'."
        )


def preflight(cohort_id: str, split_strategy: str, outer_fold: int) -> None:

    cohort_id = _cohort(cohort_id)
    _require_source_assets(cohort_id)
    folds = folds_directory(cohort_id, split_strategy=split_strategy, root=REPO_ROOT)
    required = [
        folds / "split_assignments.parquet",
        folds / "fold_genes" / f"fold_{outer_fold}_genes.parquet",
        folds / "training_labels" / f"fold_{outer_fold}.parquet",
        METHOD_ROOT / "support_gene_list.csv",
    ]
    absent = [path for path in required if not path.is_file()]
    if absent:
        raise ContractError(
            "EssTFNet cannot run this fold, the following inputs are absent: "
            + ", ".join(str(path) for path in absent)
        )
    import torch

    from adapter import EssTFNet
    from data import load_fold

    DEVICE = _beside("train").DEVICE

    if not torch.cuda.is_available():
        raise ContractError("EssTFNet requires CUDA and torch reports none available")
    device = torch.device(DEVICE)
    features = feature_directory(
        METHOD_ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
    )
    model = model_directory(
        METHOD_ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
    )
    print("method_id esstfnet")
    print(f"cohort_id {cohort_id} split_strategy {split_strategy} outer_fold {outer_fold}")
    print(f"device {device} {torch.cuda.get_device_name(device)}")
    for path in _source_assets(cohort_id):
        print(f"source {path}")
    print(f"features {features} {'present' if features.is_dir() else 'absent'}")
    print(f"model {model} {'present' if model.is_dir() else 'absent'}")


def features(cohort_id: str, split_strategy: str, outer_fold: int) -> None:

    import json

    cohort_id = _cohort(cohort_id)
    _require_source_assets(cohort_id)
    directory = feature_directory(
        METHOD_ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
    )
    products = [directory / name for name in FEATURE_PRODUCTS]
    state = _feature_state(products)
    if state == "complete":
        print(f"already materialized {directory}")
        return
    if state == "partial":
        raise ContractError(_partial_features(directory, products))
    print(
        json.dumps(
            _method_modules().build_fold(
                cohort_id, outer_fold, split_strategy=split_strategy
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def train(cohort_id: str, split_strategy: str, outer_fold: int) -> None:

    _method_modules()
    train_fold = _beside("train").train_fold

    cohort_id = _cohort(cohort_id)
    fold_features = feature_directory(
        METHOD_ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
    )
    absent = [
        fold_features / name
        for name in FEATURE_PRODUCTS
        if not (fold_features / name).is_file()
    ]
    if absent:
        raise ContractError(
            "the fold view is incomplete, absent: "
            + ", ".join(str(path) for path in absent)
            + ". Run 'features' for this fold first."
        )
    print(
        train_fold(
            cohort_id=cohort_id,
            split_strategy=split_strategy,
            outer_fold=outer_fold,
            model_dir=_require_absent(
                model_directory(
                    METHOD_ROOT,
                    cohort_id,
                    split_strategy=split_strategy,
                    outer_fold=outer_fold,
                ),
                "the model directory of this fold",
            ),
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

    _method_modules()
    score_genes = _beside("train").score_genes

    card_path = model_dir / "model_card.json"
    if not card_path.is_file():
        raise ContractError(f"{model_dir} holds no model_card.json")
    read_model_card(card_path)
    print(
        score_genes(
            model_dir=model_dir,
            gene_keys=_read_gene_keys(genes),
            output_dir=_require_absent(output, "the predict output directory"),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("preflight", "check that this fold has every input it needs"),
        ("features", "select and standardize this fold's columns"),
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
        print(f"esstfnet {arguments.command}: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
