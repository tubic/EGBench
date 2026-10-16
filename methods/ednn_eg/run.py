#!/usr/bin/env python3
"""Entry point of eDNN-EG: preflight, features, train and predict for one fold of one task.
Run: python methods/ednn_eg/run.py {preflight,features,train} --dataset <task> --split-strategy family_5fold --outer-fold <k>
     python methods/ednn_eg/run.py predict --model-dir <fold model dir> --genes <gene_key parquet> --output <new dir>
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
sys.path.insert(0, str(METHOD_ROOT / "code" / "src"))

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


FEATURE_WORK = "feature_fasttext"
TRAIN_WORK = "stack"

FEATURE_PRODUCTS = (
    "features.parquet",
    "selected_cds.fa",
    "feature_manifest.json",
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


def _source_cds(cohort_id: str) -> Path:
    return METHOD_ROOT / "feature" / "source" / f"{cohort_id}_cds_sequences.fa.gz"


def _work_directory(cohort_id: str, split_strategy: str, outer_fold: int) -> Path:
    return (
        METHOD_ROOT / "work" / cohort_id / split_strategy / f"fold_{outer_fold}"
    )


def _require_absent(directory: Path, what: str) -> Path:

    if directory.exists():
        raise ContractError(
            f"{what} already exists: {directory}. Remove it before running this "
            f"fold again; this entry point never overwrites."
        )
    return directory


def _fold_inputs(cohort_id: str, split_strategy: str, outer_fold: int) -> list[Path]:
    folds = folds_directory(cohort_id, split_strategy=split_strategy, root=REPO_ROOT)
    return [
        folds / "split_assignments.parquet",
        folds / "training_labels" / f"fold_{outer_fold}.parquet",
    ]


def preflight(cohort_id: str, split_strategy: str, outer_fold: int) -> None:

    cohort_id = _cohort(cohort_id)
    required = [_source_cds(cohort_id), *_fold_inputs(cohort_id, split_strategy, outer_fold)]
    absent = [path for path in required if not path.is_file()]
    if absent:
        raise ContractError(
            "eDNN-EG cannot run this fold, the following inputs are absent: "
            + ", ".join(str(path) for path in absent)
        )
    import fasttext
    import sklearn

    from ednn_eg.models import require_cuda
    from ednn_eg.runner import DEVICE

    _, device = require_cuda(DEVICE)
    features = feature_directory(
        METHOD_ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
    )
    model = model_directory(
        METHOD_ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
    )
    print("method_id ednn_eg")
    print(f"cohort_id {cohort_id} split_strategy {split_strategy} outer_fold {outer_fold}")
    print(f"device {device}")
    print(f"coding sequences {_source_cds(cohort_id)}")
    print(f"features {features} {'present' if features.is_dir() else 'absent'}")
    print(f"model {model} {'present' if model.is_dir() else 'absent'}")


def features(cohort_id: str, split_strategy: str, outer_fold: int) -> None:

    cohort_id = _cohort(cohort_id)
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
    build_features = _beside("build_features")
    print(build_features.build_fold(cohort_id, outer_fold, split_strategy))


def train(cohort_id: str, split_strategy: str, outer_fold: int) -> None:

    from ednn_eg.runner import train_fold

    cohort_id = _cohort(cohort_id)
    fold_features = feature_directory(
        METHOD_ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
    )
    selected_cds = fold_features / "selected_cds.fa"
    if not selected_cds.is_file():
        raise ContractError(
            f"the fold view is absent: {selected_cds}. Run 'features' for this "
            f"fold first."
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
            cds_fasta=selected_cds,
            folds_dir=folds_directory(
                cohort_id, split_strategy=split_strategy, root=REPO_ROOT
            ),
            model_dir=model_dir,
            work_dir=_work_directory(cohort_id, split_strategy, outer_fold)
            / TRAIN_WORK,
            cohort_id=cohort_id,
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

    from ednn_eg.runner import score_genes

    build_features = _beside("build_features")

    card_path = model_dir / "model_card.json"
    if not card_path.is_file():
        raise ContractError(f"{model_dir} holds no model_card.json")
    cohort_id = str(read_model_card(card_path)["cohort_id"])
    keys = _read_gene_keys(genes)
    source = _source_cds(cohort_id)
    if not source.is_file():
        raise ContractError(f"the coding sequences of {cohort_id} are absent: {source}")
    sequences = build_features.read_source_cds(source)
    print(
        score_genes(
            model_dir=model_dir,
            gene_keys=keys,
            sequences=sequences,
            output_dir=_require_absent(output, "the predict output directory"),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("preflight", "check that this fold has every input it needs"),
        ("features", "fit this fold's representation and cut its view"),
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
        print(f"ednn_eg {arguments.command}: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
