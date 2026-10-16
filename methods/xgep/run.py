#!/usr/bin/env python3
"""Entry point of XGEP-XGB: preflight, features, train and predict for one fold of one task.
Run: python methods/xgep/run.py {preflight,features,train} --dataset <task> --split-strategy family_5fold --outer-fold <k>
     python methods/xgep/run.py predict --model-dir <fold model dir> --genes <gene_key parquet> --output <new dir>
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from egbench.contracts import (
    COHORT_IDS,
    OUTER_FOLDS,
    SPLIT_STRATEGIES,
    ContractError,
)
from egbench.data import model_directory


@contextlib.contextmanager
def _method_root_on_path():
    sys.path.insert(0, str(METHOD_ROOT))
    try:
        yield
    finally:
        sys.path.remove(str(METHOD_ROOT))


def _require_absent(directory: Path, what: str) -> Path:
    if directory.exists():
        raise ContractError(
            f"{what} already exists: {directory}. Remove it before running "
            f"this fold again; this entry point never overwrites."
        )
    return directory


def cmd_preflight(cohort_id: str, split_strategy: str, outer_fold: int) -> None:
    with _method_root_on_path():
        import train

    train.preflight(cohort_id, split_strategy, outer_fold)


def cmd_features(cohort_id: str, split_strategy: str, outer_fold: int) -> None:
    with _method_root_on_path():
        import build_features

    path, already_materialized = build_features.build_fold_features(
        cohort_id, split_strategy, outer_fold
    )
    if already_materialized:
        print(f"already materialized: {path}")
    else:
        print(path)


def cmd_train(cohort_id: str, split_strategy: str, outer_fold: int) -> None:
    with _method_root_on_path():
        import train

    model_dir = _require_absent(
        model_directory(
            METHOD_ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
        ),
        "the model directory of this fold",
    )
    print(train.train_fold(cohort_id, split_strategy, outer_fold, model_dir))


def cmd_predict(model_dir: Path, genes: Path, output: Path) -> None:
    with _method_root_on_path():
        import train

    print(
        train.predict(
            model_dir, genes, _require_absent(output, "the predict output directory")
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("preflight", "check that this fold has every input it needs"),
        ("features", "cut this fold's view of the static embedding table"),
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
            cmd_predict(arguments.model_dir, arguments.genes, arguments.output)
        else:
            {
                "preflight": cmd_preflight,
                "features": cmd_features,
                "train": cmd_train,
            }[arguments.command](
                arguments.dataset, arguments.split_strategy, arguments.outer_fold
            )
    except ContractError as error:
        print(f"xgep {arguments.command}: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
