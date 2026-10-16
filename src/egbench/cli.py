"""Command line: egbench evaluate <fold dir> scores one fold of one method; egbench report --workers N writes the benchmark tables."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from egbench.contracts import PREDICTION_FILES, ContractError
from egbench.evaluate import evaluate_fold


def _fold_directory(path: Path) -> Path:
    if path.is_dir():
        return path
    if path.is_file() and path.name == PREDICTION_FILES["blind"]:
        return path.parent
    raise ContractError(
        f"{path}: pass the fold directory of a method or its "
        f"{PREDICTION_FILES['blind']}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="egbench", description="EGBench evaluation.")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser(
        "evaluate",
        help="score one fold of one method and print its evaluation record",
    )
    evaluate.add_argument(
        "predictions",
        type=Path,
        help="a fold directory of a method, or its blind_predictions.parquet",
    )
    report = commands.add_parser(
        "report",
        help="score every predictor on every task and write the benchmark tables",
    )
    report.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory, by default analysis/benchmark_report/results",
    )
    report.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of worker processes",
    )
    return parser


def _progress(line: str) -> None:
    print(line, flush=True)


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "report":
        from egbench.report import build

        written = build(arguments.output, workers=arguments.workers, progress=_progress)
        _progress(f"wrote {len(written)} files")
        return 0
    try:
        record = evaluate_fold(_fold_directory(arguments.predictions))
    except ContractError as error:
        print(f"egbench {arguments.command}: {error}", file=sys.stderr)
        return 1
    json.dump(record, sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    return 0
