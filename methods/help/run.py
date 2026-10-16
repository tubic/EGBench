#!/usr/bin/env python3
"""Entry point of HELP: preflight, features, train and predict for one fold of one task.
Run: python methods/help/run.py {preflight,features,train} --dataset <task> --split-strategy family_5fold --outer-fold <k>
     python methods/help/run.py predict --model-dir <fold model dir> --genes <gene_key parquet> --output <new dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
sys.path[:] = [
    entry for entry in sys.path if Path(entry or ".").resolve() != METHOD_ROOT
]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.append(str(METHOD_ROOT / "code" / "vendor"))

from egbench.artifacts import (
    RunRecord,
    software_versions,
    write_model_card,
    write_predict_manifest,
    write_predictions,
    write_resource_usage,
    write_run_manifest,
)
from egbench.contracts import (
    COHORT_IDS,
    OUTER_FOLDS,
    SEED,
    SPLIT_STRATEGIES,
    ContractError,
    read_model_card,
)
from egbench.data import (
    feature_directory,
    fold_gene_keys,
    model_directory,
    read_split_assignments,
    read_training_labels,
)

from svelearn.models.splitvotingens import SplitVotingEnsembleLGBM

METHOD_ID = "help"

SOURCE_MATRIX = Path("feature/source/help_features_gene_map.parquet")
FEATURE_DIMENSIONS = 3447

N_VOTERS = 10
VOTING = "soft"
VOTER_JOBS = 1
LGBM_THREADS = 16

DECISION_THRESHOLD = 0.5

FEATURE_FILE = "features.parquet"
WEIGHTS_FILE = "weights/split_voting_ensemble.joblib"


def _fold_features(cohort_id: str, split_strategy: str, outer_fold: int) -> Path:
    return (
        feature_directory(
            METHOD_ROOT,
            cohort_id,
            split_strategy=split_strategy,
            outer_fold=outer_fold,
        )
        / FEATURE_FILE
    )


def _source_matrix_path() -> Path:
    path = METHOD_ROOT / SOURCE_MATRIX
    if not path.is_file():
        raise ContractError(
            f"the HELP feature matrix is absent: {path}. It is the frozen "
            f"global matrix assembled by code/build_help_features.py from the "
            f"blocks the authors published under doi:10.5281/zenodo.12597679"
        )
    return path


def _read_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "gene_key" not in frame.columns:
        raise ContractError(f"{path}: no 'gene_key' column")
    frame["gene_key"] = frame["gene_key"].astype(str)
    if frame["gene_key"].duplicated().any():
        raise ContractError(f"{path}: duplicate gene_key values")
    columns = [column for column in frame.columns if column != "gene_key"]
    if len(columns) != FEATURE_DIMENSIONS:
        raise ContractError(
            f"{path}: HELP is {FEATURE_DIMENSIONS} features wide, this table "
            f"has {len(columns)}"
        )
    return frame.set_index("gene_key")


def _rows(matrix: pd.DataFrame, gene_keys: list[str], source: str) -> np.ndarray:
    missing = [key for key in gene_keys if key not in matrix.index]
    if missing:
        raise ContractError(
            f"{source}: {len(missing)} of {len(gene_keys)} genes are absent "
            f"from the HELP feature table, the first ten are {missing[:10]}. "
            f"HELP scores a gene from the row the authors published for it, so "
            f"a gene they left out cannot be scored"
        )
    return matrix.loc[gene_keys].to_numpy(dtype=np.float64)


def _split_build_hash(cohort_id: str, split_strategy: str, outer_fold: int) -> str:
    assignments = read_split_assignments(
        cohort_id, split_strategy=split_strategy, root=REPO_ROOT
    )
    values = sorted(
        set(assignments.loc[assignments["outer_fold"].eq(outer_fold), "split_build_hash"])
    )
    if len(values) != 1:
        raise ContractError(
            f"fold {outer_fold} of {cohort_id}/{split_strategy} carries "
            f"{len(values)} split_build_hash values: {values}"
        )
    return values[0]


def _sample_counts(cohort_id: str, split_strategy: str, outer_fold: int) -> dict[str, int]:
    return {
        role: len(
            fold_gene_keys(
                cohort_id,
                outer_fold,
                role,
                split_strategy=split_strategy,
                root=REPO_ROOT,
            )
        )
        for role in ("train", "validation", "test")
    }


def _parameters(fitted_dimensions: int = FEATURE_DIMENSIONS) -> dict[str, object]:
    return {
        "n_voters": N_VOTERS,
        "voting": VOTING,
        "voter_n_jobs": VOTER_JOBS,
        "lightgbm_threads": LGBM_THREADS,
        "lightgbm_deterministic": True,
        "lightgbm_force_row_wise": True,
        "feature_dimensions": int(fitted_dimensions),
        "fitting_roles": ["train"],
    }


def _versions() -> dict[str, str]:
    return software_versions(np, pd, sklearn, lightgbm, joblib)


def fit(train_x: np.ndarray, train_y: np.ndarray) -> SplitVotingEnsembleLGBM:
    model = SplitVotingEnsembleLGBM(
        n_voters=N_VOTERS,
        voting=VOTING,
        n_jobs=VOTER_JOBS,
        random_state=SEED,
    )
    model.base_estimator.set_params(
        n_jobs=LGBM_THREADS, deterministic=True, force_row_wise=True
    )
    model.fit(train_x, train_y)
    return model


def positive_scores(model: SplitVotingEnsembleLGBM, x: np.ndarray) -> np.ndarray:
    positive = int(np.where(model.label_encoder_.classes_ == 1)[0][0])
    return np.clip(model.predict_proba(x)[:, positive].astype(np.float64), 0.0, 1.0)


def training_rows(
    cohort_id: str, split_strategy: str, outer_fold: int
) -> tuple[list[str], np.ndarray]:
    labels = read_training_labels(
        cohort_id,
        outer_fold,
        "train",
        split_strategy=split_strategy,
        root=REPO_ROOT,
    )
    return (
        labels["gene_key"].astype(str).tolist(),
        labels["label"].to_numpy(dtype=np.int64),
    )


def preflight(cohort_id: str, split_strategy: str, outer_fold: int) -> dict[str, object]:
    source = _source_matrix_path()
    output = model_directory(
        METHOD_ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
    )
    if output.exists():
        raise ContractError(
            f"the products of this fold are already there: {output}. A run "
            f"writes a directory that does not yet exist; remove it to run again"
        )
    features = _fold_features(cohort_id, split_strategy, outer_fold)
    return {
        "method_id": METHOD_ID,
        "cohort_id": cohort_id,
        "split_strategy": split_strategy,
        "outer_fold": outer_fold,
        "split_build_hash": _split_build_hash(cohort_id, split_strategy, outer_fold),
        "sample_counts": _sample_counts(cohort_id, split_strategy, outer_fold),
        "source_matrix": str(source.relative_to(METHOD_ROOT)),
        "fold_features_present": features.is_file(),
        "model_directory": str(output.relative_to(METHOD_ROOT)),
    }


def features(cohort_id: str, split_strategy: str, outer_fold: int) -> dict[str, object]:
    sys.path.insert(0, str(REPO_ROOT / "methods"))
    from family_feature_layout import read_authority_gene_sets

    source = _source_matrix_path()
    target = _fold_features(cohort_id, split_strategy, outer_fold)
    directory = target.parent
    if target.is_file():
        return {"status": "already materialized", "features": str(target)}
    if directory.exists() and any(directory.iterdir()):
        raise ContractError(
            f"{directory} holds something other than {FEATURE_FILE}. Remove the "
            f"directory to build this fold again"
        )
    keys = read_authority_gene_sets(REPO_ROOT, cohort_id)[outer_fold]
    matrix = _read_matrix(source)
    missing = [key for key in keys if key not in matrix.index]
    if missing:
        raise ContractError(
            f"{source} lacks {len(missing)} of the {len(keys)} genes of "
            f"{cohort_id} fold {outer_fold}, the first ten are {missing[:10]}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    canonical = _fold_features(cohort_id, SPLIT_STRATEGIES[0], 0)
    if target != canonical and canonical.is_file():
        os.link(canonical, target)
    else:
        matrix.loc[keys].reset_index().to_parquet(target, index=False)
    return {"status": "materialized", "features": str(target), "genes": len(keys)}


def train(cohort_id: str, split_strategy: str, outer_fold: int) -> dict[str, object]:
    output = model_directory(
        METHOD_ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
    )
    if output.exists():
        raise ContractError(
            f"the products of this fold are already there: {output}. A run "
            f"writes a directory that does not yet exist; remove it to run again"
        )
    feature_path = _fold_features(cohort_id, split_strategy, outer_fold)
    if not feature_path.is_file():
        raise ContractError(
            f"the fold features are absent: {feature_path}. Run "
            f"'run.py features --dataset {cohort_id} --split-strategy "
            f"{split_strategy} --outer-fold {outer_fold}' first"
        )
    matrix = _read_matrix(feature_path)
    feature_columns = list(matrix.columns)
    train_keys, train_y = training_rows(cohort_id, split_strategy, outer_fold)
    validation_keys = fold_gene_keys(
        cohort_id, outer_fold, "validation", split_strategy=split_strategy, root=REPO_ROOT
    ).astype(str).tolist()
    test_keys = fold_gene_keys(
        cohort_id, outer_fold, "test", split_strategy=split_strategy, root=REPO_ROOT
    ).astype(str).tolist()

    train_x = _rows(matrix, train_keys, "train role")
    validation_x = _rows(matrix, validation_keys, "validation role")
    test_x = _rows(matrix, test_keys, "test role")

    with RunRecord() as record:
        model = fit(train_x, train_y)
        validation_scores = positive_scores(model, validation_x)
        test_scores = positive_scores(model, test_x)

    weights = output / WEIGHTS_FILE
    weights.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"ensemble": model, "feature_columns": feature_columns}, weights)

    write_predictions(
        output,
        "blind",
        test_keys,
        test_scores,
        decision_threshold=DECISION_THRESHOLD,
    )
    write_predictions(
        output,
        "validation",
        validation_keys,
        validation_scores,
        decision_threshold=DECISION_THRESHOLD,
    )
    write_run_manifest(
        output,
        method_id=METHOD_ID,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        split_build_hash=_split_build_hash(cohort_id, split_strategy, outer_fold),
        software_versions=_versions(),
        parameters=_parameters(int(train_x.shape[1])),
        sample_counts={
            "train": len(train_keys),
            "validation": len(validation_keys),
            "test": len(test_keys),
        },
        **record.timestamps,
    )
    write_resource_usage(
        output, record, device="cpu", gpu_name=None, peak_gpu_memory_bytes=0
    )
    write_model_card(
        output,
        method_id=METHOD_ID,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        split_build_hash=_split_build_hash(cohort_id, split_strategy, outer_fold),
        decision_threshold=DECISION_THRESHOLD,
        artifacts={"weights": WEIGHTS_FILE, "transforms": []},
        required_source_assets=[
            {
                "path": (METHOD_ROOT / SOURCE_MATRIX)
                .relative_to(REPO_ROOT)
                .as_posix(),
                "purpose": (
                    "The frozen global HELP feature matrix; inference reads a "
                    "gene's row from it and cannot score a gene it omits."
                ),
            }
        ],
        software_versions=_versions(),
    )
    return {"model_directory": str(output), "test_genes": len(test_keys)}


def predict(model_dir: Path, genes: Path, output: Path) -> dict[str, object]:
    card = read_model_card(model_dir / "model_card.json")
    payload = joblib.load(model_dir / str(card["artifacts"]["weights"]))
    model = payload["ensemble"]
    feature_columns = list(payload["feature_columns"])

    requested = pd.read_parquet(genes)
    if list(requested.columns) != ["gene_key"]:
        raise ContractError(
            f"{genes}: the gene list holds exactly one column, 'gene_key', "
            f"found {list(requested.columns)}"
        )
    gene_keys = requested["gene_key"].astype(str).tolist()
    if len(set(gene_keys)) != len(gene_keys):
        raise ContractError(f"{genes}: duplicate gene_key values")

    matrix = _read_matrix(_source_matrix_path())
    x = _rows(matrix[feature_columns], gene_keys, str(genes))
    with RunRecord() as record:
        scores = positive_scores(model, x)
    threshold = float(card["decision_threshold"])
    write_predictions(
        output, "blind", gene_keys, scores, decision_threshold=threshold
    )
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
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
