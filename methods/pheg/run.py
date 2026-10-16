#!/usr/bin/env python3
"""Entry point of Pheg: preflight, features, train and predict for one fold of one task.
Run: python methods/pheg/run.py {preflight,features,train} --dataset <task> --split-strategy family_5fold --outer-fold <k>
     python methods/pheg/run.py predict --model-dir <fold model dir> --genes <gene_key parquet> --output <new dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
sys.path[:] = [
    entry for entry in sys.path if Path(entry or ".").resolve() != METHOD_ROOT
]
sys.path.insert(0, str(METHOD_ROOT / "code"))
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
    read_split_assignments,
)

from svm_rfe_cbr import SVMRFECBRConfig, rank_features
from train import (
    C_GRID,
    SELECTED_FEATURE_COUNT,
    SVM_MAX_ITERATIONS,
    SVM_TOLERANCE,
    fit_and_score,
    load_data,
    probability,
)

METHOD_ID = "pheg"

RAW_SOURCE = Path("feature/source/generated/pheg_fv45_4545.parquet")
RAW_DIMENSIONS = 4545

DECISION_THRESHOLD = 0.5

FEATURE_FILE = "features.parquet"
SCALER_FILE = "scaler_state.npz"
SELECTION_FILE = "selected_features.json"
FEATURE_MANIFEST_FILE = "feature_manifest.json"
FEATURE_PRODUCTS = (
    FEATURE_FILE,
    SCALER_FILE,
    SELECTION_FILE,
    FEATURE_MANIFEST_FILE,
)

WEIGHTS_FILE = "weights/linear_svm.joblib"
CALIBRATOR_FILE = "weights/platt_calibrator.joblib"
TRANSFORM_FILES = (f"weights/{SCALER_FILE}", f"weights/{SELECTION_FILE}")


def _raw_source_path() -> Path:
    path = METHOD_ROOT / RAW_SOURCE
    if not path.is_file():
        raise ContractError(
            f"the raw FV4,5 matrix is absent: {path}. It is the {RAW_DIMENSIONS}"
            f"-dimensional Z-curve source declared in "
            f"feature/source, and both the standardisation "
            f"and the SVM-RFE+CBR ranking are fitted on it"
        )
    return path


def _fold_directory(cohort_id: str, split_strategy: str, outer_fold: int) -> Path:
    return feature_directory(
        METHOD_ROOT,
        cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )


def _fold_tables(cohort_id: str, split_strategy: str, outer_fold: int) -> tuple[Path, Path]:
    folds = folds_directory(cohort_id, split_strategy=split_strategy, root=REPO_ROOT)
    return (
        folds / "split_assignments.parquet",
        folds / "training_labels" / f"fold_{outer_fold}.parquet",
    )


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


def _parameters() -> dict[str, object]:
    return {
        "raw_feature_dimensions": RAW_DIMENSIONS,
        "selected_feature_dimensions": SELECTED_FEATURE_COUNT,
        "c_grid": list(C_GRID),
        "svm_loss": "squared_hinge",
        "svm_penalty": "l2",
        "svm_dual": False,
        "svm_tolerance": SVM_TOLERANCE,
        "svm_max_iterations": SVM_MAX_ITERATIONS,
        "c_selection": "validation_roc_auc_then_smaller_c",
        "calibration": "platt_sigmoid_on_validation_margins",
        "fitting_roles": ["train"],
    }


def _versions() -> dict[str, str]:
    return software_versions(np, pd, sklearn, joblib)


def build_fold(cohort_id: str, split_strategy: str, outer_fold: int) -> Path:
    assignments_path, labels_path = _fold_tables(cohort_id, split_strategy, outer_fold)
    assignments = pd.read_parquet(assignments_path)
    labels = pd.read_parquet(labels_path)
    condition = (
        assignments["cohort_id"].astype(str).eq(cohort_id)
        & assignments["split_strategy"].astype(str).eq(split_strategy)
        & pd.to_numeric(assignments["outer_fold"], errors="raise").eq(outer_fold)
    )
    context = assignments.loc[condition, ["gene_key", "role"]].copy()
    if context.empty or context["gene_key"].duplicated().any():
        raise ContractError("split context is empty or duplicated")
    if set(context["role"].astype(str)) != {"train", "validation", "test"}:
        raise ContractError("split context requires train, validation and test")
    context = context.sort_values("gene_key", kind="mergesort")
    train_labels = labels.loc[
        labels["cohort_id"].astype(str).eq(cohort_id)
        & labels["split_strategy"].astype(str).eq(split_strategy)
        & pd.to_numeric(labels["outer_fold"], errors="raise").eq(outer_fold)
        & labels["role"].astype(str).eq("train"),
        ["gene_key", "label"],
    ].copy()
    if (
        train_labels["gene_key"].duplicated().any()
        or set(train_labels["label"].astype(int)) != {0, 1}
    ):
        raise ContractError("the train-role labels must be unique and binary")

    raw_path = _raw_source_path()
    raw = pd.read_parquet(raw_path)
    if "gene_key" not in raw or raw["gene_key"].duplicated().any():
        raise ContractError(f"{raw_path}: unique gene_key values are required")
    names = [column for column in raw.columns if column != "gene_key"]
    if len(names) != RAW_DIMENSIONS:
        raise ContractError(
            f"{raw_path}: Pheg reads {RAW_DIMENSIONS} raw descriptors, this "
            f"table has {len(names)}"
        )
    indexed = raw.set_index("gene_key")
    keys = context["gene_key"].astype(str).tolist()
    missing = sorted(set(keys).difference(indexed.index))
    if missing:
        raise ContractError(
            f"{raw_path} lacks {len(missing)} of the fold's genes, the first ten "
            f"are {missing[:10]}"
        )
    matrix = indexed.loc[keys, names].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(matrix).all():
        raise ContractError(f"{raw_path} holds non-finite values")

    train_positions = np.flatnonzero(context["role"].eq("train").to_numpy())
    train_lookup = dict(train_labels.itertuples(index=False, name=None))
    y_train = np.asarray(
        [train_lookup[keys[index]] for index in train_positions], dtype=np.int64
    )
    scaler = StandardScaler().fit(matrix[train_positions])
    standardized = scaler.transform(matrix)
    selection_keys = [str(keys[index]) for index in train_positions]
    selection_labels = np.asarray(y_train, dtype="int64").reshape(-1)
    position = {key: index for index, key in enumerate(keys)}
    selection_rows = np.asarray(
        [position[key] for key in selection_keys], dtype=np.int64
    )
    ranking = rank_features(
        standardized[selection_rows], selection_labels, SVMRFECBRConfig()
    )
    selected_indices = np.asarray(ranking.ranking[:SELECTED_FEATURE_COUNT], dtype=int)
    selected_names = [names[index] for index in selected_indices]

    output = _fold_directory(cohort_id, split_strategy, outer_fold)
    output.mkdir(parents=True, exist_ok=True)
    features = pd.DataFrame(standardized[:, selected_indices], columns=selected_names)
    features.insert(0, "gene_key", keys)
    features.to_parquet(output / FEATURE_FILE, index=False)
    np.savez_compressed(
        output / SCALER_FILE,
        means=scaler.mean_,
        scales=scaler.scale_,
        feature_names=np.asarray(names),
    )
    (output / SELECTION_FILE).write_text(
        json.dumps(
            {
                "outer_fold": outer_fold,
                "fit_role": "train",
                "raw_feature_dimensions": RAW_DIMENSIONS,
                "selected_feature_dimensions": SELECTED_FEATURE_COUNT,
                "selected_feature_names": selected_names,
                "constant_feature_indices": list(ranking.constant_features),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / FEATURE_MANIFEST_FILE).write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "pheg_fold_features",
                "method_id": METHOD_ID,
                "dataset": cohort_id,
                "cohort_id": cohort_id,
                "split_strategy": split_strategy,
                "outer_fold": outer_fold,
                "row_count": len(keys),
                "standardization": "StandardScaler fit on the train-role genes only",
                "selection": "SVM-RFE+CBR fit on the train-role labels only",
                "feature_dimensions": SELECTED_FEATURE_COUNT,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def preflight(cohort_id: str, split_strategy: str, outer_fold: int) -> dict[str, object]:
    source = _raw_source_path()
    assignments_path, labels_path = _fold_tables(cohort_id, split_strategy, outer_fold)
    for path in (assignments_path, labels_path):
        if not path.is_file():
            raise ContractError(f"the split authority is absent: {path}")
    output = model_directory(
        METHOD_ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
    )
    if output.exists():
        raise ContractError(
            f"the products of this fold are already there: {output}. A run "
            f"writes a directory that does not yet exist; remove it to run again"
        )
    directory = _fold_directory(cohort_id, split_strategy, outer_fold)
    return {
        "method_id": METHOD_ID,
        "cohort_id": cohort_id,
        "split_strategy": split_strategy,
        "outer_fold": outer_fold,
        "split_build_hash": _split_build_hash(cohort_id, split_strategy, outer_fold),
        "raw_source": str(source.relative_to(METHOD_ROOT)),
        "fold_feature_products": {
            name: (directory / name).is_file() for name in FEATURE_PRODUCTS
        },
        "model_directory": str(output.relative_to(METHOD_ROOT)),
    }


def features(cohort_id: str, split_strategy: str, outer_fold: int) -> dict[str, object]:
    directory = _fold_directory(cohort_id, split_strategy, outer_fold)
    present = [name for name in FEATURE_PRODUCTS if (directory / name).is_file()]
    if len(present) == len(FEATURE_PRODUCTS):
        return {"status": "already materialized", "directory": str(directory)}
    if present:
        raise ContractError(
            f"{directory} holds part of this fold's view, {present}, and is "
            f"missing {[n for n in FEATURE_PRODUCTS if n not in present]}. "
            f"Remove the directory and run 'features' again to rebuild it"
        )
    build_fold(cohort_id, split_strategy, outer_fold)
    return {"status": "materialized", "directory": str(directory)}


def train(cohort_id: str, split_strategy: str, outer_fold: int) -> dict[str, object]:
    output = model_directory(
        METHOD_ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
    )
    if output.exists():
        raise ContractError(
            f"the products of this fold are already there: {output}. A run "
            f"writes a directory that does not yet exist; remove it to run again"
        )
    directory = _fold_directory(cohort_id, split_strategy, outer_fold)
    absent = [name for name in FEATURE_PRODUCTS if not (directory / name).is_file()]
    if absent:
        raise ContractError(
            f"{directory} is missing {absent}. Run 'run.py features --dataset "
            f"{cohort_id} --split-strategy {split_strategy} --outer-fold "
            f"{outer_fold}' first"
        )
    assignments_path, labels_path = _fold_tables(cohort_id, split_strategy, outer_fold)
    arrays, _, keys = load_data(
        directory / FEATURE_FILE,
        labels_path,
        assignments_path,
        cohort_id,
        split_strategy,
        outer_fold,
    )

    with RunRecord() as record:
        fitted = fit_and_score(arrays)

    weights = output / WEIGHTS_FILE
    weights.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted["model"], weights)
    joblib.dump(fitted["calibrator"], output / CALIBRATOR_FILE)
    for name in (SCALER_FILE, SELECTION_FILE):
        shutil.copyfile(directory / name, output / "weights" / name)

    write_predictions(
        output,
        "blind",
        keys["test"],
        fitted["test_scores"],
        decision_threshold=DECISION_THRESHOLD,
    )
    write_predictions(
        output,
        "validation",
        keys["validation"],
        fitted["validation_scores"],
        decision_threshold=DECISION_THRESHOLD,
    )
    identity = {
        "method_id": METHOD_ID,
        "cohort_id": cohort_id,
        "split_strategy": split_strategy,
        "outer_fold": outer_fold,
        "split_build_hash": _split_build_hash(cohort_id, split_strategy, outer_fold),
    }
    write_run_manifest(
        output,
        **identity,
        software_versions=_versions(),
        parameters=_parameters(),
        sample_counts={role: len(value) for role, value in keys.items()},
        selected_c=float(fitted["selected_c"]),
        **record.timestamps,
    )
    write_resource_usage(
        output, record, device="cpu", gpu_name=None, peak_gpu_memory_bytes=0
    )
    write_model_card(
        output,
        **identity,
        decision_threshold=DECISION_THRESHOLD,
        artifacts={
            "weights": WEIGHTS_FILE,
            "transforms": [CALIBRATOR_FILE, *TRANSFORM_FILES],
        },
        required_source_assets=[
            {
                "path": _raw_source_path().relative_to(REPO_ROOT).as_posix(),
                "purpose": (
                    f"The raw {RAW_DIMENSIONS}-dimensional Z-curve descriptors "
                    f"of every gene; inference standardises and selects them "
                    f"with the two transforms stored under weights/."
                ),
            }
        ],
        software_versions=_versions(),
    )
    return {
        "model_directory": str(output),
        "selected_c": float(fitted["selected_c"]),
        "test_genes": len(keys["test"]),
    }


def fold_view(model_dir: Path, gene_keys: list[str]) -> np.ndarray:
    selection = json.loads(
        (model_dir / "weights" / SELECTION_FILE).read_text(encoding="utf-8")
    )
    columns = list(selection["selected_feature_names"])
    with np.load(model_dir / "weights" / SCALER_FILE, allow_pickle=False) as state:
        names = [str(name) for name in state["feature_names"]]
        means = np.asarray(state["means"], dtype=np.float64)
        scales = np.asarray(state["scales"], dtype=np.float64)
    position = {name: index for index, name in enumerate(names)}
    selected = np.asarray([position[name] for name in columns], dtype=np.int64)

    source = _raw_source_path()
    raw = pd.read_parquet(source, columns=["gene_key", *columns])
    raw["gene_key"] = raw["gene_key"].astype(str)
    raw = raw.set_index("gene_key")
    missing = [key for key in gene_keys if key not in raw.index]
    if missing:
        raise ContractError(
            f"{len(missing)} of {len(gene_keys)} genes are absent from "
            f"{source}, the first ten are {missing[:10]}. Pheg scores a gene "
            f"from its Z-curve descriptors and has none for a gene the raw "
            f"matrix omits"
        )
    values = raw.loc[gene_keys, columns].to_numpy(dtype=np.float64, copy=True)
    values -= means[selected]
    values /= scales[selected]
    return values


def predict(model_dir: Path, genes: Path, output: Path) -> dict[str, object]:
    card = read_model_card(model_dir / "model_card.json")
    model = joblib.load(model_dir / str(card["artifacts"]["weights"]))
    calibrator = joblib.load(model_dir / CALIBRATOR_FILE)

    requested = pd.read_parquet(genes)
    if list(requested.columns) != ["gene_key"]:
        raise ContractError(
            f"{genes}: the gene list holds exactly one column, 'gene_key', "
            f"found {list(requested.columns)}"
        )
    gene_keys = requested["gene_key"].astype(str).tolist()
    if len(set(gene_keys)) != len(gene_keys):
        raise ContractError(f"{genes}: duplicate gene_key values")

    x = fold_view(model_dir, gene_keys)

    with RunRecord() as record:
        scores = probability(calibrator, model.decision_function(x))
    write_predictions(
        output,
        "blind",
        gene_keys,
        scores,
        decision_threshold=float(card["decision_threshold"]),
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
