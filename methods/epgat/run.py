#!/usr/bin/env python3
"""Entry point of EPGAT: preflight, features, train and predict for one fold of one task.
Run: python methods/epgat/run.py {preflight,features,train} --dataset <task> --split-strategy family_5fold --outer-fold <k>
     python methods/epgat/run.py predict --model-dir <fold model dir> --genes <gene_key parquet> --output <new dir>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
sys.path[:] = [
    entry for entry in sys.path if Path(entry or ".").resolve() != METHOD_ROOT
]
sys.path.insert(0, str(METHOD_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

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
    MODEL_CARD_FILE,
    OUTER_FOLDS,
    PREDICTION_FILES,
    RESOURCE_USAGE_FILE,
    RUN_MANIFEST_FILE,
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

from build_features import (
    build_fold as build_fold_features,
    induced_degree,
)
from contracts import (
    FORMAL_SELECTED_DIMENSIONS,
    ContractError as MethodContractError,
)
from fold_expression import build_expression_fold
from train import build_edge_index, run_training, score_graph

METHOD_ID = "epgat"

STATIC_SOURCE = Path("feature/source/generated/epgat_static.parquet")
EXPRESSION_SOURCE = Path("feature/source/expression/tcga_gene_by_sample.parquet")
EDGE_SOURCE = Path("feature/source/network/biogrid_physical_edges.parquet")

DECISION_THRESHOLD = 0.5

FEATURE_FILE = "features.parquet"
FEATURE_MANIFEST_FILE = "feature_manifest.json"
EXPRESSION_STATE_FILE = "expression_pca_50_state.npz"
FEATURE_PRODUCTS = (
    FEATURE_FILE,
    FEATURE_MANIFEST_FILE,
    EXPRESSION_STATE_FILE,
    "expression_pca_50_manifest.json",
    "expression_pca_50_train.parquet",
    "expression_pca_50_validation.parquet",
    "expression_pca_50_test.parquet",
)

WEIGHTS_FILE = "weights/model.pt"
TRANSFORM_FILE = f"weights/{EXPRESSION_STATE_FILE}"
SELECTION_FILE = "weights/feature_selection.json"

STATIC_PROJECTION = "feature/source/generated/{cohort_id}_static_features.parquet"

RUN_PRODUCTS = (
    *PREDICTION_FILES.values(),
    RUN_MANIFEST_FILE,
    RESOURCE_USAGE_FILE,
    MODEL_CARD_FILE,
    "weights",
)


def _source_paths() -> dict[str, Path]:
    paths = {
        "static": METHOD_ROOT / STATIC_SOURCE,
        "expression": METHOD_ROOT / EXPRESSION_SOURCE,
        "edges": METHOD_ROOT / EDGE_SOURCE,
    }
    missing = sorted(
        str(path.relative_to(METHOD_ROOT)) for path in paths.values() if not path.is_file()
    )
    if missing:
        raise ContractError(
            f"EPGAT cannot assemble a fold without its sources, {len(missing)} "
            f"of which are absent: {missing}. They belong under "
            f"feature/source"
        )
    return paths


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


def _versions() -> dict[str, str]:
    return software_versions(np, pd, torch)


def _static_projection_path(cohort_id: str) -> Path:
    return METHOD_ROOT / STATIC_PROJECTION.format(cohort_id=cohort_id)


def _require_no_run(output: Path) -> None:
    present = [name for name in RUN_PRODUCTS if (output / name).exists()]
    if present:
        raise ContractError(
            f"the products of this fold are already there: {output} holds "
            f"{present}. A run writes into a directory that holds none of them; "
            f"remove them to run again"
        )


def _project_static(cohort_id: str, split_strategy: str) -> Path:
    source = METHOD_ROOT / STATIC_SOURCE
    static = pd.read_parquet(source)
    if "gene_key" not in static or static["gene_key"].duplicated().any():
        raise ContractError(f"{source}: unique gene_key values are required")
    assignments = read_split_assignments(
        cohort_id, split_strategy=split_strategy, root=REPO_ROOT
    )
    keys = sorted(assignments["gene_key"].astype(str).unique())
    indexed = static.set_index("gene_key")
    projected = indexed.reindex(keys).fillna(0.0)
    missing = ~pd.Index(keys).isin(indexed.index)
    for column in ("orthology_missing", "localization_missing"):
        if column not in projected:
            raise ContractError(f"{source} lacks {column}")
        projected.loc[missing, column] = 1.0
    table = projected.reset_index().rename(columns={"index": "gene_key"})
    destination = _static_projection_path(cohort_id)
    if destination.is_file() and pd.read_parquet(destination).equals(table):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    table.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def _required_source_assets(cohort_id: str) -> list[dict[str, str]]:
    return [
        {
            "path": _static_projection_path(cohort_id)
            .relative_to(REPO_ROOT)
            .as_posix(),
            "purpose": (
                "The orthology and localisation columns of every cohort gene; "
                "its rows are also the nodes of every fold's graph, in order."
            ),
        },
        {
            "path": (METHOD_ROOT / EXPRESSION_SOURCE).relative_to(REPO_ROOT).as_posix(),
            "purpose": (
                "The TCGA expression matrix the stored components are applied to."
            ),
        },
        {
            "path": (METHOD_ROOT / EDGE_SOURCE).relative_to(REPO_ROOT).as_posix(),
            "purpose": (
                "The physical interaction edges the attention runs over, and "
                "the degree column is induced from."
            ),
        },
    ]


def _fold_selection(directory: Path) -> list[dict[str, object]]:
    manifest = json.loads(
        (directory / FEATURE_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    selection = [
        {
            "source_column": str(entry["source_column"]),
            "train_mean": float(entry["train_mean"]),
            "train_scale": float(entry["train_scale"]),
        }
        for entry in manifest["selected_features"]
    ]
    names = [entry["source_column"] for entry in selection]
    if len(names) != FORMAL_SELECTED_DIMENSIONS or len(set(names)) != len(names):
        raise ContractError(
            f"{directory / FEATURE_MANIFEST_FILE} names {len(names)} distinct "
            f"columns, EPGAT selects {FORMAL_SELECTED_DIMENSIONS}"
        )
    return selection


def build_fold(cohort_id: str, split_strategy: str, outer_fold: int) -> Path:
    source = _source_paths()
    assignments_path, labels_path = _fold_tables(cohort_id, split_strategy, outer_fold)
    directory = _fold_directory(cohort_id, split_strategy, outer_fold)
    directory.mkdir(parents=True, exist_ok=True)
    static = _project_static(cohort_id, split_strategy)
    build_expression_fold(
        expression_path=source["expression"],
        assignments_path=assignments_path,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        output_directory=directory,
    )
    build_fold_features(
        assignments_path=assignments_path,
        labels_path=labels_path,
        static_path=static,
        edges_path=source["edges"],
        expression_train_path=directory / "expression_pca_50_train.parquet",
        expression_validation_path=directory / "expression_pca_50_validation.parquet",
        expression_test_path=directory / "expression_pca_50_test.parquet",
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        output_path=directory / FEATURE_FILE,
        manifest_path=directory / FEATURE_MANIFEST_FILE,
        formal=True,
    )
    return directory


def preflight(cohort_id: str, split_strategy: str, outer_fold: int) -> dict[str, object]:
    _source_paths()
    assignments_path, labels_path = _fold_tables(cohort_id, split_strategy, outer_fold)
    for path in (assignments_path, labels_path):
        if not path.is_file():
            raise ContractError(f"the split authority is absent: {path}")
    if not torch.cuda.is_available():
        raise ContractError(
            "EPGAT trains on CUDA and no device is visible; the CPU is not a "
            "fallback for a run of this benchmark"
        )
    output = model_directory(
        METHOD_ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold
    )
    _require_no_run(output)
    directory = _fold_directory(cohort_id, split_strategy, outer_fold)
    return {
        "method_id": METHOD_ID,
        "cohort_id": cohort_id,
        "split_strategy": split_strategy,
        "outer_fold": outer_fold,
        "split_build_hash": _split_build_hash(cohort_id, split_strategy, outer_fold),
        "fold_feature_products": {
            name: (directory / name).is_file() for name in FEATURE_PRODUCTS
        },
        "cuda_devices": torch.cuda.device_count(),
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
    _require_no_run(output)
    directory = _fold_directory(cohort_id, split_strategy, outer_fold)
    absent = [name for name in FEATURE_PRODUCTS if not (directory / name).is_file()]
    if absent:
        raise ContractError(
            f"{directory} is missing {absent}. Run 'run.py features --dataset "
            f"{cohort_id} --split-strategy {split_strategy} --outer-fold "
            f"{outer_fold}' first"
        )
    assignments_path, labels_path = _fold_tables(cohort_id, split_strategy, outer_fold)
    edges = METHOD_ROOT / EDGE_SOURCE

    with RunRecord() as record:
        result = run_training(
            features_path=directory / FEATURE_FILE,
            edges_path=edges,
            assignments_path=assignments_path,
            labels_path=labels_path,
            cohort_id=cohort_id,
            split_strategy=split_strategy,
            outer_fold=outer_fold,
        )
        weights = output / WEIGHTS_FILE
        weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": result["state_dict"],
                "input_dimensions": result["input_dimensions"],
            },
            weights,
        )
        shutil.copyfile(directory / EXPRESSION_STATE_FILE, output / TRANSFORM_FILE)
        (output / SELECTION_FILE).write_text(
            json.dumps(_fold_selection(directory), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        node_keys, probabilities = _score_nodes(output, cohort_id, weights)
    position = {key: index for index, key in enumerate(node_keys)}
    scores = {
        role: probabilities[[position[key] for key in result["gene_keys"][role]]]
        for role in ("test", "validation")
    }
    device = torch.device(result["device"])
    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )

    write_predictions(
        output,
        "blind",
        result["gene_keys"]["test"],
        scores["test"],
        decision_threshold=DECISION_THRESHOLD,
    )
    write_predictions(
        output,
        "validation",
        result["gene_keys"]["validation"],
        scores["validation"],
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
        parameters=result["parameters"],
        sample_counts=result["sample_counts"],
        best_epoch=result["best_epoch"],
        best_validation_loss=result["best_validation_loss"],
        epochs_run=result["epochs_run"],
        **record.timestamps,
    )
    write_resource_usage(
        output,
        record,
        device=result["device"],
        gpu_name=result["gpu_name"],
        peak_gpu_memory_bytes=peak_gpu_memory_bytes,
    )
    write_model_card(
        output,
        **identity,
        decision_threshold=DECISION_THRESHOLD,
        artifacts={
            "weights": WEIGHTS_FILE,
            "transforms": [TRANSFORM_FILE, SELECTION_FILE],
        },
        required_source_assets=_required_source_assets(cohort_id),
        software_versions=_versions(),
    )
    return {
        "model_directory": str(output),
        "best_epoch": result["best_epoch"],
        "epochs_run": result["epochs_run"],
        "test_genes": result["sample_counts"]["test"],
    }


def fold_view(model_dir: Path, cohort_id: str) -> tuple[list[str], np.ndarray]:
    selection = json.loads((model_dir / SELECTION_FILE).read_text(encoding="utf-8"))
    static_path = _static_projection_path(cohort_id)
    expression_path = METHOD_ROOT / EXPRESSION_SOURCE
    edges_path = METHOD_ROOT / EDGE_SOURCE
    absent = [
        path
        for path in (static_path, expression_path, edges_path)
        if not path.is_file()
    ]
    if absent:
        raise ContractError(
            "EPGAT inference rebuilds this fold's node table and "
            f"{len(absent)} of its sources are absent: "
            + ", ".join(str(path) for path in absent)
        )
    static = pd.read_parquet(static_path)
    static["gene_key"] = static["gene_key"].astype(str)
    node_keys = static["gene_key"].tolist()
    if node_keys != sorted(node_keys) or len(set(node_keys)) != len(node_keys):
        raise ContractError(f"{static_path}: gene_key must be unique and sorted")

    columns: dict[str, np.ndarray] = {
        name: static[name].to_numpy(dtype=np.float64)
        for name in static.columns
        if name != "gene_key"
    }
    for index, values in enumerate(
        _expression_components(model_dir, expression_path, node_keys).T, start=1
    ):
        columns[f"expression_pc_{index:03d}"] = values
    degree = induced_degree(pd.read_parquet(edges_path), node_keys)
    columns["ppi_induced_degree"] = (
        degree.set_index("gene_key")
        .loc[node_keys, "ppi_induced_degree"]
        .to_numpy(dtype=np.float64)
    )

    source_names = list(columns)
    perturbed = np.column_stack([columns[name] for name in source_names])
    columns = {
        name: perturbed[:, position]
        for position, name in enumerate(source_names)
    }

    unknown = [
        entry["source_column"]
        for entry in selection
        if entry["source_column"] not in columns
    ]
    if unknown:
        raise ContractError(
            f"{model_dir / SELECTION_FILE} names {len(unknown)} columns this "
            f"cohort's sources do not produce, the first ten are {unknown[:10]}"
        )
    values = np.empty((len(node_keys), len(selection)), dtype=np.float64)
    for position, entry in enumerate(selection):
        values[:, position] = columns[entry["source_column"]]
        values[:, position] -= float(entry["train_mean"])
        values[:, position] /= float(entry["train_scale"])
    return node_keys, values.astype(np.float32)


def _expression_components(
    model_dir: Path, expression_path: Path, node_keys: list[str]
) -> np.ndarray:
    with np.load(model_dir / TRANSFORM_FILE, allow_pickle=False) as state:
        medians = np.asarray(state["medians"], dtype=np.float64)
        means = np.asarray(state["means"], dtype=np.float64)
        scales = np.asarray(state["scales"], dtype=np.float64)
        components = np.asarray(state["components"], dtype=np.float64)
        offset = np.asarray(state["pca_mean"], dtype=np.float64) @ components.T
    expression = pd.read_parquet(expression_path)
    expression = expression.set_index(expression["gene_key"].astype(str)).drop(
        columns="gene_key"
    )
    missing = [key for key in node_keys if key not in expression.index]
    if missing:
        raise ContractError(
            f"{len(missing)} of {len(node_keys)} nodes lack a TCGA expression "
            f"row, the first ten are {missing[:10]}"
        )
    raw = expression.loc[node_keys].to_numpy(dtype=np.float64, copy=True)
    logged = np.log2(raw + 1.0)
    imputed = np.where(np.isnan(logged), medians, logged)
    return ((imputed - means) / scales) @ components.T - offset


def _score_nodes(
    model_dir: Path, cohort_id: str, weights: Path
) -> tuple[list[str], np.ndarray]:
    stored = torch.load(weights, map_location="cpu", weights_only=True)
    node_keys, values = fold_view(model_dir, cohort_id)
    edge_index = build_edge_index(pd.read_parquet(METHOD_ROOT / EDGE_SOURCE), node_keys)
    return node_keys, score_graph(
        stored["state_dict"], int(stored["input_dimensions"]), values, edge_index
    )


def predict(model_dir: Path, genes: Path, output: Path) -> dict[str, object]:
    card = read_model_card(model_dir / MODEL_CARD_FILE)

    requested = pd.read_parquet(genes)
    if list(requested.columns) != ["gene_key"]:
        raise ContractError(
            f"{genes}: the gene list holds exactly one column, 'gene_key', "
            f"found {list(requested.columns)}"
        )
    gene_keys = requested["gene_key"].astype(str).tolist()
    if len(set(gene_keys)) != len(gene_keys):
        raise ContractError(f"{genes}: duplicate gene_key values")

    with RunRecord() as record:
        node_keys, probabilities = _score_nodes(
            model_dir,
            str(card["cohort_id"]),
            model_dir / str(card["artifacts"]["weights"]),
        )
    position = {key: index for index, key in enumerate(node_keys)}
    missing = [key for key in gene_keys if key not in position]
    if missing:
        raise ContractError(
            f"{len(missing)} of {len(gene_keys)} genes are not nodes of this "
            f"fold's graph, the first ten are {missing[:10]}. EPGAT reads a "
            f"gene's expression, orthology and localisation and its edges in "
            f"the interaction network, and a gene outside the cohort has "
            f"neither"
        )
    scores = probabilities[[position[key] for key in gene_keys]]
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
    except (ContractError, MethodContractError) as error:
        print(f"{METHOD_ID}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
