#!/usr/bin/env python3
"""Entry point of DeEPsnap: preflight, features, train and predict for one fold of one task.
Run: python methods/deepsnap/run.py {preflight,features,train} --dataset <task> --split-strategy family_5fold --outer-fold <k>
     python methods/deepsnap/run.py predict --model-dir <fold model dir> --genes <gene_key parquet> --output <new dir>
"""

from __future__ import annotations

import argparse
import json
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

from build_features import (
    assemble_columns,
    build_features as build_fold_features,
    parse_sources,
)
from contracts import (
    FORMAL_DIMENSIONS,
    FORMAL_SNAPSHOTS,
    GO_DIMENSIONS,
)
from contracts import ContractError as MethodContractError
from model import DeEPsnapMLP
from train import (
    inference_device,
    run_training,
    snapshot_probabilities,
)

METHOD_ID = "deepsnap"

SOURCE_ASSETS = {
    "gene_map": "feature/source/gene_reference/gene_id_map.parquet",
    "sequence": "feature/source/generated/deepsnap_sequence_89.parquet",
    "node2vec": "feature/source/network/node2vec_64.parquet",
    "edges": "feature/source/network/biogrid_physical_edges.parquet",
    "goa": "feature/source/annotations/goa_human.gaf.gz",
    "corum": "feature/source/annotations/corum_human_complexes.tsv",
    "pfam": "feature/source/annotations/ensembl_116_human_pfam.tsv",
    "manifest": "feature/source/annotations/source_manifest.json",
}

DECISION_THRESHOLD = 0.5

FEATURE_FILE = "features.parquet"
FEATURE_MANIFEST_FILE = "feature_manifest.json"
FEATURE_PRODUCTS = (FEATURE_FILE, FEATURE_MANIFEST_FILE)

WEIGHTS_DIRECTORY = "weights"
SCALER_FILE = "weights/scaler.npz"
GO_TERMS_FILE = "weights/selected_go_terms.json"


def _source_paths() -> dict[str, Path]:
    paths = {name: METHOD_ROOT / relative for name, relative in SOURCE_ASSETS.items()}
    missing = sorted(
        str(path.relative_to(METHOD_ROOT)) for path in paths.values() if not path.is_file()
    )
    if missing:
        raise ContractError(
            f"DeEPsnap cannot assemble a fold without its source databases, "
            f"{len(missing)} of which are absent: {missing}. Their checksums are "
            f"in feature/source/annotations/source_manifest.json"
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


def _required_source_assets() -> list[dict[str, str]]:
    purposes = {
        "gene_map": (
            "maps a gene_key onto the symbol and protein identifiers the three "
            "annotation databases are keyed by"
        ),
        "sequence": "the 89 sequence descriptors of every gene",
        "node2vec": (
            "the 64-dimensional node2vec embedding of the interaction network"
        ),
        "edges": (
            "the physical interaction edges the GO enrichment and the CORUM "
            "columns read a gene's neighbourhood from"
        ),
        "goa": (
            "the GO annotations the hundred enrichment columns are computed "
            "against"
        ),
        "corum": "the protein complexes the two CORUM columns count",
        "pfam": "the protein domains the three Pfam columns count",
    }
    return [
        {
            "path": (METHOD_ROOT / SOURCE_ASSETS[name])
            .relative_to(REPO_ROOT)
            .as_posix(),
            "purpose": purpose,
        }
        for name, purpose in purposes.items()
    ]


def _fold_go_terms(directory: Path) -> list[str]:
    manifest = json.loads(
        (directory / FEATURE_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    terms = [str(entry["go_id"]) for entry in manifest["selected_go_terms"]]
    if len(terms) != GO_DIMENSIONS or len(set(terms)) != len(terms):
        raise ContractError(
            f"{directory / FEATURE_MANIFEST_FILE} names {len(terms)} distinct "
            f"GO terms, DeEPsnap ranks {GO_DIMENSIONS}"
        )
    return terms


def build_fold(cohort_id: str, split_strategy: str, outer_fold: int) -> Path:
    source = _source_paths()
    assignments_path, labels_path = _fold_tables(cohort_id, split_strategy, outer_fold)
    directory = _fold_directory(cohort_id, split_strategy, outer_fold)
    build_fold_features(
        gene_map_path=source["gene_map"],
        assignments_path=assignments_path,
        labels_path=labels_path,
        sequence_path=source["sequence"],
        node2vec_path=source["node2vec"],
        edges_path=source["edges"],
        goa_path=source["goa"],
        corum_path=source["corum"],
        pfam_path=source["pfam"],
        source_manifest_path=source["manifest"],
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
            "DeEPsnap trains on CUDA and no device is visible; the CPU is not a "
            "fallback for a run of this benchmark"
        )
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
        "feature_dimensions": FORMAL_DIMENSIONS,
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

    with RunRecord() as record:
        result = run_training(
            features_path=directory / FEATURE_FILE,
            assignments_path=assignments_path,
            labels_path=labels_path,
            cohort_id=cohort_id,
            split_strategy=split_strategy,
            outer_fold=outer_fold,
        )

    weights = output / WEIGHTS_DIRECTORY
    weights.mkdir(parents=True, exist_ok=True)
    for index, state in enumerate(result["snapshots"], start=1):
        torch.save(
            {
                "state_dict": state,
                "input_dimensions": len(result["feature_columns"]),
                "snapshot_epoch": index * result["parameters"]["snapshot_cycle"],
            },
            weights / f"snapshot_{index:02d}.pt",
        )
    np.savez(
        output / SCALER_FILE,
        mean=result["means"],
        scale=result["scales"],
        feature_columns=np.asarray(result["feature_columns"]),
    )
    (output / GO_TERMS_FILE).write_text(
        json.dumps(
            {"selected_go_terms": _fold_go_terms(directory)}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )

    write_predictions(
        output,
        "blind",
        result["gene_keys"]["test"],
        result["scores"]["test"],
        decision_threshold=DECISION_THRESHOLD,
    )
    write_predictions(
        output,
        "validation",
        result["gene_keys"]["validation"],
        result["scores"]["validation"],
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
        positive_weight=result["positive_weight"],
        **record.timestamps,
    )
    write_resource_usage(
        output,
        record,
        device=result["device"],
        gpu_name=result["gpu_name"],
        peak_gpu_memory_bytes=result["peak_gpu_memory_bytes"],
    )
    write_model_card(
        output,
        **identity,
        decision_threshold=DECISION_THRESHOLD,
        artifacts={
            "weights": WEIGHTS_DIRECTORY,
            "transforms": [SCALER_FILE, GO_TERMS_FILE],
        },
        required_source_assets=_required_source_assets(),
        software_versions=_versions(),
    )
    return {
        "model_directory": str(output),
        "snapshots": len(result["snapshots"]),
        "test_genes": result["sample_counts"]["test"],
    }


def fold_view(model_dir: Path, gene_keys: list[str]) -> pd.DataFrame:
    terms = json.loads((model_dir / GO_TERMS_FILE).read_text(encoding="utf-8"))
    source = _source_paths()
    return assemble_columns(
        gene_keys,
        parse_sources(
            gene_map_path=source["gene_map"],
            edges_path=source["edges"],
            goa_path=source["goa"],
            corum_path=source["corum"],
            pfam_path=source["pfam"],
            formal=True,
        ),
        [str(term) for term in terms["selected_go_terms"]],
        sequence_path=source["sequence"],
        node2vec_path=source["node2vec"],
    )


def predict(model_dir: Path, genes: Path, output: Path) -> dict[str, object]:
    card = read_model_card(model_dir / "model_card.json")
    state = np.load(model_dir / SCALER_FILE, allow_pickle=False)
    means = state["mean"]
    scales = state["scale"]
    columns = [str(name) for name in state["feature_columns"]]
    snapshot_paths = sorted(
        (model_dir / str(card["artifacts"]["weights"])).glob("snapshot_*.pt")
    )
    if len(snapshot_paths) != FORMAL_SNAPSHOTS:
        raise ContractError(
            f"{model_dir}: the ensemble is {FORMAL_SNAPSHOTS} snapshots, "
            f"{len(snapshot_paths)} are stored"
        )

    requested = pd.read_parquet(genes)
    if list(requested.columns) != ["gene_key"]:
        raise ContractError(
            f"{genes}: the gene list holds exactly one column, 'gene_key', "
            f"found {list(requested.columns)}"
        )
    gene_keys = requested["gene_key"].astype(str).tolist()
    if len(set(gene_keys)) != len(gene_keys):
        raise ContractError(f"{genes}: duplicate gene_key values")

    device = inference_device()
    raw = fold_view(model_dir, gene_keys)[columns].to_numpy(dtype=np.float32)
    standardized = torch.as_tensor(
        (raw - means) / scales, dtype=torch.float32, device=device
    )
    model = DeEPsnapMLP(input_dimensions=len(columns)).to(device)
    states = [
        torch.load(path, map_location=device, weights_only=True)["state_dict"]
        for path in snapshot_paths
    ]

    with RunRecord() as record:
        scores = snapshot_probabilities(model, states, standardized)
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
