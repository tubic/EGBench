#!/usr/bin/env python3
"""Entry point of Mahi: preflight, features, train and predict for one fold of one task.
Run: python methods/mahi/run.py {preflight,features,train} --dataset <task> --split-strategy family_5fold --outer-fold <k>
     python methods/mahi/run.py predict --model-dir <fold model dir> --genes <gene_key parquet> --output <new dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
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

import build_features
from local_dataset import (
    DEFAULT_EMBEDDING_SOURCE,
    HGNC_SOURCE,
    read_hgnc_mapping,
)

METHOD_ID = "mahi"

THREADS = 4
BOOSTER_PARAMETERS = {
    "booster": "gbtree",
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}
WEIGHTS_FILE = "weights/model.json"


def _feature_path(cohort_id: str, split_strategy: str, outer_fold: int) -> Path:
    return (
        feature_directory(
            ROOT,
            cohort_id,
            split_strategy=split_strategy,
            outer_fold=outer_fold,
        )
        / "features.parquet"
    )


def _required_source_assets() -> list[dict[str, str]]:
    return [
        {
            "path": DEFAULT_EMBEDDING_SOURCE.relative_to(REPO_ROOT).as_posix(),
            "purpose": (
                "The author-published global Mahi embedding, keyed by Entrez "
                "ID; a gene absent from it cannot be scored."
            ),
        },
        {
            "path": HGNC_SOURCE.relative_to(REPO_ROOT).as_posix(),
            "purpose": "Maps a gene_key (HGNC ID) to the Entrez ID the embedding table is keyed by.",
        },
    ]


def _load_embedding_matrix(path: Path) -> pd.DataFrame:
    features = pd.read_parquet(path)
    if list(features.columns) != ["gene_key", "embedding"]:
        raise ContractError("Mahi features must hold gene_key plus one embedding column")
    if features["gene_key"].duplicated().any():
        raise ContractError("Mahi features contain duplicate gene_key values")
    matrix = np.stack(features["embedding"].to_numpy()).astype(np.float32, copy=False)
    return pd.DataFrame(matrix, index=features["gene_key"].astype(str))


def load_fold(cohort_id: str, split_strategy: str, outer_fold: int) -> dict[str, tuple]:
    feature_path = _feature_path(cohort_id, split_strategy, outer_fold)
    if not feature_path.is_file():
        raise ContractError(
            f"the fold features are absent: {feature_path}. Run 'run.py features "
            f"--dataset {cohort_id} --split-strategy {split_strategy} --outer-fold "
            f"{outer_fold}' first"
        )
    embedding = _load_embedding_matrix(feature_path)
    partitions: dict[str, tuple] = {}
    for role in ("train", "validation", "test"):
        keys = (
            fold_gene_keys(
                cohort_id, outer_fold, role, split_strategy=split_strategy, root=REPO_ROOT
            )
            .astype(str)
            .tolist()
        )
        missing = sorted(set(keys).difference(embedding.index))
        if missing:
            raise ContractError(f"{len(missing)} {role} genes lack Mahi embeddings")
        if role == "test":
            y = None
        else:
            labels = read_training_labels(
                cohort_id, outer_fold, role, split_strategy=split_strategy, root=REPO_ROOT
            )
            label_lookup = dict(zip(labels["gene_key"].astype(str), labels["label"].astype(int)))
            y = np.asarray([label_lookup[key] for key in keys], dtype=np.int64)
        partitions[role] = (
            keys,
            embedding.loc[keys].to_numpy(dtype=np.float32),
            y,
        )
    return partitions


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_value = -2.0
    for threshold in np.arange(0.01, 1.0, 0.01):
        value = float(matthews_corrcoef(labels, (scores >= threshold).astype(np.int64)))
        if value > best_value:
            best_threshold = float(round(threshold, 2))
            best_value = value
    return best_threshold, best_value


def _split_build_hash(cohort_id: str, split_strategy: str, outer_fold: int) -> str:
    assignments = read_split_assignments(cohort_id, split_strategy=split_strategy, root=REPO_ROOT)
    values = sorted(
        set(assignments.loc[assignments["outer_fold"].eq(outer_fold), "split_build_hash"])
    )
    if len(values) != 1:
        raise ContractError(
            f"fold {outer_fold} of {cohort_id}/{split_strategy} carries "
            f"{len(values)} split_build_hash values: {values}"
        )
    return values[0]


def preflight(cohort_id: str, split_strategy: str, outer_fold: int) -> dict[str, object]:
    output = model_directory(ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold)
    if output.exists():
        raise ContractError(
            f"the products of this fold are already there: {output}. A run "
            f"writes a directory that does not yet exist; remove it to run again"
        )
    feature_path = _feature_path(cohort_id, split_strategy, outer_fold)
    return {
        "method_id": METHOD_ID,
        "cohort_id": cohort_id,
        "split_strategy": split_strategy,
        "outer_fold": outer_fold,
        "split_build_hash": _split_build_hash(cohort_id, split_strategy, outer_fold),
        "fold_features_present": feature_path.is_file(),
        "model_directory": str(output.relative_to(ROOT)),
    }


def features(cohort_id: str, split_strategy: str, outer_fold: int) -> dict[str, object]:
    return build_features.build_family_fold(cohort_id, outer_fold, split_strategy=split_strategy)


def train(cohort_id: str, split_strategy: str, outer_fold: int) -> dict[str, object]:
    output = model_directory(ROOT, cohort_id, split_strategy=split_strategy, outer_fold=outer_fold)
    if output.exists():
        raise ContractError(
            f"the products of this fold are already there: {output}. A run "
            f"writes a directory that does not yet exist; remove it to run again"
        )
    partitions = load_fold(cohort_id, split_strategy, outer_fold)
    train_keys, train_x, train_y = partitions["train"]
    validation_keys, validation_x, validation_y = partitions["validation"]
    test_keys, test_x, _ = partitions["test"]
    positives = int((train_y == 1).sum())
    negatives = int((train_y == 0).sum())
    if positives == 0 or negatives == 0:
        raise ContractError("the train role must carry both classes")
    scale_pos_weight = negatives / positives

    with RunRecord() as record:
        model = XGBClassifier(
            **BOOSTER_PARAMETERS,
            scale_pos_weight=scale_pos_weight,
            n_jobs=THREADS,
            verbosity=0,
            random_state=SEED,
        )
        model.fit(train_x, train_y)
        validation_scores = model.predict_proba(validation_x)[:, 1].astype(np.float64)
        test_scores = model.predict_proba(test_x)[:, 1].astype(np.float64)

    threshold, validation_mcc = select_threshold(validation_y, validation_scores)

    weights_path = output / WEIGHTS_FILE
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(weights_path))

    write_predictions(output, "blind", test_keys, test_scores, decision_threshold=threshold)
    write_predictions(
        output, "validation", validation_keys, validation_scores, decision_threshold=threshold
    )

    versions = software_versions(np, pd, sklearn, xgboost)
    split_build_hash = _split_build_hash(cohort_id, split_strategy, outer_fold)
    write_run_manifest(
        output,
        method_id=METHOD_ID,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        split_build_hash=split_build_hash,
        software_versions=versions,
        parameters={
            "threads": THREADS,
            **BOOSTER_PARAMETERS,
            "scale_pos_weight": scale_pos_weight,
            "feature_dimensions": int(train_x.shape[1]),
            "fitting_roles": ["train"],
        },
        sample_counts={
            "train": len(train_keys),
            "validation": len(validation_keys),
            "test": len(test_keys),
        },
        threshold_selection_metric="validation_mcc",
        validation_metrics={
            "average_precision": float(average_precision_score(validation_y, validation_scores)),
            "roc_auc": float(roc_auc_score(validation_y, validation_scores)),
            "mcc_at_selected_threshold": validation_mcc,
        },
        **record.timestamps,
    )
    write_resource_usage(output, record, device="cpu", gpu_name=None, peak_gpu_memory_bytes=0)
    write_model_card(
        output,
        method_id=METHOD_ID,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        split_build_hash=split_build_hash,
        decision_threshold=threshold,
        artifacts={"weights": WEIGHTS_FILE, "transforms": []},
        required_source_assets=_required_source_assets(),
        software_versions=versions,
    )
    return {"model_directory": str(output), "test_genes": len(test_keys)}


def _global_embeddings() -> dict[str, np.ndarray]:
    if not DEFAULT_EMBEDDING_SOURCE.is_file():
        raise ContractError(
            f"the Mahi global embedding table is absent: {DEFAULT_EMBEDDING_SOURCE}. "
            f"It is a source asset of this method under feature/source"
        )
    table = pd.read_parquet(DEFAULT_EMBEDDING_SOURCE, columns=["entrez_id", "tissue", "embedding"])
    table = table.loc[table["tissue"].astype(str).eq("global")]
    embeddings: dict[str, np.ndarray] = {}
    for entrez_id, vector in zip(table["entrez_id"], table["embedding"]):
        if isinstance(entrez_id, (int, np.integer)):
            key = str(int(entrez_id))
        elif isinstance(entrez_id, (float, np.floating)):
            key = str(int(entrez_id))
        else:
            key = str(entrez_id).strip()
        embeddings[key] = np.asarray(vector, dtype=np.float32)
    return embeddings


def predict(model_dir: Path, genes: Path, output: Path) -> dict[str, object]:
    card = read_model_card(model_dir / "model_card.json")
    requested = pd.read_parquet(genes)
    if list(requested.columns) != ["gene_key"]:
        raise ContractError(
            f"{genes}: the gene list holds exactly one column, 'gene_key', "
            f"found {list(requested.columns)}"
        )
    gene_keys = requested["gene_key"].astype(str).tolist()
    if len(set(gene_keys)) != len(gene_keys):
        raise ContractError(f"{genes}: duplicate gene_key values")

    hgnc_to_entrez = read_hgnc_mapping()
    embeddings = _global_embeddings()

    missing: list[str] = []
    rows: list[np.ndarray] = []
    for key in gene_keys:
        entrez_id = hgnc_to_entrez.get(key)
        vector = embeddings.get(entrez_id) if entrez_id is not None else None
        if vector is None:
            missing.append(key)
        else:
            rows.append(vector)
    if missing:
        raise ContractError(
            f"{len(missing)} of {len(gene_keys)} requested genes are absent from "
            f"the Mahi global embedding table, the first ten are {missing[:10]}. "
            f"Mahi scores a gene from the fixed row the authors published for "
            f"its Entrez ID and cannot score a gene outside that table"
        )
    x = np.stack(rows).astype(np.float32)

    model = XGBClassifier()
    model.load_model(str(model_dir / str(card["artifacts"]["weights"])))
    with RunRecord() as record:
        scores = model.predict_proba(x)[:, 1].astype(np.float64)
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
