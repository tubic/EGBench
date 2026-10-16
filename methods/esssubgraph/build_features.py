#!/usr/bin/env python3
"""Write the EssSubgraph fold features: TCGA expression principal components fitted on the train role.
Imported by run.py; not run directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from fold_expression import build_expression_fold
from local_dataset import METHOD_ROOT


REPO_ROOT = METHOD_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from egbench.contracts import COHORT_IDS
from egbench.data import (
    feature_directory,
    folds_directory,
)


def _support_path() -> Path:
    return METHOD_ROOT / "support_gene_list.csv"


def _read_support() -> set[str]:
    support_path = _support_path()
    if not support_path.is_file():
        raise FileNotFoundError(
            f"EssSubgraph source asset is absent: {support_path}. It is checked "
            f"into the repository beside build_features.py."
        )
    frame = pd.read_csv(support_path, dtype="string", keep_default_na=False)
    if list(frame.columns) != ["dataset", "gene_key", "gene_symbol"]:
        raise ValueError("support_gene_list.csv must have dataset,gene_key,gene_symbol columns")
    keys = frame["gene_key"].astype(str)
    if keys.empty:
        raise ValueError("support_gene_list.csv is empty")
    return set(keys)


def _split_build_hash(
    folds: Path, split_strategy: str, outer_fold: int, keys: set[str]
) -> str:
    assignments = pd.read_parquet(
        folds / "split_assignments.parquet",
        columns=["gene_key", "outer_fold", "split_strategy", "split_build_hash"],
    )
    scope = assignments.loc[
        assignments["outer_fold"].eq(outer_fold)
        & assignments["split_strategy"].astype(str).eq(split_strategy)
    ]
    hashes = set(scope["split_build_hash"].astype(str))
    if len(hashes) != 1 or set(scope["gene_key"].astype(str)) != keys:
        raise ValueError("authority fold and split assignments disagree")
    return hashes.pop()


def build_fold_features(
    cohort_id: str, outer_fold: int, *, split_strategy: str
) -> tuple[Path, bool]:
    if cohort_id not in COHORT_IDS:
        raise ValueError(f"unsupported cohort_id: {cohort_id!r}")
    output_directory = feature_directory(
        METHOD_ROOT,
        cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )
    destination = output_directory / "features.parquet"
    if destination.is_file():
        return destination, True
    if output_directory.exists():
        raise FileExistsError(
            f"{output_directory} exists but does not hold features.parquet; "
            f"remove it before cutting this fold's view again"
        )
    expression_path = (
        METHOD_ROOT / "feature" / "source" / "expression" / "tcga_gene_by_sample.parquet"
    )
    if not expression_path.is_file():
        raise FileNotFoundError(
            f"EssSubgraph source asset is absent: {expression_path}. It is a "
            f"frozen input under feature/source."
        )
    support = _read_support()
    folds = folds_directory(
        cohort_id, split_strategy=split_strategy, root=REPO_ROOT
    )
    fold_path = folds / "fold_genes" / f"fold_{outer_fold}_genes.parquet"
    fold = pd.read_parquet(fold_path, columns=["gene_key", "role"])
    keys = set(fold["gene_key"].astype(str))
    if len(keys) != len(fold) or not keys.issubset(support):
        raise ValueError(f"{fold_path} is not a supported complete fold")
    split_build_hash = _split_build_hash(folds, split_strategy, outer_fold, keys)
    manifest = build_expression_fold(
        expression_path=expression_path,
        fold_genes_path=fold_path,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        output_directory=output_directory,
    )
    features = pd.concat(
        [
            pd.read_parquet(output_directory / f"expression_pca_50_{role}.parquet")
            for role in ("train", "validation", "test")
        ],
        ignore_index=True,
    ).sort_values("gene_key", kind="mergesort")
    if (
        len(features) != len(keys)
        or features["gene_key"].duplicated().any()
        or features.shape[1] != 51
    ):
        raise ValueError(
            "EssSubgraph fold feature output violates its 50-dimensional contract"
        )
    features.to_parquet(destination, index=False)
    manifest["split_build_hash"] = split_build_hash
    manifest["support_gene_count"] = len(support)
    (output_directory / "expression_pca_50_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination, False
