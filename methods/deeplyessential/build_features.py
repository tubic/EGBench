#!/usr/bin/env python3
"""Write the DeeplyEssential fold view of the 89-dimensional sequence matrix.
Imported by run.py; not run directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

METHOD_ID = "deeplyessential"
METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from egbench.contracts import COHORT_IDS, SPLIT_STRATEGIES
from egbench.data import feature_directory, folds_directory

FEATURE_DIMENSIONS = 89

def read_source_matrix() -> pd.DataFrame:
    source = METHOD_ROOT / "feature" / "source" / "deeplyessential_sequence_89.parquet"
    if not source.is_file():
        raise FileNotFoundError(
            f"DeeplyEssential source asset is absent: {source}. It is a frozen "
            f"input under feature/source."
        )
    matrix = pd.read_parquet(source)
    if matrix.shape[1] != FEATURE_DIMENSIONS + 1:
        raise ValueError(
            "DeeplyEssential source must contain gene_key plus 89 features"
        )
    return matrix.sort_values("gene_key", kind="mergesort").reset_index(drop=True)


def fold_feature_directory(
    cohort_id: str, split_strategy: str, outer_fold: int
) -> Path:
    return feature_directory(
        METHOD_ROOT,
        cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )


def build_fold_features(
    cohort_id: str, split_strategy: str, outer_fold: int
) -> tuple[Path, bool]:
    if cohort_id not in COHORT_IDS:
        raise ValueError(f"unsupported dataset: {cohort_id!r}")
    output_directory = fold_feature_directory(cohort_id, split_strategy, outer_fold)
    destination = output_directory / "features.parquet"
    if destination.is_file():
        return destination, True
    if output_directory.exists():
        raise FileExistsError(
            f"{output_directory} exists but does not hold features.parquet; "
            f"remove it before cutting this fold's view again"
        )
    authority_path = (
        folds_directory(cohort_id, split_strategy="family_5fold", root=REPO_ROOT)
        / "fold_genes"
        / f"fold_{outer_fold}_genes.parquet"
    )
    if not authority_path.is_file():
        raise FileNotFoundError(authority_path)
    authority = pd.read_parquet(authority_path, columns=["gene_key"])
    keys = sorted(authority["gene_key"].astype(str))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{authority_path} contains duplicate gene_key values")
    matrix = read_source_matrix().set_index("gene_key", drop=False)
    missing = sorted(set(keys) - set(matrix.index))
    if missing:
        raise ValueError(
            f"{len(missing)} authority genes lack DeeplyEssential source "
            f"features: {missing[:5]}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    canonical = fold_feature_directory(cohort_id, SPLIT_STRATEGIES[0], 0) / "features.parquet"
    if destination != canonical and canonical.is_file():
        os.link(canonical, destination)
    else:
        subset = matrix.loc[keys].reset_index(drop=True)
        subset.to_parquet(destination, index=False)
    return destination, False
