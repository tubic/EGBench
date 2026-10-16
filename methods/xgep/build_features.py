#!/usr/bin/env python3
"""Write the XGEP fold view of the 150-dimensional matrix, and rebuild that matrix from the authors' embeddings.
Rebuild from methods/xgep: python build_features.py rebuild-source
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
sys.path.insert(0, str(METHOD_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from egbench.contracts import COHORT_IDS, SPLIT_STRATEGIES
from egbench.data import feature_directory, folds_directory

METHOD_ID = "xgep"
FEATURE_DIMENSIONS = 150
SOURCE_ROOT = METHOD_ROOT / "feature" / "source"
SOURCE_MATRIX = SOURCE_ROOT / "xgep_150.parquet"
SOURCE_AUDIT = SOURCE_ROOT / "xgep_150_audit.json"
GENE_MAP = SOURCE_ROOT / "gene_id_map.parquet"
VENDOR_ROOT = SOURCE_ROOT / "vendor"
VENDOR_FILES = (
    "Essential_genes_pancancer_emb_len150",
    "Non_essential_genes_pancancer_emb_len150",
    "Genes_for_prediction_pancancer_emb_len150",
)


def _ensembl_mapping(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=["ensembl_gene_id", "gene_key"])
    if frame.isna().any().any():
        raise ValueError(f"XGEP gene map contains null identifiers: {path}")
    frame = frame.astype(str)
    conflicts = frame.groupby("ensembl_gene_id")["gene_key"].nunique()
    if (conflicts > 1).any():
        examples = sorted(conflicts[conflicts > 1].index)[:5]
        raise ValueError(f"XGEP gene map contains conflicting IDs: {examples}")
    return dict(zip(frame["ensembl_gene_id"], frame["gene_key"]))


def rebuild_source() -> None:
    from method_features.legacy_features import build_xgep_pancancer_150, write_xgep_150

    for filename in VENDOR_FILES:
        path = VENDOR_ROOT / filename
        if not path.is_file():
            raise FileNotFoundError(path)
    build = build_xgep_pancancer_150(VENDOR_ROOT, _ensembl_mapping(GENE_MAP))
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    write_xgep_150(build, SOURCE_MATRIX, SOURCE_AUDIT)


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
    if not SOURCE_MATRIX.is_file():
        raise FileNotFoundError(
            f"missing XGEP source matrix: {SOURCE_MATRIX}. Rebuild it with "
            f"'python build_features.py rebuild-source' first."
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
    matrix = pd.read_parquet(SOURCE_MATRIX)
    if matrix.columns[0] != "gene_key" or matrix.shape[1] != FEATURE_DIMENSIONS + 1:
        raise ValueError(
            f"{SOURCE_MATRIX} must hold gene_key plus {FEATURE_DIMENSIONS} features"
        )
    matrix = matrix.set_index("gene_key", drop=False)
    missing = sorted(set(keys) - set(matrix.index))
    if missing:
        raise ValueError(
            f"{len(missing)} authority genes lack XGEP source features: {missing[:5]}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    canonical = fold_feature_directory(cohort_id, SPLIT_STRATEGIES[0], 0) / "features.parquet"
    if destination != canonical and canonical.is_file():
        os.link(canonical, destination)
    else:
        subset = matrix.loc[keys].reset_index(drop=True)
        subset.to_parquet(destination, index=False)
    return destination, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("rebuild-source",))
    parser.parse_args()
    rebuild_source()


if __name__ == "__main__":
    main()
