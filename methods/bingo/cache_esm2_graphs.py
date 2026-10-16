#!/usr/bin/env python3
"""Build the shared BiNGO ESM-2 residue-embedding and contact-graph cache for the DEG and DepMap candidate genes.
Run from methods/bingo: python cache_esm2_graphs.py --device cuda:0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
CANDIDATE_GENE_LISTS = REPO_ROOT / "data/candidates"


ASSET_DIRECTORY = METHOD_ROOT / "feature/source/model_assets"
CACHE_DIRECTORY = METHOD_ROOT / "feature/source/esm2_graph_cache"


def acquire_assets() -> None:

    subprocess.run(
        [
            sys.executable,
            str(METHOD_ROOT / "code/acquire_assets.py"),
            "--asset-dir",
            str(ASSET_DIRECTORY),
        ],
        check=True,
        cwd=METHOD_ROOT / "code",
    )


def build_cache(device: str) -> None:

    subprocess.run(
        [
            sys.executable,
            str(METHOD_ROOT / "code/extract_features.py"),
            "--deg-gene-list",
            str(CANDIDATE_GENE_LISTS / "deg_common_feature_available.parquet"),
            "--depmap-gene-list",
            str(CANDIDATE_GENE_LISTS / "depmap_common_feature_available.parquet"),
            "--deg-protein-fasta",
            str(METHOD_ROOT / "feature/source/deg_protein_sequences.fa.gz"),
            "--depmap-protein-fasta",
            str(METHOD_ROOT / "feature/source/depmap_protein_sequences.fa.gz"),
            "--asset-dir",
            str(ASSET_DIRECTORY),
            "--cache-dir",
            str(CACHE_DIRECTORY),
            "--output",
            str(CACHE_DIRECTORY / "cache_manifest.json"),
            "--deg-index-output",
            str(CACHE_DIRECTORY / "deg_cache_index.parquet"),
            "--depmap-index-output",
            str(CACHE_DIRECTORY / "depmap_cache_index.parquet"),
            "--device",
            device,
        ],
        check=True,
        cwd=METHOD_ROOT / "code",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    if not (ASSET_DIRECTORY / "esm2_t33_650M_UR50D.pt").is_file():
        acquire_assets()
    build_cache(arguments.device)


if __name__ == "__main__":
    main()
