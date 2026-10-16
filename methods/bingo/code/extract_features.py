"""Run ESM-2 over the DEG and DepMap proteins and write the BiNGO graph cache and its indices.
Called by cache_esm2_graphs.py.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import pandas as pd
import torch

if __package__:
    from .cache_features import (
        METHOD_ROOT,
        _read_gene_list,
        _write_json_atomic,
        _write_parquet_atomic,
        build_cache_from_sequences,
        gene_key_sha256,
        read_protein_fasta,
    )
    from .contracts import require_method_path, sha256_file
else:
    from cache_features import (
        METHOD_ROOT,
        _read_gene_list,
        _write_json_atomic,
        _write_parquet_atomic,
        build_cache_from_sequences,
        gene_key_sha256,
        read_protein_fasta,
    )
    from contracts import require_method_path, sha256_file


REPO_ROOT = METHOD_ROOT.parents[1]
AUTHORITY_GENE_LISTS = {
    "DEG": REPO_ROOT / "data/candidates/deg_common_feature_available.parquet",
    "DepMap": REPO_ROOT / "data/candidates/depmap_common_feature_available.parquet",
}
DEFAULT_DEG_FASTA = METHOD_ROOT / "feature/source/deg_protein_sequences.fa.gz"
DEFAULT_DEPMAP_FASTA = METHOD_ROOT / "feature/source/depmap_protein_sequences.fa.gz"
DEFAULT_ASSET_DIR = METHOD_ROOT / "feature/source/model_assets"
DEFAULT_CACHE_DIR = METHOD_ROOT / "feature/source/esm2_graph_cache"
DEFAULT_DEG_INDEX = DEFAULT_CACHE_DIR / "deg_cache_index.parquet"
DEFAULT_DEPMAP_INDEX = DEFAULT_CACHE_DIR / "depmap_cache_index.parquet"


def merge_gene_sequences(
    *,
    deg_genes: Sequence[str],
    depmap_genes: Sequence[str],
    deg_sequences: Mapping[str, str],
    depmap_sequences: Mapping[str, str],
) -> tuple[tuple[str, ...], dict[str, str], tuple[str, ...]]:

    deg_set = {str(gene).strip() for gene in deg_genes}
    depmap_set = {str(gene).strip() for gene in depmap_genes}
    missing_deg = sorted(deg_set.difference(deg_sequences))
    missing_depmap = sorted(depmap_set.difference(depmap_sequences))
    if missing_deg:
        raise ValueError(f"{len(missing_deg)} DEG genes lack protein sequences")
    if missing_depmap:
        raise ValueError(f"{len(missing_depmap)} DepMap genes lack protein sequences")

    overlap = tuple(sorted(deg_set & depmap_set))
    mismatched = [
        gene
        for gene in overlap
        if str(deg_sequences[gene]).upper() != str(depmap_sequences[gene]).upper()
    ]
    if mismatched:
        preview = ", ".join(mismatched[:5])
        suffix = "" if len(mismatched) <= 5 else " and more"
        raise ValueError(f"overlapping gene sequences disagree: {preview}{suffix}")

    union = tuple(sorted(deg_set | depmap_set))
    merged = {
        gene: str(deg_sequences[gene]).upper()
        if gene in deg_set
        else str(depmap_sequences[gene]).upper()
        for gene in union
    }
    return union, merged, overlap


def _source_record(path: Path, genes: Sequence[str]) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(REPO_ROOT).as_posix(),
        "gene_count": len(genes),
        "gene_key_sha256": gene_key_sha256(genes),
    }


def _require_authority_gene_list(path: Path, dataset: str) -> Path:
    resolved = path.expanduser().resolve()
    expected = AUTHORITY_GENE_LISTS[dataset].resolve()
    if resolved != expected:
        raise ValueError(
            f"{dataset} gene list must be the frozen authority {expected}, got {resolved}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _write_dataset_index(
    *,
    shared_index: pd.DataFrame,
    genes: Sequence[str],
    shared_cache_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    selected = shared_index.loc[shared_index["gene_key"].isin(set(genes))].copy()
    selected = selected.sort_values("gene_key", kind="mergesort")
    if len(selected) != len(set(genes)):
        missing = sorted(set(genes).difference(selected["gene_key"].astype(str)))
        raise ValueError(f"shared cache index lacks {len(missing)} requested genes")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for cache_file in selected["cache_file"].astype(str):
        graph = shared_cache_dir / "graphs" / cache_file
        if not graph.is_file():
            raise FileNotFoundError(graph)
    _write_parquet_atomic(selected, output_path)
    return {
        "path": output_path.resolve().relative_to(METHOD_ROOT).as_posix(),
        "gene_count": len(selected),
        "sha256": sha256_file(output_path),
    }


def _canonical_graph_subpath(value: object) -> str:
    raw = str(value).strip()
    if not raw or "\\" in raw:
        raise ValueError(f"invalid Bingo shared cache_file: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid Bingo shared cache_file: {raw!r}")
    parts = path.parts[1:] if path.parts[:1] == ("graphs",) else path.parts
    if not parts or parts[0] == "graphs":
        raise ValueError(f"invalid Bingo shared cache_file: {raw!r}")
    return PurePosixPath(*parts).as_posix()


def build_shared_cache(
    *,
    deg_gene_list: Path,
    depmap_gene_list: Path,
    deg_protein_fasta: Path = DEFAULT_DEG_FASTA,
    depmap_protein_fasta: Path = DEFAULT_DEPMAP_FASTA,
    asset_dir: Path = DEFAULT_ASSET_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output: Path | None = None,
    deg_index_output: Path = DEFAULT_DEG_INDEX,
    depmap_index_output: Path = DEFAULT_DEPMAP_INDEX,
    device: torch.device = torch.device("cuda:0"),
    token_budget: int = 1024,
    max_batch_size: int = 2,
) -> dict[str, Any]:

    deg_gene_list = _require_authority_gene_list(Path(deg_gene_list), "DEG")
    depmap_gene_list = _require_authority_gene_list(Path(depmap_gene_list), "DepMap")
    deg_protein_fasta = require_method_path(
        Path(deg_protein_fasta), "deg_protein_fasta"
    )
    depmap_protein_fasta = require_method_path(
        Path(depmap_protein_fasta), "depmap_protein_fasta"
    )
    asset_dir = require_method_path(Path(asset_dir), "asset_dir")
    cache_dir = require_method_path(Path(cache_dir), "cache_dir")
    deg_index_output = require_method_path(
        Path(deg_index_output), "deg_index_output"
    )
    depmap_index_output = require_method_path(
        Path(depmap_index_output), "depmap_index_output"
    )
    output = require_method_path(
        Path(output) if output is not None else cache_dir / "cache_manifest.json",
        "output",
    )

    deg_genes = _read_gene_list(deg_gene_list)
    depmap_genes = _read_gene_list(depmap_gene_list)
    deg_sequences = read_protein_fasta(deg_protein_fasta)
    depmap_sequences = read_protein_fasta(depmap_protein_fasta)
    union, sequences, overlap = merge_gene_sequences(
        deg_genes=deg_genes,
        depmap_genes=depmap_genes,
        deg_sequences=deg_sequences,
        depmap_sequences=depmap_sequences,
    )

    manifest = build_cache_from_sequences(
        genes=union,
        sequences=sequences,
        asset_dir=asset_dir,
        cache_dir=cache_dir,
        output=output,
        index_output=cache_dir / "cache_index.parquet",
        source_paths=(deg_protein_fasta, depmap_protein_fasta),
        gene_sources=(
            _source_record(deg_gene_list, deg_genes),
            _source_record(depmap_gene_list, depmap_genes),
        ),
        device=torch.device(device),
        token_budget=token_budget,
        max_batch_size=max_batch_size,
    )

    shared_index_path = cache_dir / "cache_index.parquet"
    shared_index = pd.read_parquet(shared_index_path)
    shared_index["cache_file"] = shared_index["cache_file"].map(
        _canonical_graph_subpath
    )
    _write_parquet_atomic(shared_index, shared_index_path)
    manifest["cache_index_sha256"] = sha256_file(shared_index_path)
    output_indexes = {
        "deg": _write_dataset_index(
            shared_index=shared_index,
            genes=deg_genes,
            shared_cache_dir=cache_dir,
            output_path=deg_index_output,
        ),
        "depmap": _write_dataset_index(
            shared_index=shared_index,
            genes=depmap_genes,
            shared_cache_dir=cache_dir,
            output_path=depmap_index_output,
        ),
    }
    manifest.update(
        {
            "dataset_gene_counts": {
                "deg": len(deg_genes),
                "depmap": len(depmap_genes),
            },
            "union_gene_count": len(union),
            "overlap_gene_count": len(overlap),
            "output_indexes": output_indexes,
        }
    )
    _write_json_atomic(output, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deg-gene-list", type=Path, required=True)
    parser.add_argument("--depmap-gene-list", type=Path, required=True)
    parser.add_argument("--deg-protein-fasta", type=Path, default=DEFAULT_DEG_FASTA)
    parser.add_argument(
        "--depmap-protein-fasta", type=Path, default=DEFAULT_DEPMAP_FASTA
    )
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deg-index-output", type=Path, default=DEFAULT_DEG_INDEX)
    parser.add_argument(
        "--depmap-index-output", type=Path, default=DEFAULT_DEPMAP_INDEX
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--token-budget", type=int, default=1024)
    parser.add_argument("--max-batch-size", type=int, default=2)
    args = parser.parse_args()
    build_shared_cache(
        deg_gene_list=args.deg_gene_list,
        depmap_gene_list=args.depmap_gene_list,
        deg_protein_fasta=args.deg_protein_fasta,
        depmap_protein_fasta=args.depmap_protein_fasta,
        asset_dir=args.asset_dir,
        cache_dir=args.cache_dir,
        output=args.output,
        deg_index_output=args.deg_index_output,
        depmap_index_output=args.depmap_index_output,
        device=torch.device(args.device),
        token_budget=args.token_budget,
        max_batch_size=args.max_batch_size,
    )


if __name__ == "__main__":
    main()
