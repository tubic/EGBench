"""Assemble the HELP feature matrix from the authors' BIO, CCcfs and node2vec tables.
Run: python methods/help/code/build_help_features.py --bio <csv> --cccfs <csv> --embn2v <csv> --gene-map <parquet> --hgnc <txt> --out <parquet>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _load_block(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.map(str)
    frame = frame.fillna(frame.mean(numeric_only=True))
    constant = frame.columns[frame.nunique() <= 1]
    frame = frame.drop(columns=constant)
    scaled = StandardScaler().fit_transform(frame.to_numpy(dtype=np.float64))
    frame = pd.DataFrame(scaled, index=frame.index, columns=frame.columns)
    frame.columns = [f"{prefix}__{i}" for i in range(frame.shape[1])]
    print(f"[{prefix}] {path.name}: {frame.shape[0]} genes x {frame.shape[1]} features "
          f"({len(constant)} constant dropped)")
    return frame


def _symbol_to_gene_key(gene_map: pd.DataFrame, hgnc_path: Path) -> dict[str, str]:
    resolver: dict[str, str] = {}
    for symbol, gene_key in zip(gene_map["symbol"].astype(str), gene_map["gene_key"].astype(str)):
        resolver[symbol.upper()] = gene_key
    primary = set(resolver)

    valid_keys = set(gene_map["gene_key"].astype(str))
    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str, low_memory=False)
    ambiguous: set[str] = set()
    alias_map: dict[str, str] = {}
    for _, row in hgnc[["hgnc_id", "alias_symbol", "prev_symbol"]].iterrows():
        gene_key = str(row["hgnc_id"])
        if gene_key not in valid_keys:
            continue
        for field in ("alias_symbol", "prev_symbol"):
            raw = row[field]
            if not isinstance(raw, str) or not raw:
                continue
            for alias in raw.split("|"):
                alias = alias.strip().upper()
                if not alias or alias in primary:
                    continue
                if alias in alias_map and alias_map[alias] != gene_key:
                    ambiguous.add(alias)
                else:
                    alias_map[alias] = gene_key
    for alias in ambiguous:
        alias_map.pop(alias, None)
    for alias, gene_key in alias_map.items():
        resolver.setdefault(alias, gene_key)
    print(f"symbol resolver: {len(primary)} primary + {len(alias_map)} alias/prev "
          f"({len(ambiguous)} ambiguous dropped)")
    return resolver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bio", type=Path, required=True)
    parser.add_argument("--cccfs", type=Path, required=True)
    parser.add_argument("--embn2v", type=Path, required=True)
    parser.add_argument("--gene-map", type=Path, required=True)
    parser.add_argument("--hgnc", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    bio = _load_block(args.bio, "bio")
    cccfs = _load_block(args.cccfs, "cc")
    n2v = _load_block(args.embn2v, "n2v")

    merged = bio.join([cccfs, n2v], how="outer")
    print(f"merged HELP matrix: {merged.shape[0]} genes x {merged.shape[1]} features")

    gene_map = pd.read_parquet(args.gene_map)
    resolver = _symbol_to_gene_key(gene_map, args.hgnc)

    merged = merged.copy()
    merged["gene_key"] = [resolver.get(str(sym).upper()) for sym in merged.index]
    resolved = merged.dropna(subset=["gene_key"])
    resolved = resolved.groupby("gene_key", sort=True).mean(numeric_only=True)

    all_keys = gene_map["gene_key"].astype(str).drop_duplicates().sort_values()
    features = resolved.reindex(all_keys)
    covered = int(features.notna().any(axis=1).sum())
    features = features.fillna(features.mean(numeric_only=True))

    out = features.reset_index().rename(columns={"index": "gene_key"})
    out.columns = ["gene_key", *[c for c in out.columns if c != "gene_key"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"wrote {args.out}: {out.shape[0]} genes x {out.shape[1] - 1} features; "
          f"HELP-covered {covered} ({100 * covered / len(all_keys):.1f}%), "
          f"imputed {len(all_keys) - covered}")


if __name__ == "__main__":
    main()
