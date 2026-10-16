#!/usr/bin/env python3
"""Derive the per-task source assets of the Context, Blood, Mouse and DepMap23Q4 tasks by selecting rows of the DepMap assets.
Run: python methods/derive_cohort_source_assets.py --cohort <task> [--method <method>]
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from egbench.contracts import COHORT_IDS

DONOR = "depmap"

PER_COHORT_ASSETS: dict[str, tuple[str, ...]] = {
    "bingo": ("feature/source/{cohort_id}_protein_sequences.fa.gz",),
    "deepcelless": ("feature/source/selected_proteins/{cohort_id}.fa.gz",),
    "ednn_eg": ("feature/source/{cohort_id}_cds_sequences.fa.gz",),
    "esstfnet": (
        "feature/source/{cohort_id}/cds_sequences.fa.gz",
        "feature/source/{cohort_id}/evo2_7b_base_embeddings.parquet",
        "feature/source/{cohort_id}/raw_candidates_86151.parquet",
    ),
    "gcnn_sfm": ("feature/source/selected_cds/{cohort_id}.fa.gz",),
    "mahi": ("feature/source/{cohort_id}/features.parquet",),
    "proteomelm": (
        "feature/source/{cohort_id}_common_extractable_genes.parquet",
        "feature/source/{cohort_id}_features.parquet",
        "feature/source/{cohort_id}_protein_sequences.fa.gz",
    ),
}


def candidate_genes(cohort_id: str) -> list[str]:

    path = (
        _REPO_ROOT
        / "data"
        / "candidates"
        / f"{cohort_id}_common_feature_available.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    keys = pd.read_parquet(path, columns=["gene_key"])["gene_key"].astype(str)
    if keys.duplicated().any():
        raise ValueError(f"{path} holds duplicate gene_key values")
    return sorted(keys)


def _read_fasta(path: Path) -> dict[str, tuple[str, str]]:
    records: dict[str, tuple[str, str]] = {}
    header: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        nonlocal header, chunks
        if header is None:
            return
        gene_key = header.split("|", 1)[0].split(None, 1)[0]
        if gene_key in records:
            raise ValueError(f"duplicate FASTA gene_key {gene_key!r} in {path}")
        records[gene_key] = (header, "".join(chunks))
        header = None
        chunks = []

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:]
            elif header is None:
                raise ValueError(f"sequence before FASTA header in {path}")
            else:
                chunks.append(line)
    flush()
    return records


def _write_fasta(
    records: dict[str, tuple[str, str]], keys: Sequence[str], destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".fa.gz", prefix=f".{destination.name}.", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as output:
                    for gene_key in keys:
                        header, sequence = records[gene_key]
                        output.write(f">{header}\n")
                        for start in range(0, len(sequence), 60):
                            output.write(sequence[start : start + 60] + "\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _donor_keys(path: Path) -> set[str]:
    if path.name.endswith(".fa.gz"):
        return set(_read_fasta(path))
    return set(pq.read_table(path, columns=["gene_key"]).to_pandas()["gene_key"].astype(str))


def _select_rows(source: Path, keys: Sequence[str], destination: Path) -> None:
    frame = pq.read_table(source).to_pandas()
    subset = (
        frame.loc[frame["gene_key"].astype(str).isin(set(keys))]
        .sort_values("gene_key", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(subset) != len(keys):
        raise ValueError(
            f"{source} holds {len(subset)} of the {len(keys)} genes requested"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet", prefix=f".{destination.name}.", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        subset.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def derive(cohort_id: str, methods: Iterable[str]) -> list[dict[str, object]]:
    if cohort_id not in COHORT_IDS:
        raise ValueError(f"unsupported cohort_id: {cohort_id!r}")
    if cohort_id == DONOR:
        raise ValueError(f"{DONOR} is the donor and is not derived from itself")
    keys = candidate_genes(cohort_id)
    report: list[dict[str, object]] = []
    for method in methods:
        method_root = _REPO_ROOT / "methods" / method
        for template in PER_COHORT_ASSETS[method]:
            donor = method_root / template.format(cohort_id=DONOR)
            target = method_root / template.format(cohort_id=cohort_id)
            if not donor.is_file():
                raise FileNotFoundError(donor)
            if target.exists():
                report.append({"target": str(target), "action": "already there"})
                continue
            available = _donor_keys(donor)
            missing = sorted(set(keys).difference(available))
            if missing:
                raise ValueError(
                    f"{donor} lacks {len(missing)} genes of {cohort_id}: {missing[:5]}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            if available == set(keys):
                os.link(donor, target)
                action = "hardlinked"
            elif donor.name.endswith(".fa.gz"):
                _write_fasta(_read_fasta(donor), keys, target)
                action = "selected"
            else:
                _select_rows(donor, keys, target)
                action = "selected"
            report.append(
                {
                    "target": str(target.relative_to(method_root)),
                    "method_id": method,
                    "donor": str(donor.relative_to(method_root)),
                    "action": action,
                    "genes": len(keys),
                }
            )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORT_IDS, required=True)
    parser.add_argument(
        "--method", choices=sorted(PER_COHORT_ASSETS), action="append",
        help="restrict the derivation to one method; repeatable, all by default",
    )
    arguments = parser.parse_args(argv)
    methods = arguments.method or sorted(PER_COHORT_ASSETS)
    print(json.dumps(derive(arguments.cohort, methods), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
