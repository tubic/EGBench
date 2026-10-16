"""Coding and protein sequence pairing for EssTFNet.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import gzip
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = METHOD_ROOT / "feature" / "source"
SEQUENCE_SOURCES = SOURCE_ROOT / "sequences"
CDS_ALPHABET = frozenset("ACGT")
PROTEIN_ALPHABET = frozenset("RSPFICETYKGANUMLWDVHQ")


def read_fasta(path: Path) -> dict[str, str]:

    records: dict[str, str] = {}
    key: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        nonlocal key, chunks
        if key is None:
            return
        if key in records:
            raise ValueError(f"duplicate FASTA key {key} in {path}")
        records[key] = "".join(chunks).upper()
        key = None
        chunks = []

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
                key = header.split("|", 1)[0].split()[0]
                if not key:
                    raise ValueError(f"empty FASTA key at {path}:{line_number}")
            elif key is None:
                raise ValueError(f"sequence before FASTA header at {path}:{line_number}")
            else:
                chunks.append(line)
    flush()
    if not records:
        raise ValueError(f"no FASTA records in {path}")
    return records


def _accepted_pair(cds: str, protein: str) -> bool:
    return bool(cds) and bool(protein) and set(cds) <= CDS_ALPHABET and set(protein) <= PROTEIN_ALPHABET


def select_sequence_pairs() -> dict[str, dict[str, str]]:

    source_records: list[tuple[str, dict[str, str], dict[str, str]]] = []
    for source_name in ("ensembl97", "gencode50"):
        source_records.append(
            (
                source_name,
                read_fasta(SEQUENCE_SOURCES / f"{source_name}.cds.fa.gz"),
                read_fasta(SEQUENCE_SOURCES / f"{source_name}.pep.fa.gz"),
            )
        )

    all_keys = sorted(
        set().union(
            *(set(cds).intersection(protein) for _, cds, protein in source_records)
        )
    )
    selected: dict[str, dict[str, str]] = {}
    for gene_key in all_keys:
        for source_name, cds_records, protein_records in source_records:
            if gene_key not in cds_records or gene_key not in protein_records:
                continue
            cds = cds_records[gene_key]
            protein = protein_records[gene_key]
            if _accepted_pair(cds, protein):
                selected[gene_key] = {
                    "cds": cds,
                    "protein": protein,
                    "sequence_source": source_name,
                }
                break
    return selected
