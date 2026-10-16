"""FASTA reading for eDNN-EG."""
from __future__ import annotations

from pathlib import Path

from .contracts import normalize_cds
from .errors import ContractViolation


def canonical_gene_key_from_header(header: str) -> str:
    fields = header.strip().split(maxsplit=1)
    if not fields:
        return ""
    token = fields[0]
    return token.split("|", maxsplit=1)[0]


def read_fasta(
    path: str | Path, *, canonical_gene_keys: bool = False
) -> dict[str, str]:
    source = Path(path)
    records: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        nonlocal current_id, chunks
        if current_id is None:
            return
        if current_id in records:
            raise ContractViolation(f"duplicate FASTA identifier: {current_id}")
        records[current_id] = normalize_cds("".join(chunks))
        current_id = None
        chunks = []

    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                raw_header = line[1:].strip()
                identifier = (
                    canonical_gene_key_from_header(raw_header)
                    if canonical_gene_keys
                    else raw_header.split(maxsplit=1)[0]
                )
                if not identifier:
                    raise ContractViolation(f"empty FASTA header in {source}")
                current_id = identifier
            else:
                if current_id is None:
                    raise ContractViolation(f"sequence encountered before FASTA header in {source}")
                chunks.append(line)
    flush()
    if not records:
        raise ContractViolation(f"no FASTA records found in {source}")
    return records
