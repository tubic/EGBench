"""Paths and HGNC mapping used by Mahi.
Imported by run.py; not run directly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


METHOD_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = METHOD_ROOT / "feature" / "source"
HGNC_SOURCE = SOURCE_ROOT / "gene_reference" / "hgnc_complete_set_2026-07-07.txt"
DEFAULT_EMBEDDING_SOURCE = SOURCE_ROOT / "embeddings" / "mahi_global_context.parquet"


def method_path(path: Path, *, label: str) -> Path:

    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [METHOD_ROOT / path, Path.cwd() / path]
    resolved_candidates = [candidate.resolve() for candidate in candidates]
    inside_candidates = [
        candidate for candidate in resolved_candidates if candidate.is_relative_to(METHOD_ROOT)
    ]
    existing_inside = [candidate for candidate in inside_candidates if candidate.exists()]
    parent_inside = [candidate for candidate in inside_candidates if candidate.parent.exists()]
    resolved = next(
        iter(existing_inside or parent_inside or inside_candidates or resolved_candidates),
    )
    if not resolved.is_relative_to(METHOD_ROOT):
        raise ValueError(f"{label} must be located inside {METHOD_ROOT}: {path}")
    return resolved


def read_hgnc_mapping() -> dict[str, str]:
    if not HGNC_SOURCE.is_file():
        raise FileNotFoundError(f"local HGNC complete set is absent: {HGNC_SOURCE}")
    frame = pd.read_csv(
        HGNC_SOURCE,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        na_filter=False,
    )
    required = {"hgnc_id", "entrez_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{HGNC_SOURCE} lacks columns: {sorted(missing)}")
    hgnc_id = frame["hgnc_id"].astype("string").str.strip()
    entrez_id = frame["entrez_id"].astype("string").str.strip()
    if hgnc_id.eq("").any() or hgnc_id.duplicated().any():
        raise ValueError("HGNC complete set has empty or duplicate hgnc_id values")
    valid_entrez = entrez_id.str.fullmatch(r"[0-9]+", na=False)
    return dict(zip(hgnc_id[valid_entrez].tolist(), entrez_id[valid_entrez].tolist()))
