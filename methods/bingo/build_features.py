"""Write the fold views (gene index and protein FASTA) of the shared BiNGO ESM-2 graph cache.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

METHOD_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(METHOD_ROOT.parents[1]))
from methods.family_feature_layout import (
    materialize_bingo_cache_indices,
    materialize_fasta_folds,
)


REPO_ROOT = METHOD_ROOT.parents[1]
SOURCE_CACHE = METHOD_ROOT / "feature" / "source" / "esm2_graph_cache"
GRAPH_ROOT = SOURCE_CACHE / "graphs"


def _sha256(path: Path) -> str:

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_install(source: Path, destination: Path) -> Path:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.resolve() == source:
        return destination
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _protein_fasta(cohort_id: str) -> Path:
    path = (
        METHOD_ROOT
        / "feature"
        / "source"
        / f"{cohort_id}_protein_sequences.fa.gz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"protein sequences of {cohort_id} are absent: {path}")
    return path


def _shared_index() -> Path:
    path = SOURCE_CACHE / "cache_index.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"the shared BiNGO cache index is absent: {path}")
    return path


def _write_provenance(index: Path) -> None:

    parent = _shared_index()
    manifest = SOURCE_CACHE / "cache_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"BiNGO cache manifest is absent: {manifest}")
    payload = {
        "schema_version": "1.0",
        "parent_index": os.path.relpath(parent, start=index.parent),
        "parent_index_sha256": _sha256(parent),
        "cache_manifest": os.path.relpath(manifest, start=index.parent),
        "cache_manifest_sha256": _sha256(manifest),
        "fold_index_sha256": _sha256(index),
    }
    (index.parent / "cache_provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _atomic_install(manifest, index.parent / "cache_manifest.json")


def build(cohort_id: str) -> list[Path]:

    if not GRAPH_ROOT.is_dir():
        raise FileNotFoundError(
            f"the shared BiNGO graph cache is absent: {GRAPH_ROOT}. It is built "
            f"once by the rebuild_command of method.yaml, "
            f"'python cache_esm2_graphs.py --device cuda:0', and features never "
            f"builds it."
        )
    materialize_fasta_folds(
        METHOD_ROOT,
        cohort_id,
        "proteins",
        repo_root=REPO_ROOT,
        fasta_path=_protein_fasta(cohort_id),
    )
    indices = materialize_bingo_cache_indices(
        METHOD_ROOT,
        cohort_id,
        repo_root=REPO_ROOT,
        source_index_path=_shared_index(),
        graph_root=GRAPH_ROOT,
    )
    for index in indices:
        _write_provenance(index)
    return indices
