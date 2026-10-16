"""Fit the fold's fastText representation and write the eDNN-EG fold view.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(ROOT / "code" / "src"))
from egbench.contracts import COHORT_IDS, SEED
from egbench.data import (
    feature_directory,
    folds_directory,
)
from ednn_eg.fasttext_backend import OfficialFastTextBackend


FASTTEXT_THREAD = 1
FEATURE_WORK = "feature_fasttext"


def read_source_cds(path: Path) -> dict[str, str]:

    return _read_fasta(path)


def _read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header: str | None = None
        sequence: list[str] = []
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(sequence)
                header = line[1:].split("|", 1)[0].split()[0]
                if not header or header in records:
                    raise ValueError(f"invalid or duplicate FASTA gene_key in {path}")
                sequence = []
            elif header is None:
                raise ValueError(f"sequence before FASTA header in {path}")
            else:
                sequence.append(line)
        if header is not None:
            records[header] = "".join(sequence)
    if not records:
        raise ValueError(f"empty FASTA: {path}")
    return records


def _write_selected_cds(path: Path, keys: list[str], sequences: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for key in keys:
            handle.write(f">{key}\n")
            sequence = sequences[key]
            for offset in range(0, len(sequence), 60):
                handle.write(sequence[offset : offset + 60] + "\n")


def _paths(
    cohort_id: str, fold: int, split_strategy: str
) -> tuple[Path, Path, Path, Path]:
    if cohort_id not in COHORT_IDS or fold not in range(5):
        raise ValueError(f"unsupported cohort_id {cohort_id!r} or fold {fold}")
    folds = folds_directory(
        cohort_id, split_strategy=split_strategy, root=REPO_ROOT
    )
    output = feature_directory(
        ROOT, cohort_id, split_strategy=split_strategy, outer_fold=fold
    )
    return (
        ROOT / "feature" / "source" / f"{cohort_id}_cds_sequences.fa.gz",
        folds / "split_assignments.parquet",
        folds / "training_labels" / f"fold_{fold}.parquet",
        output,
    )


def _context(
    assignments: pd.DataFrame, cohort_id: str, fold: int, split_strategy: str
) -> pd.DataFrame:
    frame = assignments.loc[
        assignments["cohort_id"].astype(str).eq(cohort_id)
        & assignments["split_strategy"].astype(str).eq(split_strategy)
        & pd.to_numeric(assignments["outer_fold"], errors="raise").eq(fold),
        ["gene_key", "role"],
    ].copy()
    if len(frame) == 0 or frame["gene_key"].duplicated().any():
        raise ValueError("the split context is empty or has duplicate gene_key values")
    if set(frame["role"].astype(str)) != {"train", "validation", "test"}:
        raise ValueError("the split context requires train, validation and test")
    return frame.sort_values("gene_key", kind="mergesort").reset_index(drop=True)


def build_fold(
    cohort_id: str,
    fold: int,
    split_strategy: str,
    *,
    backend_class=OfficialFastTextBackend,
) -> Path:

    fasta_path, assignments_path, labels_path, output = _paths(
        cohort_id, fold, split_strategy
    )
    sequences = _read_fasta(fasta_path)
    context = _context(pd.read_parquet(assignments_path), cohort_id, fold, split_strategy)
    labels = pd.read_parquet(labels_path)
    train = context.loc[context["role"].eq("train"), ["gene_key"]].merge(
        labels.loc[
            labels["cohort_id"].astype(str).eq(cohort_id)
            & labels["split_strategy"].astype(str).eq(split_strategy)
            & pd.to_numeric(labels["outer_fold"], errors="raise").eq(fold)
            & labels["role"].astype(str).eq("train"),
            ["gene_key", "label"],
        ],
        on="gene_key",
        how="left",
        validate="one_to_one",
    )
    if train["label"].isna().any() or set(train["label"].astype(int)) != {0, 1}:
        raise ValueError("outer-train labels must be complete and binary")
    keys = context["gene_key"].astype(str).tolist()
    missing = sorted(set(keys).difference(sequences))
    if missing:
        raise ValueError(f"CDS source lacks {len(missing)} authority genes: {missing[:5]}")
    train_keys = train["gene_key"].astype(str).tolist()
    backend = backend_class(
        seed=SEED,
        thread=FASTTEXT_THREAD,
        work_dir=(
            ROOT
            / "work"
            / cohort_id
            / split_strategy
            / f"fold_{fold}"
            / FEATURE_WORK
        ),
    )
    backend.fit(train_keys, [sequences[key] for key in train_keys], train["label"].to_numpy())
    values = backend.transform(keys, [sequences[key] for key in keys])
    if values.shape != (len(keys), 100) or not np.isfinite(values).all():
        raise ValueError("official fastText did not produce a finite 100-dimensional matrix")
    output.mkdir(parents=True, exist_ok=True)
    feature_columns = [f"fasttext_{index:03d}" for index in range(1, 101)]
    features = pd.DataFrame(values, columns=feature_columns)
    features.insert(0, "gene_key", keys)
    features_path = output / "features.parquet"
    selected_cds_path = output / "selected_cds.fa"
    features.to_parquet(features_path, index=False)
    _write_selected_cds(selected_cds_path, keys, sequences)
    backend_metadata = dict(backend.metadata)
    manifest = {
        "schema_version": "1.0",
        "kind": "ednn_eg_fold_features",
        "method_id": "ednn_eg",
        "dataset": cohort_id,
        "outer_fold": fold,
        "cohort_id": cohort_id,
        "split_strategy": split_strategy,
        "row_count": len(keys),
        "feature_dimensions": 100,
        "fit_role": "train",
        "backend": backend_metadata,
        "note": "Outer-train-fitted fastText representation for this fold.",
    }
    (output / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return features_path
