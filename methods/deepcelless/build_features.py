#!/usr/bin/env python3
"""Write the per-fold protein FASTA views of DeepCellEss.
Imported by run.py; not run directly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]

_spec = importlib.util.spec_from_file_location(
    "family_feature_layout", REPO_ROOT / "methods" / "family_feature_layout.py"
)
family_feature_layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(family_feature_layout)
materialize_fasta_folds = family_feature_layout.materialize_fasta_folds
read_authority_gene_sets = family_feature_layout.read_authority_gene_sets


METHOD_ID = "deepcelless"


def _source_path(cohort_id: str) -> Path:
    return ROOT / "feature" / "source" / "selected_proteins" / f"{cohort_id}.fa.gz"


def _validate_support_list(cohort_id: str) -> None:
    path = ROOT / "support_gene_list.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is part of this method and must be present"
        )
    import pandas as pd

    frame = pd.read_csv(path)
    if list(frame.columns) != ["dataset", "gene_key", "gene_symbol"]:
        raise ValueError("support_gene_list.csv has the wrong schema")
    required = set(read_authority_gene_sets(REPO_ROOT, cohort_id)[0])
    observed = frame["gene_key"].astype(str)
    if not required.issubset(set(observed)):
        raise ValueError(f"support_gene_list.csv does not cover {cohort_id} authority")


def build(cohort_id: str) -> dict[str, object]:
    _validate_support_list(cohort_id)
    paths = materialize_fasta_folds(
        ROOT,
        cohort_id,
        "proteins",
        repo_root=REPO_ROOT,
        fasta_path=_source_path(cohort_id),
    )
    outputs = {cohort_id: [path.relative_to(ROOT).as_posix() for path in paths]}
    return {"method_id": METHOD_ID, "outputs": outputs}
