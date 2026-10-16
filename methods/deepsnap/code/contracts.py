"""Constants, source URLs and table checks shared by the DeEPsnap feature builder and trainer.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from egbench.contracts import COHORT_IDS


GOA_HUMAN_URL = "https://current.geneontology.org/annotations/goa_human.gaf.gz"
CORUM_API_ROOT = "https://mips.helmholtz-muenchen.de/fastapi-corum"
CORUM_RELEASE_URL = f"{CORUM_API_ROOT}/public/releases/current"
CORUM_HUMAN_URL = (
    f"{CORUM_API_ROOT}/public/file/download_current_file?file_id=human&file_format=txt"
)
ENSEMBL_BIOMART_URL = "https://www.ensembl.org/biomart/martservice"
BIOMART_QUERY = (
    '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="0" '
    'count="" datasetConfigVersion="0.6"><Dataset name="hsapiens_gene_ensembl" '
    'interface="default"><Attribute name="ensembl_gene_id"/>'
    '<Attribute name="ensembl_peptide_id"/><Attribute name="pfam"/>'
    '<Attribute name="pfam_start"/><Attribute name="pfam_end"/></Dataset></Query>'
)

SEQUENCE_DIMENSIONS = 89
NETWORK_DIMENSIONS = 64
GO_DIMENSIONS = 100
COMPLEX_DIMENSIONS = 2
PFAM_DIMENSIONS = 3
FORMAL_DIMENSIONS = 258
FORMAL_EPOCHS = 50
SNAPSHOT_CYCLE = 5
FORMAL_SNAPSHOTS = 10

_HGNC_RE = re.compile(r"^HGNC:[1-9][0-9]*$")


class ContractError(ValueError):
    pass


def formal_coverage_for_assignments(
    assignments_path: str | Path,
    cohort_id: str,
) -> int:

    source = Path(assignments_path)
    if not source.is_file():
        dataset_root = source.resolve().parents[2]
        audit_path = dataset_root / "processed" / "cohorts" / "cohort_audit.parquet"
        if audit_path.is_file():
            audit = pd.read_parquet(audit_path)
            selected = audit.loc[audit["cohort_id"].astype(str).eq(cohort_id), "n_genes"]
            if len(selected) == 1 and int(selected.iloc[0]) > 0:
                return int(selected.iloc[0])
    assignments = read_frame(source)
    require_columns(assignments, ("gene_key", "cohort_id"), "split assignments")
    selected = assignments.loc[assignments["cohort_id"].astype(str).eq(cohort_id), "gene_key"]
    if selected.empty:
        raise ContractError(f"assignment table has no cohort {cohort_id!r}")
    keys = canonical_gene_keys(selected, "split assignments")
    coverage = int(keys.nunique())
    if coverage < 1:
        raise ContractError(f"cohort {cohort_id!r} is empty")
    return coverage


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def strip_ensembl_version(value: object) -> str:
    return str(value).strip().split(".", 1)[0]


def read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    separator = "\t" if source.suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(source, sep=separator)


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ContractError(f"{name} is missing required columns: {missing}")


def canonical_gene_keys(values: pd.Series, name: str) -> pd.Series:
    result = values.astype(str)
    invalid = result.loc[~result.map(lambda value: bool(_HGNC_RE.fullmatch(value)))]
    if not invalid.empty:
        raise ContractError(f"{name} contains invalid gene_key {invalid.iloc[0]!r}")
    return result


def select_context(
    assignments: pd.DataFrame,
    *,
    cohort_id: str,
    split_strategy: str,
    outer_fold: int,
) -> pd.DataFrame:
    require_columns(
        assignments,
        ("gene_key", "cohort_id", "split_strategy", "outer_fold", "role"),
        "split assignments",
    )
    result = assignments.loc[
        assignments["cohort_id"].eq(cohort_id)
        & assignments["split_strategy"].eq(split_strategy)
        & assignments["outer_fold"].eq(outer_fold),
        ["gene_key", "role"],
    ].copy()
    if result.empty:
        raise ContractError("requested split context is empty")
    result["gene_key"] = canonical_gene_keys(result["gene_key"], "split assignments")
    if result["gene_key"].duplicated().any():
        raise ContractError("split context has duplicate gene assignments")
    roles = set(result["role"].astype(str))
    if roles != {"train", "validation", "test"}:
        raise ContractError(f"split context roles are invalid: {roles}")
    return result.sort_values("gene_key", kind="mergesort").reset_index(drop=True)


def train_labels_only(
    labels: pd.DataFrame,
    context: pd.DataFrame,
    *,
    cohort_id: str | None = None,
    split_strategy: str | None = None,
    outer_fold: int | None = None,
) -> pd.DataFrame:

    contextual_columns = {
        "gene_key",
        "cohort_id",
        "split_strategy",
        "outer_fold",
        "role",
        "label",
    }
    if contextual_columns.issubset(labels.columns):
        if cohort_id is None or split_strategy is None or outer_fold is None:
            raise ContractError("contextual training labels require an explicit split context")
        view = labels.loc[
            labels["cohort_id"].eq(cohort_id)
            & labels["split_strategy"].eq(split_strategy)
            & labels["outer_fold"].eq(outer_fold)
            & labels["role"].eq("train"),
            ["gene_key", "label"],
        ].rename(columns={"label": "label_binary"})
    else:
        require_columns(labels, ("gene_key", "label_binary"), "labels")
        view = labels.loc[:, ["gene_key", "label_binary"]].copy()
    view["gene_key"] = canonical_gene_keys(view["gene_key"], "labels")
    if view["gene_key"].duplicated().any():
        raise ContractError("labels contain duplicate gene keys")
    train = context.loc[context["role"].eq("train"), ["gene_key"]]
    result = train.merge(view, on="gene_key", how="left", validate="one_to_one")
    if result["label_binary"].isna().any() or not result["label_binary"].isin([0, 1]).all():
        raise ContractError("a train-role gene has no label, or its label is not binary")
    if result["label_binary"].nunique() != 2:
        raise ContractError("the train-role labels must contain both classes")
    return result


def validate_source_manifest(
    manifest_path: str | Path,
    sources: dict[str, str | Path],
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = manifest.get("sources")
    if not isinstance(records, dict):
        raise ContractError("source manifest has no source records")
    for key, path in sources.items():
        record = records.get(key)
        if not isinstance(record, dict) or "sha256" not in record:
            raise ContractError(f"source manifest has no checksum for {key}")
        if sha256_file(path) != record["sha256"]:
            raise ContractError(f"source checksum mismatch for {key}")
    return manifest
