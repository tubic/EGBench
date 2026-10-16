"""Constants, source URLs and table checks shared by the EPGAT feature builder and trainer.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


COMPARA_RELEASE = 115
COMPARA_URL = (
    "https://ftp.ensembl.org/pub/release-115/tsv/ensembl-compara/"
    "homologies/homo_sapiens/Compara.115.protein_default.homologies.tsv.gz"
)
COMPARTMENTS_URL = (
    "https://download.jensenlab.org/human_compartment_integrated_full.tsv"
)

HUMAN_COMPARTMENTS: tuple[tuple[str, str], ...] = (
    ("cytoskeleton", "GO:0005856"),
    ("cytosol", "GO:0005829"),
    ("endoplasmic_reticulum", "GO:0005783"),
    ("endosome", "GO:0005768"),
    ("extracellular_space", "GO:0005576"),
    ("golgi_apparatus", "GO:0005794"),
    ("lysosome", "GO:0005764"),
    ("mitochondrion", "GO:0005739"),
    ("nucleus", "GO:0005634"),
    ("peroxisome", "GO:0005777"),
    ("plasma_membrane", "GO:0005886"),
)

FORMAL_COVERAGE = 5_311
FORMAL_SELECTED_DIMENSIONS = 150
MAX_EPOCHS = 1_000
EARLY_STOPPING_PATIENCE = 100

_HGNC_RE = re.compile(r"^HGNC:[1-9][0-9]*$")


class ContractError(ValueError):
    pass


def formal_coverage_for_assignments(
    assignments_path: str | Path,
    cohort_id: str,
) -> int:

    source = Path(assignments_path).resolve()
    if source.is_file():
        assignments = pd.read_parquet(source)
        if {"cohort_id", "gene_key"}.issubset(assignments.columns):
            contextual = assignments.loc[
                assignments["cohort_id"].astype(str).eq(str(cohort_id)), "gene_key"
            ]
            if not contextual.empty:
                coverage = int(contextual.astype(str).nunique())
                if coverage < 1:
                    raise ContractError(f"split assignments has no genes for {cohort_id}")
                return coverage
    dataset_root = source.parents[2]
    profile_path = dataset_root / "config" / "dataset_profile.yaml"
    if not profile_path.is_file():
        return FORMAL_COVERAGE
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or not isinstance(profile.get("dataset_id"), str):
        raise ContractError("dataset profile has no valid dataset_id")
    cohort_audit_path = dataset_root / "processed" / "cohorts" / "cohort_audit.parquet"
    if not cohort_audit_path.is_file():
        raise ContractError("profile cohort audit is missing")
    audit = pd.read_parquet(cohort_audit_path)
    selected = audit.loc[audit["cohort_id"].astype(str).eq(cohort_id), "n_genes"]
    if len(selected) != 1:
        raise ContractError(f"profile cohort audit has no unique row for {cohort_id}")
    coverage = int(selected.iloc[0])
    if coverage < 1:
        raise ContractError(f"profile cohort {cohort_id} is empty")
    return coverage


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def strip_ensembl_version(value: object) -> str:
    return str(value).strip().split(".", 1)[0]


def read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    separator = "\t" if source.suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(source, sep=separator)


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ContractError(f"{name} is missing required columns: {missing}")


def canonical_gene_keys(values: pd.Series, name: str) -> pd.Series:
    result = values.astype(str)
    invalid = result.loc[~result.map(lambda value: bool(_HGNC_RE.fullmatch(value)))]
    if not invalid.empty:
        raise ContractError(f"{name} contains a non-canonical gene_key: {invalid.iloc[0]!r}")
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
    selected = assignments.loc[
        assignments["cohort_id"].eq(cohort_id)
        & assignments["split_strategy"].eq(split_strategy)
        & assignments["outer_fold"].eq(outer_fold),
        ["gene_key", "role"],
    ].copy()
    if selected.empty:
        raise ContractError("the requested split context has no assignments")
    selected["gene_key"] = canonical_gene_keys(selected["gene_key"], "assignments")
    if selected["gene_key"].duplicated().any():
        raise ContractError("split context contains duplicate gene assignments")
    observed_roles = set(selected["role"].astype(str))
    if observed_roles != {"train", "validation", "test"}:
        raise ContractError(f"split roles must be train/validation/test, found {observed_roles}")
    return selected.sort_values("gene_key", kind="mergesort").reset_index(drop=True)


def _labels_for_role(
    labels: pd.DataFrame,
    context: pd.DataFrame,
    role: str,
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
        label_view = labels.loc[
            labels["cohort_id"].eq(cohort_id)
            & labels["split_strategy"].eq(split_strategy)
            & labels["outer_fold"].eq(outer_fold)
            & labels["role"].eq(role),
            ["gene_key", "label"],
        ].rename(columns={"label": "label_binary"})
    else:
        require_columns(labels, ("gene_key", "label_binary"), "labels")
        label_view = labels.loc[:, ["gene_key", "label_binary"]].copy()
    label_view["gene_key"] = canonical_gene_keys(label_view["gene_key"], "labels")
    if label_view["gene_key"].duplicated().any():
        raise ContractError("labels contain duplicate gene_key values")
    role_keys = context.loc[context["role"].eq(role), ["gene_key"]]
    result = role_keys.merge(label_view, on="gene_key", how="left", validate="one_to_one")
    if result["label_binary"].isna().any():
        raise ContractError(f"one or more {role}-role genes have no label")
    values = result["label_binary"].to_numpy()
    if not pd.Series(values).isin([0, 1]).all():
        raise ContractError(f"the {role}-role labels must be binary")
    if len(set(int(value) for value in values)) != 2:
        raise ContractError(f"the {role}-role labels must contain both classes")
    return result


def train_labels_only(
    labels: pd.DataFrame,
    context: pd.DataFrame,
    *,
    cohort_id: str | None = None,
    split_strategy: str | None = None,
    outer_fold: int | None = None,
) -> pd.DataFrame:

    return _labels_for_role(
        labels,
        context,
        "train",
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )


def validation_labels_only(
    labels: pd.DataFrame,
    context: pd.DataFrame,
    *,
    cohort_id: str | None = None,
    split_strategy: str | None = None,
    outer_fold: int | None = None,
) -> pd.DataFrame:

    return _labels_for_role(
        labels,
        context,
        "validation",
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )


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
