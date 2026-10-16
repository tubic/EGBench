"""Shared file helpers for the XGEP source rebuild."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tarfile
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd


BIOGRID_REQUIRED_COLUMNS = (
    "Entrez Gene Interactor A",
    "Entrez Gene Interactor B",
    "Organism ID Interactor A",
    "Organism ID Interactor B",
    "Experimental System",
    "Experimental System Type",
    "Publication Source",
)
_TCGA_SAMPLE = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-(?P<sample_type>[0-9]{2})[A-Z]-")
_TCGA_PRIMARY_TUMOR_CODES = frozenset({"01", "03", "09"})
_EXPRESSION_MEMBER = re.compile(r"(?:^|/)[A-Z0-9]+__geneExpT\.txt$")


@dataclass(frozen=True)
class BioGridResult:
    edges: pd.DataFrame
    node_features: pd.DataFrame
    audit: dict[str, object]


@dataclass(frozen=True)
class ExpressionResult:
    matrix: pd.DataFrame
    samples: pd.DataFrame
    audit: dict[str, object]


def _canonical_gene_key(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("HGNC:") and text[5:].isdigit():
        return f"HGNC:{int(text[5:])}"
    return None


def build_biogrid_physical(
    archive: Path,
    entrez_to_hgnc: Mapping[str, str],
    gene_universe: Sequence[str],
    *,
    chunksize: int = 100_000,
) -> BioGridResult:

    mapping = {
        str(entrez).strip(): canonical
        for entrez, hgnc in entrez_to_hgnc.items()
        if (canonical := _canonical_gene_key(hgnc)) is not None
    }
    universe = sorted(set(gene_universe))
    universe_set = set(universe)
    if any(_canonical_gene_key(item) != item for item in universe):
        raise ValueError("gene_universe contains a non-canonical HGNC ID")

    edge_records: dict[tuple[str, str], dict[str, object]] = {}
    total_rows = human_physical_rows = mapped_rows = universe_filtered_rows = self_loops = 0
    with zipfile.ZipFile(archive) as bundle:
        candidates = sorted(
            name
            for name in bundle.namelist()
            if "Homo_sapiens" in name and name.endswith(".tab3.txt")
        )
        if len(candidates) != 1:
            raise ValueError(f"expected one Homo sapiens TAB3 file, found {candidates!r}")
        with bundle.open(candidates[0]) as handle:
            chunks = pd.read_csv(
                handle,
                sep="\t",
                dtype=str,
                usecols=list(BIOGRID_REQUIRED_COLUMNS),
                chunksize=chunksize,
                keep_default_na=False,
            )
            for chunk in chunks:
                total_rows += len(chunk)
                filtered = chunk.loc[
                    (chunk["Organism ID Interactor A"] == "9606")
                    & (chunk["Organism ID Interactor B"] == "9606")
                    & (chunk["Experimental System Type"].str.lower() == "physical")
                ]
                human_physical_rows += len(filtered)
                for row in filtered.itertuples(index=False, name=None):
                    values = dict(zip(BIOGRID_REQUIRED_COLUMNS, row, strict=True))
                    left = mapping.get(values["Entrez Gene Interactor A"])
                    right = mapping.get(values["Entrez Gene Interactor B"])
                    if left is None or right is None:
                        continue
                    mapped_rows += 1
                    if left not in universe_set or right not in universe_set:
                        universe_filtered_rows += 1
                        continue
                    if left == right:
                        self_loops += 1
                        continue
                    a, b = sorted((left, right))
                    record = edge_records.setdefault(
                        (a, b),
                        {"systems": set(), "pubmed": set(), "source_record_count": 0},
                    )
                    system = values["Experimental System"].strip()
                    publication_source = values["Publication Source"].strip()
                    pubmed = (
                        publication_source.split(":", 1)[1]
                        if publication_source.upper().startswith("PUBMED:")
                        else publication_source
                    )
                    if system:
                        record["systems"].add(system)
                    if pubmed:
                        record["pubmed"].add(pubmed)
                    record["source_record_count"] = int(record["source_record_count"]) + 1

    edges = pd.DataFrame.from_records(
        [
            {
                "gene_key_a": a,
                "gene_key_b": b,
                "experimental_systems": "|".join(sorted(record["systems"])),
                "pubmed_ids": "|".join(sorted(record["pubmed"])),
                "source_record_count": int(record["source_record_count"]),
            }
            for (a, b), record in sorted(edge_records.items())
        ],
        columns=(
            "gene_key_a",
            "gene_key_b",
            "experimental_systems",
            "pubmed_ids",
            "source_record_count",
        ),
    )
    graph = nx.Graph()
    graph.add_nodes_from(universe)
    graph.add_edges_from(edges[["gene_key_a", "gene_key_b"]].itertuples(index=False, name=None))
    component_size: dict[str, int] = {}
    for component in nx.connected_components(graph):
        size = len(component)
        component_size.update({gene: size for gene in component})
    node_features = pd.DataFrame(
        [
            {
                "gene_key": gene,
                "degree": int(graph.degree(gene)),
                "log1p_degree": math.log1p(graph.degree(gene)),
                "component_size": component_size[gene],
                "is_isolated": graph.degree(gene) == 0,
            }
            for gene in universe
        ]
    )
    audit: dict[str, object] = {
        "total_source_rows": total_rows,
        "human_physical_rows": human_physical_rows,
        "mapped_rows_before_universe_filter": mapped_rows,
        "rows_removed_by_universe_filter": universe_filtered_rows,
        "mapped_rows_before_deduplication": mapped_rows - universe_filtered_rows,
        "self_loops_removed": self_loops,
        "unique_undirected_edges": len(edges),
        "universe_nodes": len(universe),
        "isolated_nodes": int(node_features["is_isolated"].sum()),
    }
    return BioGridResult(edges=edges, node_features=node_features, audit=audit)


def _sample_key(sample_id: str) -> str:
    return sample_id[:16]


def _read_tumor_member(archive: Path) -> tuple[pd.DataFrame, str]:
    with tarfile.open(archive, "r:*") as bundle:
        members = sorted(
            member
            for member in bundle.getmembers()
            if member.isfile() and _EXPRESSION_MEMBER.search(member.name)
        )
        if len(members) != 1:
            raise ValueError(
                f"expected one tumor expression member in {archive.name}, found {len(members)}"
            )
        extracted = bundle.extractfile(members[0])
        if extracted is None:
            raise ValueError(f"could not read {members[0].name}")
        frame = pd.read_csv(extracted, sep="\t", dtype={"GeneSymbol": str, "EntrezID": str})
        return frame, members[0].name


def build_tcga_expression(
    archives: Iterable[Path],
    entrez_to_hgnc: Mapping[str, str],
) -> ExpressionResult:

    mapping = {
        str(entrez).strip(): canonical
        for entrez, hgnc in entrez_to_hgnc.items()
        if (canonical := _canonical_gene_key(hgnc)) is not None
    }
    by_gene: list[pd.DataFrame] = []
    sample_rows: list[dict[str, str]] = []
    duplicate_aliquots: dict[str, list[str]] = defaultdict(list)
    member_names: list[str] = []
    excluded_samples_by_archive: dict[str, list[str]] = {}

    for archive in sorted(archives, key=lambda path: path.name):
        frame, member_name = _read_tumor_member(archive)
        member_names.append(member_name)
        required = {"GeneSymbol", "EntrezID"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{archive.name} lacks GeneSymbol/EntrezID")
        sample_columns = [str(column) for column in frame.columns if column not in required]
        sample_types: dict[str, str] = {}
        malformed: list[str] = []
        for sample in sample_columns:
            match = _TCGA_SAMPLE.match(sample)
            if match is None:
                malformed.append(sample)
            else:
                sample_types[sample] = match.group("sample_type")
        if malformed:
            raise ValueError(
                f"{archive.name} contains malformed TCGA sample IDs: {malformed[:3]!r}"
            )
        primary_columns = [
            sample for sample in sample_columns if sample_types[sample] in _TCGA_PRIMARY_TUMOR_CODES
        ]
        excluded = sorted(set(sample_columns) - set(primary_columns))
        excluded_samples_by_archive[archive.name] = excluded
        if not primary_columns:
            raise ValueError(f"{archive.name} contains no primary-tumor samples")

        keep_by_sample: dict[str, str] = {}
        for sample in sorted(primary_columns):
            key = _sample_key(sample)
            duplicate_aliquots[key].append(sample)
            keep_by_sample.setdefault(key, sample)
        kept = sorted(keep_by_sample.values())
        cancer_type = archive.name.removesuffix("rnaSeq.tar").upper()
        sample_rows.extend(
            {
                "sample_id": sample,
                "tcga_sample_key": _sample_key(sample),
                "cancer_type": cancer_type,
                "sample_type": "primary_tumor",
                "source_archive": archive.name,
            }
            for sample in kept
        )

        numeric = frame[["EntrezID", *kept]].copy()
        numeric["gene_key"] = numeric["EntrezID"].map(mapping)
        numeric = numeric.dropna(subset=["gene_key"]).drop(columns="EntrezID")
        for column in kept:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
        numeric = numeric.groupby("gene_key", sort=True, as_index=True).mean()
        by_gene.append(numeric)

    if not by_gene:
        raise ValueError("no TCGA tumor expression archives were supplied")
    matrix = pd.concat(by_gene, axis=1, join="outer", verify_integrity=True).sort_index()
    matrix.index.name = "gene_key"
    matrix = matrix.reset_index()
    samples = pd.DataFrame.from_records(sample_rows).sort_values("sample_id").reset_index(drop=True)
    if samples["sample_id"].duplicated().any():
        duplicates = samples.loc[samples["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"sample IDs occur in multiple archives: {duplicates[:3]!r}")
    dropped = {
        key: sorted(values)[1:] for key, values in duplicate_aliquots.items() if len(values) > 1
    }
    values = matrix.drop(columns="gene_key").to_numpy(dtype=float)
    finite_or_nan = np.isfinite(values) | np.isnan(values)
    if not finite_or_nan.all() or np.nanmin(values) < 0:
        raise ValueError("TCGA expression matrix contains invalid or negative values")
    audit = {
        "expression_unit": "RSEM normalized count (upper quartile)",
        "log_transformed": False,
        "orientation": "gene_by_sample",
        "source_members": member_names,
        "genes": len(matrix),
        "samples": len(samples),
        "cancer_type_counts": {
            key: int(value)
            for key, value in samples.groupby("cancer_type").size().sort_index().items()
        },
        "duplicate_aliquots_dropped": dropped,
        "accepted_tcga_sample_type_codes": sorted(_TCGA_PRIMARY_TUMOR_CODES),
        "non_primary_samples_excluded": excluded_samples_by_archive,
    }
    return ExpressionResult(matrix=matrix, samples=samples, audit=audit)


def write_frame_atomic(
    frame: pd.DataFrame,
    path: Path,
    *,
    sort_by: Sequence[str],
    compression_level: int = 9,
) -> str:

    missing = set(sort_by) - set(frame.columns)
    if missing:
        raise ValueError(f"sort columns are absent: {sorted(missing)!r}")
    ordered = frame.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        ordered.to_parquet(
            temporary,
            index=False,
            engine="pyarrow",
            compression="zstd",
            compression_level=compression_level,
        )
        with temporary.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        os.replace(temporary, path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(payload: Mapping[str, object], path: Path) -> str:
    serialized = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        temporary = Path(temporary_name)
        os.replace(temporary, path)
        return hashlib.sha256(serialized).hexdigest()
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
