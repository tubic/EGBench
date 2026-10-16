"""Build the 258-dimensional DeEPsnap fold features, and its node2vec block.
Rebuild the node2vec block from methods/deepsnap: python code/build_features.py build-node2vec
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

METHOD_ROOT = Path(__file__).resolve().parents[1]

NODE2VEC_OUTPUT = "feature/source/network/node2vec_64.parquet"
NODE2VEC_EDGES = "feature/source/network/biogrid_physical_edges.parquet"
NODE2VEC_GENE_UNIVERSE = "feature/source/gene_reference/gene_id_map.parquet"

try:
    from .contracts import (
        COHORT_IDS,
        COMPLEX_DIMENSIONS,
        FORMAL_DIMENSIONS,
        GO_DIMENSIONS,
        NETWORK_DIMENSIONS,
        PFAM_DIMENSIONS,
        SEQUENCE_DIMENSIONS,
        ContractError,
        canonical_gene_keys,
        formal_coverage_for_assignments,
        read_frame,
        require_columns,
        select_context,
        sha256_file,
        stable_json_hash,
        strip_ensembl_version,
        train_labels_only,
        validate_source_manifest,
    )
except ImportError:
    from contracts import (
        COHORT_IDS,
        COMPLEX_DIMENSIONS,
        FORMAL_DIMENSIONS,
        GO_DIMENSIONS,
        NETWORK_DIMENSIONS,
        PFAM_DIMENSIONS,
        SEQUENCE_DIMENSIONS,
        ContractError,
        canonical_gene_keys,
        formal_coverage_for_assignments,
        read_frame,
        require_columns,
        select_context,
        sha256_file,
        stable_json_hash,
        strip_ensembl_version,
        train_labels_only,
        validate_source_manifest,
    )


NODE2VEC_DIMENSIONS = 64
NODE2VEC_WALK_LENGTH = 80
NODE2VEC_WALKS_PER_NODE = 10
NODE2VEC_WINDOW = 10
NODE2VEC_NEGATIVE = 5
NODE2VEC_EPOCHS = 5
NODE2VEC_SEED = 2026


def _node2vec_token_hash(token: str) -> int:
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big") & 0x7FFFFFFF


def _node2vec_graph(edges: pd.DataFrame, universe: pd.Series) -> tuple:
    nodes = tuple(str(key).strip() for key in universe)
    if not all(nodes) or len(set(nodes)) != len(nodes):
        raise ValueError("the node2vec gene universe must be non-empty and free of duplicates")
    nodes = tuple(sorted(nodes))
    pairs: set[tuple[str, str]] = set()
    for left, right in edges.itertuples(index=False, name=None):
        left, right = str(left).strip(), str(right).strip()
        if left != right:
            pairs.add((left, right) if left < right else (right, left))
    outside = {node for pair in pairs for node in pair}.difference(nodes)
    if outside:
        raise ValueError(
            f"{len(outside)} edge endpoints fall outside the gene universe: {sorted(outside)[:5]}"
        )
    neighbours: dict[str, list[str]] = {node: [] for node in nodes}
    for left, right in pairs:
        neighbours[left].append(right)
        neighbours[right].append(left)
    return nodes, {node: tuple(sorted(values)) for node, values in neighbours.items()}


class _Node2VecWalks:

    def __init__(self, nodes: tuple, neighbours: dict) -> None:
        self.nodes = nodes
        self.neighbours = neighbours
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        total = len(self.nodes) * NODE2VEC_WALKS_PER_NODE
        report_every = max(total // 10, 1)
        generator = np.random.Generator(np.random.PCG64(NODE2VEC_SEED))
        walked = 0
        for _ in range(NODE2VEC_WALKS_PER_NODE):
            for start in self.nodes:
                walk = [start]
                while len(walk) < NODE2VEC_WALK_LENGTH:
                    candidates = self.neighbours[walk[-1]]
                    if not candidates:
                        break
                    walk.append(candidates[int(generator.integers(0, len(candidates)))])
                walked += 1
                if walked % report_every == 0:
                    print(
                        f"node2vec pass {self.passes}/{NODE2VEC_EPOCHS + 1}: "
                        f"{walked}/{total} walks",
                        flush=True,
                    )
                yield walk


def build_node2vec() -> Path:
    from gensim.models import Word2Vec

    edges_path = METHOD_ROOT / NODE2VEC_EDGES
    universe_path = METHOD_ROOT / NODE2VEC_GENE_UNIVERSE
    for path in (edges_path, universe_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    nodes, neighbours = _node2vec_graph(
        pd.read_parquet(edges_path, columns=["gene_key_a", "gene_key_b"]),
        pd.read_parquet(universe_path, columns=["gene_key"])["gene_key"],
    )
    print(
        f"node2vec: {len(nodes)} genes, "
        f"{sum(len(value) for value in neighbours.values()) // 2} edges",
        flush=True,
    )
    model = Word2Vec(
        sentences=_Node2VecWalks(nodes, neighbours),
        vector_size=NODE2VEC_DIMENSIONS,
        window=NODE2VEC_WINDOW,
        min_count=0,
        sample=0.0,
        workers=1,
        sg=1,
        hs=0,
        negative=NODE2VEC_NEGATIVE,
        epochs=NODE2VEC_EPOCHS,
        seed=NODE2VEC_SEED,
        sorted_vocab=1,
        hashfxn=_node2vec_token_hash,
    )
    matrix = np.vstack([model.wv[node] for node in nodes])
    if not np.isfinite(matrix).all():
        raise ValueError("node2vec produced a non-finite embedding")
    embedding = pd.DataFrame(
        matrix,
        columns=[f"node2vec_{index:03d}" for index in range(1, NODE2VEC_DIMENSIONS + 1)],
    )
    embedding.insert(0, "gene_key", list(nodes))
    destination = METHOD_ROOT / NODE2VEC_OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    embedding.to_parquet(
        destination,
        index=False,
        engine="pyarrow",
        compression="zstd",
        compression_level=9,
    )
    return destination


def _unique_map(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    ambiguous: set[str] = set()
    for identifier, gene_key in pairs:
        previous = result.get(identifier)
        if previous is not None and previous != gene_key:
            ambiguous.add(identifier)
        else:
            result[identifier] = gene_key
    for identifier in ambiguous:
        result.pop(identifier, None)
    return result


def gene_maps(gene_map: pd.DataFrame) -> dict[str, dict[str, str]]:
    require_columns(
        gene_map,
        ("gene_key", "symbol", "entrez_id", "ensembl_gene_id", "ensembl_protein_id"),
        "gene map",
    )
    frame = gene_map.loc[
        :,
        ["gene_key", "symbol", "entrez_id", "ensembl_gene_id", "ensembl_protein_id"],
    ].copy()
    frame["gene_key"] = canonical_gene_keys(frame["gene_key"], "gene map")
    if frame["gene_key"].duplicated().any():
        raise ContractError("gene map contains duplicate gene keys")
    symbol_pairs: list[tuple[str, str]] = []
    entrez_pairs: list[tuple[str, str]] = []
    gene_pairs: list[tuple[str, str]] = []
    protein_pairs: list[tuple[str, str]] = []
    for row in frame.itertuples(index=False):
        if pd.notna(row.symbol) and str(row.symbol).strip():
            symbol_pairs.append((str(row.symbol).strip().upper(), row.gene_key))
        if pd.notna(row.entrez_id) and str(row.entrez_id).strip():
            entrez_pairs.append((str(row.entrez_id).strip(), row.gene_key))
        if pd.notna(row.ensembl_gene_id) and str(row.ensembl_gene_id).strip():
            gene_pairs.append((strip_ensembl_version(row.ensembl_gene_id), row.gene_key))
        if pd.notna(row.ensembl_protein_id) and str(row.ensembl_protein_id).strip():
            protein_pairs.append((strip_ensembl_version(row.ensembl_protein_id), row.gene_key))
    return {
        "symbol": _unique_map(symbol_pairs),
        "entrez": _unique_map(entrez_pairs),
        "ensembl_gene": _unique_map(gene_pairs),
        "ensembl_protein": _unique_map(protein_pairs),
    }


def parse_goa(
    path: str | Path,
    gene_map: pd.DataFrame,
    *,
    formal: bool = False,
) -> tuple[dict[str, set[str]], dict[str, object]]:
    maps = gene_maps(gene_map)
    annotations: dict[str, set[str]] = defaultdict(set)
    total_rows = 0
    cc_rows = 0
    mapped_rows = 0
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("!"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 15:
                raise ContractError(f"GOA GAF row {line_number} has fewer than 15 fields")
            total_rows += 1
            if fields[8] != "C" or "NOT" in fields[3].split("|"):
                continue
            cc_rows += 1
            candidates = [fields[2], *fields[10].split("|")]
            mapped = {
                maps["symbol"][candidate.strip().upper()]
                for candidate in candidates
                if candidate.strip().upper() in maps["symbol"]
            }
            if len(mapped) != 1:
                continue
            annotations[next(iter(mapped))].add(fields[4].strip())
            mapped_rows += 1
    unique_terms = set().union(*annotations.values()) if annotations else set()
    if formal and (
        total_rows < 100_000
        or cc_rows < 20_000
        or mapped_rows < 10_000
        or len(unique_terms) < 500
    ):
        raise ContractError("GOA input is too small to be the complete human GAF database")
    report = {
        "source": str(path),
        "source_sha256": sha256_file(path),
        "total_annotation_rows": total_rows,
        "cellular_component_rows": cc_rows,
        "mapped_rows": mapped_rows,
        "mapped_genes": len(annotations),
        "mapped_cc_terms": len(unique_terms),
        "mapping": "approved HGNC symbol or unambiguous GOA symbol synonym",
        "qualifier_policy": "exclude NOT",
        "ontology_propagation": "none; use the complete GOA human CC annotations as supplied",
    }
    return dict(annotations), report


def parse_corum(
    path: str | Path,
    gene_map: pd.DataFrame,
    *,
    formal: bool = False,
) -> tuple[dict[str, set[str]], dict[str, object]]:
    maps = gene_maps(gene_map)
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    require_columns(frame, ("complex_id", "organism", "subunits_gene_name"), "CORUM")
    human = frame.loc[frame["organism"].str.contains("Human", case=False, na=False)]
    complexes: dict[str, set[str]] = {}
    mapped_subunits = 0
    for complex_id, raw_members in human[["complex_id", "subunits_gene_name"]].itertuples(
        index=False, name=None
    ):
        members = {
            maps["symbol"][token.strip().upper()]
            for token in str(raw_members).split(";")
            if token.strip().upper() in maps["symbol"]
        }
        if members:
            complexes[str(complex_id)] = members
            mapped_subunits += len(members)
    if formal and (len(frame) < 5_000 or len(human) < 5_000 or len(complexes) < 4_000):
        raise ContractError("CORUM input is too small to be the complete human complex database")
    report = {
        "source": str(path),
        "source_sha256": sha256_file(path),
        "total_rows": len(frame),
        "human_rows": len(human),
        "mapped_complexes": len(complexes),
        "mapped_subunit_assignments": mapped_subunits,
        "mapping": "CORUM subunits_gene_name to unique approved HGNC symbol",
    }
    return complexes, report


def _resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    aliases = {
        "gene": ("Gene stable ID", "ensembl_gene_id"),
        "protein": ("Protein stable ID", "ensembl_peptide_id", "ensembl_protein_id"),
        "pfam": ("Pfam ID", "pfam"),
        "start": ("Pfam start", "pfam_start"),
        "end": ("Pfam end", "pfam_end"),
    }
    resolved: dict[str, str] = {}
    for key, candidates in aliases.items():
        matches = [column for column in candidates if column in frame]
        if len(matches) != 1:
            raise ContractError(f"BioMart Pfam export needs one {key} column from {candidates}")
        resolved[key] = matches[0]
    return resolved


def parse_pfam(
    path: str | Path,
    gene_map: pd.DataFrame,
    *,
    formal: bool = False,
) -> tuple[dict[str, set[tuple[str, int, int]]], dict[str, object]]:
    maps = gene_maps(gene_map)
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    columns = _resolve_columns(frame)
    domains: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    valid_rows = 0
    for protein, pfam_id, raw_start, raw_end in frame[
        [columns["protein"], columns["pfam"], columns["start"], columns["end"]]
    ].itertuples(index=False, name=None):
        gene_key = maps["ensembl_protein"].get(strip_ensembl_version(protein))
        if gene_key is None or not pfam_id.strip():
            continue
        try:
            start = int(raw_start)
            end = int(raw_end)
        except ValueError as exc:
            raise ContractError("BioMart Pfam start/end must be integers") from exc
        if start <= 0 or end < start:
            raise ContractError("BioMart Pfam coordinates are invalid")
        domains[gene_key].add((pfam_id.strip(), start, end))
        valid_rows += 1
    if formal and (len(frame) < 100_000 or valid_rows < 20_000 or len(domains) < 5_000):
        raise ContractError("BioMart input is too small to be the complete human Pfam export")
    report = {
        "source": str(path),
        "source_sha256": sha256_file(path),
        "total_rows": len(frame),
        "canonical_protein_domain_rows": valid_rows,
        "mapped_canonical_proteins": len(domains),
        "mapping": "exact versionless Ensembl peptide ID to the project canonical protein",
        "domain_instance": "unique (Pfam ID, start, end) on the canonical protein",
    }
    return dict(domains), report


def adjacency_from_edges(edges: pd.DataFrame, universe: set[str]) -> dict[str, set[str]]:
    require_columns(edges, ("gene_key_a", "gene_key_b"), "PPI edges")
    adjacency = {gene_key: set() for gene_key in universe}
    for left, right in edges[["gene_key_a", "gene_key_b"]].itertuples(index=False, name=None):
        left, right = str(left), str(right)
        if left == right or left not in universe or right not in universe:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def rank_go_terms(
    annotations: dict[str, set[str]],
    train_labels: pd.DataFrame,
    *,
    top_k: int,
) -> list[tuple[str, int]]:
    positives = train_labels.loc[train_labels["label_binary"].eq(1), "gene_key"]
    counts: dict[str, int] = defaultdict(int)
    for gene_key in positives:
        for term in annotations.get(gene_key, ()):
            counts[term] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) < top_k or ranked[top_k - 1][1] <= 0:
        raise ContractError(f"outer-train positives annotate fewer than {top_k} GO CC terms")
    return ranked[:top_k]


def go_enrichment_features(
    gene_keys: list[str],
    adjacency: dict[str, set[str]],
    annotations: dict[str, set[str]],
    top_terms: list[str],
    universe: set[str],
) -> np.ndarray:
    members: dict[str, set[str]] = {term: set() for term in top_terms}
    for gene_key, terms in annotations.items():
        if gene_key not in universe:
            continue
        for term in terms:
            if term in members:
                members[term].add(gene_key)
    population = len(universe)
    result = np.zeros((len(gene_keys), len(top_terms)), dtype=np.float64)
    for row_index, gene_key in enumerate(gene_keys):
        query = {gene_key, *adjacency.get(gene_key, set())}
        draws = len(query)
        for column_index, term in enumerate(top_terms):
            overlap = len(query.intersection(members[term]))
            if overlap == 0:
                continue
            probability = float(hypergeom.sf(overlap - 1, population, len(members[term]), draws))
            if probability > 0:
                result[row_index, column_index] = -math.log10(probability)
            else:
                log_probability = float(
                    hypergeom.logsf(overlap - 1, population, len(members[term]), draws)
                )
                result[row_index, column_index] = (
                    -log_probability / math.log(10)
                    if math.isfinite(log_probability)
                    else -math.log10(np.finfo(np.float64).tiny)
                )
    return result


def complex_features(
    gene_keys: list[str],
    adjacency: dict[str, set[str]],
    complexes: dict[str, set[str]],
) -> np.ndarray:
    membership_count: dict[str, int] = defaultdict(int)
    for members in complexes.values():
        for gene_key in members:
            membership_count[gene_key] += 1
    result = np.zeros((len(gene_keys), 2), dtype=np.float64)
    for row_index, gene_key in enumerate(gene_keys):
        neighbors = adjacency.get(gene_key, set())
        result[row_index, 0] = membership_count.get(gene_key, 0)
        if neighbors:
            result[row_index, 1] = sum(
                membership_count.get(neighbor, 0) for neighbor in neighbors
            ) / len(neighbors)
    return result


def pfam_features(
    gene_keys: list[str],
    domains: dict[str, set[tuple[str, int, int]]],
) -> np.ndarray:
    proteins_by_type: dict[str, set[str]] = defaultdict(set)
    for gene_key, instances in domains.items():
        for pfam_id, _, _ in instances:
            proteins_by_type[pfam_id].add(gene_key)
    frequency = {pfam_id: len(genes) for pfam_id, genes in proteins_by_type.items()}
    result = np.zeros((len(gene_keys), 3), dtype=np.float64)
    for row_index, gene_key in enumerate(gene_keys):
        instances = sorted(domains.get(gene_key, set()))
        result[row_index, 0] = len(instances)
        result[row_index, 1] = len({pfam_id for pfam_id, _, _ in instances})
        result[row_index, 2] = sum(1.0 / frequency[pfam_id] for pfam_id, _, _ in instances)
    return result


def _fixed_feature_view(path: str | Path, expected_dimensions: int, name: str) -> pd.DataFrame:
    frame = read_frame(path)
    require_columns(frame, ("gene_key",), name)
    columns = [column for column in frame if column != "gene_key"]
    if len(columns) != expected_dimensions:
        raise ContractError(f"{name} must contain exactly {expected_dimensions} features")
    if frame["gene_key"].duplicated().any():
        raise ContractError(f"{name} contains duplicate genes")
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ContractError(f"{name} contains non-finite values")
    return frame


@dataclass(frozen=True)
class ParsedSources:

    universe: set[str]
    adjacency: dict[str, set[str]]
    annotations: dict[str, set[str]]
    complexes: dict[str, set[str]]
    domains: dict[str, set[tuple[str, int, int]]]
    reports: dict[str, object]


def parse_sources(
    *,
    gene_map_path: str | Path,
    edges_path: str | Path,
    goa_path: str | Path,
    corum_path: str | Path,
    pfam_path: str | Path,
    formal: bool = False,
) -> ParsedSources:
    gene_map = read_frame(gene_map_path)
    maps = gene_maps(gene_map)
    universe = set(maps["symbol"].values())
    adjacency = adjacency_from_edges(read_frame(edges_path), universe)
    annotations, goa_report = parse_goa(goa_path, gene_map, formal=formal)
    complexes, corum_report = parse_corum(corum_path, gene_map, formal=formal)
    domains, pfam_report = parse_pfam(pfam_path, gene_map, formal=formal)
    return ParsedSources(
        universe=universe,
        adjacency=adjacency,
        annotations=annotations,
        complexes=complexes,
        domains=domains,
        reports={"goa": goa_report, "corum": corum_report, "pfam": pfam_report},
    )


def assemble_columns(
    gene_keys: list[str],
    sources: ParsedSources,
    top_terms: list[str],
    *,
    sequence_path: str | Path,
    node2vec_path: str | Path,
) -> pd.DataFrame:
    go_values = go_enrichment_features(
        gene_keys, sources.adjacency, sources.annotations, top_terms, sources.universe
    )
    complex_values = complex_features(gene_keys, sources.adjacency, sources.complexes)
    domain_values = pfam_features(gene_keys, sources.domains)
    sequence = _fixed_feature_view(sequence_path, SEQUENCE_DIMENSIONS, "DeepHE sequence view")
    network = _fixed_feature_view(node2vec_path, NETWORK_DIMENSIONS, "node2vec view")
    assembled = pd.DataFrame({"gene_key": gene_keys}).merge(
        sequence, on="gene_key", how="left", validate="one_to_one"
    )
    assembled = assembled.merge(network, on="gene_key", how="left", validate="one_to_one")
    base_columns = [column for column in assembled if column != "gene_key"]
    if assembled[base_columns].isna().any().any():
        raise ContractError("sequence or node2vec input does not cover every DeEPsnap cohort gene")
    go_frame = pd.DataFrame(
        go_values,
        columns=[f"go_enrichment_{index + 1:03d}" for index in range(len(top_terms))],
    )
    assembled = pd.concat([assembled.reset_index(drop=True), go_frame], axis=1)
    assembled["corum_membership_count"] = complex_values[:, 0]
    assembled["corum_neighbor_complex_ratio_sum"] = complex_values[:, 1]
    assembled["pfam_domain_instance_count"] = domain_values[:, 0]
    assembled["pfam_unique_domain_type_count"] = domain_values[:, 1]
    assembled["pfam_sidf"] = domain_values[:, 2]
    return assembled


def build_features(
    *,
    gene_map_path: str | Path,
    assignments_path: str | Path,
    labels_path: str | Path,
    sequence_path: str | Path,
    node2vec_path: str | Path,
    edges_path: str | Path,
    goa_path: str | Path,
    corum_path: str | Path,
    pfam_path: str | Path,
    cohort_id: str,
    split_strategy: str,
    outer_fold: int,
    output_path: str | Path,
    manifest_path: str | Path,
    source_manifest_path: str | Path | None = None,
    top_go: int = GO_DIMENSIONS,
    formal: bool = False,
) -> dict[str, object]:
    if formal and source_manifest_path is None:
        raise ContractError("formal DeEPsnap feature construction requires the acquisition source manifest")
    source_manifest = None
    if source_manifest_path is not None:
        source_manifest = validate_source_manifest(
            source_manifest_path,
            {
                "goa_human": goa_path,
                "corum_human": corum_path,
                "ensembl_biomart_pfam": pfam_path,
            },
        )
    context = select_context(
        read_frame(assignments_path),
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )
    expected_coverage = formal_coverage_for_assignments(assignments_path, cohort_id)
    if formal and (cohort_id not in COHORT_IDS or len(context) != expected_coverage):
        raise ContractError(
            f"formal DeEPsnap requires {expected_coverage} authoritative cohort genes"
        )
    if formal and top_go != GO_DIMENSIONS:
        raise ContractError("formal DeEPsnap requires exactly 100 GO features")
    train_labels = train_labels_only(
        read_frame(labels_path),
        context,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )
    sources = parse_sources(
        gene_map_path=gene_map_path,
        edges_path=edges_path,
        goa_path=goa_path,
        corum_path=corum_path,
        pfam_path=pfam_path,
        formal=formal,
    )
    selection_keys = train_labels["gene_key"].astype(str).tolist()
    selection_values = np.asarray(
        train_labels["label_binary"].to_numpy(), dtype="int64"
    ).reshape(-1)
    ranked_terms = rank_go_terms(
        sources.annotations,
        pd.DataFrame({"gene_key": selection_keys, "label_binary": selection_values}),
        top_k=top_go,
    )
    top_terms = [term for term, _ in ranked_terms]
    assembled = assemble_columns(
        context["gene_key"].tolist(),
        sources,
        top_terms,
        sequence_path=sequence_path,
        node2vec_path=node2vec_path,
    )
    feature_columns = [column for column in assembled if column != "gene_key"]
    expected = SEQUENCE_DIMENSIONS + NETWORK_DIMENSIONS + top_go + COMPLEX_DIMENSIONS + PFAM_DIMENSIONS
    if len(feature_columns) != expected:
        raise AssertionError("DeEPsnap feature assembly produced the wrong dimension")
    if formal and expected != FORMAL_DIMENSIONS:
        raise ContractError("formal DeEPsnap input must contain exactly 258 features")
    if not np.isfinite(assembled[feature_columns].to_numpy(dtype=np.float64)).all():
        raise ContractError("DeEPsnap features contain non-finite values")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    assembled.sort_values("gene_key", kind="mergesort").to_parquet(output, index=False)
    train_label_payload = sorted(
        (str(row.gene_key), int(row.label_binary)) for row in train_labels.itertuples(index=False)
    )
    manifest = {
        "schema_version": "1.0",
        "kind": "deepsnap_fold_features",
        "cohort_id": cohort_id,
        "split_strategy": split_strategy,
        "outer_fold": outer_fold,
        "row_count": len(context),
        "feature_count": len(feature_columns),
        "dimension_contract": {
            "sequence": SEQUENCE_DIMENSIONS,
            "node2vec": NETWORK_DIMENSIONS,
            "go": top_go,
            "corum": COMPLEX_DIMENSIONS,
            "pfam": PFAM_DIMENSIONS,
        },
        "top_go_selection": "GOA human CC terms ranked by count among outer-train positive genes only; ties by GO ID",
        "selected_go_terms": [
            {"rank": index + 1, "go_id": term, "outer_train_positive_count": count}
            for index, (term, count) in enumerate(ranked_terms)
        ],
        "go_enrichment": "one-sided hypergeometric survival probability over the full project protein universe; query is gene plus direct BioGRID neighbors; score=-log10(p)",
        "corum_features": [
            "number of mapped CORUM complexes containing the gene",
            "sum across complexes of the fraction of direct neighbors in that complex",
        ],
        "pfam_features": [
            "canonical-protein Pfam domain instance count",
            "canonical-protein distinct Pfam type count",
            "sum over domain instances of inverse mapped-protein domain frequency",
        ],
        "missing_annotation_policy": "zero only after a complete real database has been parsed and mapped",
        "train_label_hash": stable_json_hash(train_label_payload),
        "source_manifest": source_manifest,
        "source_reports": sources.reports,
        "output": str(output),
    }
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build-node2vec",))
    parser.parse_args()
    print(build_node2vec(), flush=True)


if __name__ == "__main__":
    main()
