"""Build the EPGAT static features and the fold features with train-role feature selection.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import gzip
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from egbench.perturbation import perturb_edges

try:
    from .contracts import (
        COMPARA_RELEASE,
        FORMAL_SELECTED_DIMENSIONS,
        HUMAN_COMPARTMENTS,
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
        COMPARA_RELEASE,
        FORMAL_SELECTED_DIMENSIONS,
        HUMAN_COMPARTMENTS,
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


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _gene_maps(gene_map: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    require_columns(
        gene_map,
        ("gene_key", "symbol", "ensembl_gene_id", "ensembl_protein_id"),
        "gene map",
    )
    frame = gene_map.loc[
        :,
        ["gene_key", "symbol", "ensembl_gene_id", "ensembl_protein_id"],
    ].copy()
    frame["gene_key"] = canonical_gene_keys(frame["gene_key"], "gene map")
    if frame["gene_key"].duplicated().any():
        raise ContractError("gene map contains duplicate gene_key values")

    def unique_mapping(column: str, *, upper: bool = False) -> dict[str, str]:
        values: dict[str, str] = {}
        ambiguous: set[str] = set()
        for key, raw in frame[["gene_key", column]].itertuples(index=False, name=None):
            if pd.isna(raw) or not str(raw).strip():
                continue
            identifier = strip_ensembl_version(raw) if column.startswith("ensembl_") else str(raw).strip()
            identifier = identifier.upper() if upper else identifier
            previous = values.get(identifier)
            if previous is not None and previous != key:
                ambiguous.add(identifier)
            else:
                values[identifier] = key
        for identifier in ambiguous:
            values.pop(identifier, None)
        return values

    return (
        unique_mapping("ensembl_gene_id"),
        unique_mapping("ensembl_protein_id"),
        unique_mapping("symbol", upper=True),
    )


def parse_compara(
    path: str | Path,
    gene_map: pd.DataFrame,
    *,
    formal: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    gene_by_ensembl, _, _ = _gene_maps(gene_map)
    gene_species: dict[str, set[str]] = defaultdict(set)
    observed_species: set[str] = set()
    total_rows = 0
    accepted_rows = 0
    required = {
        "gene_stable_id",
        "species",
        "homology_type",
        "homology_species",
    }
    for chunk in pd.read_csv(path, sep="\t", compression="infer", chunksize=250_000, dtype=str):
        missing = required.difference(chunk.columns)
        if missing:
            raise ContractError(f"Compara TSV is missing columns: {sorted(missing)}")
        total_rows += len(chunk)
        mask = (
            chunk["species"].eq("homo_sapiens")
            & chunk["homology_species"].ne("homo_sapiens")
            & chunk["homology_type"].str.contains("ortholog", case=False, na=False)
        )
        selected = chunk.loc[mask, ["gene_stable_id", "homology_species"]]
        accepted_rows += len(selected)
        for source_gene, species in selected.itertuples(index=False, name=None):
            species_name = str(species).strip()
            if not species_name:
                continue
            observed_species.add(species_name)
            gene_key = gene_by_ensembl.get(strip_ensembl_version(source_gene))
            if gene_key is not None:
                gene_species[gene_key].add(species_name)
    if formal and (total_rows < 1_000_000 or accepted_rows < 100_000 or len(observed_species) < 100):
        raise ContractError("Compara input is too small to be the complete release-115 human protein homology export")
    species_order = sorted(observed_species)
    gene_keys = sorted(gene_map["gene_key"].astype(str).unique())
    matrix = np.zeros((len(gene_keys), len(species_order)), dtype=np.uint8)
    species_index = {species: index for index, species in enumerate(species_order)}
    for row_index, gene_key in enumerate(gene_keys):
        for species in gene_species.get(gene_key, ()):
            matrix[row_index, species_index[species]] = 1
    columns = [f"ortholog__{_slug(species)}" for species in species_order]
    frame = pd.DataFrame(matrix, columns=columns)
    frame.insert(0, "gene_key", gene_keys)
    frame["orthology_missing"] = (matrix.sum(axis=1) == 0).astype(np.uint8)
    report = {
        "source": str(path),
        "source_sha256": sha256_file(path),
        "ensembl_compara_release": COMPARA_RELEASE,
        "total_rows": total_rows,
        "accepted_orthology_rows": accepted_rows,
        "species_dimensions": len(species_order),
        "mapped_genes": int((frame["orthology_missing"] == 0).sum()),
        "species": species_order,
    }
    return frame, report


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="strict")
    return path.open("rt", encoding="utf-8", errors="strict")


def parse_compartments(
    path: str | Path,
    gene_map: pd.DataFrame,
    *,
    formal: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    _, gene_by_protein, gene_by_symbol = _gene_maps(gene_map)
    target_go = {go_id: name for name, go_id in HUMAN_COMPARTMENTS}
    scores: dict[tuple[str, str], float] = {}
    total_rows = 0
    target_rows = 0
    mapped_rows = 0
    source = Path(path)
    with _open_text(source) as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 5:
                raise ContractError(
                    f"JensenLab compartments row {line_number} has {len(fields)} fields; expected 5"
                )
            total_rows += 1
            protein_id, preferred_name, go_id, _, raw_score = fields
            if go_id not in target_go:
                continue
            target_rows += 1
            gene_key = gene_by_protein.get(strip_ensembl_version(protein_id))
            if gene_key is None:
                gene_key = gene_by_symbol.get(preferred_name.strip().upper())
            if gene_key is None:
                continue
            try:
                score = float(raw_score)
            except ValueError as exc:
                raise ContractError(f"invalid JensenLab score on row {line_number}") from exc
            if not math.isfinite(score) or score < 0 or score > 5:
                raise ContractError(f"JensenLab score is outside [0,5] on row {line_number}")
            mapped_rows += 1
            key = (gene_key, go_id)
            scores[key] = max(score, scores.get(key, 0.0))
    if formal and (total_rows < 1_000_000 or target_rows < 10_000 or mapped_rows < 5_000):
        raise ContractError("JensenLab input is too small to be the complete human integrated export")
    gene_keys = sorted(gene_map["gene_key"].astype(str).unique())
    data: dict[str, list[float]] = {"gene_key": gene_keys}
    for name, go_id in HUMAN_COMPARTMENTS:
        data[f"compartment__{name}"] = [scores.get((gene_key, go_id), 0.0) for gene_key in gene_keys]
    frame = pd.DataFrame(data)
    value_columns = [column for column in frame if column.startswith("compartment__")]
    frame["localization_missing"] = frame[value_columns].eq(0).all(axis=1).astype(np.uint8)
    report = {
        "source": str(path),
        "source_sha256": sha256_file(path),
        "format": "protein_id, preferred_name, GO_ID, GO_name, integrated_score",
        "score_contract": "native JensenLab integrated confidence in [0,5]; maximum per mapped gene and overview compartment",
        "total_rows": total_rows,
        "overview_compartment_rows": target_rows,
        "mapped_rows": mapped_rows,
        "mapped_genes": int((frame["localization_missing"] == 0).sum()),
        "compartments": [{"name": name, "go_id": go_id} for name, go_id in HUMAN_COMPARTMENTS],
    }
    return frame, report


def build_static(
    *,
    gene_map_path: str | Path,
    compara_path: str | Path,
    compartments_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    source_manifest_path: str | Path | None = None,
    formal: bool = False,
) -> dict[str, object]:
    if formal and source_manifest_path is None:
        raise ContractError("formal EPGAT static construction requires the acquisition source manifest")
    source_manifest = None
    if source_manifest_path is not None:
        source_manifest = validate_source_manifest(
            source_manifest_path,
            {
                "ensembl_compara_115": compara_path,
                "jensenlab_human_compartments": compartments_path,
            },
        )
    gene_map = read_frame(gene_map_path)
    orthology, orthology_report = parse_compara(compara_path, gene_map, formal=formal)
    localization, localization_report = parse_compartments(
        compartments_path,
        gene_map,
        formal=formal,
    )
    static = orthology.merge(localization, on="gene_key", validate="one_to_one")
    if static.drop(columns="gene_key").isna().any().any():
        raise ContractError("static EPGAT features contain missing numeric values")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    static.to_parquet(output, index=False)
    manifest = {
        "schema_version": "1.0",
        "kind": "epgat_static_features",
        "row_count": len(static),
        "feature_count": len(static.columns) - 1,
        "output": str(output),
        "source_manifest": source_manifest,
        "orthology": orthology_report,
        "localization": localization_report,
    }
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _expression_frame(role_paths: dict[str, str | Path], context: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    reference_columns: list[str] | None = None
    for role in ("train", "validation", "test"):
        frame = read_frame(role_paths[role])
        require_columns(frame, ("gene_key",), f"{role} expression PCA")
        columns = [column for column in frame.columns if column != "gene_key"]
        if reference_columns is None:
            reference_columns = columns
        elif columns != reference_columns:
            raise ContractError("expression PCA columns differ across roles")
        expected = set(context.loc[context["role"].eq(role), "gene_key"])
        observed = set(frame["gene_key"].astype(str))
        if observed != expected:
            raise ContractError(f"{role} expression PCA genes do not match the fixed split role")
        pieces.append(frame)
    assert reference_columns is not None
    combined = pd.concat(pieces, ignore_index=True)
    if combined["gene_key"].duplicated().any():
        raise ContractError("expression PCA role files overlap")
    values = combined[reference_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ContractError("expression PCA contains non-finite values")
    return combined


def induced_degree(
    edges: pd.DataFrame,
    gene_keys: Iterable[str],
    *,
    training_nodes: Iterable[str] | None = None,
) -> pd.DataFrame:
    require_columns(edges, ("gene_key_a", "gene_key_b"), "PPI edges")
    edges = perturb_edges(
        edges,
        training_nodes=None if training_nodes is None else list(training_nodes),
    )
    universe = set(gene_keys)
    degree = {gene_key: 0 for gene_key in universe}
    seen: set[tuple[str, str]] = set()
    for left, right in edges[["gene_key_a", "gene_key_b"]].itertuples(index=False, name=None):
        left, right = str(left), str(right)
        if left == right or left not in universe or right not in universe:
            continue
        edge = (left, right) if left < right else (right, left)
        if edge in seen:
            continue
        seen.add(edge)
        degree[left] += 1
        degree[right] += 1
    return pd.DataFrame(
        {"gene_key": sorted(universe), "ppi_induced_degree": [degree[key] for key in sorted(universe)]}
    )


def _pearson_train_only(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    centered_y = labels.astype(np.float64) - labels.mean()
    y_norm = np.sqrt(np.sum(centered_y * centered_y))
    centered_x = values - values.mean(axis=0, keepdims=True)
    x_norm = np.sqrt(np.sum(centered_x * centered_x, axis=0))
    denominator = x_norm * y_norm
    result = np.zeros(values.shape[1], dtype=np.float64)
    valid = denominator > 0
    result[valid] = (centered_x[:, valid].T @ centered_y) / denominator[valid]
    return result


def build_fold(
    *,
    assignments_path: str | Path,
    labels_path: str | Path,
    static_path: str | Path,
    edges_path: str | Path,
    expression_train_path: str | Path,
    expression_validation_path: str | Path,
    expression_test_path: str | Path,
    cohort_id: str,
    split_strategy: str,
    outer_fold: int,
    output_path: str | Path,
    manifest_path: str | Path,
    top_k: int = FORMAL_SELECTED_DIMENSIONS,
    formal: bool = False,
) -> dict[str, object]:
    context = select_context(
        read_frame(assignments_path),
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )
    expected_coverage = formal_coverage_for_assignments(assignments_path, cohort_id)
    if formal and len(context) != expected_coverage:
        raise ContractError(
            f"formal EPGAT requires the complete {cohort_id} family context with {expected_coverage} genes"
        )
    labels = read_frame(labels_path)
    train_labels = train_labels_only(
        labels,
        context,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )
    expression = _expression_frame(
        {
            "train": expression_train_path,
            "validation": expression_validation_path,
            "test": expression_test_path,
        },
        context,
    )
    static = read_frame(static_path)
    require_columns(
        static,
        ("gene_key", "orthology_missing", "localization_missing"),
        "EPGAT static features",
    )
    degree = induced_degree(
        read_frame(edges_path),
        context["gene_key"],
        training_nodes=context.loc[
            context["role"].isin(("train", "validation")), "gene_key"
        ].tolist(),
    )
    assembled = (
        context.merge(expression, on="gene_key", validate="one_to_one")
        .merge(static, on="gene_key", how="left", validate="one_to_one")
        .merge(degree, on="gene_key", validate="one_to_one")
    )
    numeric_columns = [column for column in assembled if column not in {"gene_key", "role"}]
    if assembled[numeric_columns].isna().any().any():
        raise ContractError("static EPGAT builder must explicitly zero-fill every gene before fold assembly")
    raw = assembled[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ContractError("assembled EPGAT features contain non-finite values")
    train_mask = assembled["role"].eq("train").to_numpy()
    means = raw[train_mask].mean(axis=0)
    scales = raw[train_mask].std(axis=0)
    scales[scales == 0] = 1.0
    standardized = (raw - means) / scales

    train_order = assembled.loc[train_mask, "gene_key"].tolist()
    label_lookup = dict(train_labels.itertuples(index=False, name=None))
    y_train = np.asarray([label_lookup[key] for key in train_order], dtype=np.int64)
    protected = ["orthology_missing", "localization_missing"]
    candidate_columns = [column for column in numeric_columns if column not in protected]
    candidate_indexes = [numeric_columns.index(column) for column in candidate_columns]
    node_keys = assembled["gene_key"].astype(str).tolist()
    selection_keys = [str(key) for key in train_order]
    selection_values = np.asarray(y_train, dtype="int64").reshape(-1)
    position = {key: index for index, key in enumerate(node_keys)}
    selection_rows = np.asarray(
        [position[key] for key in selection_keys], dtype=np.int64
    )
    correlations = _pearson_train_only(
        standardized[selection_rows][:, candidate_indexes], selection_values
    )
    needed = top_k - len(protected)
    if needed <= 0 or needed > len(candidate_columns):
        raise ContractError("requested selected dimension is incompatible with available EPGAT features")
    order = sorted(range(len(candidate_columns)), key=lambda i: (-correlations[i], candidate_columns[i]))
    selected_columns = [candidate_columns[index] for index in order[:needed]] + protected
    selected_indexes = [numeric_columns.index(column) for column in selected_columns]
    selected_values = standardized[:, selected_indexes].astype(np.float32)
    output_columns = [f"epgat_{index:03d}" for index in range(1, top_k + 1)]
    output_frame = pd.DataFrame(selected_values, columns=output_columns)
    output_frame.insert(0, "gene_key", assembled["gene_key"].tolist())
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.sort_values("gene_key", kind="mergesort").to_parquet(output, index=False)

    selected_details = []
    correlation_lookup = dict(zip(candidate_columns, correlations, strict=True))
    for column in selected_columns:
        index = numeric_columns.index(column)
        selected_details.append(
            {
                "source_column": column,
                "output_column": output_columns[len(selected_details)],
                "train_mean": float(means[index]),
                "train_scale": float(scales[index]),
                "train_pearson": None if column in protected else float(correlation_lookup[column]),
                "protected_missingness": column in protected,
            }
        )
    train_label_payload = sorted((key, int(label_lookup[key])) for key in train_order)
    manifest = {
        "schema_version": "1.0",
        "kind": "epgat_fold_features",
        "cohort_id": cohort_id,
        "split_strategy": split_strategy,
        "outer_fold": outer_fold,
        "row_count": len(context),
        "role_counts": context["role"].value_counts().sort_index().to_dict(),
        "raw_feature_count": len(numeric_columns),
        "selected_feature_count": top_k,
        "feature_selection": "descending signed Pearson correlation using outer-train labels only; two missingness indicators forced into the selected view",
        "standardization": "outer-train mean and population standard deviation only",
        "train_label_hash": stable_json_hash(train_label_payload),
        "selected_features": selected_details,
        "output": str(output),
    }
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
