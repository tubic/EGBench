"""Write the family-split fold views of a method's static feature matrix, FASTA file or BiNGO graph-cache index.
Imported by the feature builders of BiNGO, DeepCellEss, GCNN-SFM, HELP and ProteomeLM-Ess.
"""
from __future__ import annotations

import gzip
import io
import os
import tempfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from egbench.contracts import COHORT_IDS, SPLIT_STRATEGIES
from egbench.data import cohort_tables_directory, feature_directory, folds_directory


FOLD_COUNT = 5
AUTHORITY_COLUMNS = ("gene_key", "gene_symbol", "cluster_id", "role")
AUTHORITY_ROLES = frozenset({"train", "validation", "test"})
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _require_cohort(cohort_id: str) -> str:
    if cohort_id not in COHORT_IDS:
        raise ValueError(f"unsupported cohort_id: {cohort_id!r}")
    return cohort_id


def _as_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _ensure_unique_keys(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    if "gene_key" not in frame.columns:
        raise ValueError(f"{path} lacks required gene_key column")
    result = frame.copy()
    if result["gene_key"].isna().any():
        raise ValueError(f"{path} contains null gene_key values")
    result["gene_key"] = result["gene_key"].astype(str)
    if (result["gene_key"] == "").any():
        raise ValueError(f"{path} contains blank gene_key values")
    duplicate = result["gene_key"].duplicated(keep=False)
    if duplicate.any():
        examples = sorted(result.loc[duplicate, "gene_key"].unique())[:5]
        raise ValueError(f"{path} contains duplicate gene_key values: {examples}")
    return result


def authority_fold_path(repo_root: Path | str, cohort_id: str, fold: int) -> Path:

    _require_cohort(cohort_id)
    if fold not in range(FOLD_COUNT):
        raise ValueError(f"fold must be in 0..{FOLD_COUNT - 1}, got {fold}")
    return (
        folds_directory(
            cohort_id, split_strategy="family_5fold", root=_as_path(repo_root)
        )
        / "fold_genes"
        / f"fold_{fold}_genes.parquet"
    )


def read_authority_gene_sets(
    repo_root: Path | str, cohort_id: str
) -> dict[int, list[str]]:

    _require_cohort(cohort_id)
    root = _as_path(repo_root)
    common_path = cohort_tables_directory(cohort_id, root=root) / "common_binary.parquet"
    if not common_path.is_file():
        raise FileNotFoundError(common_path)
    common = _ensure_unique_keys(pd.read_parquet(common_path), common_path)
    common_keys = sorted(common["gene_key"].tolist())

    gene_sets: dict[int, list[str]] = {}
    stable_metadata: pd.DataFrame | None = None
    for fold in range(FOLD_COUNT):
        path = authority_fold_path(root, cohort_id, fold)
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = pd.read_parquet(path)
        missing = set(AUTHORITY_COLUMNS).difference(raw.columns)
        if missing:
            raise ValueError(f"{path} lacks authority columns: {sorted(missing)}")
        frame = _ensure_unique_keys(raw, path)
        for column in ("gene_symbol", "cluster_id", "role"):
            if frame[column].isna().any():
                raise ValueError(f"{path} contains null {column} values")
            frame[column] = frame[column].astype(str)
            if (frame[column].str.strip() == "").any():
                raise ValueError(f"{path} contains blank {column} values")
        observed_roles = frozenset(frame["role"].unique())
        if observed_roles != AUTHORITY_ROLES:
            raise ValueError(
                f"{path} role values must be exactly {sorted(AUTHORITY_ROLES)}, "
                f"got {sorted(observed_roles)}"
            )
        role_counts = frame.groupby("cluster_id", sort=False)["role"].nunique()
        split_clusters = role_counts[role_counts != 1]
        if not split_clusters.empty:
            raise ValueError(
                f"{path} assigns a cluster to multiple roles: "
                f"{sorted(split_clusters.index.astype(str))[:5]}"
            )

        keys = sorted(frame["gene_key"].tolist())
        if keys != common_keys:
            missing_keys = sorted(set(common_keys).difference(keys))
            extra_keys = sorted(set(keys).difference(common_keys))
            raise ValueError(
                f"{path} gene set differs from {common_path}: "
                f"missing={missing_keys[:5]}, extra={extra_keys[:5]}"
            )
        metadata = (
            frame.loc[:, ["gene_key", "gene_symbol", "cluster_id"]]
            .sort_values("gene_key", kind="mergesort")
            .reset_index(drop=True)
        )
        if stable_metadata is None:
            stable_metadata = metadata
        elif not metadata.equals(stable_metadata):
            raise ValueError(
                f"{path} gene_symbol or cluster_id mapping differs across folds"
            )
        gene_sets[fold] = keys
    return gene_sets


def _replace_with_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir():
            raise IsADirectoryError(destination)
        destination.unlink()
    os.link(source, destination)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet",
        prefix=f".{destination.name}.",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_static_matrix(frame: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = _ensure_unique_keys(frame, path)
    if frame.columns.duplicated().any():
        raise ValueError(f"{path} contains duplicate columns")
    feature_columns = [column for column in frame.columns if column != "gene_key"]
    if not feature_columns:
        raise ValueError(f"{path} has no feature columns")
    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(
            f"{path} contains non-numeric feature columns: {non_numeric[:5]}"
        )
    values = frame.loc[:, feature_columns].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite feature values")
    return frame, feature_columns


def materialize_static_feature_folds(
    method_root: Path | str,
    cohort_id: str,
    *,
    matrix_path: Path | str,
    repo_root: Path | str = _REPO_ROOT,
) -> tuple[list[Path], int]:

    _require_cohort(cohort_id)
    root = _as_path(method_root)
    source = _as_path(matrix_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    matrix, feature_columns = _validate_static_matrix(pd.read_parquet(source), source)
    matrix = matrix.set_index("gene_key", drop=False)
    authority = read_authority_gene_sets(repo_root, cohort_id)
    outputs: list[Path] = []
    first_output: Path | None = None
    first_keys: list[str] | None = None
    for split_strategy in SPLIT_STRATEGIES:
        for fold, keys in authority.items():
            missing = sorted(set(keys).difference(matrix.index))
            if missing:
                raise ValueError(
                    f"{source} lacks {len(missing)} authority genes for {cohort_id} "
                    f"fold {fold}: {missing[:5]}"
                )
            destination = (
                feature_directory(
                    root,
                    cohort_id,
                    split_strategy=split_strategy,
                    outer_fold=fold,
                )
                / "features.parquet"
            )
            if first_output is not None and keys == first_keys:
                _replace_with_hardlink(first_output, destination)
            else:
                subset = matrix.loc[keys].reset_index(drop=True)
                _atomic_parquet(subset, destination)
                _validate_materialized_table(destination, keys, "features.parquet")
                if first_output is None:
                    first_output = destination
                    first_keys = keys
            outputs.append(destination)
    return outputs, len(feature_columns)


def _open_fasta(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _read_fasta(path: Path) -> dict[str, tuple[str, str]]:

    if not path.is_file():
        raise FileNotFoundError(path)
    records: dict[str, tuple[str, str]] = {}
    header: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        nonlocal header, chunks
        if header is None:
            return
        gene_key = header.split("|", 1)[0].split(None, 1)[0]
        if not gene_key:
            raise ValueError(f"empty FASTA gene_key in {path}")
        if gene_key in records:
            raise ValueError(f"duplicate FASTA gene_key {gene_key!r} in {path}")
        sequence = "".join(chunks)
        if not sequence:
            raise ValueError(f"empty FASTA sequence for {gene_key!r} in {path}")
        records[gene_key] = (header, sequence)
        header = None
        chunks = []

    with _open_fasta(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:]
                if not header:
                    raise ValueError(f"empty FASTA header at {path}:{line_number}")
            elif header is None:
                raise ValueError(
                    f"sequence before FASTA header at {path}:{line_number}"
                )
            else:
                chunks.append(line)
    flush()
    if not records:
        raise ValueError(f"no FASTA records in {path}")
    return records


def _write_fasta_gzip(
    records: Mapping[str, tuple[str, str]],
    keys: Iterable[str],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".fa.gz",
        prefix=f".{destination.name}.",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with io.TextIOWrapper(
                    compressed, encoding="utf-8", newline="\n"
                ) as output:
                    for gene_key in keys:
                        header, sequence = records[gene_key]
                        output.write(f">{header}\n")
                        for start in range(0, len(sequence), 60):
                            output.write(sequence[start : start + 60] + "\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize_fasta_folds(
    method_root: Path | str,
    cohort_id: str,
    sequence_kind: str,
    *,
    fasta_path: Path | str,
    repo_root: Path | str = _REPO_ROOT,
) -> list[Path]:

    _require_cohort(cohort_id)
    output_names = {
        "proteins": "selected_proteins.fa.gz",
        "cds": "selected_cds.fa.gz",
    }
    if sequence_kind not in output_names:
        raise ValueError("sequence_kind must be 'proteins' or 'cds'")
    root = _as_path(method_root)
    source = _as_path(fasta_path)
    records = _read_fasta(source)
    authority = read_authority_gene_sets(repo_root, cohort_id)
    outputs: list[Path] = []
    first_output: Path | None = None
    first_keys: list[str] | None = None
    for split_strategy in SPLIT_STRATEGIES:
        for fold, keys in authority.items():
            missing = sorted(set(keys).difference(records))
            if missing:
                raise ValueError(
                    f"{source} lacks {len(missing)} authority genes for {cohort_id} "
                    f"fold {fold}: {missing[:5]}"
                )
            destination = (
                feature_directory(
                    root,
                    cohort_id,
                    split_strategy=split_strategy,
                    outer_fold=fold,
                )
                / output_names[sequence_kind]
            )
            if first_output is not None and keys == first_keys:
                _replace_with_hardlink(first_output, destination)
            else:
                _write_fasta_gzip(records, keys, destination)
                observed = _read_fasta(destination)
                if sorted(observed) != keys:
                    raise ValueError(f"FASTA validation failed for {destination}")
                if first_output is None:
                    first_output = destination
                    first_keys = keys
            outputs.append(destination)
    return outputs


def _resolve_bingo_graph_root(
    method_root: Path, graph_root: Path | str | None
) -> Path:
    allowed = (
        method_root / "feature" / "source" / "esm2_graph_cache" / "graphs"
    )
    allowed_absolute = Path(os.path.abspath(allowed))
    allowed_resolved = allowed_absolute.resolve()
    if allowed_resolved != allowed_absolute:
        raise ValueError(f"BiNGO graph directory must not traverse symlinks: {allowed}")

    provided = allowed if graph_root is None else Path(graph_root).expanduser()
    provided_absolute = Path(os.path.abspath(provided))
    if provided_absolute != allowed_absolute:
        raise ValueError(
            f"BiNGO graph_root must be exactly {allowed_absolute}, "
            f"got {provided_absolute}"
        )
    if provided_absolute.resolve() != allowed_absolute:
        raise ValueError(
            f"BiNGO graph_root must not traverse symlinks: {provided_absolute}"
        )
    if not allowed_absolute.is_dir():
        raise FileNotFoundError(allowed_absolute)
    return allowed_absolute


def _normalized_graph_subpath(cache_file: object) -> PurePosixPath:
    raw = str(cache_file).strip()
    if not raw:
        raise ValueError("cache_file must be a non-empty relative graph subpath")
    if "\\" in raw:
        raise ValueError(f"cache_file must use POSIX separators: {raw!r}")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(
            f"cache_file must stay within the shared graph directory: {raw!r}"
        )
    normalized = PurePosixPath(*candidate.parts)
    if normalized == PurePosixPath("."):
        raise ValueError(f"cache_file does not name a graph file: {raw!r}")
    return normalized


def materialize_bingo_cache_indices(
    method_root: Path | str,
    cohort_id: str,
    *,
    source_index_path: Path | str,
    repo_root: Path | str = _REPO_ROOT,
    graph_root: Path | str | None = None,
) -> list[Path]:

    _require_cohort(cohort_id)
    root = _as_path(method_root)
    source = _as_path(source_index_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    graph_directory = _resolve_bingo_graph_root(root, graph_root)
    index = _ensure_unique_keys(pd.read_parquet(source), source)
    if "cache_file" not in index.columns:
        raise ValueError(f"{source} lacks required cache_file column")
    index = index.set_index("gene_key", drop=False)
    authority = read_authority_gene_sets(repo_root, cohort_id)

    needed_keys = sorted(set().union(*(set(keys) for keys in authority.values())))
    missing = sorted(set(needed_keys).difference(index.index))
    if missing:
        raise ValueError(
            f"{source} lacks {len(missing)} cache rows for {cohort_id}: {missing[:5]}"
        )
    graph_subpaths: dict[str, PurePosixPath] = {}
    target_sources: dict[PurePosixPath, str] = {}
    for gene_key in needed_keys:
        original = str(index.at[gene_key, "cache_file"]).strip()
        relative = _normalized_graph_subpath(original)
        prior = target_sources.get(relative)
        if prior is not None and prior != original:
            raise ValueError(
                f"distinct cache_file values normalize to the same target: "
                f"{prior!r} and {original!r}"
            )
        target_sources[relative] = original
        target = graph_directory.joinpath(*relative.parts)
        resolved_target = target.resolve()
        if not resolved_target.is_relative_to(graph_directory):
            raise ValueError(f"cache graph escapes the shared graph directory: {target}")
        if not target.is_file():
            raise FileNotFoundError(
                f"cache graph referenced by {source} is missing: {target}"
            )
        graph_subpaths[gene_key] = relative

    outputs: list[Path] = []
    first_output: Path | None = None
    first_table: pd.DataFrame | None = None
    for split_strategy in SPLIT_STRATEGIES:
        for fold, keys in authority.items():
            destination = (
                feature_directory(
                    root,
                    cohort_id,
                    split_strategy=split_strategy,
                    outer_fold=fold,
                )
                / "cache_index.parquet"
            )
            subset = index.loc[keys].reset_index(drop=True).copy()
            subset["cache_file"] = [
                os.path.relpath(
                    graph_directory.joinpath(*graph_subpaths[gene_key].parts),
                    start=destination.parent,
                )
                for gene_key in subset["gene_key"]
            ]
            if (
                first_output is not None
                and first_table is not None
                and subset.equals(first_table)
            ):
                _replace_with_hardlink(first_output, destination)
            else:
                _atomic_parquet(subset, destination)
                _validate_materialized_table(destination, keys, "cache_index.parquet")
                _validate_cache_paths(destination, graph_directory)
                if first_output is None:
                    first_output = destination
                    first_table = subset
            outputs.append(destination)
    return outputs


def _validate_materialized_table(
    path: Path, expected_keys: Sequence[str], artifact_name: str
) -> None:
    frame = _ensure_unique_keys(pd.read_parquet(path), path)
    observed = sorted(frame["gene_key"].tolist())
    if observed != list(expected_keys):
        raise ValueError(
            f"{artifact_name} at {path} does not match its authority gene set"
        )
    if artifact_name == "features.parquet":
        _validate_static_matrix(frame, path)


def _validate_cache_paths(path: Path, graph_directory: Path) -> None:
    frame = pd.read_parquet(path)
    if "cache_file" not in frame.columns:
        raise ValueError(f"{path} lacks cache_file")
    missing: list[str] = []
    for value in frame["cache_file"].astype(str):
        relative = Path(value)
        if relative.is_absolute():
            raise ValueError(f"{path} contains an absolute cache_file: {value}")
        target = (path.parent / relative).resolve()
        if not target.is_relative_to(graph_directory):
            raise ValueError(f"{path} cache_file escapes shared graphs: {value}")
        if not target.is_file():
            missing.append(value)
    if missing:
        raise FileNotFoundError(f"{path} references missing cache graph: {missing[0]}")
