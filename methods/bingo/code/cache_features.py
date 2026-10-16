"""ESM-2 inference and contact-graph construction for the BiNGO graph cache.
Imported by code/extract_features.py.
"""
from __future__ import annotations

import gc
import gzip
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch

if __package__:
    from .contracts import require_method_path, sha256_file
    from .graph import (
        BINGO_MODEL_NAME,
        CACHE_SCHEMA,
        CONTACT_RETAIN_FRACTION,
        ESM_EMBEDDING_DIM,
        ESM_LAYER,
        MAX_PROTEIN_LENGTH,
        high_contact_edges,
    )
else:
    from contracts import require_method_path, sha256_file
    from graph import (
        BINGO_MODEL_NAME,
        CACHE_SCHEMA,
        CONTACT_RETAIN_FRACTION,
        ESM_EMBEDDING_DIM,
        ESM_LAYER,
        MAX_PROTEIN_LENGTH,
        high_contact_edges,
    )


METHOD_ROOT = Path(__file__).resolve().parents[1]
_ALLOWED_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYBXZUOJ")
_MODEL_LOADER = Callable[[Path, torch.device], tuple[Any, Any]]
_PENDING_ITEM = tuple[str, Path, Path, str]


def _validate_batch_parameters(token_budget: int, max_batch_size: int) -> None:
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")


def _group_pending_batches(
    pending: Sequence[_PENDING_ITEM],
    *,
    token_budget: int,
    max_batch_size: int,
) -> tuple[tuple[_PENDING_ITEM, ...], ...]:

    _validate_batch_parameters(token_budget, max_batch_size)
    batches: list[tuple[_PENDING_ITEM, ...]] = []
    current: list[_PENDING_ITEM] = []
    current_max_length = 0
    for item in sorted(pending, key=lambda value: value[0]):
        truncated_length = len(item[3])
        candidate_size = len(current) + 1
        candidate_max_length = max(current_max_length, truncated_length)
        exceeds_batch_limit = candidate_size > max_batch_size
        exceeds_token_budget = (
            candidate_max_length + 2
        ) * candidate_size > token_budget
        if current and (exceeds_batch_limit or exceeds_token_budget):
            batches.append(tuple(current))
            current = []
            current_max_length = 0
        current.append(item)
        current_max_length = max(current_max_length, truncated_length)
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def cache_filename(gene_key: str) -> str:

    return hashlib.sha256(gene_key.encode("utf-8")).hexdigest()[:24] + ".pt"


def _method_relative(path: Path) -> str:
    return path.resolve().relative_to(METHOD_ROOT).as_posix()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _load_exact_model(asset_dir: Path, device: torch.device) -> tuple[Any, Any]:
    try:
        import esm
    except ImportError as exc:
        raise RuntimeError("Bingo cache generation requires fair-esm==2.0.0") from exc
    model_path = asset_dir / f"{BINGO_MODEL_NAME}.pt"
    contact_path = asset_dir / f"{BINGO_MODEL_NAME}-contact-regression.pt"
    if not model_path.is_file() or not contact_path.is_file():
        raise FileNotFoundError(
            "exact ESM-2 model and contact-regression weights are both required"
        )
    model_data = torch.load(model_path, map_location="cpu", weights_only=False)
    regression_data = torch.load(contact_path, map_location="cpu", weights_only=False)
    model, alphabet = esm.pretrained.load_model_and_alphabet_core(
        BINGO_MODEL_NAME, model_data, regression_data
    )
    if int(model.num_layers) != ESM_LAYER or int(model.embed_dim) != ESM_EMBEDDING_DIM:
        raise RuntimeError("loaded ESM architecture does not match esm2_t33_650M_UR50D")
    model.eval().requires_grad_(False)
    model.to(device)
    return model, alphabet


def _read_gene_list(path: Path) -> tuple[str, ...]:

    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        frame = pd.read_parquet(path, columns=["gene_key"])
    except Exception as exc:
        raise ValueError(f"gene list lacks readable gene_key column: {path}") from exc
    values = frame["gene_key"]
    if values.isna().any():
        raise ValueError(f"gene list contains null gene_key values: {path}")
    genes = tuple(sorted({str(value).strip() for value in values}))
    if not genes or any(not gene for gene in genes):
        raise ValueError(f"gene list contains no usable gene_key values: {path}")
    return genes


def gene_key_sha256(genes: Sequence[str]) -> str:

    canonical = "\n".join(sorted(set(map(str, genes)))) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_protein_fasta(path: Path) -> dict[str, str]:

    if not path.is_file():
        raise FileNotFoundError(path)
    opener = gzip.open if path.suffix.lower() in {".gz", ".gzip"} else Path.open
    records: dict[str, list[str]] = {}
    current: str | None = None
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                gene_key = line[1:].split("|", 1)[0].split()[0]
                if not gene_key or gene_key in records:
                    raise ValueError(
                        f"invalid or duplicate FASTA key at line {line_number}"
                    )
                records[gene_key] = []
                current = gene_key
            elif current is None:
                raise ValueError(
                    f"FASTA sequence precedes a header at line {line_number}"
                )
            else:
                records[current].append(line.upper())
    sequences = {key: "".join(parts) for key, parts in records.items()}
    if not sequences or any(not sequence for sequence in sequences.values()):
        raise ValueError("FASTA contains no records or an empty sequence")
    invalid = sorted(
        {
            aa
            for sequence in sequences.values()
            for aa in sequence
            if aa not in _ALLOWED_RESIDUES
        }
    )
    if invalid:
        raise ValueError(f"FASTA contains unsupported residues: {invalid}")
    return sequences


def _model_file_identities(asset_dir: Path) -> dict[str, dict[str, Any]]:
    files = {
        "esm_model": asset_dir / f"{BINGO_MODEL_NAME}.pt",
        "esm_contact_regression": asset_dir
        / f"{BINGO_MODEL_NAME}-contact-regression.pt",
    }
    identities: dict[str, dict[str, Any]] = {}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        identities[name] = {
            "filename": path.name,
            "path": _method_relative(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return identities


def _valid_cached(
    path: Path,
    gene_key: str,
    *,
    protein_sequence_sha256: str,
    esm_model_sha256: str,
    esm_contact_regression_sha256: str,
) -> dict[str, Any] | None:

    if not path.is_file():
        return None
    try:
        item = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(item, Mapping):
        return None
    if (
        item.get("gene_key") != gene_key
        or item.get("cache_schema") != CACHE_SCHEMA
        or item.get("model_name") != BINGO_MODEL_NAME
        or item.get("esm_layer") != ESM_LAYER
        or item.get("contact_retain_fraction") != CONTACT_RETAIN_FRACTION
        or item.get("contact_direction") != "largest_probabilities"
        or item.get("edge_directionality") != "bidirectional"
        or item.get("protein_sequence_sha256") != protein_sequence_sha256
        or item.get("esm_model_sha256") != esm_model_sha256
        or item.get("esm_contact_regression_sha256") != esm_contact_regression_sha256
        or not isinstance(item.get("x"), torch.Tensor)
        or not isinstance(item.get("edge_index"), torch.Tensor)
    ):
        return None
    x = item["x"]
    edge_index = item["edge_index"]
    cached_length = item.get("cached_length")
    original_length = item.get("original_length")
    if not isinstance(cached_length, int) or not isinstance(original_length, int):
        return None
    if cached_length != min(original_length, MAX_PROTEIN_LENGTH):
        return None
    if item.get("truncated") is not (original_length > MAX_PROTEIN_LENGTH):
        return None
    if (
        x.ndim != 2
        or x.shape != (cached_length, ESM_EMBEDDING_DIM)
        or x.dtype != torch.float32
        or not torch.isfinite(x).all()
    ):
        return None
    expected_directed_edges = (
        0
        if cached_length == 1
        else 2
        * max(
            1,
            math.ceil(
                cached_length * (cached_length - 1) / 2 * CONTACT_RETAIN_FRACTION
            ),
        )
    )
    if (
        edge_index.ndim != 2
        or edge_index.shape != (2, expected_directed_edges)
        or edge_index.dtype not in (torch.int32, torch.int64)
    ):
        return None
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= cached_length
    ):
        return None
    return item


def _cache_row(
    relative: Path, item: Mapping[str, Any], output_path: Path
) -> dict[str, Any]:
    try:
        graph_subpath = relative.relative_to("graphs")
    except ValueError as error:
        raise ValueError(
            f"cache graph path must be relative to the graphs directory: {relative}"
        ) from error
    return {
        "gene_key": str(item["gene_key"]),
        "cache_file": graph_subpath.as_posix(),
        "original_length": int(item["original_length"]),
        "cached_length": int(item["cached_length"]),
        "truncated": bool(item["truncated"]),
        "node_count": int(item["x"].shape[0]),
        "directed_edge_count": int(item["edge_index"].shape[1]),
        "contact_threshold": float(item["contact_threshold"]),
        "cache_sha256": sha256_file(output_path),
    }


def _write_progress(
    path: Path,
    *,
    genes: Sequence[str],
    completed: int,
    last_gene_key: str | None,
    status: str,
) -> None:
    _write_json_atomic(
        path,
        {
            "status": status,
            "completed": completed,
            "total": len(genes),
            "requested_gene_count": len(genes),
            "last_gene_key": last_gene_key,
            "cache_schema": CACHE_SCHEMA,
        },
    )


def build_cache_from_sequences(
    *,
    genes: Sequence[str],
    sequences: Mapping[str, str],
    asset_dir: Path,
    cache_dir: Path,
    device: torch.device,
    token_budget: int = 1024,
    max_batch_size: int = 2,
    output: Path | None = None,
    index_output: Path | None = None,
    source_paths: Sequence[Path] = (),
    gene_sources: Sequence[Mapping[str, Any]] = (),
    model_loader: _MODEL_LOADER | None = None,
) -> dict[str, Any]:

    _validate_batch_parameters(token_budget, max_batch_size)
    asset_dir = require_method_path(Path(asset_dir), "asset_dir")
    cache_dir = require_method_path(Path(cache_dir), "cache_dir")
    output_path = require_method_path(
        Path(output) if output is not None else cache_dir / "cache_manifest.json",
        "output",
    )
    index_path = require_method_path(
        Path(index_output)
        if index_output is not None
        else cache_dir / "cache_index.parquet",
        "index_output",
    )
    input_paths = tuple(
        require_method_path(Path(path), "protein_fasta") for path in source_paths
    )
    if not input_paths:
        raise ValueError("at least one protein FASTA is required")

    ordered_genes = tuple(sorted({str(gene).strip() for gene in genes}))
    if not ordered_genes or any(not gene for gene in ordered_genes):
        raise ValueError("cache gene set is empty or contains an empty gene_key")
    missing = sorted(set(ordered_genes).difference(sequences))
    if missing:
        raise ValueError(f"{len(missing)} cache genes lack selected protein sequences")

    model_files = _model_file_identities(asset_dir)
    sequence_files = [
        {
            "path": _method_relative(path),
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in input_paths
    ]
    cache_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = cache_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = cache_dir / "progress.json"

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    pending: list[tuple[str, Path, Path, str]] = []
    reused_count = 0
    for gene_key in ordered_genes:
        sequence = str(sequences[gene_key]).upper()
        truncated = sequence[:MAX_PROTEIN_LENGTH]
        protein_sequence_sha256 = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        relative = Path("graphs") / cache_filename(gene_key)
        output_graph = cache_dir / relative
        item = _valid_cached(
            output_graph,
            gene_key,
            protein_sequence_sha256=protein_sequence_sha256,
            esm_model_sha256=model_files["esm_model"]["sha256"],
            esm_contact_regression_sha256=model_files["esm_contact_regression"]["sha256"],
        )
        if item is None:
            pending.append((gene_key, output_graph, relative, truncated))
        else:
            rows.append(_cache_row(relative, item, output_graph))
            reused_count += 1

    _write_progress(
        progress_path,
        genes=ordered_genes,
        completed=reused_count,
        last_gene_key=None,
        status="running" if pending else "complete",
    )

    peak_gpu_memory = 0
    gpu_name: str | None = None
    cuda_device_index: int | None = None
    computed_count = 0
    batches = _group_pending_batches(
        pending,
        token_budget=token_budget,
        max_batch_size=max_batch_size,
    )
    if pending:
        if device.type != "cuda" and model_loader is None:
            raise RuntimeError("formal Bingo ESM cache generation requires CUDA")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if device.type == "cuda":
            cuda_device_index = (
                torch.cuda.current_device() if device.index is None else int(device.index)
            )
            torch.cuda.set_device(cuda_device_index)
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(cuda_device_index)
            gpu_name = torch.cuda.get_device_name(cuda_device_index)
        loader = _load_exact_model if model_loader is None else model_loader
        model, alphabet = loader(asset_dir, device)
        converter = alphabet.get_batch_converter()
        for batch in batches:
            batch_keys = tuple(item[0] for item in batch)
            try:
                _, _, tokens = converter([(item[0], item[3]) for item in batch])
                tokens = tokens.to(device, non_blocking=True)
                with torch.inference_mode():
                    result = model(
                        tokens,
                        repr_layers=[ESM_LAYER],
                        return_contacts=True,
                    )
            except Exception as exc:
                keys = ", ".join(batch_keys)
                raise RuntimeError(
                    f"failed ESM batch for gene_key(s): {keys}"
                ) from exc

            for index, (gene_key, output_graph, relative, truncated) in enumerate(
                batch
            ):
                sequence = str(sequences[gene_key]).upper()
                protein_sequence_sha256 = hashlib.sha256(
                    sequence.encode("ascii")
                ).hexdigest()
                try:
                    x = result["representations"][ESM_LAYER][
                        index, 1 : len(truncated) + 1
                    ].float().cpu()
                    contacts = result["contacts"][
                        index, : len(truncated), : len(truncated)
                    ].float().cpu()
                    if x.shape != (len(truncated), ESM_EMBEDDING_DIM):
                        raise RuntimeError(
                            f"unexpected residue embedding shape for {gene_key}"
                        )
                    if contacts.shape != (len(truncated), len(truncated)):
                        raise RuntimeError(f"unexpected contact shape for {gene_key}")
                    edge_index, threshold = high_contact_edges(
                        contacts, CONTACT_RETAIN_FRACTION
                    )
                    item = {
                        "gene_key": gene_key,
                        "cache_schema": CACHE_SCHEMA,
                        "model_name": BINGO_MODEL_NAME,
                        "esm_layer": ESM_LAYER,
                        "original_length": len(sequence),
                        "cached_length": len(truncated),
                        "truncated": len(sequence) > MAX_PROTEIN_LENGTH,
                        "protein_sequence_sha256": protein_sequence_sha256,
                        "esm_model_sha256": model_files["esm_model"]["sha256"],
                        "esm_contact_regression_sha256": model_files[
                            "esm_contact_regression"
                        ]["sha256"],
                        "contact_retain_fraction": CONTACT_RETAIN_FRACTION,
                        "contact_direction": "largest_probabilities",
                        "edge_directionality": "bidirectional",
                        "contact_threshold": threshold,
                        "x": x,
                        "edge_index": edge_index.to(dtype=torch.int32),
                    }
                    temporary = output_graph.with_name(output_graph.name + ".partial")
                    torch.save(item, temporary)
                    os.replace(temporary, output_graph)
                except Exception as exc:
                    raise RuntimeError(
                        f"failed to cache gene_key={gene_key}"
                    ) from exc
                rows.append(_cache_row(relative, item, output_graph))
                computed_count += 1
                _write_progress(
                    progress_path,
                    genes=ordered_genes,
                    completed=len(rows),
                    last_gene_key=gene_key,
                    status="running" if len(rows) < len(ordered_genes) else "complete",
                )
            del result, tokens
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if device.type == "cuda":
            if cuda_device_index is None:
                raise AssertionError("CUDA device index was not initialized")
            peak_gpu_memory = int(torch.cuda.max_memory_allocated(cuda_device_index))

    index_frame = pd.DataFrame(rows).sort_values("gene_key", kind="mergesort")
    _write_parquet_atomic(index_frame, index_path)
    manifest: dict[str, Any] = {
        "method_id": "bingo",
        "cache_schema": CACHE_SCHEMA,
        "model_name": BINGO_MODEL_NAME,
        "esm_layer": ESM_LAYER,
        "embedding_dim": ESM_EMBEDDING_DIM,
        "esm_inference_dtype": "float32",
        "cached_embedding_dtype": "float32",
        "maximum_protein_length": MAX_PROTEIN_LENGTH,
        "contact_retain_fraction": CONTACT_RETAIN_FRACTION,
        "contact_direction": "largest_probabilities",
        "edge_directionality": "bidirectional",
        "requested_gene_count": len(ordered_genes),
        "cache_completed_count": len(index_frame),
        "complete": len(index_frame) == len(ordered_genes),
        "reused_cache_count": reused_count,
        "computed_cache_count": computed_count,
        "token_budget": token_budget,
        "max_batch_size": max_batch_size,
        "batch_count": len(batches),
        "gene_key_sha256": gene_key_sha256(ordered_genes),
        "gene_sources": [dict(source) for source in gene_sources],
        "input_protein_fastas": sequence_files,
        "model_files": model_files,
        "output_index": _method_relative(index_path),
        "progress": _method_relative(progress_path),
        "cache_index_sha256": sha256_file(index_path),
        "device": str(device),
        "gpu_name": gpu_name,
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "wall_seconds": time.perf_counter() - started,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
    }
    _write_json_atomic(output_path, manifest)
    return manifest


def build_cache(
    *,
    gene_list: Path,
    protein_fasta: Path,
    asset_dir: Path,
    cache_dir: Path,
    device: torch.device,
    token_budget: int = 1024,
    max_batch_size: int = 2,
    output: Path | None = None,
    index_output: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:

    gene_list = require_method_path(Path(gene_list), "gene_list")
    protein_fasta = require_method_path(Path(protein_fasta), "protein_fasta")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    all_genes = _read_gene_list(gene_list)
    genes = all_genes if limit is None else all_genes[:limit]
    sequences = read_protein_fasta(protein_fasta)
    return build_cache_from_sequences(
        genes=genes,
        sequences=sequences,
        asset_dir=asset_dir,
        cache_dir=cache_dir,
        device=torch.device(device),
        token_budget=token_budget,
        max_batch_size=max_batch_size,
        output=output,
        index_output=index_output,
        source_paths=(protein_fasta,),
        gene_sources=(
            {
                "path": _method_relative(gene_list),
                "gene_count": len(genes),
                "gene_key_sha256": gene_key_sha256(genes),
            },
        ),
    )
