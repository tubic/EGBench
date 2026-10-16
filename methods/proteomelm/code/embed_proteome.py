#!/usr/bin/env python3
"""Embed a proteome with ESM-C 600M and ProteomeLM-L and write layer-8 gene features.
Called by build_features.py; needs the ProteomeLM source release on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


METHOD_ROOT = Path(__file__).resolve().parents[1]
ESMC_RUNTIME_ROOT = METHOD_ROOT / "feature/source/esmc_runtime"
ESMC_WEIGHTS = ESMC_RUNTIME_ROOT / "data/weights/esmc_600m_2024_12_v0.pth"
FEATURE_DIMENSIONS = 1152
DEFAULT_ESM_BLOCK = 512
DEFAULT_ESM_CACHE_CHUNK_GENES = 512


def _method_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(METHOD_ROOT):
        raise ValueError(f"{label} must be located inside {METHOD_ROOT}: {resolved}")
    return resolved


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    sequences: list[str] = []
    seen: set[str] = set()
    current_label: str | None = None
    current_parts: list[str] = []

    def emit() -> None:
        if current_label is None:
            return
        sequence = "".join(current_parts).replace(" ", "").replace("\r", "").upper()
        if not sequence:
            raise ValueError(f"FASTA record has an empty sequence: {current_label}")
        if current_label not in seen:
            seen.add(current_label)
            labels.append(current_label)
            sequences.append(sequence[:4096])

    with _open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                emit()
                header = line[1:].strip().split()[0]
                current_label = header.split("|")[0].strip()
                if not current_label:
                    raise ValueError("FASTA record has an empty gene key")
                current_parts = []
            elif current_label is None:
                raise ValueError(f"FASTA sequence appears before a header in {path}")
            else:
                current_parts.append(line)
        emit()

    if not labels:
        raise ValueError(f"FASTA contains no records: {path}")
    return labels, sequences


def read_gene_list(path: Path) -> list[str]:
    frame = pd.read_parquet(path, columns=["gene_key"])
    values = frame["gene_key"]
    if values.isna().any():
        raise ValueError(f"gene list contains null gene_key values: {path}")
    labels = [str(value).strip() for value in values.tolist()]
    if any(not label for label in labels):
        raise ValueError(f"gene list contains an empty gene_key: {path}")
    if len(labels) != len(set(labels)):
        raise ValueError(f"gene list contains duplicate gene_key values: {path}")
    if not labels:
        raise ValueError(f"gene list is empty: {path}")
    return labels


def select_common_genes(
    gene_list: Iterable[str], fasta_labels: Iterable[str],
) -> list[str]:
    fasta_set = set(fasta_labels)
    return [gene_key for gene_key in gene_list if gene_key in fasta_set]


def _cache_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{index}" for index in range(FEATURE_DIMENSIONS)]


def _read_embedding_cache(path: Path, prefix: str) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    frame = pd.read_parquet(path)
    value_columns = _cache_columns(prefix)
    required = ["gene_key", "sequence_sha256", *value_columns]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"cache {path} is missing columns: {missing}")
    frame = frame.loc[:, required].copy()
    frame["gene_key"] = frame["gene_key"].astype(str)
    frame["sequence_sha256"] = frame["sequence_sha256"].astype(str)
    if frame["gene_key"].duplicated().any():
        raise ValueError(f"cache contains duplicate gene_key values: {path}")
    values = frame.loc[:, value_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"cache contains non-finite values: {path}")
    return frame


def _atomic_to_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_embedding_cache(
    path: Path,
    existing: pd.DataFrame | None,
    labels: list[str],
    sequences: list[str],
    embeddings: np.ndarray,
    prefix: str,
    *,
    sort_by_gene: bool = True,
) -> pd.DataFrame:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.shape != (len(labels), FEATURE_DIMENSIONS):
        raise ValueError(f"expected {FEATURE_DIMENSIONS}-dimensional embeddings, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("refusing to cache non-finite embeddings")
    fresh = pd.DataFrame(values, columns=_cache_columns(prefix))
    fresh.insert(0, "sequence_sha256", [_sequence_sha256(seq) for seq in sequences])
    fresh.insert(0, "gene_key", labels)
    combined = pd.concat([existing, fresh], ignore_index=True) if existing is not None else fresh
    combined = combined.drop_duplicates("gene_key", keep="last")
    if sort_by_gene:
        combined = combined.sort_values("gene_key", kind="mergesort")
    combined = combined.reset_index(drop=True)
    _atomic_to_parquet(combined, path)
    return combined


def _cached_embeddings(
    cache: pd.DataFrame | None,
    labels: list[str],
    sequences: list[str],
    prefix: str,
) -> tuple[np.ndarray, list[int]]:
    values = np.empty((len(labels), FEATURE_DIMENSIONS), dtype=np.float32)
    missing: list[int] = []
    by_gene = cache.set_index("gene_key") if cache is not None else None
    value_columns = _cache_columns(prefix)
    for index, (gene_key, sequence) in enumerate(zip(labels, sequences)):
        if by_gene is None or gene_key not in by_gene.index:
            missing.append(index)
            continue
        row = by_gene.loc[gene_key]
        if row["sequence_sha256"] != _sequence_sha256(sequence):
            missing.append(index)
            continue
        values[index] = row.loc[value_columns].to_numpy(dtype=np.float32)
    return values, missing


def _require_finite_matrix(values: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIMENSIONS:
        raise ValueError(f"{name} must have shape (n, {FEATURE_DIMENSIONS}), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def esmc_encode(model: Any, sequences: list[str], device: str, token_budget: int = 2048) -> Any:
    import torch

    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    pad_token_id = model.tokenizer.pad_token_id
    outputs: list[torch.Tensor] = []

    def flush(batch: list[str]) -> torch.Tensor:
        input_ids = model._tokenize(batch).long().to(device)
        with torch.no_grad():
            output = model(input_ids)
            token_embeddings = output.embeddings
            mask = input_ids != pad_token_id
            token_embeddings = token_embeddings.masked_fill(~mask.unsqueeze(-1), 0.0)
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
            return (token_embeddings.sum(dim=1) / counts).float().cpu()

    batch: list[str] = []
    token_count = 0
    for sequence in sequences:
        if batch and token_count + len(sequence) > token_budget:
            outputs.append(flush(batch))
            batch, token_count = [], 0
        batch.append(sequence)
        token_count += len(sequence)
    if batch:
        outputs.append(flush(batch))
    return torch.cat(outputs, dim=0)


def load_esmc(device: str) -> Any:
    if not ESMC_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"ESM-C checkpoint is absent: {ESMC_WEIGHTS}. "
            "Materialize it with acquire_assets.py --include-esmc before inference."
        )
    os.environ["INFRA_PROVIDER"] = "local"
    import torch
    from esm.pretrained import ESMC_600M_202412

    os.chdir(ESMC_RUNTIME_ROOT)
    model = ESMC_600M_202412(device=device, use_flash_attn=False).eval()
    return model.to(device, dtype=torch.bfloat16)


def load_or_compute_esm(
    labels: list[str],
    sequences: list[str],
    cache_path: Path,
    device: str,
    token_budget: int,
    cache_chunk_genes: int,
) -> np.ndarray:
    if cache_chunk_genes <= 0:
        raise ValueError("cache_chunk_genes must be positive")
    cache = _read_embedding_cache(cache_path, "esm")
    values, missing = _cached_embeddings(cache, labels, sequences, "esm")
    if not missing:
        print(f"ESM-C cache: {cache_path} ({len(labels)} rows reused)")
        return values

    import torch

    print(f"ESM-C cache: computing {len(missing)} missing rows")
    model = load_esmc(device)
    try:
        for start in range(0, len(missing), cache_chunk_genes):
            chunk_indices = missing[start : start + cache_chunk_genes]
            chunk_labels = [labels[index] for index in chunk_indices]
            chunk_sequences = [sequences[index] for index in chunk_indices]
            computed = esmc_encode(model, chunk_sequences, device, token_budget)
            computed_np = _require_finite_matrix(
                computed.detach().cpu().numpy(), name="ESM-C embeddings"
            )
            if computed_np.shape[0] != len(chunk_labels):
                raise ValueError(
                    f"ESM-C returned {computed_np.shape[0]} rows for "
                    f"{len(chunk_labels)} sequences"
                )
            for row_index, cache_index in enumerate(chunk_indices):
                values[cache_index] = computed_np[row_index]
            cache = _write_embedding_cache(
                cache_path,
                cache,
                chunk_labels,
                chunk_sequences,
                computed_np,
                "esm",
            )
            completed = min(start + len(chunk_indices), len(missing))
            print(f"ESM-C cache: persisted {completed}/{len(missing)} newly computed rows")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return values


def _valid_block(
    path: Path,
    labels: list[str],
    sequences: list[str],
) -> np.ndarray | None:
    frame = _read_embedding_cache(path, "plm")
    if frame is None:
        return None
    expected_hashes = [_sequence_sha256(sequence) for sequence in sequences]
    actual_labels = frame["gene_key"].tolist()
    actual_hashes = frame["sequence_sha256"].tolist()
    if actual_labels != labels or actual_hashes != expected_hashes:
        return None
    return frame.loc[:, _cache_columns("plm")].to_numpy(dtype=np.float32)


def _write_block(
    path: Path,
    labels: list[str],
    sequences: list[str],
    values: np.ndarray,
) -> None:
    _write_embedding_cache(
        path, None, labels, sequences, values, "plm", sort_by_gene=False
    )


def contextualize_in_blocks(
    labels: list[str],
    sequences: list[str],
    esm_embeddings: np.ndarray,
    block_size: int,
    block_cache_dir: Path,
    contextualize: Callable[[np.ndarray], np.ndarray],
    cache_prefix: str = "",
) -> np.ndarray:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    esm_values = _require_finite_matrix(esm_embeddings, name="ESM-C embeddings")
    if len(esm_values) != len(labels) or len(sequences) != len(labels):
        raise ValueError("labels, sequences, and ESM-C embeddings must have matching lengths")

    block_cache_dir.mkdir(parents=True, exist_ok=True)
    parts: list[np.ndarray] = []
    for start in range(0, len(labels), block_size):
        end = min(start + block_size, len(labels))
        block_path = block_cache_dir / f"{cache_prefix}block_{start:08d}_{end:08d}.parquet"
        block_labels = labels[start:end]
        block_sequences = sequences[start:end]
        cached = _valid_block(block_path, block_labels, block_sequences)
        if cached is not None:
            print(f"ProteomeLM block {start}:{end}: reused {block_path}")
            parts.append(cached)
            continue
        computed = _require_finite_matrix(
            contextualize(esm_values[start:end]), name=f"ProteomeLM block {start}:{end}"
        )
        if computed.shape[0] != end - start:
            raise ValueError(
                f"ProteomeLM block {start}:{end} returned {computed.shape[0]} rows"
            )
        _write_block(block_path, block_labels, block_sequences, computed)
        print(f"ProteomeLM block {start}:{end}: wrote {block_path}")
        parts.append(computed)
    return np.concatenate(parts, axis=0)


def _write_final_features(path: Path, labels: list[str], values: np.ndarray) -> None:
    matrix = _require_finite_matrix(values, name="ProteomeLM features")
    if matrix.shape[0] != len(labels) or len(labels) != len(set(labels)):
        raise ValueError("final features must have one unique row per selected gene_key")
    frame = pd.DataFrame(matrix, columns=_cache_columns("plm"))
    frame.insert(0, "gene_key", labels)
    if frame.shape != (len(labels), FEATURE_DIMENSIONS + 1):
        raise ValueError(f"unexpected final feature shape: {frame.shape}")
    _atomic_to_parquet(frame, path)
    print(f"wrote {path}: {len(labels)} genes x {FEATURE_DIMENSIONS} dims")


def _contextualizer(model: Any, device: str, layer: int) -> Callable[[np.ndarray], np.ndarray]:
    import torch

    def run(block: np.ndarray) -> np.ndarray:
        inputs_embeds = torch.from_numpy(block).unsqueeze(0).to(
            device, dtype=getattr(model, "dtype", torch.bfloat16)
        )
        with torch.no_grad():
            outputs = model(
                inputs_embeds=inputs_embeds,
                group_embeds=None,
                output_hidden_states=True,
                output_attentions=False,
            )
        hidden_states = outputs.hidden_states
        if hidden_states is None or layer >= len(hidden_states):
            raise ValueError(f"ProteomeLM did not return requested layer {layer}")
        return hidden_states[layer].squeeze(0).float().cpu().numpy()

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proteins", type=Path, required=True, help="gzip or plain protein FASTA")
    parser.add_argument("--gene-list", type=Path, required=True, help="method-local Parquet with gene_key")
    parser.add_argument("--proteomelm-dir", type=Path, required=True, help="method-local ProteomeLM-L weights directory")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True, help="method-local features.parquet")
    parser.add_argument("--esm-cache", type=Path, required=True, help="method-local ESM-C base embedding cache")
    parser.add_argument("--block-cache-dir", type=Path, required=True, help="method-local ProteomeLM block cache directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--esm-block",
        type=int,
        default=DEFAULT_ESM_BLOCK,
        help="ProteomeLM block size; default 512 matches max_position_embeddings=512 in the public model config",
    )
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--esm-cache-chunk-genes", type=int, default=DEFAULT_ESM_CACHE_CHUNK_GENES)
    args = parser.parse_args()

    if args.esm_block <= 0 or args.esm_block > DEFAULT_ESM_BLOCK:
        raise ValueError("--esm-block must be in the range 1..512 for ProteomeLM-L")
    if args.token_budget <= 0:
        raise ValueError("--token-budget must be positive")
    if args.esm_cache_chunk_genes <= 0:
        raise ValueError("--esm-cache-chunk-genes must be positive")
    if args.layer < 0:
        raise ValueError("--layer must be non-negative")
    proteins = _method_path(args.proteins, "protein FASTA")
    gene_list_path = _method_path(args.gene_list, "gene list")
    model_directory = _method_path(args.proteomelm_dir, "ProteomeLM-L directory")
    output_path = _method_path(args.out, "output")
    esm_cache_path = _method_path(args.esm_cache, "ESM-C cache")
    block_cache_dir = _method_path(args.block_cache_dir, "ProteomeLM block cache")
    if not proteins.is_file():
        raise FileNotFoundError(proteins)
    if not gene_list_path.is_file():
        raise FileNotFoundError(gene_list_path)
    if not model_directory.is_dir():
        raise FileNotFoundError(model_directory)

    requested = read_gene_list(gene_list_path)
    fasta_labels, fasta_sequences = read_fasta(proteins)
    sequence_by_gene = dict(zip(fasta_labels, fasta_sequences))
    labels = select_common_genes(requested, fasta_labels)
    if not labels:
        raise ValueError("gene list and protein FASTA have no common gene_key values")
    missing = len(requested) - len(labels)
    if missing:
        print(f"gene-list/FASTA intersection: {len(labels)} genes ({missing} gene-list rows absent from FASTA)")
    sequences = [sequence_by_gene[gene_key] for gene_key in labels]
    print(f"proteins: {len(labels)} unique common genes")

    esm_embeddings = load_or_compute_esm(
        labels,
        sequences,
        esm_cache_path,
        args.device,
        args.token_budget,
        args.esm_cache_chunk_genes,
    )

    model: Any | None = None

    def contextualize(block: np.ndarray) -> np.ndarray:
        nonlocal model
        if model is None:
            import torch
            from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM

            model = (
                ProteomeLMForMaskedLM.from_pretrained(str(model_directory), local_files_only=True)
                .to(args.device, dtype=torch.bfloat16)
                .eval()
            )
        return _contextualizer(model, args.device, args.layer)(block)

    contextualized = contextualize_in_blocks(
        labels,
        sequences,
        esm_embeddings,
        args.esm_block,
        block_cache_dir,
        contextualize,
        cache_prefix=f"layer{args.layer}_",
    )
    if model is not None:
        del model
    _write_final_features(output_path, labels, contextualized)


if __name__ == "__main__":
    main()
