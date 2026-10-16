#!/usr/bin/env python3
"""Embed coding sequences with Evo2 7B base for EssTFNet.
Run from methods/esstfnet: python code/embed_dna.py --dataset <task> --gene-list support_gene_list.csv --device cuda:0
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.lib import ArrowInvalid


METHOD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_ROOT.parents[1] / "src"))
from egbench.contracts import COHORT_IDS
MODEL_NAME = "evo2_7b_base"
LAYER_NAME = "blocks.28.mlp.l3"
EMBEDDING_DIM = 4096
MODEL_CONTEXT_NT = 8192
DEFAULT_DEVICE = "cuda:0"
POOLING_CONTRACT = (
    "complete_cds_nonoverlap_nt_windows_official_tokenizer_"
    "all_layer_tokens_sum_divide_total_token_count_no_special_tokens"
)
DEFAULT_CACHE_DIR = METHOD_ROOT / "feature" / "source" / "generated" / "evo2_7b_base_mean_pool"
DEFAULT_MODEL_CACHE_DIR = METHOD_ROOT / "feature" / "source" / "model_assets" / "huggingface"
MODEL_CHECKPOINT_PATH = METHOD_ROOT / "feature" / "source" / "model_assets" / "evo2_7b_base.pt"
EMBEDDING_TYPE = pa.list_(pa.float16(), EMBEDDING_DIM)
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("gene_key", pa.string()),
        pa.field("embedding", EMBEDDING_TYPE),
    ]
)


class CacheConflictError(ValueError):
    pass


CACHE_SCHEMA = pa.schema(
    [
        pa.field("gene_key", pa.string()),
        pa.field("dataset", pa.string()),
        pa.field("cds_sha256", pa.string()),
        pa.field("model_name", pa.string()),
        pa.field("layer", pa.string()),
        pa.field("window_size", pa.int32()),
        pa.field("pooling_contract", pa.string()),
        pa.field("embedding", EMBEDDING_TYPE),
    ]
)


def _resolve_method_path(path: Path, label: str) -> Path:
    resolved = (METHOD_ROOT / path if not path.is_absolute() else path).expanduser().resolve()
    if not resolved.is_relative_to(METHOD_ROOT):
        raise ValueError(f"{label} must be inside {METHOD_ROOT}: {resolved}")
    return resolved


def _relative_method_path(path: Path) -> str:
    return path.relative_to(METHOD_ROOT).as_posix()


def _open_fasta(path: Path) -> Iterator[str]:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def read_fasta(path: Path) -> dict[str, str]:

    records: dict[str, str] = {}
    current_key: str | None = None
    current_parts: list[str] = []

    def emit() -> None:
        nonlocal current_key, current_parts
        if current_key is None:
            return
        sequence = "".join("".join(part.split()) for part in current_parts).upper()
        if not sequence:
            raise ValueError(f"FASTA record has an empty CDS sequence: {current_key}")
        if current_key in records:
            raise ValueError(f"FASTA contains duplicate gene_key: {current_key}")
        records[current_key] = sequence
        current_key = None
        current_parts = []

    with _open_fasta(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                emit()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"FASTA has an empty header at {path}:{line_number}")
                token = header.split()[0]
                current_key = token.split("|", 1)[0].strip()
                if not current_key:
                    raise ValueError(f"FASTA has an empty gene_key at {path}:{line_number}")
            elif current_key is None:
                raise ValueError(f"FASTA sequence appears before a header at {path}:{line_number}")
            else:
                current_parts.append(line)
        emit()

    if not records:
        raise ValueError(f"FASTA contains no records: {path}")
    return records


def read_gene_list(path: Path) -> list[str]:

    path = _resolve_method_path(path, "gene list")
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        frame = (
            pd.read_csv(path, dtype="string", keep_default_na=False)
            if path.suffix.lower() == ".csv"
            else pd.read_parquet(path, columns=["gene_key"])
        )
    except Exception as error:
        raise ValueError(f"unable to read gene_key from {path}") from error
    if "gene_key" not in frame.columns:
        raise ValueError(f"{path} lacks gene_key")
    if frame["gene_key"].isna().any():
        raise ValueError(f"{path} contains null gene_key values")

    keys = sorted({str(value).strip() for value in frame["gene_key"].tolist()})
    if any(not key for key in keys):
        raise ValueError(f"{path} contains an empty gene_key")
    if not keys:
        raise ValueError(f"{path} contains no gene_key values")
    return sorted(keys)


def _default_fasta(cohort_id: str) -> Path:
    if cohort_id not in COHORT_IDS:
        raise ValueError(f"unsupported cohort_id: {cohort_id}")
    return METHOD_ROOT / "feature" / "source" / cohort_id / "cds_sequences.fa.gz"


def _default_output(cohort_id: str) -> Path:
    if cohort_id not in COHORT_IDS:
        raise ValueError(f"unsupported cohort_id: {cohort_id}")
    return (
        METHOD_ROOT / "feature" / "source" / cohort_id / f"{MODEL_NAME}_embeddings.parquet"
    )


def _validate_gene_list_coverage(keys: list[str], sequences: Mapping[str, str], fasta_path: Path) -> None:
    missing = sorted(set(keys).difference(sequences))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"gene list is not completely covered by {fasta_path}; "
            f"missing {len(missing)} gene_key values, including {preview}"
        )


def validate_default_dataset_overlap(dataset_name: str, fasta_path: Path, sequences: Mapping[str, str]) -> None:

    default_path = _default_fasta(dataset_name).resolve()
    if fasta_path.resolve() != default_path:
        return
    mismatches: list[str] = []
    for other_name in COHORT_IDS:
        if other_name == dataset_name:
            continue
        other_path = METHOD_ROOT / "feature" / "source" / other_name / "cds_sequences.fa.gz"
        if not other_path.is_file():
            continue
        other_sequences = read_fasta(other_path)
        mismatches.extend(
            gene_key
            for gene_key in set(sequences).intersection(other_sequences)
            if sequences[gene_key] != other_sequences[gene_key]
        )
    mismatches = sorted(set(mismatches))
    if mismatches:
        preview = ", ".join(mismatches[:5])
        raise ValueError(
            f"DEG/DepMap CDS sequences disagree for {len(mismatches)} overlapping gene_key values, "
            f"including {preview}"
        )


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, gene_key: str) -> Path:

    filename = hashlib.sha256(gene_key.encode("utf-8")).hexdigest() + ".parquet"
    return cache_dir / filename


def _embedding_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float16)
    if array.shape != (EMBEDDING_DIM,):
        raise ValueError(f"{name} must have shape ({EMBEDDING_DIM},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _atomic_write_table(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        pq.write_table(table, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_cache(
    path: Path,
    dataset_name: str,
    gene_key: str,
    cds_sha256: str,
    window_size: int,
) -> np.ndarray | None:

    if not path.is_file():
        return None
    try:
        table = pq.read_table(path)
        required = set(CACHE_SCHEMA.names)
        if not required.issubset(table.column_names) or table.num_rows != 1:
            return None
        row = {name: table[name][0].as_py() for name in CACHE_SCHEMA.names}
        if row["dataset"] != dataset_name and row["cds_sha256"] != cds_sha256:
            raise CacheConflictError(
                f"shared Evo2 cache has a different CDS for {gene_key} from dataset "
                f"{row['dataset']} while processing {dataset_name}"
            )
        if (
            row["gene_key"] != gene_key
            or row["cds_sha256"] != cds_sha256
            or row["model_name"] != MODEL_NAME
            or row["layer"] != LAYER_NAME
            or int(row["window_size"]) != window_size
            or row["pooling_contract"] != POOLING_CONTRACT
        ):
            return None
        return _embedding_array(row["embedding"], f"cache {path}")
    except CacheConflictError:
        raise
    except (OSError, ArrowInvalid, ValueError, TypeError, KeyError, IndexError):
        return None


def _write_cache(
    path: Path,
    dataset_name: str,
    gene_key: str,
    cds_sha256: str,
    window_size: int,
    embedding: np.ndarray,
) -> None:
    vector = _embedding_array(embedding, f"embedding for {gene_key}")
    table = pa.Table.from_arrays(
        [
            pa.array([gene_key], type=pa.string()),
            pa.array([dataset_name], type=pa.string()),
            pa.array([cds_sha256], type=pa.string()),
            pa.array([MODEL_NAME], type=pa.string()),
            pa.array([LAYER_NAME], type=pa.string()),
            pa.array([window_size], type=pa.int32()),
            pa.array([POOLING_CONTRACT], type=pa.string()),
            pa.array([vector.tolist()], type=EMBEDDING_TYPE),
        ],
        schema=CACHE_SCHEMA,
    )
    _atomic_write_table(table, path)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    candidate = value
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "float"):
        candidate = candidate.float()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy"):
        candidate = candidate.numpy()
    return np.asarray(candidate)


def _layer_token_matrix(embeddings: Any) -> np.ndarray:
    if not isinstance(embeddings, Mapping):
        raise ValueError("Evo2 embeddings must be a mapping keyed by layer name")
    if LAYER_NAME not in embeddings:
        raise ValueError(f"Evo2 output lacks requested layer {LAYER_NAME}")
    matrix = _tensor_to_numpy(embeddings[LAYER_NAME])
    if matrix.ndim == 3:
        if matrix.shape[0] != 1:
            raise ValueError(f"expected one Evo2 input sequence, got shape {matrix.shape}")
        matrix = matrix[0]
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIM or matrix.shape[0] == 0:
        raise ValueError(
            f"{LAYER_NAME} must have shape (tokens, {EMBEDDING_DIM}), got {matrix.shape}"
        )
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{LAYER_NAME} contains non-finite values")
    return matrix


def _layer_token_sum(embeddings: Any, torch_module: Any) -> tuple[np.ndarray, int]:

    if not isinstance(embeddings, Mapping):
        raise ValueError("Evo2 embeddings must be a mapping keyed by layer name")
    if LAYER_NAME not in embeddings:
        raise ValueError(f"Evo2 output lacks requested layer {LAYER_NAME}")

    candidate = embeddings[LAYER_NAME]
    is_tensor = getattr(torch_module, "is_tensor", None)
    if callable(is_tensor) and is_tensor(candidate):
        matrix = candidate
        if matrix.ndim == 3:
            if matrix.shape[0] != 1:
                raise ValueError(f"expected one Evo2 input sequence, got shape {tuple(matrix.shape)}")
            matrix = matrix[0]
        if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIM or matrix.shape[0] == 0:
            raise ValueError(
                f"{LAYER_NAME} must have shape (tokens, {EMBEDDING_DIM}), "
                f"got {tuple(matrix.shape)}"
            )
        token_count = int(matrix.shape[0])
        vector = matrix.float().sum(dim=0)
        if not bool(torch_module.isfinite(vector).all().item()):
            raise ValueError(f"{LAYER_NAME} contains non-finite values")
        return np.asarray(vector.cpu().numpy(), dtype=np.float64), token_count

    matrix = _layer_token_matrix(embeddings)
    return matrix.sum(axis=0), int(matrix.shape[0])


def embed_gene(
    model: Any,
    sequence: str,
    device: str,
    window_size: int,
    *,
    torch_module: Any | None = None,
) -> np.ndarray:

    if not sequence:
        raise ValueError("cannot embed an empty CDS sequence")
    if not 0 < window_size <= MODEL_CONTEXT_NT:
        raise ValueError(f"window_size must be in [1, {MODEL_CONTEXT_NT}], got {window_size}")
    if torch_module is None:
        import torch as torch_module

    total = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    token_count = 0
    for start in range(0, len(sequence), window_size):
        window = sequence[start : start + window_size]
        token_ids = model.tokenizer.tokenize(window)
        if len(token_ids) == 0:
            raise ValueError(f"official tokenizer returned no tokens for CDS window starting at {start}")
        input_ids = torch_module.tensor(token_ids, dtype=torch_module.int).unsqueeze(0).to(device)
        with torch_module.no_grad():
            _, embeddings = model(
                input_ids,
                return_embeddings=True,
                layer_names=[LAYER_NAME],
            )
        window_sum, window_token_count = _layer_token_sum(embeddings, torch_module)
        total += window_sum
        token_count += window_token_count

    if token_count == 0:
        raise ValueError("Evo2 produced no token vectors for the CDS")
    return _embedding_array(total / token_count, "pooled embedding")


def _require_cuda(device: str) -> None:
    if not device.startswith("cuda"):
        raise RuntimeError(
            f"Evo2-7B-base formal embedding requires CUDA, got --device {device!r}"
        )
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Evo2-7B-base formal embedding requires a CUDA GPU, but torch.cuda.is_available() is false"
            )
        if ":" in device:
            device_index = int(device.split(":", 1)[1])
            if device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible"
                )
    except ModuleNotFoundError as error:
        raise RuntimeError("Evo2-7B-base formal embedding requires PyTorch with CUDA support") from error
    except ValueError as error:
        raise RuntimeError(f"invalid CUDA device {device!r}") from error


def load_evo2_model(device: str) -> Any:

    _require_cuda(device)
    if not MODEL_CHECKPOINT_PATH.is_file() or MODEL_CHECKPOINT_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            f"official Evo2-7B-base checkpoint is absent: {MODEL_CHECKPOINT_PATH}"
        )
    DEFAULT_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(DEFAULT_MODEL_CACHE_DIR)
    from evo2 import Evo2

    model = Evo2("evo2_7b_base", local_path=str(MODEL_CHECKPOINT_PATH))
    if hasattr(model, "model") and hasattr(model.model, "eval"):
        model.model.eval()
    return model


def _output_matches(path: Path, keys: list[str], values: np.ndarray) -> bool:
    if not path.is_file():
        return False
    try:
        table = pq.read_table(path)
        if not table.schema.equals(OUTPUT_SCHEMA) or table.num_rows != len(keys):
            return False
        output_keys = [str(value) for value in table["gene_key"].to_pylist()]
        if output_keys != keys:
            return False
        output_values = np.asarray(table["embedding"].to_pylist(), dtype=np.float16)
        return output_values.shape == values.shape and np.array_equal(output_values, values)
    except (OSError, ValueError, TypeError, ArrowInvalid):
        return False


def _write_output(path: Path, keys: list[str], values: np.ndarray) -> None:
    matrix = np.asarray(values, dtype=np.float16)
    if matrix.shape != (len(keys), EMBEDDING_DIM):
        raise ValueError(f"output embeddings must have shape ({len(keys)}, {EMBEDDING_DIM}), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("refusing to write non-finite output embeddings")
    table = pa.Table.from_arrays(
        [
            pa.array(keys, type=pa.string()),
            pa.array(matrix.tolist(), type=EMBEDDING_TYPE),
        ],
        schema=OUTPUT_SCHEMA,
    )
    _atomic_write_table(table, path)


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.manifest.json")


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def embed_dataset(
    dataset_name: str,
    gene_list: Path,
    *,
    fasta: Path | None = None,
    output: Path | None = None,
    cache_dir: Path | None = None,
    device: str = DEFAULT_DEVICE,
    window_size: int = MODEL_CONTEXT_NT,
    resume: bool = False,
    model: Any | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:

    if dataset_name not in COHORT_IDS:
        raise ValueError(f"unsupported cohort_id: {dataset_name}")
    if not 0 < window_size <= MODEL_CONTEXT_NT:
        raise ValueError(f"window_size must be in [1, {MODEL_CONTEXT_NT}], got {window_size}")

    gene_list_path = _resolve_method_path(gene_list, "gene list")
    fasta_path = _resolve_method_path(fasta or _default_fasta(dataset_name), "FASTA")
    output_path = _resolve_method_path(output or _default_output(dataset_name), "output")
    cache_path = _resolve_method_path(cache_dir or DEFAULT_CACHE_DIR, "cache directory")
    if not fasta_path.is_file():
        raise FileNotFoundError(fasta_path)

    keys = read_gene_list(gene_list_path)
    sequences = read_fasta(fasta_path)
    _validate_gene_list_coverage(keys, sequences, fasta_path)
    validate_default_dataset_overlap(dataset_name, fasta_path, sequences)

    values = np.empty((len(keys), EMBEDDING_DIM), dtype=np.float16)
    pending: list[tuple[int, str, str, Path]] = []
    reused_count = 0
    for index, gene_key in enumerate(keys):
        sequence = sequences[gene_key]
        cds_sha256 = _sequence_sha256(sequence)
        gene_cache_path = _cache_path(cache_path, gene_key)
        cached = _read_cache(gene_cache_path, dataset_name, gene_key, cds_sha256, window_size)
        if cached is None:
            pending.append((index, gene_key, sequence, gene_cache_path))
        else:
            values[index] = cached
            reused_count += 1

    computed_count = 0
    if pending:
        if model is None:
            model = load_evo2_model(device)
        for index, gene_key, sequence, gene_cache_path in pending:
            vector = embed_gene(
                model,
                sequence,
                device,
                window_size,
                torch_module=torch_module,
            )
            _write_cache(
                gene_cache_path,
                dataset_name,
                gene_key,
                _sequence_sha256(sequence),
                window_size,
                vector,
            )
            values[index] = vector
            computed_count += 1

    manifest_path = _manifest_path(output_path)
    resumed_output = resume and _output_matches(output_path, keys, values)
    if not resumed_output:
        _write_output(output_path, keys, values)

    manifest = {
        "dataset": dataset_name,
        "rows": len(keys),
        "embedding_dim": EMBEDDING_DIM,
        "model": MODEL_NAME,
        "layer": LAYER_NAME,
        "window_size": window_size,
        "pooling_contract": POOLING_CONTRACT,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": device,
        "model_cache_dir": _relative_method_path(DEFAULT_MODEL_CACHE_DIR),
        "model_checkpoint": _relative_method_path(MODEL_CHECKPOINT_PATH),
        "complete_cds": True,
        "special_tokens": False,
        "computed": computed_count,
        "reused": reused_count,
        "resumed_output": resumed_output,
        "paths": {
            "fasta": _relative_method_path(fasta_path),
            "gene_list": _relative_method_path(gene_list_path),
            "output": _relative_method_path(output_path),
            "cache_dir": _relative_method_path(cache_path),
            "manifest": _relative_method_path(manifest_path),
        },
    }
    _write_manifest(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=COHORT_IDS, required=True)
    parser.add_argument("--gene-list", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, help="optional method-local gzip/plain CDS FASTA")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--window-size", type=int, default=MODEL_CONTEXT_NT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        _require_cuda(args.device)
        manifest = embed_dataset(
            args.dataset,
            args.gene_list,
            fasta=args.fasta,
            output=args.output,
            cache_dir=args.cache_dir,
            device=args.device,
            window_size=args.window_size,
            resume=args.resume,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(_relative_method_path(_manifest_path(_resolve_method_path(args.output or _default_output(args.dataset), "output"))))
    del manifest


if __name__ == "__main__":
    main()
