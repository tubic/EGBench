"""Fold and inference inputs of the EssTFNet branches.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]

from local_dataset import select_sequence_pairs

from adapter import DNA_ALPHABET, PROTEIN_ALPHABET

sys.path.insert(0, str(METHOD_ROOT.parents[1] / "src"))
from egbench.contracts import ContractError
from egbench.data import feature_directory, folds_directory


PAD_TOKEN = 0
_EXAMPLE_LIMIT = 10


@dataclass(frozen=True)
class Partition:
    gene_keys: list[str]
    dna_tokens: torch.Tensor
    protein_tokens: torch.Tensor
    features: torch.Tensor
    language_model_embedding: torch.Tensor
    labels: np.ndarray | None


def tokenize(sequence: str, max_len: int, alphabet: str) -> np.ndarray:

    lookup = {residue: position + 1 for position, residue in enumerate(alphabet)}
    tokens = np.full(max_len, PAD_TOKEN, dtype=np.uint8)
    for position, residue in enumerate(sequence[:max_len]):
        tokens[position] = lookup[residue]
    return tokens


def _array_column(path: Path, column: str) -> dict[str, np.ndarray]:
    frame = pd.read_parquet(path)
    if list(frame.columns) != ["gene_key", column]:
        raise ValueError(f"{path} must hold gene_key plus {column}")
    return dict(zip(frame["gene_key"].astype(str), frame[column]))


def _matrix(lookup: dict[str, np.ndarray], keys: list[str], label: str) -> np.ndarray:
    missing = [key for key in keys if key not in lookup]
    if missing:
        raise ValueError(f"{label} lacks {len(missing)} fold genes")
    return np.stack([np.asarray(lookup[key], dtype=np.float32) for key in keys])


def standardize(
    matrices: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:

    mean = matrices["train"].mean(axis=0, dtype=np.float64)
    scale = matrices["train"].std(axis=0, dtype=np.float64)
    scale[scale == 0.0] = 1.0
    standardized = {
        role: ((matrix - mean) / scale).astype(np.float32)
        for role, matrix in matrices.items()
    }
    return standardized, {"mean": mean, "scale": scale}


def _sequence_pairs(keys: list[str]) -> dict[str, dict[str, str]]:
    sequences = select_sequence_pairs()
    absent = sorted(set(keys).difference(sequences))
    if absent:
        raise ContractError(
            f"EssTFNet scores a gene from its coding and peptide sequence and "
            f"{len(absent)} of the {len(keys)} requested genes have no valid "
            f"pair: {absent[:_EXAMPLE_LIMIT]}"
        )
    return sequences


def _token_tensors(
    keys: list[str],
    sequences: dict[str, dict[str, str]],
    max_len: int,
    protein_max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dna = np.stack(
        [tokenize(sequences[key]["cds"], max_len, DNA_ALPHABET) for key in keys]
    )
    protein = np.stack(
        [
            tokenize(sequences[key]["protein"], protein_max_len, PROTEIN_ALPHABET)
            for key in keys
        ]
    )
    return torch.from_numpy(dna), torch.from_numpy(protein)


def load_fold(
    cohort_id: str,
    split_strategy: str,
    outer_fold: int,
    max_len: int,
    protein_max_len: int,
) -> tuple[dict[str, Partition], dict[str, np.ndarray]]:

    directory = feature_directory(
        METHOD_ROOT,
        cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )
    folds = folds_directory(
        cohort_id,
        split_strategy=split_strategy,
        root=METHOD_ROOT.parents[1],
    )
    context = pd.read_parquet(
        folds / "fold_genes" / f"fold_{outer_fold}_genes.parquet",
        columns=["gene_key", "role"],
    )
    labels = pd.read_parquet(folds / "training_labels" / f"fold_{outer_fold}.parquet")
    label_context = labels.loc[
        labels["split_strategy"].eq(split_strategy) & labels["outer_fold"].eq(outer_fold)
    ]
    if context.empty or label_context.empty:
        raise ValueError(
            f"no {split_strategy} rows for {cohort_id}/fold_{outer_fold} in {folds}"
        )
    label_lookup = dict(
        zip(label_context["gene_key"], label_context["label"].astype(int), strict=True)
    )
    physicochemical = _array_column(
        directory / "physicochemical_features.parquet", "features"
    )
    language_model = _array_column(directory / "evo2_embeddings.parquet", "embedding")
    roles = ("train", "validation", "test")
    role_keys = {
        role: context.loc[context["role"].eq(role), "gene_key"].astype(str).tolist()
        for role in roles
    }
    sequences = _sequence_pairs(
        [key for keys in role_keys.values() for key in keys]
    )
    embeddings, evo2_statistics = standardize(
        {
            role: _matrix(language_model, keys, "evo2_embeddings.parquet")
            for role, keys in role_keys.items()
        }
    )
    partitions = {}
    for role in roles:
        keys = role_keys[role]
        dna_tokens, protein_tokens = _token_tensors(
            keys, sequences, max_len, protein_max_len
        )
        partitions[role] = Partition(
            gene_keys=keys,
            dna_tokens=dna_tokens,
            protein_tokens=protein_tokens,
            features=torch.from_numpy(
                _matrix(physicochemical, keys, "physicochemical_features.parquet")
            ),
            language_model_embedding=torch.from_numpy(embeddings[role]),
            labels=None
            if role == "test"
            else np.asarray([label_lookup[key] for key in keys], dtype=np.int64),
        )
    return partitions, evo2_statistics


def inference_partition(
    cohort_id: str,
    gene_keys: list[str],
    max_len: int,
    protein_max_len: int,
    *,
    physicochemical_state: Path,
    evo2_state: Path,
) -> Partition:

    from build_features import SOURCE_ROOT, _scan_evo2_rows, _scan_raw_rows

    keys = [str(key) for key in gene_keys]
    sequences = _sequence_pairs(keys)
    physicochemical = np.load(physicochemical_state)
    selected = np.asarray(physicochemical["selected_raw_indices"], dtype=np.int64)
    wanted = {key: index for index, key in enumerate(keys)}
    raw = SOURCE_ROOT / cohort_id / "raw_candidates_86151.parquet"
    evo2 = SOURCE_ROOT / cohort_id / "evo2_7b_base_embeddings.parquet"
    absent = [path for path in (raw, evo2) if not path.is_file()]
    if absent:
        raise ContractError(
            "the whole-genome EssTFNet assets inference reads are absent: "
            + ", ".join(str(path) for path in absent)
        )
    features = _scan_raw_rows(raw, wanted, selected).astype(np.float64)
    features -= np.asarray(physicochemical["mean"], dtype=np.float64)
    features /= np.asarray(physicochemical["scale"], dtype=np.float64)
    embeddings = _scan_evo2_rows(evo2, wanted).astype(np.float64)
    evo2_scaler = np.load(evo2_state)
    embeddings -= np.asarray(evo2_scaler["mean"], dtype=np.float64)
    embeddings /= np.asarray(evo2_scaler["scale"], dtype=np.float64)
    dna_tokens, protein_tokens = _token_tensors(
        keys, sequences, max_len, protein_max_len
    )
    return Partition(
        gene_keys=keys,
        dna_tokens=dna_tokens,
        protein_tokens=protein_tokens,
        features=torch.from_numpy(features.astype(np.float32)),
        language_model_embedding=torch.from_numpy(embeddings.astype(np.float32)),
        labels=None,
    )
