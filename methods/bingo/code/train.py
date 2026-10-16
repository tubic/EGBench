"""Train one BiNGO fold and score genes with a trained fold.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn
import torch

from acquire_assets import EXPECTED_SHA256
from contracts import (
    METHOD_ID,
    Partition,
    load_prepared_split,
    require_method_path,
    sha256_file,
)
from graph import (
    BINGO_MODEL_NAME,
    CACHE_SCHEMA,
    CONTACT_RETAIN_FRACTION,
    ESM_EMBEDDING_DIM,
    ESM_LAYER,
    MAX_PROTEIN_LENGTH,
    SEED,
    BingoGAT,
    TrainConfig,
    architecture_contract,
    fit,
    predict_scores,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from egbench.artifacts import (
    RunRecord,
    software_versions,
    write_model_card,
    write_predict_manifest,
    write_predictions,
    write_resource_usage,
    write_run_manifest,
)
from egbench.contracts import ContractError, read_model_card


DEVICE = "cuda:0"

TRAIN_CONFIG = TrainConfig()

WEIGHTS_DIRECTORY = "weights"
MODEL_FILE = "best_model.pt"

_EXAMPLE_LIMIT = 10


class GraphCacheDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        cache_dir: Path,
        partition: Partition,
        *,
        graph_root: Path | None = None,
    ) -> None:
        try:
            from torch_geometric.data import Data
        except ImportError as exc:
            raise RuntimeError("Bingo requires torch-geometric") from exc
        self._data_type = Data
        self.cache_dir = cache_dir.resolve()
        self.graph_root = self.cache_dir if graph_root is None else graph_root.resolve()
        manifest_path = self.cache_dir / "cache_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_files = manifest.get("model_files", {})
        try:
            self._esm_model_sha256 = str(model_files["esm_model"]["sha256"])
            self._esm_contact_sha256 = str(
                model_files["esm_contact_regression"]["sha256"]
            )
        except (KeyError, TypeError) as error:
            raise ValueError("cache manifest lacks exact ESM model hashes") from error
        index_path = self.cache_dir / "cache_index.parquet"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        index = pd.read_parquet(index_path)
        required = {
            "gene_key",
            "cache_file",
            "node_count",
            "directed_edge_count",
            "cache_sha256",
        }
        if missing := sorted(required.difference(index.columns)):
            raise ValueError(f"cache index lacks columns: {missing}")
        if index["gene_key"].astype(str).duplicated().any():
            raise ValueError("cache index has duplicate gene keys")
        lookup = index.assign(gene_key=index["gene_key"].astype(str)).set_index(
            "gene_key", verify_integrity=True
        )
        missing_genes = sorted(set(partition.gene_keys).difference(lookup.index))
        if missing_genes:
            raise ContractError(
                f"BiNGO scores a gene from its ESM-2 contact graph and "
                f"{len(missing_genes)} of the {len(partition.gene_keys)} "
                f"requested genes have none: "
                f"{missing_genes[:_EXAMPLE_LIMIT]}. The graphs "
                f"are built by the rebuild_command of method.yaml."
            )
        self.gene_keys = partition.gene_keys
        self.cluster_ids = partition.cluster_ids
        self.labels = partition.labels
        self._paths = tuple(
            self.graph_root / str(lookup.at[gene_key, "cache_file"])
            for gene_key in self.gene_keys
        )
        if missing_files := [str(path) for path in self._paths if not path.is_file()]:
            raise FileNotFoundError(
                f"Bingo cache files are missing: {missing_files[:3]}"
            )

    def __len__(self) -> int:
        return len(self.gene_keys)

    def __getitem__(self, index: int) -> Any:
        item = torch.load(self._paths[index], map_location="cpu", weights_only=False)
        if item.get("gene_key") != self.gene_keys[index]:
            raise ValueError("Bingo cache file/gene mapping mismatch")
        expected_metadata = {
            "cache_schema": CACHE_SCHEMA,
            "model_name": BINGO_MODEL_NAME,
            "esm_layer": ESM_LAYER,
            "contact_retain_fraction": CONTACT_RETAIN_FRACTION,
            "contact_direction": "largest_probabilities",
            "edge_directionality": "bidirectional",
            "esm_model_sha256": self._esm_model_sha256,
            "esm_contact_regression_sha256": self._esm_contact_sha256,
        }
        for key, expected in expected_metadata.items():
            if item.get(key) != expected:
                raise ValueError(f"Bingo graph cache has invalid {key}")
        if (
            not isinstance(item.get("x"), torch.Tensor)
            or item["x"].dtype != torch.float32
        ):
            raise ValueError("cached residue embeddings must be float32 tensors")
        x = item["x"]
        edge_index = torch.as_tensor(item["edge_index"], dtype=torch.long)
        if x.ndim != 2 or x.shape[1] != ESM_EMBEDDING_DIM:
            raise ValueError("cached residue embeddings have the wrong shape")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("cached contact edges have the wrong shape")
        data = self._data_type(x=x, edge_index=edge_index)
        if self.labels is not None:
            data.y = torch.tensor([self.labels[index]], dtype=torch.float32)
        return data


def _loader(dataset: GraphCacheDataset, *, shuffle: bool) -> Any:

    try:
        from torch_geometric.loader import DataLoader
    except ImportError as exc:
        raise RuntimeError("Bingo requires torch-geometric") from exc
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=TRAIN_CONFIG.batch_size,
        shuffle=shuffle,
        num_workers=TRAIN_CONFIG.num_workers,
        pin_memory=True,
        persistent_workers=TRAIN_CONFIG.num_workers > 0,
        generator=generator,
        drop_last=False,
    )


def _validate_full_cache(
    cache_dir: Path,
    expected_genes: Sequence[str],
    *,
    expected_shared_root: Path | None = None,
) -> dict[str, Any]:
    provenance_path = cache_dir / "cache_provenance.json"
    manifest_path = cache_dir / "cache_manifest.json"
    index_path = cache_dir / "cache_index.parquet"
    if not provenance_path.is_file() or not manifest_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            "Bingo cache provenance, manifest, and fold index are all required"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    required_provenance = {
        "parent_index",
        "parent_index_sha256",
        "cache_manifest",
        "cache_manifest_sha256",
        "fold_index_sha256",
    }
    if missing := sorted(required_provenance.difference(provenance)):
        raise ValueError(f"Bingo cache provenance lacks fields: {missing}")
    parent_index_path = (cache_dir / str(provenance["parent_index"])).resolve()
    source_manifest_path = (cache_dir / str(provenance["cache_manifest"])).resolve()
    shared_graph_root: Path | None = None
    if expected_shared_root is not None:
        shared_root = expected_shared_root.resolve()
        shared_graph_root = (shared_root / "graphs").resolve()
        expected_parent = shared_root / "cache_index.parquet"
        expected_manifest = shared_root / "cache_manifest.json"
        if parent_index_path != expected_parent:
            raise ValueError(
                "Bingo provenance does not reference the canonical shared parent index"
            )
        if source_manifest_path != expected_manifest:
            raise ValueError(
                "Bingo provenance does not reference the canonical shared cache manifest"
            )
    if not parent_index_path.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError("BiNGO shared parent index or source manifest is absent")
    if sha256_file(parent_index_path) != str(provenance["parent_index_sha256"]):
        raise ValueError("Bingo shared parent index hash mismatch")
    if sha256_file(source_manifest_path) != str(provenance["cache_manifest_sha256"]):
        raise ValueError("Bingo source cache manifest hash mismatch")
    if sha256_file(index_path) != str(provenance["fold_index_sha256"]):
        raise ValueError("Bingo fold cache index hash mismatch")
    if sha256_file(manifest_path) != str(provenance["cache_manifest_sha256"]):
        raise ValueError("Bingo fold cache manifest differs from source manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_values = {
        "cache_schema": CACHE_SCHEMA,
        "model_name": BINGO_MODEL_NAME,
        "maximum_protein_length": MAX_PROTEIN_LENGTH,
        "esm_inference_dtype": "float32",
        "cached_embedding_dtype": "float32",
        "contact_retain_fraction": CONTACT_RETAIN_FRACTION,
        "contact_direction": "largest_probabilities",
        "edge_directionality": "bidirectional",
        "complete": True,
    }
    for key, expected in required_values.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Bingo cache manifest {key}={manifest.get(key)!r}; expected {expected!r}"
            )
    expected_model_files = {
        "esm_model": f"{BINGO_MODEL_NAME}.pt",
        "esm_contact_regression": f"{BINGO_MODEL_NAME}-contact-regression.pt",
    }
    model_files = manifest.get("model_files")
    if not isinstance(model_files, dict):
        raise ValueError("Bingo cache manifest lacks model_files")
    for key, filename in expected_model_files.items():
        record = model_files.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"Bingo cache manifest lacks {key} identity")
        if record.get("filename") != filename:
            raise ValueError(f"Bingo cache manifest has invalid {key} filename")
        if record.get("sha256") != EXPECTED_SHA256[filename]:
            raise ValueError(f"Bingo cache manifest has invalid {key} sha256")
    if manifest.get("cache_index_sha256") != sha256_file(parent_index_path):
        raise ValueError("Bingo shared parent index does not match its manifest")
    index = pd.read_parquet(index_path)
    parent_index = pd.read_parquet(parent_index_path)
    for count_field in ("requested_gene_count", "cache_completed_count"):
        if manifest.get(count_field) != len(parent_index):
            raise ValueError(
                f"Bingo cache manifest {count_field} does not match parent rows"
            )
    if "gene_key" not in index or "gene_key" not in parent_index:
        raise ValueError("Bingo fold and parent indices require gene_key")
    if index["gene_key"].astype(str).duplicated().any():
        raise ValueError("Bingo fold cache index contains duplicate gene keys")
    if parent_index["gene_key"].astype(str).duplicated().any():
        raise ValueError("Bingo shared parent index contains duplicate gene keys")
    observed = set(index["gene_key"].astype(str))
    expected = set(map(str, expected_genes))
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise ValueError(
            "Bingo cache gene set mismatch: "
            f"{len(missing)} missing and {len(extra)} extra genes"
        )
    fold = index.assign(gene_key=index["gene_key"].astype(str)).set_index("gene_key")
    parent = parent_index.assign(
        gene_key=parent_index["gene_key"].astype(str)
    ).set_index("gene_key")
    missing_parent = sorted(observed.difference(parent.index))
    if missing_parent:
        raise ValueError(
            f"Bingo shared parent index lacks {len(missing_parent)} fold genes"
        )
    if set(fold.columns) != set(parent.columns):
        raise ValueError("Bingo fold and parent cache index columns differ")
    aligned_parent = parent.loc[fold.index]
    for column in fold.columns:
        if column == "cache_file":
            continue
        if not fold[column].equals(aligned_parent[column]):
            raise ValueError(f"Bingo fold cache index differs from parent in {column}")
    for gene_key in fold.index:
        fold_target = (cache_dir / str(fold.at[gene_key, "cache_file"])).resolve()
        parent_target = (
            parent_index_path.parent
            / "graphs"
            / str(aligned_parent.at[gene_key, "cache_file"])
        ).resolve()
        if shared_graph_root is not None and not parent_target.is_relative_to(
            shared_graph_root
        ):
            raise ValueError(
                f"Bingo parent cache_file escapes the shared graph root for {gene_key}"
            )
        if fold_target != parent_target:
            raise ValueError(
                f"Bingo fold cache_file does not rebase parent row for {gene_key}"
            )
    for row in index.itertuples(index=False):
        graph_path = cache_dir / str(row.cache_file)
        if not graph_path.is_file():
            raise FileNotFoundError(graph_path)
        observed_hash = sha256_file(graph_path)
        if observed_hash != str(row.cache_sha256):
            raise ValueError(f"Bingo cache hash mismatch for {row.gene_key}")
    return manifest


def _versions() -> dict[str, str]:
    import pyarrow
    import torch_geometric

    return software_versions(np, pd, pyarrow, sklearn, torch, torch_geometric)


def _required_source_assets() -> list[dict[str, str]]:
    return [
        {
            "path": (
                "methods/bingo/feature/source/esm2_graph_cache/graphs"
            ),
            "purpose": (
                "one ESM-2 residue embedding and contact graph per gene; the "
                "network reads a gene's graph and never its sequence"
            ),
        },
        {
            "path": (
                "methods/bingo/feature/source/esm2_graph_cache/"
                "cache_index.parquet"
            ),
            "purpose": (
                "names the graph file of every cached gene and the digest it "
                "was written with"
            ),
        },
    ]


def _model_card_fields(
    *,
    cohort: str,
    split_strategy: str,
    outer_fold: int,
    split_build_hash: str,
    threshold: float,
    versions: dict[str, str],
) -> dict[str, object]:
    return {
        "method_id": METHOD_ID,
        "cohort_id": cohort,
        "split_strategy": split_strategy,
        "outer_fold": int(outer_fold),
        "split_build_hash": split_build_hash,
        "decision_threshold": float(threshold),
        "artifacts": {
            "weights": f"{WEIGHTS_DIRECTORY}/{MODEL_FILE}",
            "transforms": [],
        },
        "required_source_assets": _required_source_assets(),
        "software_versions": versions,
    }


def _save_model(model: BingoGAT, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_FILE
    torch.save(
        {
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            "architecture": architecture_contract(),
        },
        path,
    )
    return path


def _load_model(path: Path, device: torch.device) -> BingoGAT:

    if not path.is_file():
        raise ContractError(f"the stored BiNGO model is absent: {path}")
    stored = torch.load(path, map_location="cpu", weights_only=False)
    architecture = stored.get("architecture")
    if architecture != architecture_contract():
        raise ContractError(
            f"{path} was written by a different architecture than the one this "
            f"code builds"
        )
    model = BingoGAT(dropout=TRAIN_CONFIG.dropout)
    model.load_state_dict(stored["state_dict"])
    return model.to(device)


def train_fold(
    *,
    folds_dir: Path,
    cache_dir: Path,
    model_dir: Path,
    cohort: str,
    split_strategy: str,
    outer_fold: int,
) -> Path:

    device = torch.device(DEVICE)
    if device.type != "cuda":
        raise ContractError("formal BiNGO training requires CUDA")
    cache_dir = require_method_path(cache_dir, "cache_dir")
    model_dir = require_method_path(model_dir, "model_dir")
    prepared = load_prepared_split(
        folds_dir,
        cohort=cohort,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )
    _validate_full_cache(
        cache_dir,
        prepared.train.gene_keys
        + prepared.validation.gene_keys
        + prepared.test.gene_keys,
        expected_shared_root=(
            Path(__file__).resolve().parents[1]
            / "feature"
            / "source"
            / "esm2_graph_cache"
        ),
    )
    train_data = GraphCacheDataset(cache_dir, prepared.train)
    validation_data = GraphCacheDataset(cache_dir, prepared.validation)
    test_data = GraphCacheDataset(cache_dir, prepared.test)
    train_loader = _loader(train_data, shuffle=True)
    validation_loader = _loader(validation_data, shuffle=False)
    test_loader = _loader(test_data, shuffle=False)

    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    with RunRecord() as record:
        result = fit(train_loader, validation_loader, device, TRAIN_CONFIG)
        _, validation_scores, validation_labels = predict_scores(
            result.model, validation_loader, device, require_labels=True
        )
        _, test_scores, test_labels = predict_scores(
            result.model, test_loader, device, require_labels=False
        )
        if validation_labels is None or test_labels is not None:
            raise ContractError("the sealed prediction label contract failed")
        torch.cuda.synchronize(device)

    threshold = float(result.validation_threshold)
    _save_model(result.model, model_dir / WEIGHTS_DIRECTORY)
    write_predictions(
        model_dir,
        "validation",
        validation_data.gene_keys,
        validation_scores,
        decision_threshold=threshold,
    )
    write_predictions(
        model_dir,
        "blind",
        test_data.gene_keys,
        test_scores,
        decision_threshold=threshold,
    )
    versions = _versions()
    write_run_manifest(
        model_dir,
        method_id=METHOD_ID,
        cohort_id=cohort,
        split_strategy=split_strategy,
        outer_fold=int(outer_fold),
        split_build_hash=prepared.split_build_hash,
        software_versions=versions,
        parameters={
            "device": str(device),
            **TRAIN_CONFIG.__dict__,
            **architecture_contract(),
        },
        sample_counts={
            "train": len(train_data),
            "validation": len(validation_data),
            "test": len(test_data),
        },
        best_epoch=result.best_epoch,
        **record.timestamps,
    )
    write_resource_usage(
        model_dir,
        record,
        device=str(device),
        gpu_name=torch.cuda.get_device_name(device),
        peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(device)),
    )
    return write_model_card(
        model_dir,
        **_model_card_fields(
            cohort=cohort,
            split_strategy=split_strategy,
            outer_fold=outer_fold,
            split_build_hash=prepared.split_build_hash,
            threshold=threshold,
            versions=versions,
        ),
    )


def score_genes(
    *,
    model_dir: Path,
    gene_keys: Sequence[str],
    shared_cache: Path,
    output_dir: Path,
) -> Path:

    card = read_model_card(model_dir / "model_card.json")
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    partition = Partition(
        gene_keys=tuple(str(key) for key in gene_keys), cluster_ids=(), labels=None
    )
    with RunRecord() as record:
        dataset = GraphCacheDataset(
            shared_cache, partition, graph_root=shared_cache / "graphs"
        )
        model = _load_model(model_dir / WEIGHTS_DIRECTORY / MODEL_FILE, device)
        _, scores, _ = predict_scores(
            model, _loader(dataset, shuffle=False), device, require_labels=False
        )
        torch.cuda.synchronize(device)

    write_predictions(
        output_dir,
        "blind",
        dataset.gene_keys,
        scores,
        decision_threshold=float(card["decision_threshold"]),
    )
    return write_predict_manifest(
        output_dir, card, record, gene_count=len(dataset.gene_keys)
    )
