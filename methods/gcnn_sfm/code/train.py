"""Train one GCNN-SFM fold and write its predictions and run records.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import gzip
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from egbench.artifacts import (
    RunRecord,
    software_versions,
    write_model_card,
    write_predictions,
    write_resource_usage,
    write_run_manifest,
)
from egbench.contracts import COHORT_IDS, SEED

if __package__:
    from .adapter import (
        DISTANCE_GROUP_COUNT,
        FEATURE_NODE_COUNT,
        FORMAL_BATCH_SIZE,
        PAIR_NODE_COUNT,
        SOURCE_C_SHA256,
        SOURCE_COMMIT,
        GCNNSFM,
        TrainConfig,
        architecture_contract,
        encode_graph,
        fit,
        predict_scores,
        set_random_seed,
        verify_official_c_fixture,
    )
else:
    from adapter import (
        DISTANCE_GROUP_COUNT,
        FEATURE_NODE_COUNT,
        FORMAL_BATCH_SIZE,
        PAIR_NODE_COUNT,
        SOURCE_C_SHA256,
        SOURCE_COMMIT,
        GCNNSFM,
        TrainConfig,
        architecture_contract,
        encode_graph,
        fit,
        predict_scores,
        set_random_seed,
        verify_official_c_fixture,
    )


METHOD_ID = "gcnn_sfm"
WEIGHTS_FILE = "weights/best_model.pt"
_CONTEXT_COLUMNS = ("cohort_id", "split_strategy", "outer_fold")
_KEY_COLUMNS = (*_CONTEXT_COLUMNS, "role", "gene_key")


@dataclass(frozen=True)
class SequencePartition:
    gene_keys: tuple[str, ...]
    cluster_ids: tuple[str, ...]
    sequences: tuple[str, ...]
    labels: np.ndarray | None


@dataclass(frozen=True)
class PreparedData:
    train: SequencePartition
    validation: SequencePartition
    test: SequencePartition
    split_build_hash: str
    minimum_cds_length: int
    maximum_cds_length: int
    source_gate: dict[str, Any]


@dataclass(frozen=True)
class EncodedPartition:
    pair_features: np.ndarray
    feature_features: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _one_value(frame: pd.DataFrame, column: str, table: str) -> Any:
    if column not in frame:
        raise ValueError(f"{table} lacks {column}")
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"{table} requires one {column} value, found {len(values)}")
    return values[0]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], table: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{table} lacks columns: {missing}")
    if not frame.columns.is_unique:
        raise ValueError(f"{table} has duplicate columns")


def read_fasta(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: dict[str, list[str]] = {}
    current: str | None = None
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt" if path.suffix == ".gz" else "r", encoding="utf-8") as handle:
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
    return sequences


def _context(
    frame: pd.DataFrame,
    cohort: str,
    split_strategy: str,
    outer_fold: int,
) -> pd.DataFrame:
    return frame.loc[
        frame["cohort_id"].astype(str).eq(cohort)
        & frame["split_strategy"].astype(str).eq(split_strategy)
        & pd.to_numeric(frame["outer_fold"], errors="coerce").eq(outer_fold)
    ].copy()


def _validate_and_join_partitions(
    assignments: pd.DataFrame,
    training_labels: pd.DataFrame,
    *,
    cohort: str,
    split_strategy: str,
    outer_fold: int,
) -> tuple[dict[str, pd.DataFrame], str]:
    if cohort not in COHORT_IDS:
        raise ValueError(f"GCNN-SFM cohort must be one of {list(COHORT_IDS)}")
    _require_columns(
        assignments,
        (*_KEY_COLUMNS, "cluster_id", "split_build_hash"),
        "split assignments",
    )
    if "label" in assignments:
        raise ValueError("split assignments must be label-free")
    _require_columns(
        training_labels,
        (*_KEY_COLUMNS, "cluster_id", "label", "split_build_hash"),
        "training labels",
    )
    label_roles = set(training_labels["role"].astype(str))
    if not label_roles.issubset({"train", "validation"}):
        raise ValueError("training labels may contain only train and validation roles")
    if assignments.duplicated(list(_KEY_COLUMNS)).any():
        raise ValueError("split assignments contain duplicate keys")
    if training_labels.duplicated(list(_KEY_COLUMNS)).any():
        raise ValueError("training labels contain duplicate keys")

    assignment_context = _context(assignments, cohort, split_strategy, outer_fold)
    label_context = _context(training_labels, cohort, split_strategy, outer_fold)
    if set(assignment_context["role"].astype(str)) != {"train", "validation", "test"}:
        raise ValueError("split context requires train, validation, and test roles")
    if set(label_context["role"].astype(str)) != {"train", "validation"}:
        raise ValueError("label context requires train and validation roles")

    compare_columns = [*_KEY_COLUMNS, "cluster_id", "split_build_hash"]
    expected = assignment_context.loc[
        assignment_context["role"].ne("test"), compare_columns
    ]
    observed = label_context.loc[:, compare_columns]
    coverage = expected.merge(
        observed,
        on=compare_columns,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not coverage["_merge"].eq("both").all():
        raise ValueError(
            "training labels do not exactly cover assigned train and validation rows"
        )

    label_values = pd.to_numeric(label_context["label"], errors="coerce")
    if label_values.isna().any() or not label_values.isin([0, 1]).all():
        raise ValueError("training labels must be binary 0/1")
    label_context["label"] = label_values.astype(np.int64)
    partitions: dict[str, pd.DataFrame] = {}
    for role in ("train", "validation"):
        assigned = assignment_context.loc[
            assignment_context["role"].eq(role), ["gene_key", "cluster_id"]
        ]
        labels = label_context.loc[
            label_context["role"].eq(role), ["gene_key", "label"]
        ]
        joined = assigned.merge(labels, on="gene_key", validate="one_to_one")
        if set(joined["label"].unique()) != {0, 1}:
            raise ValueError(f"{role} partition must contain both binary classes")
        partitions[role] = joined
    partitions["test"] = assignment_context.loc[
        assignment_context["role"].eq("test"), ["gene_key", "cluster_id"]
    ].copy()
    if any(partition.empty for partition in partitions.values()):
        raise ValueError("one or more split partitions are empty")

    gene_sets = {
        role: set(partition["gene_key"].astype(str))
        for role, partition in partitions.items()
    }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = gene_sets[left] & gene_sets[right]
        if overlap:
            raise ValueError(f"gene leakage between {left} and {right}: {len(overlap)}")
    if split_strategy == "family_5fold":
        cluster_sets = {
            role: set(partition["cluster_id"].astype(str))
            for role, partition in partitions.items()
        }
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            overlap = cluster_sets[left] & cluster_sets[right]
            if overlap:
                raise ValueError(
                    f"family cluster leakage between {left} and {right}: {len(overlap)}"
                )

    split_hash = str(
        _one_value(assignment_context, "split_build_hash", "split context")
    )
    if (
        str(_one_value(label_context, "split_build_hash", "label context"))
        != split_hash
    ):
        raise ValueError("split build hashes disagree")
    return partitions, split_hash


def load_prepared_data(
    assignments_path: Path,
    training_labels_path: Path,
    sequence_path: Path,
    cohort: str,
    split_strategy: str,
    outer_fold: int,
) -> PreparedData:
    source_gate = verify_official_c_fixture()
    if not source_gate["all_passed"]:
        raise RuntimeError(f"official C encoder fixture gate failed: {source_gate}")
    assignments_path = assignments_path.resolve()
    training_labels_path = training_labels_path.resolve()
    sequence_path = sequence_path.resolve()
    training_labels = pd.read_parquet(training_labels_path)
    assignments = pd.read_parquet(assignments_path)
    frames, split_hash = _validate_and_join_partitions(
        assignments,
        training_labels,
        cohort=cohort,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )
    sequence_by_key = read_fasta(sequence_path)
    all_keys = set().union(
        *(set(frame["gene_key"].astype(str)) for frame in frames.values())
    )
    missing = sorted(all_keys.difference(sequence_by_key))
    if missing:
        raise ValueError(f"{len(missing)} split genes lack a selected CDS")
    invalid = sorted(
        {
            base
            for key in all_keys
            for base in set(sequence_by_key[key]).difference("ACGT")
        }
    )
    if invalid:
        raise ValueError(f"formal CDS FASTA contains unsupported bases: {invalid}")

    def materialize(role: str) -> SequencePartition:
        frame = frames[role]
        keys = tuple(frame["gene_key"].astype(str))
        return SequencePartition(
            gene_keys=keys,
            cluster_ids=tuple(frame["cluster_id"].astype(str)),
            sequences=tuple(sequence_by_key[key] for key in keys),
            labels=(
                frame["label"].to_numpy(dtype=np.int64) if "label" in frame else None
            ),
        )

    selected_lengths = [len(sequence_by_key[key]) for key in all_keys]
    if min(selected_lengths) < 9:
        raise ValueError("GCNN-SFM requires CDS sequences long enough for k=3,d=3")
    return PreparedData(
        train=materialize("train"),
        validation=materialize("validation"),
        test=materialize("test"),
        split_build_hash=split_hash,
        minimum_cds_length=min(selected_lengths),
        maximum_cds_length=max(selected_lengths),
        source_gate=source_gate,
    )


def _encode_partition(partition: SequencePartition) -> EncodedPartition:
    count = len(partition.gene_keys)
    pair = np.empty((count, PAIR_NODE_COUNT, DISTANCE_GROUP_COUNT), dtype=np.float32)
    feature = np.empty((count, FEATURE_NODE_COUNT, 1), dtype=np.float32)
    for index, sequence in enumerate(partition.sequences):
        graph = encode_graph(sequence)
        pair[index] = graph.pair_node_features
        feature[index] = graph.feature_node_features
    if not np.isfinite(pair).all() or not np.isfinite(feature).all():
        raise ValueError("GCNN-SFM encoding produced non-finite values")
    return EncodedPartition(pair_features=pair, feature_features=feature)


def _required_labels(partition: SequencePartition, role: str) -> np.ndarray:
    if partition.labels is None:
        raise ValueError(f"{role} labels are required")
    return partition.labels


def _validation_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (scores >= threshold).astype(np.int64)
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "mcc_at_selected_threshold": float(matthews_corrcoef(labels, predictions)),
        "accuracy_at_selected_threshold": float(np.mean(predictions == labels)),
        "selected_threshold": float(threshold),
    }


def _prediction_frame(
    gene_keys: Sequence[str], scores: np.ndarray
) -> pd.DataFrame:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(gene_keys),) or not np.isfinite(values).all():
        raise ValueError("prediction scores must be finite and aligned with gene keys")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("GCNN-SFM prediction scores must be probabilities")
    return pd.DataFrame({"gene_key": tuple(gene_keys), "y_score": values})


def _formal_cuda(device_name: str) -> torch.device:
    if not device_name.startswith("cuda"):
        raise RuntimeError("formal GCNN-SFM training requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("formal GCNN-SFM training requires available CUDA")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    return device


def inference_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _input_paths(
    assignments_path: Path, training_labels_path: Path, sequence_path: Path
) -> dict[str, Path]:
    return {
        "training_labels": training_labels_path.resolve(),
        "split_assignments": assignments_path.resolve(),
        "selected_cds": sequence_path.resolve(),
    }


def run_formal_training(
    data: PreparedData,
    *,
    assignments_path: Path,
    training_labels_path: Path,
    sequence_path: Path,
    output_dir: Path,
    cohort: str,
    split_strategy: str,
    outer_fold: int,
    device_name: str,
    max_epochs: int,
    patience: int,
    required_source_assets: list[dict[str, str]],
) -> dict[str, Any]:
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    source_gate = verify_official_c_fixture()
    if not source_gate["all_passed"]:
        raise RuntimeError(f"official C encoder fixture gate failed: {source_gate}")
    device = _formal_cuda(device_name)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    inputs = _input_paths(assignments_path, training_labels_path, sequence_path)
    input_hashes_before = {name: _sha256(path) for name, path in inputs.items()}
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    with RunRecord() as record:
        train_encoded = _encode_partition(data.train)
        validation_encoded = _encode_partition(data.validation)
        test_encoded = _encode_partition(data.test)
        set_random_seed(SEED)
        model = GCNNSFM()
        config = TrainConfig(
            batch_size=FORMAL_BATCH_SIZE,
            max_epochs=max_epochs,
            patience=patience,
            device=str(device),
        )
        result = fit(
            model,
            train_encoded.pair_features,
            train_encoded.feature_features,
            _required_labels(data.train, "train"),
            validation_encoded.pair_features,
            validation_encoded.feature_features,
            _required_labels(data.validation, "validation"),
            config=config,
        )
        validation_scores = predict_scores(
            result.model,
            validation_encoded.pair_features,
            validation_encoded.feature_features,
            device=str(device),
            batch_size=FORMAL_BATCH_SIZE,
        )
        test_scores = predict_scores(
            result.model,
            test_encoded.pair_features,
            test_encoded.feature_features,
            device=str(device),
            batch_size=FORMAL_BATCH_SIZE,
        )
        torch.cuda.synchronize(device)

    for name, path in inputs.items():
        if _sha256(path) != input_hashes_before[name]:
            raise RuntimeError(f"input changed during training: {name}")
    validation_metrics = _validation_metrics(
        _required_labels(data.validation, "validation"),
        validation_scores,
        result.threshold,
    )
    _prediction_frame(data.test.gene_keys, test_scores)
    _prediction_frame(data.validation.gene_keys, validation_scores)

    weights_path = output_dir / WEIGHTS_FILE
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                name: parameter.detach().cpu()
                for name, parameter in result.model.state_dict().items()
            },
            "threshold": result.threshold,
            "architecture": architecture_contract(),
            "config": asdict(config),
            "source_commit": SOURCE_COMMIT,
            "source_c_sha256": SOURCE_C_SHA256,
        },
        weights_path,
    )
    pd.DataFrame(result.history).to_csv(
        output_dir / "training_history.csv", index=False
    )

    write_predictions(
        output_dir,
        "blind",
        data.test.gene_keys,
        test_scores,
        decision_threshold=result.threshold,
    )
    write_predictions(
        output_dir,
        "validation",
        data.validation.gene_keys,
        validation_scores,
        decision_threshold=result.threshold,
    )

    versions = software_versions(np, pd, sklearn, torch)
    manifest = {
        "method_id": METHOD_ID,
        "cohort_id": cohort,
        "split_strategy": split_strategy,
        "outer_fold": outer_fold,
        "split_build_hash": data.split_build_hash,
        "software_versions": versions,
        "parameters": {**asdict(config), **architecture_contract()},
        "sample_counts": {
            "train": len(data.train.gene_keys),
            "validation": len(data.validation.gene_keys),
            "test": len(data.test.gene_keys),
        },
        "sequence_contract": {
            "minimum_cds_length": data.minimum_cds_length,
            "maximum_cds_length": data.maximum_cds_length,
            "alphabet": "ACGT",
        },
        "source_gate": {
            "source_commit": source_gate["source_commit"],
            "source_c_sha256": source_gate["source_c_sha256"],
            "case_count": len(source_gate["checks"]),
            "required_match": "byte_exact_float64",
        },
        "selection_contract": {
            "checkpoint": "validation_cross_entropy_only",
            "threshold": "validation_accuracy_only",
            "weighted_sampling": "training_labels_only",
            "blind_role_keys": "split_assignments_only",
        },
        "validation_metrics": validation_metrics,
        **record.timestamps,
    }
    write_run_manifest(output_dir, **manifest)
    write_resource_usage(
        output_dir,
        record,
        device=str(device),
        gpu_name=torch.cuda.get_device_name(device),
        peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(device)),
    )
    write_model_card(
        output_dir,
        method_id=METHOD_ID,
        cohort_id=cohort,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
        split_build_hash=data.split_build_hash,
        decision_threshold=result.threshold,
        artifacts={"weights": WEIGHTS_FILE, "transforms": []},
        required_source_assets=required_source_assets,
        software_versions=versions,
    )
    return manifest


def preflight_summary(data: PreparedData) -> dict[str, Any]:
    return {
        "method_id": METHOD_ID,
        "sample_counts": {
            "train": len(data.train.gene_keys),
            "validation": len(data.validation.gene_keys),
            "test": len(data.test.gene_keys),
        },
        "split_build_hash": data.split_build_hash,
        "minimum_cds_length": data.minimum_cds_length,
        "maximum_cds_length": data.maximum_cds_length,
        "batch_size": FORMAL_BATCH_SIZE,
        "source_commit": SOURCE_COMMIT,
        "source_c_sha256": SOURCE_C_SHA256,
    }
