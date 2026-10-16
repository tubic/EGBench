"""Train one eDNN-EG fold and score genes with a trained fold.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn

from .adapter import EDNNEGAdapter
from .calibration import PlattScoreCalibrator
from .contracts import FORMAL_CALIBRATION_SPEC, architecture_manifest
from .frozen_data import FrozenSplit, load_frozen_split
from .models import configure_cuda_determinism_environment, require_cuda

REPO_ROOT = Path(__file__).resolve().parents[5]
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
from egbench.contracts import (
    SEED,
    ContractError,
    read_model_card,
)

METHOD_ID = "ednn_eg"

DEVICE = "cuda"
FASTTEXT_THREAD = 1

WEIGHTS_DIRECTORY = "weights"
CALIBRATOR_FILE = "calibrator.joblib"

_EXAMPLE_LIMIT = 10


def _validated_scores(
    gene_keys: Sequence[str], probabilities: np.ndarray
) -> np.ndarray:

    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(gene_keys) != scores.size:
        raise ContractError(
            f"eDNN-EG scored {scores.size} values for {len(gene_keys)} genes"
        )
    if len(set(gene_keys)) != len(gene_keys):
        raise ContractError("eDNN-EG scored a gene twice")
    if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ContractError("eDNN-EG produced a score outside [0, 1] or non-finite")
    return scores


def _versions(torch) -> dict[str, str]:
    import fasttext
    import pyarrow

    return software_versions(np, pd, pyarrow, sklearn, torch, fasttext)


def _required_source_assets(cohort_id: str) -> list[dict[str, str]]:
    return [
        {
            "path": (
                f"methods/ednn_eg/feature/source/"
                f"{cohort_id}_cds_sequences.fa.gz"
            ),
            "purpose": (
                "coding sequences; the stored fastText model turns a gene's "
                "coding sequence into the hundred values the stack scores"
            ),
        }
    ]


def _model_card_fields(
    *,
    cohort_id: str,
    split_strategy: str,
    outer_fold: int,
    split_build_hash: str,
    versions: dict[str, str],
) -> dict[str, object]:
    return {
        "method_id": METHOD_ID,
        "cohort_id": cohort_id,
        "split_strategy": split_strategy,
        "outer_fold": int(outer_fold),
        "split_build_hash": split_build_hash,
        "decision_threshold": float(FORMAL_CALIBRATION_SPEC.threshold),
        "artifacts": {
            "weights": WEIGHTS_DIRECTORY,
            "transforms": [
                f"{WEIGHTS_DIRECTORY}/fasttext.bin",
                f"{WEIGHTS_DIRECTORY}/{CALIBRATOR_FILE}",
            ],
        },
        "required_source_assets": _required_source_assets(cohort_id),
        "software_versions": versions,
    }


def train_fold(
    *,
    cds_fasta: Path,
    folds_dir: Path,
    model_dir: Path,
    work_dir: Path,
    cohort_id: str,
    split_strategy: str,
    outer_fold: int,
) -> Path:

    data: FrozenSplit = load_frozen_split(
        cds_fasta,
        folds_dir / "training_labels" / f"fold_{outer_fold}.parquet",
        folds_dir / "split_assignments.parquet",
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )
    if data.split_build_hash is None:
        raise ContractError(
            f"{folds_dir}: the split assignments carry no split_build_hash, so a "
            f"run of them could not say which build of the split it used"
        )
    configure_cuda_determinism_environment()
    torch, device = require_cuda(DEVICE)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    with RunRecord() as record:
        adapter = EDNNEGAdapter(
            work_dir=work_dir,
            seed=SEED,
            device=str(device),
            fasttext_thread=FASTTEXT_THREAD,
        )
        adapter.fit(
            data.train.sequence_map(),
            data.train.label_map(),
            groups=data.train.group_map(),
            split_strategy=split_strategy,
        )
        validation = adapter.predict_scores(data.validation.sequence_map())
        calibrator = PlattScoreCalibrator(seed=SEED).fit(
            validation.meta_scores,
            np.asarray(data.validation.labels, dtype=np.int64),
        )
        validation_probabilities = calibrator.transform(validation.meta_scores)
        blind = adapter.predict_scores(data.test.sequence_map())
        blind_probabilities = calibrator.transform(blind.meta_scores)
        torch.cuda.synchronize(device)

    weights = model_dir / WEIGHTS_DIRECTORY
    adapter.save(weights)
    joblib.dump(calibrator, weights / CALIBRATOR_FILE)

    threshold = float(FORMAL_CALIBRATION_SPEC.threshold)
    write_predictions(
        model_dir,
        "validation",
        validation.ids,
        _validated_scores(validation.ids, validation_probabilities),
        decision_threshold=threshold,
    )
    write_predictions(
        model_dir,
        "blind",
        blind.ids,
        _validated_scores(blind.ids, blind_probabilities),
        decision_threshold=threshold,
    )
    versions = _versions(torch)
    write_run_manifest(
        model_dir,
        method_id=METHOD_ID,
        cohort_id=cohort_id,
        split_strategy=split_strategy,
        outer_fold=int(outer_fold),
        split_build_hash=data.split_build_hash,
        software_versions=versions,
        parameters={
            "device": str(device),
            "fasttext_thread": FASTTEXT_THREAD,
            **architecture_manifest(),
        },
        sample_counts=data.outer_audit["sample_counts"],
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
            cohort_id=cohort_id,
            split_strategy=split_strategy,
            outer_fold=outer_fold,
            split_build_hash=data.split_build_hash,
            versions=versions,
        ),
    )


def score_genes(
    *,
    model_dir: Path,
    gene_keys: Sequence[str],
    sequences: dict[str, str],
    output_dir: Path,
) -> Path:

    card = read_model_card(model_dir / "model_card.json")
    configure_cuda_determinism_environment()
    _, device = require_cuda(DEVICE)
    missing = [key for key in gene_keys if key not in sequences]
    if missing:
        raise ContractError(
            f"eDNN-EG scores a gene from its coding sequence and "
            f"{len(missing)} of the {len(gene_keys)} requested genes have none: "
            f"{missing[:_EXAMPLE_LIMIT]}"
        )
    weights = model_dir / WEIGHTS_DIRECTORY
    with RunRecord() as record:
        adapter = EDNNEGAdapter.load(weights, seed=SEED, device=str(device))
        calibrator: PlattScoreCalibrator = joblib.load(weights / CALIBRATOR_FILE)
        batch = adapter.predict_scores({key: sequences[key] for key in gene_keys})
        probabilities = calibrator.transform(batch.meta_scores)

    threshold = float(card["decision_threshold"])
    write_predictions(
        output_dir,
        "blind",
        batch.ids,
        _validated_scores(batch.ids, probabilities),
        decision_threshold=threshold,
    )
    return write_predict_manifest(
        output_dir, card, record, gene_count=len(gene_keys)
    )
