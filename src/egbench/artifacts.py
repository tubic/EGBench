"""Writers for the files a method run leaves in its fold directory."""
from __future__ import annotations

import json
import platform
import resource
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from importlib import metadata
from pathlib import Path
from types import ModuleType, TracebackType

import numpy as np
import pandas as pd

from egbench.contracts import (
    MODEL_CARD_FILE,
    PREDICT_MANIFEST_FILE,
    PREDICTION_COLUMNS,
    PREDICTION_FILES,
    PREDICTION_NAMES,
    RESOURCE_USAGE_FILE,
    RUN_MANIFEST_FILE,
    ContractError,
    validate_model_card,
    validate_prediction_table,
    validate_resource_usage,
    validate_run_manifest,
)


_RSS_UNIT_BYTES = 1 if sys.platform == "darwin" else 1024


def software_versions(*modules: ModuleType) -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for module in modules:
        if not isinstance(module, ModuleType):
            raise ContractError(
                f"software_versions takes module objects, got "
                f"{type(module).__name__} ({module!r}). A version number that is "
                f"not a module's own is not a software version"
            )
        name = module.__name__
        version = getattr(module, "__version__", None)
        if version is None:
            try:
                version = metadata.version(name)
            except metadata.PackageNotFoundError:
                version = None
        if not isinstance(version, str) or not version:
            raise ContractError(
                f"software_versions: module {name!r} states no version string, "
                f"its '__version__' is {version!r} and no distribution of that "
                f"name is installed"
            )
        versions[name] = version
    return versions


class RunRecord:

    def __init__(self) -> None:
        self._started_at: datetime | None = None
        self._finished_at: datetime | None = None
        self._started_cpu: float | None = None
        self._cpu_seconds: float | None = None
        self._peak_rss_bytes: int | None = None

    def __enter__(self) -> RunRecord:
        if self._started_at is not None:
            raise ContractError(
                "RunRecord: this record has already been used; a second run "
                "takes a second record"
            )
        self._started_at = datetime.now().astimezone()
        self._started_cpu = _cpu_seconds()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._started_cpu is None:
            raise ContractError(
                "RunRecord: the record was left without having been entered, "
                "use it as 'with RunRecord() as run:'"
            )
        self._finished_at = datetime.now().astimezone()
        self._cpu_seconds = _cpu_seconds() - self._started_cpu
        self._peak_rss_bytes = _peak_rss_bytes()

    def _read(self, attribute: str) -> object:
        value = getattr(self, f"_{attribute}")
        if value is None:
            raise ContractError(
                f"RunRecord: '{attribute}' is not known yet, use the record as "
                f"'with RunRecord() as run:' and read what it holds after the "
                f"block"
            )
        return value

    @property
    def started_at(self) -> str:
        started: datetime = self._read("started_at")
        return started.isoformat()

    @property
    def finished_at(self) -> str:
        finished: datetime = self._read("finished_at")
        return finished.isoformat()

    @property
    def timestamps(self) -> dict[str, str]:
        return {"started_at": self.started_at, "finished_at": self.finished_at}

    @property
    def cpu_seconds(self) -> float:
        return float(self._read("cpu_seconds"))

    @property
    def peak_rss_bytes(self) -> int:
        return int(self._read("peak_rss_bytes"))


def _peak_rss_bytes() -> int:
    return int(
        max(
            resource.getrusage(who).ru_maxrss
            for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
        )
        * _RSS_UNIT_BYTES
    )


def _cpu_seconds() -> float:
    return float(
        sum(
            usage.ru_utime + usage.ru_stime
            for usage in (
                resource.getrusage(resource.RUSAGE_SELF),
                resource.getrusage(resource.RUSAGE_CHILDREN),
            )
        )
    )


def _directory(directory: str | Path) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_predictions(
    directory: str | Path,
    name: str,
    gene_keys: Sequence[str] | pd.Index | pd.Series,
    y_score: Iterable[float] | np.ndarray | pd.Series,
    *,
    decision_threshold: float,
) -> Path:
    if name not in PREDICTION_NAMES:
        raise ContractError(
            f"a prediction table is written as one of {list(PREDICTION_NAMES)}, "
            f"got {name!r}"
        )
    if not isinstance(decision_threshold, float) or not np.isfinite(
        decision_threshold
    ):
        raise ContractError(
            f"write_predictions: 'decision_threshold' must be a finite float, "
            f"got {decision_threshold!r}"
        )
    keys = pd.Series(list(gene_keys), dtype="object", name="gene_key")
    scores = np.asarray(y_score, dtype="float64")
    if scores.ndim != 1:
        raise ContractError(
            f"write_predictions: 'y_score' must be one score per gene, its "
            f"shape is {scores.shape}"
        )
    if len(keys) != scores.size:
        raise ContractError(
            f"write_predictions: 'gene_keys' holds {len(keys)} genes and "
            f"'y_score' holds {scores.size} scores"
        )
    frame = pd.DataFrame(
        {
            "gene_key": keys,
            "y_score": scores,
            "y_pred": (scores >= decision_threshold).astype("int8"),
        }
    )[list(PREDICTION_COLUMNS)]
    file_name = PREDICTION_FILES[name]
    validate_prediction_table(
        frame,
        table=str(Path(directory) / file_name),
        decision_threshold=decision_threshold,
    )
    path = _directory(directory) / file_name
    frame.to_parquet(path, index=False)
    return path


def write_run_manifest(directory: str | Path, **fields: object) -> Path:
    validate_run_manifest(fields, source=RUN_MANIFEST_FILE)
    return _write_json(_directory(directory) / RUN_MANIFEST_FILE, fields)


def write_resource_usage(
    directory: str | Path,
    record: RunRecord,
    *,
    device: str,
    gpu_name: str | None,
    peak_gpu_memory_bytes: int,
) -> Path:
    usage = {
        "cpu_seconds": record.cpu_seconds,
        "peak_rss_bytes": record.peak_rss_bytes,
        "device": device,
        "gpu_name": gpu_name,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
    }
    validate_resource_usage(usage, source=RESOURCE_USAGE_FILE)
    return _write_json(_directory(directory) / RESOURCE_USAGE_FILE, usage)


def write_model_card(directory: str | Path, **fields: object) -> Path:
    validate_model_card(fields, source=MODEL_CARD_FILE)
    return _write_json(_directory(directory) / MODEL_CARD_FILE, fields)


def write_predict_manifest(
    directory: str | Path,
    model_card: Mapping[str, object],
    record: RunRecord,
    *,
    gene_count: int,
) -> Path:
    if (
        not isinstance(gene_count, int)
        or isinstance(gene_count, bool)
        or gene_count <= 0
    ):
        raise ContractError(
            f"predict_manifest.json: 'gene_count' must be a positive integer, "
            f"got {gene_count!r}"
        )
    payload = {**model_card, "gene_count": gene_count, **record.timestamps}
    validate_model_card(payload, source=PREDICT_MANIFEST_FILE)
    return _write_json(_directory(directory) / PREDICT_MANIFEST_FILE, payload)
