"""Cohorts, blind prediction readers, gene resampling and Holm correction shared by the analyses.

Imported by the analysis scripts; it runs nothing by itself.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from egbench.contracts import PREDICTION_FILES, SEED, TargetKind
from egbench.data import (
    PROJECT_ROOT,
    cohort_target,
    model_directory,
    prediction_cohort,
    primary_scope,
    read_evaluation_truth,
)
from egbench.evaluate import scope_frame
from egbench.metrics.discrimination import auprc, average_ranks
from egbench.metrics.regression import spearman as spearman_metric
from egbench.stats import confidence_interval, holm_adjusted, paired_difference

METHODS: tuple[str, ...] = (
    "bingo", "deepcelless", "deephe", "deeplyessential", "deepsnap", "ednn_eg",
    "epgat", "esssubgraph", "esstfnet", "gcnn_sfm", "help", "mahi", "pheg",
    "proteomelm", "xgep",
)
BINARY_COHORTS: tuple[str, ...] = ("deg", "depmap", "context", "blood", "mouse")
ALL_COHORTS: tuple[str, ...] = ("deg", "depmap", "context", "gene_effect", "blood", "mouse")
COHORT_NAMES: dict[str, str] = {
    "deg": "DEG", "depmap": "DepMap", "context": "Context", "gene_effect": "GeneEffect",
    "blood": "Blood", "mouse": "Mouse",
}
SPLIT = "family_5fold"
FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)
METHODS_DIRECTORY = PROJECT_ROOT / "methods"
ANALYSIS_DIRECTORY = PROJECT_ROOT / "analysis"
GENE_KEY = "gene_key"
SCORE = "y_score"
FOLD = "fold"
_Z = 1.959963984540054


def target_of(cohort: str) -> str:
    return cohort_target(cohort).column


def kind_of(cohort: str) -> TargetKind:
    return cohort_target(cohort).kind


def primary_metric(cohort: str):
    if kind_of(cohort) is TargetKind.BINARY:
        return "auprc", auprc
    return "spearman", spearman_metric


def blind_scores(method: str, cohort: str, fold: int) -> pd.Series:
    directory = model_directory(
        METHODS_DIRECTORY / method, prediction_cohort(cohort), split_strategy=SPLIT, outer_fold=fold
    )
    frame = pd.read_parquet(directory / PREDICTION_FILES["blind"], columns=[GENE_KEY, SCORE])
    if frame[GENE_KEY].duplicated().any():
        raise ValueError(f"{directory}: a gene is scored twice")
    return pd.Series(frame[SCORE].to_numpy("float64"), index=pd.Index(frame[GENE_KEY].astype("object")))


def fold_truth(cohort: str, fold: int) -> pd.DataFrame:
    return read_evaluation_truth(cohort, fold, split_strategy=SPLIT).reset_index(drop=True)


def test_frame(method: str, cohort: str, fold: int) -> pd.DataFrame:
    directory = model_directory(
        METHODS_DIRECTORY / method, prediction_cohort(cohort), split_strategy=SPLIT, outer_fold=fold
    )
    scores = pd.read_parquet(directory / PREDICTION_FILES["blind"], columns=[GENE_KEY, SCORE])
    truth = read_evaluation_truth(cohort, fold, split_strategy=SPLIT)
    merged = truth[[GENE_KEY, target_of(cohort)]].merge(scores, on=GENE_KEY, how="left", validate="one_to_one")
    if merged[SCORE].isna().any():
        raise ValueError(f"{method} {cohort} fold {fold}: a test gene has no score")
    return scope_frame(merged, cohort_id=cohort, scope=primary_scope(cohort)).reset_index(drop=True)


def pooled(method: str, cohort: str) -> pd.DataFrame:
    frames = [test_frame(method, cohort, fold).assign(**{FOLD: fold}) for fold in FOLDS]
    return pd.concat(frames, ignore_index=True)


def percentile(values: np.ndarray) -> np.ndarray:
    return average_ranks(np.asarray(values, dtype="float64")) / float(len(values))


class Resampler:
    def __init__(self, cells: np.ndarray, seed: int = SEED) -> None:
        cells = np.asarray(cells)
        order = np.argsort(cells, kind="stable")
        sorted_cells = cells[order]
        boundaries = np.flatnonzero(np.diff(sorted_cells)) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(cells)]])
        sizes = ends - starts
        self._order = order
        self._offsets = np.repeat(starts, sizes)
        self._highs = np.repeat(sizes, sizes)
        self._generator = np.random.default_rng(seed)

    def draw(self) -> np.ndarray:
        inside = self._generator.integers(0, self._highs)
        return self._order[self._offsets + inside]


def summarise(point: float, replicates: np.ndarray) -> tuple[float, float]:
    values = np.asarray(replicates, dtype="float64")
    values = values[~np.isnan(values)]
    if math.isnan(point) or values.size < 2:
        return float("nan"), float("nan")
    error = float(values.std(ddof=1))
    if error == 0.0:
        return 0.0, (1.0 if point == 0.0 else float("nan"))
    return _Z * error, math.erfc(abs(point) / (error * math.sqrt(2.0)))


def holm(frame: pd.DataFrame, group_columns: list[str], p_column: str = "p") -> pd.Series:
    adjusted = pd.Series(np.nan, index=frame.index)
    for _, rows in frame.groupby(group_columns, sort=False, dropna=False):
        adjusted.loc[rows.index] = holm_adjusted(rows[p_column].to_numpy("float64"))
    return adjusted


def fast_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(np.asarray(scores, dtype="float64"))
    return (float(ranks[positives].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def interval(y: np.ndarray, s: np.ndarray, metric, fold: np.ndarray, kind: TargetKind) -> tuple[float, float]:
    result = confidence_interval(y, s, metric, fold=fold, target_kind=kind)
    return float(result.point), float(result.half_width)


def paired(y: np.ndarray, a: np.ndarray, b: np.ndarray, metric, fold: np.ndarray,
           kind: TargetKind) -> tuple[float, float, float]:
    result = paired_difference(y, a, b, metric, fold=fold, target_kind=kind)
    return float(result.difference), float(result.half_width), float(result.p_value)


def resampled_statistics(frame: pd.DataFrame, statistics, count: int, label: str) -> dict[str, tuple[float, float, float]]:
    point = statistics(frame, None)
    cells = pd.factorize(pd.MultiIndex.from_arrays([frame[FOLD], frame["label"]]))[0]
    resampler = Resampler(cells, seed=SEED)
    replicates = {name: np.empty(count) for name in point}
    progress = Progress(count, label)
    for r in range(count):
        values = statistics(frame, resampler.draw())
        for name, value in values.items():
            replicates[name][r] = value
        progress.tick()
    return {name: (value, *summarise(value, replicates[name])) for name, value in point.items()}


class Progress:
    def __init__(self, total: int, label: str) -> None:
        self.total = total
        self.label = label
        self.done = 0
        self.started = time.time()
        self.step = max(1, total // 10)

    def tick(self, note: str = "") -> None:
        self.done += 1
        if self.done % self.step == 0 or self.done == self.total:
            elapsed = time.time() - self.started
            print(f"[{self.label}] {self.done}/{self.total} done, {elapsed:.0f} s {note}", flush=True)


def process_pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
