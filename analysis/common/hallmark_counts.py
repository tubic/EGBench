"""The four hallmarks of essential genes (CORUM complex member, yeast orthologue, PPI degree and TCGA expression at
or above the task median) for every labelled gene of a task, and the resampling helpers of the hallmark analyses.

Imported by hallmark_exceptions, prospective_validation, dependency_strength_context and mouse_lethal_stages.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from egbench.contracts import SEED
from egbench.data import feature_directory, read_cohort_labels
from egbench.metrics.discrimination import average_ranks
from egbench.stats import RESAMPLE_COUNT

from common.feature_blocks import induced_degree, read_indexed
from common.hallmarks import expression_median, gene_age
from common.stats import METHODS_DIRECTORY, SPLIT, Progress, Resampler, summarise

HALLMARKS: tuple[str, ...] = ("complex", "old", "connected", "expressed")
BINARY_TASKS: tuple[str, ...] = ("deg", "depmap", "context", "blood", "mouse")
TASK_NAMES = {"deg": "DEG", "depmap": "DepMap", "context": "Context", "blood": "Blood", "mouse": "Mouse",
              "gene_effect": "GeneEffect", "depmap_23q4": "DepMap 23Q4"}
RICH = 3
YEAST = "ortholog__saccharomyces_cerevisiae"
EPGAT_GENERATED = METHODS_DIRECTORY / "epgat" / "feature" / "source" / "generated"


def fold_table(method: str, task: str, fold: int = 0):
    return feature_directory(METHODS_DIRECTORY / method, task, split_strategy=SPLIT, outer_fold=fold) / "features.parquet"


def static_features(task: str):
    return EPGAT_GENERATED / f"{task}_static_features.parquet"


def hallmark_frame(genes: pd.Index, corum_count: np.ndarray, yeast: np.ndarray, degree: np.ndarray,
                   expression: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    thresholds = {"ppi_median": float(np.median(degree)), "tcga_median": float(np.median(expression))}
    frame = pd.DataFrame({"gene_key": np.asarray(genes, dtype="object")})
    frame["corum_count"] = corum_count
    frame["ppi_degree"] = degree
    frame["tcga_median"] = expression
    frame["complex"] = (corum_count >= 1).astype("int64")
    frame["old"] = (yeast == 1).astype("int64")
    frame["connected"] = (degree >= thresholds["ppi_median"]).astype("int64")
    frame["expressed"] = (expression >= thresholds["tcga_median"]).astype("int64")
    frame["count"] = frame[list(HALLMARKS)].sum(axis=1).astype("int64")
    return frame, thresholds


def hallmark_table(task: str) -> tuple[pd.DataFrame, dict[str, float]]:
    keys = pd.read_parquet(fold_table("help", task), columns=["gene_key"])["gene_key"].astype(str)
    genes = pd.Index(sorted(keys), dtype="object", name="gene_key")
    labels = read_cohort_labels(task).set_index("gene_key")["label"]
    labels.index = labels.index.astype(str)
    yeast = read_indexed(static_features(task), columns=[YEAST])[YEAST].reindex(genes).to_numpy("float64")
    corum = read_indexed(fold_table("deepsnap", task), columns=["corum_membership_count"])
    corum = corum["corum_membership_count"].reindex(genes).to_numpy("float64")
    frame, thresholds = hallmark_frame(genes, corum, yeast, induced_degree(genes), expression_median(genes))
    frame.insert(1, "label", labels.reindex(genes).to_numpy("float64"))
    if frame["label"].isna().any():
        raise ValueError(f"{task}: a gene has no label")
    frame["label"] = frame["label"].astype("int64")
    age, _ = gene_age(pd.Index(genes.astype(str), name="gene_key"))
    frame["age_layer"] = age.to_numpy()
    return frame, thresholds


def cells_of(*columns: np.ndarray) -> np.ndarray:
    return pd.factorize(pd.MultiIndex.from_arrays([np.asarray(c) for c in columns]))[0]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = average_ranks(np.asarray(a, dtype="float64"))
    rb = average_ranks(np.asarray(b, dtype="float64"))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denominator = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return float("nan") if denominator == 0.0 else float(ra @ rb) / denominator


def bootstrap(cells: np.ndarray, statistic, label: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    point = statistic(None)
    resampler = Resampler(cells, seed=SEED)
    replicates = {name: np.empty(RESAMPLE_COUNT) for name in point}
    progress = Progress(RESAMPLE_COUNT, label)
    for r in range(RESAMPLE_COUNT):
        values = statistic(resampler.draw())
        for name in point:
            replicates[name][r] = values[name]
        progress.tick()
    rows = []
    for name, value in point.items():
        half_width, p = summarise(value, replicates[name])
        rows.append({"statistic": name, "point": value, "half_width": half_width, "p": p})
    return pd.DataFrame(rows), replicates


def one_per_family(frame: pd.DataFrame, generator: np.random.Generator) -> pd.DataFrame:
    order = generator.permutation(len(frame))
    return frame.iloc[order].drop_duplicates("cluster_id").sort_index()


def skip_draws(generator: np.random.Generator, sizes) -> None:
    for size in sizes:
        generator.permutation(int(size))


def share_row(mask: pd.Series) -> tuple[int, int, float, float]:
    size, count = int(len(mask)), int(mask.sum())
    share = count / size if size else float("nan")
    half = 1.96 * np.sqrt(share * (1 - share) / size) if size else float("nan")
    return count, size, share, half
