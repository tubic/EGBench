"""Fit one gradient boosting model on the feature groups each of seven network or expression methods reads, and report its AUROC in each literature quartile.

Run from the release root after feature_groups/blocks.py: python analysis/research_attention/input_models.py [--workers N]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from egbench.stats import RESAMPLE_COUNT
from egbench.strata import QUARTILE_BANDS, quartile_bands, strata_table

from common.feature_blocks import (
    BLOCKS, COHORT_NAMES, FOLDS, METHOD_NAMES, WORKERS, block_width, fit_set, run_pool, score_frame, test_rows,
)
from common.stats import ANALYSIS_DIRECTORY, fast_auroc, percentile, process_pool, resampled_statistics, write_table

WORK = ANALYSIS_DIRECTORY / "research_attention" / "work"
RESULTS = ANALYSIS_DIRECTORY / "research_attention" / "results"
COHORTS: tuple[str, ...] = ("deg", "depmap")
INPUT_METHODS: tuple[str, ...] = ("help", "mahi", "xgep", "deephe", "deepsnap", "epgat", "esssubgraph")
INPUT_BLOCKS: dict[str, tuple[str, ...]] = {
    method: tuple(block.key for block in BLOCKS if method in block.readers) for method in INPUT_METHODS
}


def fit_job(job: tuple[str, str, int]) -> dict:
    cohort, method, fold = job
    return fit_set(cohort, fold, INPUT_BLOCKS[method])


def build(cohort: str) -> pd.DataFrame:
    frame = pd.read_csv(WORK / f"input_scores_{cohort}.csv")
    frame["gene_key"] = frame["gene_key"].astype(str)
    for method in INPUT_METHODS:
        column = np.empty(len(frame))
        for fold in FOLDS:
            rows = (frame["fold"] == fold).to_numpy()
            column[rows] = percentile(frame.loc[rows, method].to_numpy("float64"))
        frame[f"input:{method}"] = column
    table = strata_table(cohort)
    bands = pd.Series(quartile_bands(table["pubmed_count"]).to_numpy(), index=table["gene_key"].astype(str))
    frame["quartile"] = bands.reindex(frame["gene_key"]).to_numpy()
    if frame["quartile"].isna().any():
        raise ValueError(f"{cohort}: a test gene has no literature quartile")
    frame["label"] = frame["label"].astype("int64")
    return frame


def statistics(frame: pd.DataFrame, take: np.ndarray | None) -> dict[str, float]:
    labels = frame["label"].to_numpy()
    bands = frame["quartile"].to_numpy()
    scores = {m: frame[f"input:{m}"].to_numpy() for m in INPUT_METHODS}
    if take is not None:
        labels, bands = labels[take], bands[take]
        scores = {m: values[take] for m, values in scores.items()}
    out: dict[str, float] = {}
    for band in QUARTILE_BANDS:
        inside = bands == band
        for method in INPUT_METHODS:
            out[f"input:{method}:{band}"] = fast_auroc(labels[inside], scores[method][inside])
    return out


def run(cohort: str) -> pd.DataFrame:
    summary = resampled_statistics(build(cohort), statistics, RESAMPLE_COUNT, f"{COHORT_NAMES[cohort]} resampling")
    return pd.DataFrame([{"cohort_id": cohort, "task": COHORT_NAMES[cohort], "statistic": name, "point": point,
                          "half_width": half, "p": p} for name, (point, half, p) in summary.items()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    arguments = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    widths = {m: sum(block_width("deg", 0, key) for key in keys) for m, keys in INPUT_BLOCKS.items()}
    for method in INPUT_METHODS:
        print(f"{METHOD_NAMES[method]} input: {', '.join(INPUT_BLOCKS[method])}, {widths[method]} features", flush=True)
    order = sorted(INPUT_METHODS, key=lambda method: -widths[method])
    jobs = [(cohort, method, fold) for method in order for cohort in COHORTS for fold in FOLDS]
    fits = run_pool(fit_job, jobs, arguments.workers, "input model fits")
    for cohort in COHORTS:
        table = None
        for method in INPUT_METHODS:
            rows = test_rows(score_frame(cohort, {fold: fits[(cohort, method, fold)] for fold in FOLDS}))
            rows = rows[["gene_key", "fold", "label", "y_score"]].rename(columns={"y_score": method})
            table = rows if table is None else table.merge(rows, on=["gene_key", "fold", "label"], validate="one_to_one")
        table.to_csv(WORK / f"input_scores_{cohort}.csv", index=False)
    with process_pool(len(COHORTS)) as pool:
        table = pd.concat(list(pool.map(run, COHORTS)), ignore_index=True)
    write_table(table, RESULTS / "input_models.csv")
    print("wrote input_models.csv", flush=True)


if __name__ == "__main__":
    main()
