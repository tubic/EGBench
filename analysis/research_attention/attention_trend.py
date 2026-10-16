"""AUROC of the 15 methods inside the four PubMed literature quartiles on the five binary tasks, with gene resampling intervals.

Run from the release root: python analysis/research_attention/attention_trend.py [--workers N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from egbench.data import primary_scope, scope_gene_keys
from egbench.stats import RESAMPLE_COUNT
from egbench.strata import QUARTILE_BANDS, quartile_bands, strata_table

from common.stats import (
    ANALYSIS_DIRECTORY, BINARY_COHORTS, FOLDS, METHODS, blind_scores, fast_auroc, fold_truth, percentile,
    process_pool, resampled_statistics, write_table,
)

RESULTS: Path = ANALYSIS_DIRECTORY / "research_attention" / "results"
TREND = RESULTS / "attention_trend.csv"


def assemble(cohort: str) -> pd.DataFrame:
    table = strata_table(cohort)
    bands = quartile_bands(table["pubmed_count"]).set_axis(table["gene_key"].astype("object"))
    kept = scope_gene_keys(cohort, primary_scope(cohort))
    frames = []
    for fold in FOLDS:
        truth = fold_truth(cohort, fold)
        keys = pd.Index(truth["gene_key"].astype("object"))
        frame = pd.DataFrame({"gene_key": keys, "fold": fold, "label": truth["label"].to_numpy("int64")})
        for method in METHODS:
            scores = blind_scores(method, cohort, fold).reindex(keys)
            if scores.isna().any():
                raise ValueError(f"{method} {cohort} fold {fold}: missing scores")
            frame[method] = percentile(scores.to_numpy())
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame["quartile"] = bands.reindex(frame["gene_key"]).to_numpy()
    if frame["quartile"].isna().any():
        raise ValueError(f"{cohort}: a tested gene has no literature quartile")
    if kept is not None:
        frame = frame[frame["gene_key"].isin(kept)].reset_index(drop=True)
    return frame


def statistics(frame: pd.DataFrame, take: np.ndarray | None) -> dict[str, float]:
    labels = frame["label"].to_numpy()
    quartiles = frame["quartile"].to_numpy()
    scores = {m: frame[m].to_numpy() for m in METHODS}
    if take is not None:
        labels, quartiles = labels[take], quartiles[take]
        scores = {m: s[take] for m, s in scores.items()}
    out: dict[str, float] = {}
    for band in QUARTILE_BANDS:
        inside = quartiles == band
        for m in METHODS:
            out[f"{m}:{band}"] = fast_auroc(labels[inside], scores[m][inside])
    for m in METHODS:
        out[f"{m}:q1_minus_q4"] = out[f"{m}:q1"] - out[f"{m}:q4"]
    return out


def run(cohort: str) -> pd.DataFrame:
    frame = assemble(cohort)
    summary = resampled_statistics(frame, statistics, RESAMPLE_COUNT, f"{cohort} resampling")
    rows = [{"cohort_id": cohort, "metric": "auroc", "statistic": name, "point": point, "half_width": half, "p": p}
            for name, (point, half, p) in summary.items()]
    counts = frame.groupby("quartile").agg(genes=("label", "size"), positives=("label", "sum")).reset_index()
    for _, row in counts.iterrows():
        for quantity in ("genes", "positives"):
            rows.append({"cohort_id": cohort, "metric": "count", "statistic": f"{quantity}:{row['quartile']}",
                         "point": float(row[quantity]), "half_width": np.nan, "p": np.nan})
    print(f"{cohort} done", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=len(BINARY_COHORTS))
    arguments = parser.parse_args()
    with process_pool(arguments.workers) as pool:
        table = pd.concat(list(pool.map(run, BINARY_COHORTS)), ignore_index=True)
    write_table(table, TREND)
    print("wrote attention_trend.csv", flush=True)


if __name__ == "__main__":
    main()
