"""AUROC of the BioGRID node2vec and TCGA expression component models in genes split by literature count and PPI degree on DEG, DepMap and Mouse.

Run from the release root after feature_groups/alone.py: python analysis/research_attention/network_degree.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from egbench.stats import RESAMPLE_COUNT, holm_adjusted
from egbench.strata import QUARTILE_BANDS, quartile_bands, strata_table

from common.feature_blocks import COHORT_NAMES, block_array, cohort_genes
from common.stats import ANALYSIS_DIRECTORY, fast_auroc, percentile, process_pool, resampled_statistics, write_table
from feature_groups.alone import alone_path

RESULTS = ANALYSIS_DIRECTORY / "research_attention" / "results"
TASKS: tuple[str, ...] = ("deg", "depmap", "mouse")
NETWORK = "biogrid_n2v"
REFERENCE = "tcga_pca"
GROUPS: tuple[str, ...] = (NETWORK, REFERENCE)
LESS, MORE = "less", "more"
LOW, HIGH = "low", "high"
LITERATURE_HALVES: tuple[str, ...] = (LESS, MORE)
DEGREE_HALVES: tuple[str, ...] = (LOW, HIGH)
M1 = f"degree_effect:{NETWORK}"
M2 = "degree_effect:network_minus_reference"


def read_scores(task: str, key: str) -> pd.DataFrame:
    frame = pd.read_csv(alone_path(task, key))
    frame["gene_key"] = frame["gene_key"].astype(str)
    return frame[frame["role"] == "test"].sort_values(["fold", "gene_key"], kind="stable").reset_index(drop=True)


def assemble(task: str) -> pd.DataFrame:
    frames = {key: read_scores(task, key) for key in GROUPS}
    template = frames[NETWORK][["gene_key", "fold", "label"]]
    if not frames[REFERENCE][["gene_key", "fold"]].equals(template[["gene_key", "fold"]]):
        raise ValueError(f"{task}: the two models score different genes")
    out = template.copy()
    out["label"] = out["label"].to_numpy("int64")
    for key, frame in frames.items():
        column = np.empty(len(frame))
        for fold in sorted(frame["fold"].unique()):
            rows = (frame["fold"] == fold).to_numpy()
            column[rows] = percentile(frame.loc[rows, "y_score"].to_numpy("float64"))
        out[key] = column
    table = strata_table(task)
    bands = pd.Series(quartile_bands(table["pubmed_count"]).to_numpy(), index=table["gene_key"].astype(str))
    out["quartile"] = bands.reindex(out["gene_key"]).to_numpy()
    if out["quartile"].isna().any():
        raise ValueError(f"{task}: a test gene has no literature quartile")
    out["literature_half"] = np.where(out["quartile"].isin(("q1", "q2")), LESS, MORE)
    degree = pd.Series(np.asarray(block_array(task, 0, "ppi_degree"))[:, 0].astype("float64"),
                       index=pd.Index(cohort_genes(task).astype(str)))
    out["degree"] = degree.reindex(out["gene_key"]).to_numpy()
    if out["degree"].isna().any():
        raise ValueError(f"{task}: a test gene has no PPI degree")
    median = float(np.median(out["degree"].to_numpy()))
    out["degree_half"] = np.where(out["degree"].to_numpy() >= median, HIGH, LOW)
    return out


def statistics(frame: pd.DataFrame, take: np.ndarray | None) -> dict[str, float]:
    labels = frame["label"].to_numpy()
    bands = frame["quartile"].to_numpy()
    literature = frame["literature_half"].to_numpy()
    degree = frame["degree_half"].to_numpy()
    scores = {key: frame[key].to_numpy() for key in GROUPS}
    if take is not None:
        labels, bands, literature, degree = labels[take], bands[take], literature[take], degree[take]
        scores = {key: values[take] for key, values in scores.items()}
    out: dict[str, float] = {}
    for key in GROUPS:
        for band in QUARTILE_BANDS:
            inside = bands == band
            out[f"{key}:{band}"] = fast_auroc(labels[inside], scores[key][inside])
        out[f"{key}:drop"] = out[f"{key}:q4"] - out[f"{key}:q1"]
        for lit in LITERATURE_HALVES:
            for half in DEGREE_HALVES:
                inside = (literature == lit) & (degree == half)
                out[f"{key}:{lit}:{half}"] = fast_auroc(labels[inside], scores[key][inside])
        out[f"degree_effect:{key}"] = float(np.mean(
            [out[f"{key}:{lit}:{HIGH}"] - out[f"{key}:{lit}:{LOW}"] for lit in LITERATURE_HALVES]))
    out[M2] = out[f"degree_effect:{NETWORK}"] - out[f"degree_effect:{REFERENCE}"]
    return out


def run(task: str) -> pd.DataFrame:
    summary = resampled_statistics(assemble(task), statistics, RESAMPLE_COUNT, f"{COHORT_NAMES[task]} resampling")
    return pd.DataFrame([{"cohort_id": task, "task": COHORT_NAMES[task], "statistic": name, "point": point,
                          "half_width": half, "p": p} for name, (point, half, p) in summary.items()])


def main() -> None:
    with process_pool(len(TASKS)) as pool:
        table = pd.concat(list(pool.map(run, TASKS)), ignore_index=True)
    table["holm_p"] = np.nan
    for name in (M1, M2):
        rows = table["statistic"].eq(name)
        table.loc[rows, "holm_p"] = holm_adjusted(table.loc[rows, "p"].to_numpy("float64"))
    write_table(table, RESULTS / "network_degree.csv")
    print("wrote network_degree.csv", flush=True)


if __name__ == "__main__":
    main()
