"""Labelled genes the 15-method common gene set excludes on DEG and DepMap: PubMed counts and the methods that cannot score them.

Run from the release root: python analysis/research_attention/excluded_genes.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from egbench.baselines import literature_counts

from common.stats import ANALYSIS_DIRECTORY, GENE_KEY, METHODS, write_table

COHORTS: tuple[str, ...] = ("deg", "depmap")
DATA = ANALYSIS_DIRECTORY / "research_attention" / "data"
RESULTS = ANALYSIS_DIRECTORY / "research_attention" / "results"
TABLE_COLUMN = "all_methods_extractable"


def availability(cohort: str) -> pd.DataFrame:
    frame = pd.read_parquet(DATA / f"{cohort}_method_availability.parquet")
    frame[GENE_KEY] = frame[GENE_KEY].astype("object")
    return frame.loc[frame["label"].notna()].reset_index(drop=True)


def describe(literature: np.ndarray, edge: float) -> dict:
    return {
        "genes": int(len(literature)),
        "pubmed_median": float(np.median(literature)) if len(literature) else np.nan,
        "least_studied_share": float(np.mean(literature <= edge)) if len(literature) else np.nan,
    }


def run(cohort: str) -> dict[str, pd.DataFrame]:
    frame = availability(cohort)
    keys = pd.Index(frame[GENE_KEY].astype("object"))
    literature = literature_counts().reindex(keys).fillna(0.0).to_numpy("float64")
    kept = frame[TABLE_COLUMN].to_numpy(dtype=bool)
    label = frame["label"].astype("int64").to_numpy()
    edge = float(np.quantile(literature[kept], 0.25))
    rows = []
    for group, mask in (("retained", kept), ("excluded", ~kept)):
        for name, selection in (("all", np.ones(len(frame), dtype=bool)), ("essential", label == 1),
                                ("nonessential", label == 0)):
            rows.append({"cohort_id": cohort, "group": group, "class_name": name, "literature_edge": edge,
                         **describe(literature[mask & selection], edge)})
    rows.append({"cohort_id": cohort, "group": "candidate", "class_name": "all", "literature_edge": edge,
                 **describe(literature, edge)})
    counts = []
    for method in METHODS:
        missing = ~frame[f"extractable__{method}"].to_numpy(dtype=bool)
        others = frame[[f"extractable__{m}" for m in METHODS if m != method]].all(axis=1).to_numpy(dtype=bool)
        counts.append({
            "cohort_id": cohort, "method": method, "excluded": int(missing.sum()),
            "excluded_positive": int((missing & (label == 1)).sum()),
            "excluded_negative": int((missing & (label == 0)).sum()),
            "only_this_method": int((missing & others).sum()),
        })
    summary = {
        "cohort_id": cohort, "candidate_labeled": int(len(frame)), "common": int(kept.sum()),
        "excluded": int((~kept).sum()), "excluded_positive": int(((~kept) & (label == 1)).sum()),
        "excluded_negative": int(((~kept) & (label == 0)).sum()),
    }
    return {"profile": pd.DataFrame(rows), "methods": pd.DataFrame(counts), "summary": pd.DataFrame([summary])}


def main() -> None:
    parts = {"profile": [], "methods": [], "summary": []}
    for cohort in COHORTS:
        for key, frame in run(cohort).items():
            parts[key].append(frame)
    for key, frames in parts.items():
        write_table(pd.concat(frames, ignore_index=True), RESULTS / f"excluded_genes_{key}.csv")
    print("wrote excluded_genes_profile.csv, excluded_genes_methods.csv and excluded_genes_summary.csv", flush=True)


if __name__ == "__main__":
    main()
