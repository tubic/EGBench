"""Exceptions to the four hallmarks on DEG and DepMap: the four groups of genes, the GO processes enriched among
non-essential genes with at least three hallmarks, and the share of each group the predictors rank in their top k.

Run from the release root with PYTHONPATH=src:analysis: python analysis/hallmark_exceptions/hallmark_groups.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from egbench.data import PROJECT_ROOT
from egbench.fusion import pooled
from egbench.modules import process_pairs, restrict_terms
from egbench.reference import GENE_EFFECT_COLUMN, deg_vote_counts, gene_effect_means
from egbench.stats import benjamini_hochberg_adjusted
from egbench.strata import QUARTILE_BANDS, quartile_bands, strata_table

from common.depmap_matrices import DEPENDENT_ABOVE, dependency_matrix
from common.feature_blocks import go_names
from common.hallmark_counts import HALLMARKS, RICH, TASK_NAMES
from common.hallmarks import gene_properties_path
from common.stats import METHODS, process_pool

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
WORK = HERE / "work"
TASKS: tuple[str, ...] = ("deg", "depmap")
FUSION = "fusion_top5"
COUNT = "hallmark_count"
PREDICTORS: tuple[str, ...] = METHODS + (FUSION, COUNT)
GROUPS: tuple[str, ...] = ("rich_essential", "rich_nonessential", "poor_essential", "poor_nonessential")
GROUP_NAMES = {
    "rich_essential": "essential genes with at least three hallmarks",
    "rich_nonessential": "non-essential genes with at least three hallmarks",
    "poor_essential": "essential genes with fewer than three hallmarks",
    "poor_nonessential": "non-essential genes with fewer than three hallmarks",
}
DEPMAP_UNIVERSE = PROJECT_ROOT / "datasets" / "DepMap" / "data" / "depmap26q1_gene_universe.parquet"
DEG_UNIVERSE = PROJECT_ROOT / "datasets" / "DEG" / "data" / "gene_universe.parquet"
DEPENDENT_SHARE = 0.1
TOP = 15
TERM_MINIMUM, TERM_MAXIMUM = 5, 200


def exception_genes_path(task: str) -> Path:
    return WORK / f"exception_genes_{task}.csv"


def assemble(task: str) -> pd.DataFrame:
    frame = pd.read_csv(gene_properties_path(task))
    frame["gene_key"] = frame["gene_key"].astype(str)
    frame["label"] = frame["label"].astype("int64")
    frame["count"] = frame["count_main"].astype("int64")
    rich = frame["count"] >= RICH
    frame["group"] = np.select(
        [rich & (frame.label == 1), rich & (frame.label == 0), ~rich & (frame.label == 1)],
        ["rich_essential", "rich_nonessential", "poor_essential"], "poor_nonessential")
    keys = pd.Index(frame["gene_key"], dtype="object")

    table = strata_table(task)
    table["gene_key"] = table["gene_key"].astype(str)
    table = table.set_index("gene_key")
    frame["quartile"] = pd.Series(quartile_bands(table["pubmed_count"]).to_numpy(), index=table.index).reindex(keys).to_numpy()
    frame["cluster_id"] = table["cluster_id"].reindex(keys).to_numpy()
    if frame["quartile"].isna().any() or frame["cluster_id"].isna().any():
        raise ValueError(f"{task}: a gene has no literature quartile or no protein family")

    depmap = pd.read_parquet(DEPMAP_UNIVERSE, columns=["gene_key", "gene_symbol", "official_common_essential",
                                                       "strongly_selective"])
    depmap["gene_key"] = depmap["gene_key"].astype(str)
    depmap = depmap.set_index("gene_key")
    frame["depmap_common_essential"] = depmap["official_common_essential"].astype("float64").reindex(keys).to_numpy()
    frame["depmap_strongly_selective"] = depmap["strongly_selective"].astype("float64").reindex(keys).to_numpy()
    votes = deg_vote_counts()
    votes.index = votes.index.astype(str)
    frame["deg_votes"] = votes.reindex(keys).to_numpy()
    symbols = pd.read_parquet(DEG_UNIVERSE, columns=["gene_key", "gene_symbol"])
    symbols = pd.concat([symbols, depmap.reset_index()[["gene_key", "gene_symbol"]]], ignore_index=True)
    symbols["gene_key"] = symbols["gene_key"].astype(str)
    frame["gene_symbol"] = symbols.drop_duplicates("gene_key").set_index("gene_key")["gene_symbol"].reindex(keys).to_numpy()

    effect = gene_effect_means()[GENE_EFFECT_COLUMN]
    effect.index = effect.index.astype(str)
    frame["gene_effect_mean"] = effect.reindex(keys).to_numpy()
    matrix = dependency_matrix()
    matrix.columns = matrix.columns.astype(str)
    measured = keys.intersection(matrix.columns)
    values = matrix[measured]
    share = (values > DEPENDENT_ABOVE).sum(axis=0) / values.notna().sum(axis=0)
    frame["dependent_share"] = share.reindex(keys).to_numpy()

    if task == "deg":
        known = frame["depmap_common_essential"].notna()
        evidence = (frame["depmap_common_essential"] == 1) | (frame["depmap_strongly_selective"] == 1)
    else:
        known = frame["deg_votes"].notna()
        evidence = frame["deg_votes"] >= 1
    frame["evidence_known"] = known.astype("int64")
    frame["evidence"] = np.where(known, evidence.astype("float64"), np.nan)
    return frame


def describe(task: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        part = frame[frame.group == group]
        row = {"cohort_id": task, "task": TASK_NAMES[task], "group": group, "group_name": GROUP_NAMES[group],
               "genes": int(len(part)), "share_of_class": float(len(part) / (frame.label == part.label.iloc[0]).sum())}
        for band in QUARTILE_BANDS:
            row[f"share_{band}"] = float((part.quartile == band).mean())
        for hallmark in HALLMARKS:
            row[f"share_{hallmark}"] = float(part[hallmark].mean())
        for k in range(5):
            row[f"count_{k}"] = int((part["count"] == k).sum())
        row["share_core_function"] = float(part.core_function.mean())
        row["ppi_degree_median"] = float(part.ppi_degree.median())
        row["tcga_median_median"] = float(part.tcga_median.median())
        row["evidence_known"] = int(part.evidence_known.sum())
        row["gene_effect_median"] = float(part.gene_effect_mean.median())
        row["dependent_share_median"] = float(part.dependent_share.median())
        row["dependent_share_at_least_0.1"] = float((part.dependent_share.dropna() >= DEPENDENT_SHARE).mean())
        row["depmap_common_essential_share"] = float((part.depmap_common_essential.dropna() == 1).mean())
        row["depmap_strongly_selective_share"] = float((part.depmap_strongly_selective.dropna() == 1).mean())
        row["deg_vote_share"] = float((part.deg_votes.dropna() >= 1).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def enrichment(task: str, frame: pd.DataFrame, foreground: str, background_label: int, names: dict[str, str]) -> pd.DataFrame:
    background = frame[frame.label == background_label].set_index("gene_key")
    terms = restrict_terms(process_pairs(), background.index, minimum=TERM_MINIMUM, maximum=TERM_MAXIMUM)
    terms = terms.astype({"gene_key": str})
    inside = (background.group == foreground)
    total, drawn = int(len(background)), int(inside.sum())
    rows = []
    for term, members in terms.groupby("term_id"):
        present = inside.reindex(members.gene_key).dropna().astype(bool)
        size, hits = int(len(present)), int(present.sum())
        rows.append({"cohort_id": task, "task": TASK_NAMES[task], "foreground": foreground, "term_id": term,
                     "term_name": names.get(term, ""), "background_genes": size, "foreground_genes": hits,
                     "fold_enrichment": (hits / size) / (drawn / total),
                     "p": float(hypergeom.sf(hits - 1, total, size, drawn))})
    table = pd.DataFrame(rows)
    table["q"] = benjamini_hochberg_adjusted(table["p"].to_numpy("float64"))
    table = table.sort_values(["q", "p", "term_id"], kind="stable").reset_index(drop=True)
    table.insert(3, "rank", np.arange(1, len(table) + 1))
    table["top15"] = table["rank"] <= TOP
    table["background_total"], table["foreground_total"] = total, drawn
    return table


def methods(task: str, frame: pd.DataFrame) -> pd.DataFrame:
    base = None
    scores: dict[str, np.ndarray] = {}
    for predictor in PREDICTORS:
        if predictor == COUNT:
            continue
        pool = pooled(predictor, task)
        pool["gene_key"] = pool["gene_key"].astype(str)
        pool = pool.sort_values(["fold", "gene_key"], kind="stable").reset_index(drop=True)
        if base is None:
            base = pool[["gene_key", "fold"]].copy()
            base["label"] = pool["label"].to_numpy("int64")
        elif not pool["gene_key"].equals(base["gene_key"]):
            raise ValueError(f"{task} {predictor}: the scored genes differ")
        scores[predictor] = pool["y_score"].to_numpy("float64")
    genes = frame.set_index("gene_key").reindex(base["gene_key"])
    if genes["label"].isna().any() or not np.array_equal(genes["label"].to_numpy("int64"), base["label"].to_numpy()):
        raise ValueError(f"{task}: the scored genes and labels differ from the gene table")
    scores[COUNT] = genes["count"].to_numpy("float64")
    labels = base["label"].to_numpy()
    fold = base["fold"].to_numpy()
    group = genes["group"].to_numpy()
    rows = []
    for predictor in PREDICTORS:
        chosen = np.zeros(len(base), dtype=bool)
        for f in np.unique(fold):
            inside = np.flatnonzero(fold == f)
            k = int(labels[inside].sum())
            order = inside[np.argsort(-scores[predictor][inside], kind="stable")]
            chosen[order[:k]] = True
        false_positive = chosen & (labels == 0)
        values = {
            "recall_poor_essential": float(chosen[group == "poor_essential"].mean()),
            "recall_rich_essential": float(chosen[group == "rich_essential"].mean()),
            "false_positive_rich_share": (float((group[false_positive] == "rich_nonessential").mean())
                                          if false_positive.any() else float("nan")),
        }
        for quantity, value in values.items():
            rows.append({"cohort_id": task, "task": TASK_NAMES[task], "predictor": predictor, "quantity": quantity,
                         "point": value})
    return pd.DataFrame(rows)


def run(task: str) -> dict[str, pd.DataFrame]:
    frame = assemble(task)
    result = {
        "genes": frame.assign(cohort_id=task),
        "counts": describe(task, frame),
        "go_rich_nonessential": enrichment(task, frame, "rich_nonessential", 0, go_names()),
        "methods": methods(task, frame),
    }
    print(f"{TASK_NAMES[task]} done", flush=True)
    return result


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with process_pool(len(TASKS)) as pool:
        results = dict(zip(TASKS, pool.map(run, TASKS)))
    for task in TASKS:
        results[task]["genes"].to_csv(exception_genes_path(task), index=False)
    counts = pd.concat([results[task]["counts"] for task in TASKS], ignore_index=True)
    counts.to_csv(RESULTS / "exception_counts.csv", index=False)
    go = pd.concat([results[task]["go_rich_nonessential"] for task in TASKS], ignore_index=True)
    go[go["top15"]].to_csv(RESULTS / "exception_go_rich_nonessential.csv", index=False)
    table = pd.concat([results[task]["methods"] for task in TASKS], ignore_index=True)
    table.to_csv(RESULTS / "exception_methods.csv", index=False)


if __name__ == "__main__":
    main()
