"""AUROC of the hallmark count inside each literature quartile, TCGA coverage of the excluded genes, and the summary of the 15 methods' Q1 minus Q4 AUROC.

Run from the release root after attention_trend.py and essential_gene_hallmarks/four_properties.py:
python analysis/research_attention/attention_followup.py
"""

from __future__ import annotations

import pandas as pd

from egbench.contracts import TargetKind
from egbench.data import PROJECT_ROOT, _EXPRESSION_FILE, _expression_medians
from egbench.metrics.discrimination import auroc
from egbench.stats import confidence_interval
from egbench.strata import QUARTILE_BANDS, quartile_bands, strata_table

from common.hallmarks import TASK_NAMES, TASKS, gene_properties_path
from common.stats import BINARY_COHORTS, METHODS, write_table
from research_attention.attention_trend import RESULTS, TREND
from research_attention.excluded_genes import DATA


def genes_with_properties(task: str) -> pd.DataFrame:
    table = strata_table(task)
    table["gene_key"] = table["gene_key"].astype(str)
    table["quartile"] = quartile_bands(table["pubmed_count"]).to_numpy()
    props = pd.read_csv(gene_properties_path(task))
    props["gene_key"] = props["gene_key"].astype(str)
    merged = table[["gene_key", "label", "quartile"]].merge(
        props[["gene_key", "label", "count_main"]], on="gene_key", how="inner", suffixes=("", "_property"),
        validate="one_to_one")
    if len(merged) != len(table) or len(merged) != len(props):
        raise ValueError(f"{task}: the strata table and the property table hold different genes")
    return merged


def properties_by_quartile(task: str) -> pd.DataFrame:
    frame = genes_with_properties(task)
    labels = frame["label"].to_numpy("int64")
    counts = frame["count_main"].to_numpy("float64")
    bands = frame["quartile"].to_numpy()
    rows = []
    for band in QUARTILE_BANDS:
        inside = bands == band
        interval = confidence_interval(labels[inside].astype("float64"), counts[inside], auroc,
                                       target_kind=TargetKind.BINARY)
        rich = counts[inside] >= 3
        rows.append({
            "cohort_id": task, "task": TASK_NAMES[task], "quartile": band,
            "genes": int(inside.sum()), "essential": int(labels[inside].sum()),
            "auroc": float(interval.point), "auroc_half_width": float(interval.half_width),
            "genes_three_or_more": int(rich.sum()), "essential_three_or_more": int(labels[inside][rich].sum()),
        })
    return pd.DataFrame(rows)


def excluded_tcga(task: str, medians: pd.Series) -> pd.DataFrame:
    table = pd.read_parquet(DATA / f"{task}_method_availability.parquet")
    table = table[table["label"].notna()].copy()
    table["gene_key"] = table["gene_key"].astype(str)
    measured = table["gene_key"].isin(medians.index)
    zero = table["gene_key"].map(medians).eq(0.0) & measured
    rows = []
    for group, mask in (("excluded", ~table["all_methods_extractable"]), ("retained", table["all_methods_extractable"])):
        n_measured = int((mask & measured).sum())
        rows.append({
            "cohort_id": task, "group": group, "genes": int(mask.sum()),
            "essential": int((table.loc[mask, "label"] == 1).sum()),
            "not_in_tcga": int((mask & ~measured).sum()), "in_tcga": n_measured,
            "median_zero": int((mask & zero).sum()),
            "median_zero_share_of_measured": float((mask & zero).sum() / n_measured) if n_measured else float("nan"),
        })
    return pd.DataFrame(rows)


def method_drops() -> pd.DataFrame:
    trend = pd.read_csv(TREND)
    trend = trend[trend["metric"] == "auroc"]
    rows = trend[trend["statistic"].isin([f"{m}:q1_minus_q4" for m in METHODS])].copy()
    if len(rows) != len(METHODS) * len(BINARY_COHORTS):
        raise ValueError(f"attention_trend holds {len(rows)} method differences")
    rows["method"] = rows["statistic"].str.split(":").str[0]
    rows["drop"] = -rows["point"]
    rows["upper_below_zero"] = rows["point"] + rows["half_width"] < 0
    return rows.groupby("cohort_id").agg(
        methods=("method", "size"), negative=("point", lambda v: int((v < 0).sum())),
        upper_below_zero=("upper_below_zero", "sum"), drop_min=("drop", "min"), drop_median=("drop", "median"),
        drop_max=("drop", "max")).reset_index()


def main() -> None:
    write_table(pd.concat([properties_by_quartile(task) for task in TASKS], ignore_index=True),
                RESULTS / "attention_properties_auroc.csv")
    medians = _expression_medians(PROJECT_ROOT / _EXPRESSION_FILE)
    medians.index = medians.index.astype(str)
    write_table(pd.concat([excluded_tcga(task, medians) for task in TASKS], ignore_index=True),
                RESULTS / "excluded_genes_tcga.csv")
    write_table(method_drops(), RESULTS / "method_drops_summary.csv")
    print("wrote attention_properties_auroc.csv, excluded_genes_tcga.csv and method_drops_summary.csv", flush=True)


if __name__ == "__main__":
    main()
