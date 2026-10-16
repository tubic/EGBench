"""Essential gene share, the four hallmarks and PPI degree inside each PubMed literature quartile on DEG and DepMap.

Run from the release root after essential_gene_hallmarks/four_properties.py: python analysis/research_attention/attention_properties.py
"""

from __future__ import annotations

import pandas as pd

from egbench.strata import QUARTILE_BANDS, quartile_bands, strata_table

from common.hallmarks import TASK_NAMES, TASKS, gene_properties_path, share_interval
from common.stats import ANALYSIS_DIRECTORY, write_table

RESULTS = ANALYSIS_DIRECTORY / "research_attention" / "results"
PROPERTIES: tuple[str, ...] = ("complex", "old", "connected", "expressed")


def genes_with_properties(task: str) -> pd.DataFrame:
    table = strata_table(task)
    table["gene_key"] = table["gene_key"].astype(str)
    table["quartile"] = quartile_bands(table["pubmed_count"]).to_numpy()
    props = pd.read_csv(gene_properties_path(task))
    props["gene_key"] = props["gene_key"].astype(str)
    merged = table.merge(props, on="gene_key", how="inner", suffixes=("", "_property"), validate="one_to_one")
    if len(merged) != len(table) or len(merged) != len(props):
        raise ValueError(f"{task}: the strata table and the property table hold different genes")
    if not (merged["label"].to_numpy("float64") == merged["label_property"].to_numpy("float64")).all():
        raise ValueError(f"{task}: the two tables disagree on a label")
    return merged


def properties_by_quartile(task: str) -> pd.DataFrame:
    merged = genes_with_properties(task)
    labels = merged["label"].to_numpy("int64")
    rows = []
    for band in QUARTILE_BANDS:
        inside = (merged["quartile"] == band).to_numpy()
        sub = merged[inside]
        share, half = share_interval(labels, inside.astype("int64"))
        row = {
            "cohort_id": task, "task": TASK_NAMES[task], "quartile": band,
            "pubmed_low": float(sub["pubmed_count"].min()), "pubmed_high": float(sub["pubmed_count"].max()),
            "genes": int(len(sub)), "essential": int(labels[inside].sum()),
            "essential_share": share, "essential_share_half_width": half,
        }
        for prop in PROPERTIES:
            row[f"{prop}_share"] = float(sub[prop].mean())
        row["none_share"] = float((sub["count_main"] == 0).mean())
        row["all_four_share"] = float((sub["count_main"] == 4).mean())
        row["ppi_degree_median"] = float(sub["ppi_degree"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    table = pd.concat([properties_by_quartile(task) for task in TASKS], ignore_index=True)
    write_table(table, RESULTS / "attention_properties.csv")
    print("wrote attention_properties.csv", flush=True)


if __name__ == "__main__":
    main()
