"""Essential gene shares by gene age layer, by PPI degree and expression quintiles, and by the number of the four hallmarks on DEG and DepMap.

Run from the release root after feature_biology.py: python analysis/essential_gene_hallmarks/four_properties.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from common.feature_blocks import WORKERS, run_pool
from common.hallmarks import (
    LAYERS, RESULTS, STATIC, TASK_NAMES, TASKS, difference_interval, feature_values_path, gene_age,
    gene_properties_path, share_interval, species_layers, write_table,
)

PROPERTIES: tuple[str, ...] = ("complex", "old", "connected", "expressed")
SCOPES = {"all": "all genes", "other": "outside the core functions"}
GENE_COLUMNS = ["gene_key", "label", "core_function", "corum_any", "yeast_ortholog", "ppi_degree", "tcga_median",
                "age_layer", "complex", "old", "connected", "expressed", "count_main", "age_oldest",
                "ppi_quintile", "tcga_quintile"]


def write_atomic(frame: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def bins(values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(values, np.arange(1, count) / count)
    return np.searchsorted(edges, values, side="left") + 1, edges


def gene_table(task: str) -> tuple[pd.DataFrame, dict[str, float], dict[str, np.ndarray]]:
    values = pd.read_csv(feature_values_path(task))
    values["gene_key"] = values["gene_key"].astype(str)
    genes = pd.Index(values["gene_key"], name="gene_key")
    age, _ = gene_age(genes)
    frame = values[["gene_key", "label", "core_function", "corum_any", "yeast_ortholog", "ppi_degree",
                    "tcga_median"]].copy()
    frame["age_layer"] = age.to_numpy()
    if not ((frame["age_layer"] == 1) == (frame["yeast_ortholog"] == 1)).all():
        raise ValueError(f"{task}: the oldest layer and the yeast orthologue column disagree")
    ppi = frame["ppi_degree"].to_numpy("float64")
    tcga = frame["tcga_median"].to_numpy("float64")
    thresholds = {"ppi_median": float(np.median(ppi)), "tcga_median": float(np.median(tcga))}
    frame["complex"] = (frame["corum_any"] == 1).astype("int64")
    frame["old"] = (frame["age_layer"] == 1).astype("int64")
    frame["connected"] = (ppi >= thresholds["ppi_median"]).astype("int64")
    frame["expressed"] = (tcga >= thresholds["tcga_median"]).astype("int64")
    frame["count_main"] = frame[list(PROPERTIES)].sum(axis=1)
    frame["age_oldest"] = frame["old"]
    edges = {}
    for key, column in (("ppi", ppi), ("tcga", tcga)):
        frame[f"{key}_quintile"], edges[key] = bins(column, 5)
    return frame[GENE_COLUMNS], thresholds, edges


_TABLES: dict[str, pd.DataFrame] = {}


def _table(task: str) -> pd.DataFrame:
    if task not in _TABLES:
        _TABLES[task] = pd.read_csv(gene_properties_path(task))
    return _TABLES[task]


def interval_job(job: tuple) -> dict:
    kind, task, scope, column, a, b = job
    frame = _table(task)
    if scope == "other":
        frame = frame[frame["core_function"] == 0]
    labels = frame["label"].to_numpy("float64")
    if kind == "share":
        member = (frame[column] == a).to_numpy()
        point, half = share_interval(labels, member)
        return {"genes": int(member.sum()), "essentials": int(labels[member].sum()), "point": point, "half_width": half}
    group_a, group_b = (frame[column] == a).to_numpy(), (frame[column] == b).to_numpy()
    point, half = difference_interval(labels, group_a, group_b)
    return {"genes_a": int(group_a.sum()), "genes_b": int(group_b.sum()), "point": point, "half_width": half}


def jobs_for(task: str) -> list[tuple]:
    jobs = [("share", task, "all", "age_layer", layer.index, None) for layer in LAYERS]
    jobs.append(("difference", task, "all", "age_oldest", 1, 0))
    for scope in SCOPES:
        jobs += [("share", task, scope, "count_main", k, None) for k in range(5)]
        jobs += [("difference", task, scope, "count_main", k + 1, k) for k in range(4)]
    return jobs


def age_table(results: dict, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for layer in LAYERS:
            r = results[("share", task, "all", "age_layer", layer.index, None)]
            rows.append({"cohort_id": task, "task": TASK_NAMES[task], "grouping": "layer", "group": layer.index,
                         "group_name": layer.name, **r,
                         "lower": r["point"] - r["half_width"], "upper": r["point"] + r["half_width"]})
        r = results[("difference", task, "all", "age_oldest", 1, 0)]
        frame = tables[task]
        rows.append({"cohort_id": task, "task": TASK_NAMES[task], "grouping": "oldest_minus_rest", "group": 1,
                     "group_name": "oldest layer minus the other eight",
                     "genes": r["genes_a"], "essentials": int(frame.loc[frame.age_layer == 1, "label"].sum()),
                     "point": r["point"], "half_width": r["half_width"],
                     "lower": r["point"] - r["half_width"], "upper": r["point"] + r["half_width"],
                     "rest_genes": r["genes_b"], "rest_essentials": int(frame.loc[frame.age_layer != 1, "label"].sum())})
    return pd.DataFrame(rows)


def grid_tables(tables: dict[str, pd.DataFrame], edges: dict[str, dict[str, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells, bin_rows = [], []
    for task in TASKS:
        frame = tables[task]
        for pq in range(1, 6):
            for tq in range(1, 6):
                part = frame[(frame.ppi_quintile == pq) & (frame.tcga_quintile == tq)]
                cells.append({"cohort_id": task, "task": TASK_NAMES[task], "ppi_quintile": pq, "tcga_quintile": tq,
                              "genes": int(len(part)), "essentials": int(part.label.sum()),
                              "share": float(part.label.mean()) if len(part) else float("nan")})
        for quantity, key in (("ppi_degree", "ppi"), ("tcga_median", "tcga")):
            cut = edges[task][key]
            for k in range(1, 6):
                part = frame[frame[f"{key}_quintile"] == k]
                bin_rows.append({
                    "cohort_id": task, "task": TASK_NAMES[task], "quantity": quantity, "bin": k,
                    "lower_exclusive": float(cut[k - 2]) if k > 1 else float("-inf"),
                    "upper_inclusive": float(cut[k - 1]) if k < 5 else float("inf"),
                    "value_min": float(part[quantity].min()) if len(part) else float("nan"),
                    "value_max": float(part[quantity].max()) if len(part) else float("nan"),
                    "genes": int(len(part)), "essentials": int(part.label.sum()),
                    "share": float(part.label.mean()) if len(part) else float("nan")})
    return pd.DataFrame(cells), pd.DataFrame(bin_rows)


def property_tables(results: dict, tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, ...]:
    groups, adjacent, coverage = [], [], []
    for task in TASKS:
        for scope, scope_name in SCOPES.items():
            base = {"cohort_id": task, "task": TASK_NAMES[task], "scope": scope, "scope_name": scope_name}
            for k in range(5):
                r = results[("share", task, scope, "count_main", k, None)]
                groups.append({**base, "properties": k, **r,
                               "lower": r["point"] - r["half_width"], "upper": r["point"] + r["half_width"]})
            for k in range(4):
                r = results[("difference", task, scope, "count_main", k + 1, k)]
                adjacent.append({**base, "higher": k + 1, "lower_group": k, "genes_higher": r["genes_a"],
                                 "genes_lower": r["genes_b"], "difference": r["point"], "half_width": r["half_width"],
                                 "lower": r["point"] - r["half_width"]})
            frame = tables[task]
            if scope == "other":
                frame = frame[frame.core_function == 0]
            essential = frame[frame.label == 1]
            coverage.append({**base, "genes": int(len(frame)), "essentials": int(len(essential)),
                             "genes_three_or_more": int((frame.count_main >= 3).sum()),
                             "essentials_three_or_more": int((essential.count_main >= 3).sum()),
                             "share_of_genes": float((frame.count_main >= 3).mean()),
                             "share_of_essentials": float((essential.count_main >= 3).mean())})
    return pd.DataFrame(groups), pd.DataFrame(adjacent), pd.DataFrame(coverage)


def core_function_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        frame = tables[task]
        for group, name, part in (("core", "core function genes", frame[frame.core_function == 1]),
                                  ("other", "other genes", frame[frame.core_function == 0])):
            rows.append({"cohort_id": task, "task": TASK_NAMES[task], "group": group, "group_name": name,
                         "genes": int(len(part)), "essentials": int(part.label.sum()),
                         "oldest_layer_share": float((part.age_layer == 1).mean()),
                         "tcga_median_of_group": float(part.tcga_median.median())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    arguments = parser.parse_args()
    write_table(species_layers(list(pd.read_parquet(STATIC).columns)), "age_layers")
    tables, thresholds, edges = {}, [], {}
    for task in TASKS:
        frame, cut, edges[task] = gene_table(task)
        write_atomic(frame, gene_properties_path(task))
        tables[task] = pd.read_csv(gene_properties_path(task))
        thresholds.append({"cohort_id": task, "task": TASK_NAMES[task], **cut})
    write_table(pd.DataFrame(thresholds), "property_thresholds")
    jobs = [job for task in TASKS for job in jobs_for(task)]
    results = run_pool(interval_job, jobs, arguments.workers, "share intervals", 30.0)
    write_table(age_table(results, tables), "age_essential_share")
    cells, bin_table = grid_tables(tables, edges)
    write_table(cells, "degree_expression_grid")
    write_table(bin_table, "degree_expression_bins")
    groups, adjacent, coverage = property_tables(results, tables)
    write_table(adjacent, "four_properties_adjacent")
    write_table(coverage, "four_properties_coverage")
    write_table(core_function_table(tables), "core_function_age_expression")
    write_atomic(groups, RESULTS / "four_properties.csv")
    print("wrote the four properties tables", flush=True)


if __name__ == "__main__":
    main()
