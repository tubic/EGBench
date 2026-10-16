"""Dependency strength and context: Spearman correlation of the 15 methods and the hallmark count with the GeneEffect
strength among all, essential and non-essential genes, and the hallmark shares and AUROCs on the five binary tasks.

Run from the release root with PYTHONPATH=src:analysis: python analysis/dependency_strength_context/strength_context.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from egbench.fusion import pooled
from egbench.stats import holm_adjusted

from common.hallmark_counts import BINARY_TASKS, HALLMARKS, RICH, TASK_NAMES, bootstrap, cells_of, hallmark_table, spearman
from common.stats import FOLDS, METHODS, fast_auroc, process_pool

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
COUNT = "hallmark_count"
STRENGTH_PREDICTORS: tuple[str, ...] = METHODS + (COUNT,)


def strength() -> pd.DataFrame:
    base, scores = None, {}
    for predictor in METHODS:
        pool = pooled(predictor, "gene_effect")
        pool["gene_key"] = pool["gene_key"].astype(str)
        pool = pool.sort_values(["fold", "gene_key"], kind="stable").reset_index(drop=True)
        if base is None:
            base = pool[["gene_key", "fold", "gene_effect"]].copy()
        elif not pool["gene_key"].equals(base["gene_key"]):
            raise ValueError(f"{predictor}: the scored genes differ")
        scores[predictor] = pool["y_score"].to_numpy("float64")
    keys = pd.Index(base["gene_key"], dtype="object")
    properties, _ = hallmark_table("depmap")
    properties["gene_key"] = properties["gene_key"].astype(str)
    properties = properties.set_index("gene_key").reindex(keys)
    if properties["label"].isna().any():
        raise ValueError("a GeneEffect gene has no DepMap label or hallmark")
    scores[COUNT] = properties["count"].to_numpy("float64")
    target = -base["gene_effect"].to_numpy("float64")
    labels = properties["label"].to_numpy("int64")
    fold = base["fold"].to_numpy()
    block = np.column_stack([scores[p] for p in STRENGTH_PREDICTORS])

    def statistic(take):
        t, y, f, b = (target, labels, fold, block) if take is None else (target[take], labels[take], fold[take], block[take])
        out = {}
        inside = {"all": np.ones(len(y), dtype=bool), "essential": y == 1, "nonessential": y == 0}
        for column, predictor in enumerate(STRENGTH_PREDICTORS):
            s = b[:, column]
            for scope, mask in inside.items():
                out[f"{predictor}:spearman_{scope}"] = float(np.mean(
                    [spearman(s[mask & (f == k)], t[mask & (f == k)]) for k in FOLDS]))
            out[f"{predictor}:auroc"] = float(np.mean([fast_auroc(y[f == k], s[f == k]) for k in FOLDS]))
        return out

    table, _ = bootstrap(cells_of(fold, labels), statistic, "dependency strength")
    table["predictor"] = table["statistic"].str.split(":").str[0]
    table["quantity"] = table["statistic"].str.split(":").str[1]
    table["genes"] = len(base)
    table["essential_genes"] = int(labels.sum())
    table["holm_p"] = np.nan
    for scope in ("essential", "nonessential"):
        rows = table["predictor"].isin(METHODS) & table["quantity"].eq(f"spearman_{scope}")
        table.loc[rows, "holm_p"] = holm_adjusted(table.loc[rows, "p"].to_numpy("float64"))
    return table


def task_frame(task: str) -> pd.DataFrame:
    frame, _ = hallmark_table(task)
    frame["gene_key"] = frame["gene_key"].astype(str)
    base = None
    for method in METHODS:
        pool = pooled(method, task)
        pool["gene_key"] = pool["gene_key"].astype(str)
        pool = pool.sort_values(["fold", "gene_key"], kind="stable").reset_index(drop=True)
        if base is None:
            base = pool[["gene_key", "fold", "label"]].copy()
        elif not pool["gene_key"].equals(base["gene_key"]):
            raise ValueError(f"{task} {method}: the scored genes differ")
        base[method] = pool["y_score"].to_numpy("float64")
    merged = base.merge(frame.drop(columns=["label"]), on="gene_key", how="left", validate="one_to_one")
    if merged["count"].isna().any():
        raise ValueError(f"{task}: a scored gene has no hallmark")
    return merged


def hallmark_job(task: str) -> pd.DataFrame:
    frame = task_frame(task)
    labels = frame["label"].to_numpy("int64")
    fold = frame["fold"].to_numpy()
    count = frame["count"].to_numpy("float64")
    rich = (count >= RICH).astype("float64")
    flags = {name: frame[name].to_numpy("float64") for name in HALLMARKS}
    method_scores = {m: frame[m].to_numpy("float64") for m in METHODS}

    def fold_auroc(y, s, f):
        values = [fast_auroc(y[f == k], s[f == k]) for k in FOLDS]
        values = [v for v in values if not np.isnan(v)]
        return float(np.mean(values)) if values else float("nan")

    def statistic(take):
        y, f, c, r = (labels, fold, count, rich) if take is None else (labels[take], fold[take], count[take], rich[take])
        positive, negative = y == 1, y == 0
        out = {"all:positive_rich_share": float(r[positive].mean()), "all:negative_rich_share": float(r[negative].mean()),
               "all:hallmark_count_auroc": fold_auroc(y, c, f)}
        for hallmark, values in flags.items():
            v = values if take is None else values[take]
            out[f"all:positive_{hallmark}_share"] = float(v[positive].mean())
            out[f"all:negative_{hallmark}_share"] = float(v[negative].mean())
        for method, values in method_scores.items():
            v = values if take is None else values[take]
            out[f"all:{method}_auroc"] = fold_auroc(y, v, f)
        return out

    table, _ = bootstrap(cells_of(fold, labels), statistic, f"hallmarks {TASK_NAMES[task]}")
    table["group"] = table["statistic"].str.split(":").str[0]
    table["quantity"] = table["statistic"].str.split(":").str[1]
    table.insert(0, "task", TASK_NAMES[task])
    table.insert(0, "cohort_id", task)
    return table


def run(job: str) -> pd.DataFrame:
    return strength() if job == "strength" else hallmark_job(job)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    jobs = ["strength", *BINARY_TASKS]
    with process_pool(len(jobs)) as pool:
        results = dict(zip(jobs, pool.map(run, jobs)))
    results["strength"].to_csv(RESULTS / "strength.csv", index=False)
    pd.concat([results[task] for task in BINARY_TASKS], ignore_index=True).to_csv(
        RESULTS / "task_hallmarks.csv", index=False)


if __name__ == "__main__":
    main()
