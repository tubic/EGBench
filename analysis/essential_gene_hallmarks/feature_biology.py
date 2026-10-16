"""Direction and size of 13 features tied to essential genes, CORUM complexes and GO biological processes enriched in essential genes on DEG and DepMap.

Run from the release root after pc_importance.py: python analysis/essential_gene_hallmarks/feature_biology.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from egbench.contracts import TargetKind
from egbench.metrics.discrimination import auroc
from egbench.modules import PROCESS, cohort_modules, complex_names, complex_pairs, restrict_terms
from egbench.stats import benjamini_hochberg_adjusted, paired_difference

from common.feature_blocks import WORKERS, block_array, block_names, cohort_genes, go_names, run_pool
from common.hallmarks import (
    CONTINUOUS, CONTINUOUS_KEYS, FEATURE_BY_KEY, FEATURES, TASK_NAMES, TASKS, WORK, feature_frame,
    feature_values_path, write_table,
)
from common.stats import holm

CORUM_MINIMUM = 5
TOP = 15


def read_values(task: str) -> pd.DataFrame:
    frame = pd.read_csv(feature_values_path(task))
    frame["gene_key"] = frame["gene_key"].astype(str)
    return frame


def share_difference(labels: np.ndarray, values: np.ndarray) -> float:
    return float(values[labels == 1].mean() - values[labels == 0].mean())


def test_job(job: tuple[str, str]) -> dict:
    task, key = job
    frame = read_values(task)
    y = frame["label"].to_numpy("float64")
    x = frame[key].to_numpy("float64")
    feature = FEATURE_BY_KEY[key]
    metric = auroc if feature.kind == CONTINUOUS else share_difference
    result = paired_difference(y, x, np.zeros_like(x), metric, target_kind=TargetKind.BINARY)
    offset = 0.5 if feature.kind == CONTINUOUS else 0.0
    row = {"cohort_id": task, "task": TASK_NAMES[task], "genes": int(len(y)), "positives": int(y.sum()),
           "feature": key, "feature_name": feature.name, "feature_group": feature.group, "kind": feature.kind,
           "statistic": "auroc" if feature.kind == CONTINUOUS else "share_difference",
           "point": float(result.difference) + offset, "half_width": float(result.half_width),
           "p": float(result.p_value)}
    if feature.kind == CONTINUOUS:
        row["positive_median"] = float(np.median(x[y == 1]))
        row["negative_median"] = float(np.median(x[y == 0]))
    else:
        row["positive_share"] = float(x[y == 1].mean())
        row["negative_share"] = float(x[y == 0].mean())
    return row


def corum_tables(task: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = restrict_terms(complex_pairs(), frame.index, minimum=CORUM_MINIMUM, maximum=float("inf"))
    names = complex_names()
    labels = frame["label"]
    rows = []
    for term, members in pairs.groupby("term_id"):
        member_labels = labels.reindex(members["gene_key"]).to_numpy("float64")
        rows.append({"cohort_id": task, "task": TASK_NAMES[task], "complex_id": term,
                     "complex_name": names.get(term, ""), "genes": int(len(member_labels)),
                     "positives": int(member_labels.sum()), "positive_share": float(member_labels.mean())})
    table = pd.DataFrame(rows).sort_values(["positive_share", "genes", "complex_id"],
                                           ascending=[False, False, True], kind="stable")
    table.insert(2, "rank", np.arange(1, len(table) + 1))
    table["top15"] = table["rank"] <= TOP
    members = set(pairs["gene_key"])
    inside = labels[labels.index.isin(members)]
    outside = labels[~labels.index.isin(members)]
    summary = pd.DataFrame([{
        "cohort_id": task, "task": TASK_NAMES[task], "complexes": int(len(table)),
        "member_genes": int(len(inside)), "member_positives": int(inside.sum()),
        "member_positive_share": float(inside.mean()),
        "other_genes": int(len(outside)), "other_positives": int(outside.sum()),
        "other_positive_share": float(outside.mean()),
        "all_positive_share": float(labels.mean()),
        "complexes_all_positive": int((table.positive_share == 1.0).sum()),
    }])
    return table, summary


def go_table(task: str, frame: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    pairs = cohort_modules(task, PROCESS)
    labels = frame["label"]
    total, positives = len(labels), int(labels.sum())
    rows = []
    for term, members in pairs.groupby("term_id"):
        member_labels = labels.reindex(members["gene_key"]).to_numpy("float64")
        size, hits = int(len(member_labels)), int(member_labels.sum())
        rows.append({"cohort_id": task, "task": TASK_NAMES[task], "term_id": term, "term_name": names.get(term, ""),
                     "genes": size, "positives": hits,
                     "fold_enrichment": (hits / size) / (positives / total),
                     "p": float(hypergeom.sf(hits - 1, total, positives, size))})
    table = pd.DataFrame(rows)
    table["q"] = benjamini_hochberg_adjusted(table["p"].to_numpy("float64"))
    table = table.sort_values(["q", "p", "term_id"], kind="stable").reset_index(drop=True)
    table.insert(2, "rank", np.arange(1, len(table) + 1))
    table["top15"] = table["rank"] <= TOP
    return table


def pc_expression(values: dict[str, pd.DataFrame]) -> pd.DataFrame:
    importance = pd.read_csv(WORK / "pc_importance.csv")
    rows = []
    for task in TASKS:
        top = importance[importance.cohort_id == task].sort_values("rank").head(3)
        median = values[task].set_index("gene_key")["tcga_median"]
        target = median.reindex(cohort_genes(task)).to_numpy("float64")
        for r in top.itertuples():
            per_fold = []
            for fold in range(5):
                names = block_names(task, fold, "tcga_pca")
                column = np.asarray(block_array(task, fold, "tcga_pca")[:, names.index(r.component)], dtype="float64")
                per_fold.append(float(pd.Series(column).corr(pd.Series(target), method="spearman")))
            rows.append({"cohort_id": task, "task": TASK_NAMES[task], "component": r.component,
                         "importance_rank": int(r.rank), "importance": float(r.importance),
                         "spearman_mean": float(np.mean(per_fold)),
                         "spearman_abs_mean": float(np.mean(np.abs(per_fold))),
                         "spearman_min": float(np.min(per_fold)), "spearman_max": float(np.max(per_fold)),
                         "spearman_by_fold": ";".join(f"{v:+.3f}" for v in per_fold)})
    return pd.DataFrame(rows)


def correlation(values: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for task in TASKS:
        matrix = values[task][list(CONTINUOUS_KEYS)].corr(method="spearman")
        matrix.insert(0, "feature", matrix.index)
        matrix.insert(0, "task", TASK_NAMES[task])
        matrix.insert(0, "cohort_id", task)
        parts.append(matrix.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    arguments = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    names = go_names()
    values, corum_parts, corum_summaries, go_parts = {}, [], [], []
    for task in TASKS:
        frame = feature_frame(task)
        corum, corum_summary = corum_tables(task, frame)
        go = go_table(task, frame, names)
        pairs = cohort_modules(task, PROCESS)
        core_genes = set(pairs[pairs["term_id"].isin(go[go.top15]["term_id"])]["gene_key"])
        frame["core_function"] = frame.index.isin(core_genes).astype("int64")
        frame.reset_index().to_csv(feature_values_path(task), index=False)
        values[task] = read_values(task)
        corum_parts.append(corum)
        corum_summaries.append(corum_summary)
        go_parts.append(go)
        print(f"[features] {TASK_NAMES[task]}: {len(frame)} genes, {int(frame.core_function.sum())} core function genes",
              flush=True)
    write_table(pd.concat(corum_parts, ignore_index=True), "corum_top")
    write_table(pd.concat(corum_summaries, ignore_index=True), "corum_summary")
    write_table(pd.concat(go_parts, ignore_index=True), "go_enrichment")
    jobs = [(task, feature.key) for task in TASKS for feature in FEATURES]
    results = run_pool(test_job, jobs, arguments.workers, "feature tests", 30.0)
    tests = pd.DataFrame([results[job] for job in jobs])
    tests["holm_p"] = holm(tests, ["cohort_id"])
    write_table(tests, "feature_direction")
    write_table(pc_expression(values), "pc_expression")
    write_table(correlation(values), "feature_correlation")
    print("wrote corum_top, corum_summary, go_enrichment, feature_direction, pc_expression, feature_correlation",
          flush=True)


if __name__ == "__main__":
    main()
