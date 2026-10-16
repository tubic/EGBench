"""Permutation importance of each TCGA expression principal component inside the core feature set's model on DEG and DepMap.

Run from the release root after the feature_groups cache exists: python analysis/essential_gene_hallmarks/pc_importance.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from egbench.contracts import SEED
from egbench.data import VALIDATION_ROLE
from egbench.metrics.discrimination import auprc

from common.feature_blocks import (
    BLOCK_KEYS, COHORT_NAMES, CORE_BLOCKS, FOLDS, FORWARD_COHORTS, PERMUTATIONS, WORKERS, block_names, design,
    fit_set, fold_roles, run_pool,
)
from common.hallmarks import WORK, write_table

PC_BLOCK = "tcga_pca"


def fold_job(job: tuple[str, tuple[str, ...], int]) -> dict:
    cohort, keys, fold = job
    model = fit_set(cohort, fold, keys, keep_model=True)["model"]
    roles = fold_roles(cohort, fold)
    x = design(cohort, fold, keys, roles[VALIDATION_ROLE].rows)
    y = roles[VALIDATION_ROLE].labels
    reference = float(auprc(y, model.predict_proba(x)[:, 1]))
    widths = [len(block_names(cohort, fold, key)) for key in keys]
    start = int(np.sum(widths[: keys.index(PC_BLOCK)]))
    names = block_names(cohort, fold, PC_BLOCK)
    block_index = BLOCK_KEYS.index(PC_BLOCK)
    rows = []
    for offset, name in enumerate(names):
        j = start + offset
        saved = x[:, j].copy()
        drops = []
        for repeat in range(PERMUTATIONS):
            generator = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(0, fold, block_index, offset, repeat)))
            x[:, j] = saved[generator.permutation(len(y))]
            drops.append(reference - float(auprc(y, model.predict_proba(x)[:, 1])))
        x[:, j] = saved
        rows.append({"cohort_id": cohort, "fold": fold, "component": name, "drop": float(np.mean(drops))})
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    arguments = parser.parse_args()
    jobs = [(c, CORE_BLOCKS, fold) for c in FORWARD_COHORTS for fold in FOLDS]
    results = run_pool(fold_job, jobs, arguments.workers, "component permutation")
    frame = pd.DataFrame([row for job in jobs for row in results[job]["rows"]])
    table = (frame.groupby(["cohort_id", "component"], sort=False)["drop"]
             .agg(importance="mean", fold_sd=lambda v: float(np.std(v, ddof=1))).reset_index())
    table = table.sort_values(["cohort_id", "importance"], ascending=[True, False], kind="stable")
    table.insert(2, "rank", table.groupby("cohort_id").cumcount() + 1)
    table.insert(1, "task", table.cohort_id.map(COHORT_NAMES))
    write_table(table, "pc_importance", WORK)
    print("wrote work/pc_importance.csv", flush=True)


if __name__ == "__main__":
    main()
