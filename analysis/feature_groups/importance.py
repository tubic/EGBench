"""Permutation importance of each feature and each feature group inside the core feature set's model on DEG and DepMap.

Run from the release root after blocks.py and cccfs_terms.py: python analysis/feature_groups/importance.py [--workers N]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from egbench.contracts import SEED
from egbench.data import VALIDATION_ROLE
from egbench.metrics.discrimination import auprc

from common.feature_blocks import (
    BLOCK_BY_KEY, BLOCK_KEYS, CATEGORIES, CCCFS_TERMS, COHORT_NAMES, CORE_BLOCKS, FOLDS, FORWARD_COHORTS,
    PERMUTATIONS, WORKERS, block_names, design, fit_set, fold_roles, run_pool, write_table,
)


def used_columns(model) -> set[int]:
    used = set()
    for iteration in model._predictors:
        for predictor in iteration:
            nodes = predictor.nodes
            used.update(int(i) for i in nodes["feature_idx"][~nodes["is_leaf"].astype(bool)])
    return used


def fold_job(job: tuple[str, tuple[str, ...], int]) -> dict:
    cohort, keys, fold = job
    model = fit_set(cohort, fold, keys, keep_model=True)["model"]
    roles = fold_roles(cohort, fold)
    x = design(cohort, fold, keys, roles[VALIDATION_ROLE].rows)
    y = roles[VALIDATION_ROLE].labels
    reference = float(auprc(y, model.predict_proba(x)[:, 1]))
    used = used_columns(model)
    offsets = np.cumsum([0] + [len(block_names(cohort, fold, k)) for k in keys])
    columns, blocks = [], []
    for position, key in enumerate(keys):
        start, stop = int(offsets[position]), int(offsets[position + 1])
        names = block_names(cohort, fold, key)
        block_index = BLOCK_KEYS.index(key)
        drops = []
        for repeat in range(PERMUTATIONS):
            generator = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(1, fold, block_index, repeat)))
            order = generator.permutation(len(y))
            shuffled = x.copy()
            shuffled[:, start:stop] = x[order, start:stop]
            drops.append(reference - float(auprc(y, model.predict_proba(shuffled)[:, 1])))
        blocks.append({"cohort_id": cohort, "fold": fold, "block": key, "columns": stop - start,
                       "used_columns": sum(1 for j in range(start, stop) if j in used),
                       "drop": float(np.mean(drops))})
        if not BLOCK_BY_KEY[key].per_column:
            continue
        seen: dict[str, int] = {}
        for offset, name in enumerate(names):
            seen[name] = seen.get(name, 0) + 1
            if seen[name] > 1:
                name = f"{name} #{offset}"
            j = start + offset
            drop = 0.0
            if j in used:
                saved = x[:, j].copy()
                values = []
                for repeat in range(PERMUTATIONS):
                    generator = np.random.default_rng(
                        np.random.SeedSequence(SEED, spawn_key=(0, fold, block_index, offset, repeat)))
                    x[:, j] = saved[generator.permutation(len(y))]
                    values.append(reference - float(auprc(y, model.predict_proba(x)[:, 1])))
                x[:, j] = saved
                drop = float(np.mean(values))
            columns.append({"cohort_id": cohort, "fold": fold, "block": key, "position": offset,
                            "feature": name, "used": j in used, "drop": drop})
    print(f"[importance] {COHORT_NAMES[cohort]} fold {fold} done", flush=True)
    return {"columns": columns, "blocks": blocks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    arguments = parser.parse_args()
    jobs = [(c, CORE_BLOCKS, fold) for c in FORWARD_COHORTS for fold in FOLDS]
    results = run_pool(fold_job, jobs, arguments.workers, "permutation importance")
    columns = pd.DataFrame([row for job in jobs for row in results[job]["columns"]])
    blocks = pd.DataFrame([row for job in jobs for row in results[job]["blocks"]])
    terms = pd.read_csv(CCCFS_TERMS)
    tiers = {int(column.split("__")[1]): tier for column, tier in zip(terms["column"], terms["tier"])}
    feature_rows = []
    for (cohort, key, feature), group in columns.groupby(["cohort_id", "block", "feature"], sort=False):
        per_fold = {int(r.fold): r.drop for r in group.itertuples()}
        tier = tiers[int(group.position.iloc[0])] if key == "cccfs" else ""
        feature_rows.append({
            "cohort_id": cohort, "task": COHORT_NAMES[cohort], "block": key, "block_name": BLOCK_BY_KEY[key].name,
            "category": CATEGORIES[BLOCK_BY_KEY[key].category], "feature": feature, "cccfs_tier": tier,
            "importance": float(sum(per_fold.values()) / len(FOLDS)),
            "folds_present": len(per_fold), "folds_used": int(group.used.sum()),
            "fold_drops": ";".join(f"{per_fold[f]:.5f}" if f in per_fold else "" for f in FOLDS),
        })
    features = pd.DataFrame(feature_rows)
    features = features.sort_values(["cohort_id", "importance"], ascending=[True, False], kind="stable")
    features.insert(2, "rank", features.groupby("cohort_id").cumcount() + 1)
    write_table(features, "core_feature_importance")
    block_rows = []
    for (cohort, key), group in blocks.groupby(["cohort_id", "block"], sort=False):
        block_rows.append({
            "cohort_id": cohort, "task": COHORT_NAMES[cohort], "block": key, "name": BLOCK_BY_KEY[key].name,
            "category": CATEGORIES[BLOCK_BY_KEY[key].category], "columns": int(group["columns"].iloc[0]),
            "used_columns_mean": float(group.used_columns.mean()),
            "importance": float(group["drop"].mean()), "fold_sd": float(group["drop"].std(ddof=1)),
            "fold_drops": ";".join(f"{d:.5f}" for d in group.sort_values("fold")["drop"]),
        })
    block_table = pd.DataFrame(block_rows).sort_values(["cohort_id", "importance"], ascending=[True, False])
    write_table(block_table, "core_block_importance")
    print("wrote core_feature_importance.csv and core_block_importance.csv", flush=True)


if __name__ == "__main__":
    main()
