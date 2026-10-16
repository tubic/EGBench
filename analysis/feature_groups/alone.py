"""Fit the gradient boosting learner on each feature group alone on the five binary tasks and compare with each task's leader.

Run from the release root: python analysis/feature_groups/alone.py [--workers N]
"""

from __future__ import annotations

import argparse

import pandas as pd

from common.feature_blocks import (
    ALONE_COHORTS, BLOCK_BY_KEY, BLOCK_KEYS, CATEGORIES, COHORT_NAMES, FOLDS, LEADERS, METHOD_NAMES, WORK, WORKERS,
    block_width, fit_set, fold_validation_auprc, leader_frame, paired_test, pooled_validation_auprc, run_pool,
    score_frame, test_intervals, write_table,
)
from common.stats import holm

ALONE = WORK / "alone"


def alone_path(cohort: str, key: str):
    return ALONE / f"{cohort}_{key}.csv"


def read_alone(cohort: str, key: str) -> pd.DataFrame:
    frame = pd.read_csv(alone_path(cohort, key))
    frame["gene_key"] = frame["gene_key"].astype(str)
    return frame


def fit_job(job: tuple[str, str, int]) -> dict:
    cohort, key, fold = job
    return fit_set(cohort, fold, (key,))


def stats_job(job: tuple[str, str]) -> dict:
    cohort, key = job
    frame = read_alone(cohort, key)
    intervals = test_intervals(frame)
    leader = leader_frame(cohort, frame)
    difference, half, p = paired_test(frame, leader["y_score"].to_numpy())
    return {
        "cohort_id": cohort, "block": key, "validation_auprc_pooled": pooled_validation_auprc(frame),
        "validation_auprc_fold_mean": fold_validation_auprc(frame),
        "auprc": intervals["auprc"][0], "auprc_half_width": intervals["auprc"][1],
        "auroc": intervals["auroc"][0], "auroc_half_width": intervals["auroc"][1],
        "minus_leader": difference, "minus_leader_half_width": half, "p": p,
    }


def leader_job(cohort: str) -> dict:
    frame = read_alone(cohort, BLOCK_KEYS[0])
    intervals = test_intervals(leader_frame(cohort, frame).assign(role="test"))
    return {"cohort_id": cohort, "leader": METHOD_NAMES[LEADERS[cohort]],
            "leader_auprc": intervals["auprc"][0], "leader_auprc_half_width": intervals["auprc"][1],
            "leader_auroc": intervals["auroc"][0], "leader_auroc_half_width": intervals["auroc"][1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    arguments = parser.parse_args()
    ALONE.mkdir(parents=True, exist_ok=True)
    widths = {key: block_width("deg", 0, key) for key in BLOCK_KEYS}
    order = sorted(BLOCK_KEYS, key=lambda key: -widths[key])
    jobs = [(cohort, key, fold) for key in order for cohort in ALONE_COHORTS for fold in FOLDS]
    fits = run_pool(fit_job, jobs, arguments.workers, "alone fits")
    rounds = []
    for cohort in ALONE_COHORTS:
        for key in BLOCK_KEYS:
            per_fold = {fold: fits[(cohort, key, fold)] for fold in FOLDS}
            score_frame(cohort, per_fold).to_csv(alone_path(cohort, key), index=False)
            rounds.append({"cohort_id": cohort, "block": key,
                           "rounds": ";".join(str(per_fold[f]["n_iter"]) for f in FOLDS)})
    stats = run_pool(stats_job, [(c, k) for c in ALONE_COHORTS for k in BLOCK_KEYS], arguments.workers,
                     "alone intervals", 30.0)
    leaders = run_pool(leader_job, list(ALONE_COHORTS), arguments.workers, "leader intervals", 30.0)
    table = pd.DataFrame([stats[(c, k)] for c in ALONE_COHORTS for k in BLOCK_KEYS])
    table = table.merge(pd.DataFrame(rounds), on=["cohort_id", "block"])
    table = table.merge(pd.DataFrame([leaders[c] for c in ALONE_COHORTS]), on="cohort_id")
    table["holm_p"] = holm(table, ["cohort_id"])
    table.insert(1, "task", table.cohort_id.map(COHORT_NAMES))
    table.insert(3, "name", table.block.map(lambda k: BLOCK_BY_KEY[k].name))
    table.insert(4, "category", table.block.map(lambda k: CATEGORIES[BLOCK_BY_KEY[k].category]))
    table.insert(5, "columns", table.block.map(widths))
    write_table(table, "block_alone")
    print("wrote block_alone.csv", flush=True)


if __name__ == "__main__":
    main()
