"""Forward selection of feature groups on DEG and DepMap by pooled validation AUPRC, eight steps.

Run from the release root after alone.py: python analysis/feature_groups/forward.py [--workers N]
"""

from __future__ import annotations

import argparse

import pandas as pd

from common.feature_blocks import (
    BLOCK_BY_KEY, BLOCK_KEYS, CATEGORIES, COHORT_NAMES, CORE_TOLERANCE, FOLDS, FORWARD_COHORTS, FORWARD_STEPS,
    RESULTS, WORK, WORKERS, block_width, fit_set, fold_validation_auprc, pooled_validation_auprc, run_pool,
    score_frame, test_intervals, write_table,
)
from feature_groups.alone import read_alone

FORWARD = WORK / "forward"


def fit_job(job: tuple[str, tuple[str, ...], int]) -> dict:
    cohort, keys, fold = job
    return fit_set(cohort, fold, keys)


def frame_of(fits: dict, cohort: str, keys: tuple[str, ...]) -> pd.DataFrame:
    return score_frame(cohort, {fold: fits[(cohort, keys, fold)] for fold in FOLDS})


def rounds_of(fits: dict, cohort: str, keys: tuple[str, ...]) -> str:
    return ";".join(str(fits[(cohort, keys, fold)]["n_iter"]) for fold in FOLDS)


def choose(values: dict[str, float]) -> str:
    return max(values, key=lambda key: (values[key], -BLOCK_KEYS.index(key)))


def interval_job(job: tuple[str, str]) -> dict:
    _, name = job
    frame = pd.read_csv(FORWARD / f"{name}.csv")
    frame["gene_key"] = frame["gene_key"].astype(str)
    return test_intervals(frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    arguments = parser.parse_args()
    FORWARD.mkdir(parents=True, exist_ok=True)
    widths = {key: block_width("deg", 0, key) for key in BLOCK_KEYS}
    alone = pd.read_csv(RESULTS / "block_alone.csv")
    path, selected = [], {}
    for cohort in FORWARD_COHORTS:
        scores = {key: read_alone(cohort, key) for key in BLOCK_KEYS}
        values = {key: pooled_validation_auprc(scores[key]) for key in BLOCK_KEYS}
        chosen = choose(values)
        scores[chosen].to_csv(FORWARD / f"{cohort}_step1.csv", index=False)
        rounds = alone[(alone.cohort_id == cohort) & (alone.block == chosen)].iloc[0]["rounds"]
        selected[cohort] = (chosen,)
        path.append({"cohort_id": cohort, "step": 1, "block": chosen, "set": (chosen,),
                     "set_columns": widths[chosen], "validation_auprc_pooled": values[chosen],
                     "validation_auprc_fold_mean": fold_validation_auprc(scores[chosen]), "rounds": rounds})
        print(f"[forward] {COHORT_NAMES[cohort]} step 1: {chosen}, validation AUPRC {values[chosen]:.4f}", flush=True)

    for step in range(2, FORWARD_STEPS + 1):
        jobs = []
        for cohort in FORWARD_COHORTS:
            remaining = [key for key in BLOCK_KEYS if key not in selected[cohort]]
            jobs += [(cohort, selected[cohort] + (key,), fold) for key in remaining for fold in FOLDS]
        jobs.sort(key=lambda job: -sum(widths[k] for k in job[1]))
        fits = run_pool(fit_job, jobs, arguments.workers, f"forward step {step}")
        for cohort in FORWARD_COHORTS:
            prefix = selected[cohort]
            remaining = [key for key in BLOCK_KEYS if key not in prefix]
            frames = {key: frame_of(fits, cohort, prefix + (key,)) for key in remaining}
            values = {key: pooled_validation_auprc(frames[key]) for key in remaining}
            chosen = choose(values)
            chosen_set = prefix + (chosen,)
            selected[cohort] = chosen_set
            frames[chosen].to_csv(FORWARD / f"{cohort}_step{step}.csv", index=False)
            path.append({"cohort_id": cohort, "step": step, "block": chosen, "set": chosen_set,
                         "set_columns": sum(widths[k] for k in chosen_set),
                         "validation_auprc_pooled": values[chosen],
                         "validation_auprc_fold_mean": fold_validation_auprc(frames[chosen]),
                         "rounds": rounds_of(fits, cohort, chosen_set)})
            print(f"[forward] {COHORT_NAMES[cohort]} step {step}: {chosen}, "
                  f"validation AUPRC {values[chosen]:.4f}", flush=True)

    path = pd.DataFrame(path)
    cores = {}
    for cohort in FORWARD_COHORTS:
        rows = path[path.cohort_id == cohort]
        best = rows.validation_auprc_pooled.max()
        cores[cohort] = int(rows[rows.validation_auprc_pooled >= best - CORE_TOLERANCE].step.min())
    names = [(c, f"{c}_step{s}") for c in FORWARD_COHORTS for s in range(1, FORWARD_STEPS + 1)]
    intervals = run_pool(interval_job, names, arguments.workers, "forward intervals", 30.0)
    path["task"] = path.cohort_id.map(COHORT_NAMES)
    path["name"] = path.block.map(lambda k: BLOCK_BY_KEY[k].name)
    path["category"] = path.block.map(lambda k: CATEGORIES[BLOCK_BY_KEY[k].category])
    path["in_core"] = [step <= cores[c] for c, step in zip(path.cohort_id, path.step)]
    for metric in ("auprc", "auroc"):
        path[f"test_{metric}"] = [intervals[(c, f"{c}_step{s}")][metric][0] for c, s in zip(path.cohort_id, path.step)]
        path[f"test_{metric}_half_width"] = [intervals[(c, f"{c}_step{s}")][metric][1]
                                             for c, s in zip(path.cohort_id, path.step)]
    path["set"] = path["set"].map(lambda keys: "+".join(keys))
    path = path.sort_values(["cohort_id", "step"], kind="stable")
    path = path[["cohort_id", "task", "step", "block", "name", "category", "set", "set_columns",
                 "validation_auprc_pooled", "validation_auprc_fold_mean", "test_auprc", "test_auprc_half_width",
                 "test_auroc", "test_auroc_half_width", "rounds", "in_core"]]
    write_table(path, "forward_path")
    for cohort in FORWARD_COHORTS:
        core = path[(path.cohort_id == cohort) & path.in_core]["block"].tolist()
        print(f"[forward] {COHORT_NAMES[cohort]} core: {', '.join(core)}", flush=True)
    print("wrote forward_path.csv", flush=True)


if __name__ == "__main__":
    main()
