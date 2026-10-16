"""Fit the core feature set (four feature groups) on the six tasks, with HELP's BIOGRID column dropped or replaced by PPI degree, and compare with each task's leader.

Run from the release root after blocks.py: python analysis/feature_groups/core_fixed.py [--workers N]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from egbench.contracts import TargetKind
from egbench.data import TEST_ROLE, TRAIN_ROLE, VALIDATION_ROLE, primary_scope, read_evaluation_truth
from egbench.evaluate import scope_frame
from egbench.metrics.discrimination import auprc, auroc
from egbench.metrics.regression import spearman
from egbench.stats import confidence_intervals, paired_difference

from common.feature_blocks import (
    ALONE_COHORTS, COHORT_NAMES, CORE_BLOCKS, FOLDS, GENE_KEY, HELP_BIO_NAMES, LEADERS, METHOD_NAMES, SPLIT, WORK,
    WORKERS, block_array, cohort_genes, fold_roles, make_learner, run_pool, write_table,
)
from common.stats import holm, pooled

VARIANTS: tuple[str, ...] = ("A", "B1", "B2")
VARIANT_NAMES = {"A": "core feature set", "B1": "BIOGRID column dropped", "B2": "BIOGRID replaced by PPI degree"}
BIOGRID = HELP_BIO_NAMES.index("BIOGRID")
GENE_EFFECT = "gene_effect"
TASKS: tuple[str, ...] = ("deg", "depmap", "context", GENE_EFFECT, "blood", "mouse")
TASK_NAMES = {**COHORT_NAMES, GENE_EFFECT: "GeneEffect"}
TASK_LEADERS = {**LEADERS, GENE_EFFECT: "mahi"}
FIXED = WORK / "fixed"
COMPARISONS: tuple[tuple[str, str], ...] = (("A", "leader"), ("B1", "leader"), ("B2", "leader"), ("B1", "A"), ("B2", "A"))


def fixed_scores_path(task: str, variant: str = "A"):
    return FIXED / f"{task}_{variant}.csv"


def variant_design(cohort: str, fold: int, variant: str, rows: np.ndarray) -> np.ndarray:
    parts = []
    for key in CORE_BLOCKS:
        values = np.asarray(block_array(cohort, fold, key)[rows], dtype="float64")
        if key == "help_bio" and variant == "B1":
            values = np.delete(values, BIOGRID, axis=1)
        elif key == "help_bio" and variant == "B2":
            values = values.copy()
            values[:, BIOGRID] = np.asarray(block_array(cohort, fold, "ppi_degree")[rows], dtype="float64")[:, 0]
        parts.append(values)
    return np.ascontiguousarray(np.hstack(parts))


def gene_effect_truth(fold: int) -> pd.DataFrame:
    truth = read_evaluation_truth(GENE_EFFECT, fold, split_strategy=SPLIT)
    truth = scope_frame(truth, cohort_id=GENE_EFFECT, scope=primary_scope(GENE_EFFECT))
    return truth[[GENE_KEY, GENE_EFFECT]].reset_index(drop=True)


def fit_job(job: tuple[str, str, int]) -> dict:
    cohort, variant, fold = job
    roles = fold_roles(cohort, fold)
    model = make_learner()
    model.fit(variant_design(cohort, fold, variant, roles[TRAIN_ROLE].rows), roles[TRAIN_ROLE].labels)
    result = {"n_iter": int(model.n_iter_)}
    for role in (VALIDATION_ROLE, TEST_ROLE):
        result[role] = model.predict_proba(variant_design(cohort, fold, variant, roles[role].rows))[:, 1]
    if cohort == "depmap":
        truth = gene_effect_truth(fold)
        rows = cohort_genes("depmap").get_indexer(truth[GENE_KEY].astype(str).to_numpy())
        if (rows < 0).any():
            raise ValueError(f"fold {fold}: {int((rows < 0).sum())} gene effect test genes are not DepMap genes")
        result[GENE_EFFECT] = model.predict_proba(variant_design("depmap", fold, variant, rows))[:, 1]
        result["gene_effect_keys"] = truth[GENE_KEY].astype(str).to_numpy("object")
        result["gene_effect_target"] = truth[GENE_EFFECT].to_numpy("float64")
    return result


def frames_of(fits: dict, variant: str) -> dict[str, pd.DataFrame]:
    frames = {}
    for cohort in ALONE_COHORTS:
        parts = []
        for fold in FOLDS:
            roles = fold_roles(cohort, fold)
            for role in (VALIDATION_ROLE, TEST_ROLE):
                parts.append(pd.DataFrame({GENE_KEY: roles[role].keys, "fold": fold, "role": role,
                                           "label": roles[role].labels,
                                           "y_score": fits[(cohort, variant, fold)][role]}))
        frames[cohort] = pd.concat(parts, ignore_index=True)
    parts = []
    for fold in FOLDS:
        fit = fits[("depmap", variant, fold)]
        parts.append(pd.DataFrame({GENE_KEY: fit["gene_effect_keys"], "fold": fold, "role": TEST_ROLE,
                                   "label": fit["gene_effect_target"], "y_score": fit[GENE_EFFECT]}))
    frames[GENE_EFFECT] = pd.concat(parts, ignore_index=True)
    return frames


def kind_of(task: str) -> TargetKind:
    return TargetKind.CONTINUOUS if task == GENE_EFFECT else TargetKind.BINARY


def primary(task: str):
    return spearman if task == GENE_EFFECT else auprc


def test_part(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["role"] == TEST_ROLE].reset_index(drop=True)


def leader_test_scores(task: str, test: pd.DataFrame) -> np.ndarray:
    leader = pooled(TASK_LEADERS[task], task)
    leader[GENE_KEY] = leader[GENE_KEY].astype(str)
    scores = leader.set_index(GENE_KEY)["y_score"].reindex(pd.Index(test[GENE_KEY], dtype="object"))
    if scores.isna().any():
        raise ValueError(f"{TASK_LEADERS[task]} does not score every test gene of {task}")
    return scores.to_numpy("float64")


def interval_job(job: tuple[str, str]) -> dict:
    task, variant = job
    test = test_part(pd.read_csv(fixed_scores_path(task, "A")))
    if variant == "leader":
        scores = leader_test_scores(task, test)
    else:
        scores = test_part(pd.read_csv(fixed_scores_path(task, variant)))["y_score"].to_numpy("float64")
    metrics = {"primary": primary(task)}
    if task != GENE_EFFECT:
        metrics["auroc"] = auroc
    result = confidence_intervals(test["label"].to_numpy("float64"), scores, metrics,
                                  fold=test["fold"].to_numpy(), target_kind=kind_of(task))
    return {name: (float(v.point), float(v.half_width)) for name, v in result.items()}


def comparison_job(job: tuple[str, str, str]) -> dict:
    task, first, second = job
    test = test_part(pd.read_csv(fixed_scores_path(task, first)))
    other = (leader_test_scores(task, test) if second == "leader"
             else test_part(pd.read_csv(fixed_scores_path(task, second)))["y_score"].to_numpy("float64"))
    result = paired_difference(test["label"].to_numpy("float64"), test["y_score"].to_numpy("float64"), other,
                               primary(task), fold=test["fold"].to_numpy(), target_kind=kind_of(task))
    return {"difference": float(result.difference), "half_width": float(result.half_width), "p": float(result.p_value)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    arguments = parser.parse_args()
    FIXED.mkdir(parents=True, exist_ok=True)
    jobs = [(cohort, variant, fold) for cohort in ALONE_COHORTS for variant in VARIANTS for fold in FOLDS]
    fits = run_pool(fit_job, jobs, arguments.workers, "core feature set fits")
    rounds = {}
    for variant in VARIANTS:
        for task, frame in frames_of(fits, variant).items():
            frame.to_csv(fixed_scores_path(task, variant), index=False)
            source = "depmap" if task == GENE_EFFECT else task
            rounds[(task, variant)] = ";".join(str(fits[(source, variant, f)]["n_iter"]) for f in FOLDS)
    interval_jobs = [(task, variant) for task in TASKS for variant in (*VARIANTS, "leader")]
    intervals = run_pool(interval_job, interval_jobs, arguments.workers, "core feature set intervals", 30.0)
    comparison_jobs = [(task, first, second) for task in TASKS for first, second in COMPARISONS]
    compared = run_pool(comparison_job, comparison_jobs, arguments.workers, "core feature set comparisons", 30.0)
    rows = []
    for task in TASKS:
        for first, second in COMPARISONS:
            own, other, result = intervals[(task, first)], intervals[(task, second)], compared[(task, first, second)]
            rows.append({
                "cohort_id": task, "task": TASK_NAMES[task],
                "metric": "spearman" if task == GENE_EFFECT else "auprc",
                "first": first, "first_name": VARIANT_NAMES[first],
                "second": METHOD_NAMES[TASK_LEADERS[task]] if second == "leader" else second,
                "first_point": own["primary"][0], "first_half_width": own["primary"][1],
                "first_auroc": own.get("auroc", (np.nan, np.nan))[0],
                "first_auroc_half_width": own.get("auroc", (np.nan, np.nan))[1],
                "second_point": other["primary"][0], "second_half_width": other["primary"][1],
                "difference": result["difference"], "half_width": result["half_width"], "p": result["p"],
                "first_rounds": rounds[(task, first)],
            })
    table = pd.DataFrame(rows)
    table["holm_p"] = holm(table, ["cohort_id"])
    write_table(table, "core_fixed")
    print("wrote core_fixed.csv", flush=True)


if __name__ == "__main__":
    main()
