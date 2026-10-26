"""The benchmark as the agent tools read it: tasks, folds, genes, the published predictions and submission tables.

Shared by ``server.py`` and the modules behind its tools; it runs nothing by itself. Tasks are named by
the cohort identifiers of ``egbench`` (``deg``, ``mouse``, ...), genes by HGNC identifiers. A
submission table has one row per gene with the columns ``gene_key``, ``y_score`` and ``outer_fold``,
and is joined to the fold tables through the functions behind the published leaderboard
(``egbench.evaluate.join_targets`` and ``scope_frame``), so it is scored on the same genes as the 15
predictors of the manuscript, whose test-fold predictions are read from
``agent/predictions/<cohort>.parquet``.
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from egbench.contracts import METHOD_IDS
from egbench.data import (
    FULL_SCOPE,
    cohort_target,
    prediction_cohort,
    primary_scope,
    read_evaluation_truth,
)
from egbench.evaluate import FOLD_COUNT, PRIMARY_METRICS, _excluded_genes, join_targets, scope_frame
from egbench.stats import CONFIDENCE_LEVEL

import external

SPLIT = "family_5fold"
GENE_KEY = "gene_key"
SCORE = "y_score"
FOLD = "outer_fold"
SUBMISSION_COLUMNS = (GENE_KEY, SCORE, FOLD)
ALPHA = 1.0 - CONFIDENCE_LEVEL
EXAMPLE_COUNT = 10
DIGITS = 4

PREDICTIONS_DIRECTORY = ROOT / "agent" / "predictions"
RESULTS_DIRECTORY = ROOT / "analysis" / "benchmark_report" / "results"
WORK_DIRECTORY = external.DATA_ROOT / "agent" / "work"
FUSION_ID = "fusion_top5"


class Task(NamedTuple):
    name: str
    question: str
    positives: str
    negatives: str | None


TASKS: dict[str, Task] = {
    "deg": Task(
        "DEG",
        "Whether a gene is common essential in human cell lines",
        "at least 5 votes across 16 Database of Essential Genes (DEG) human datasets",
        "0 votes",
    ),
    "depmap": Task(
        "DepMap",
        "Whether a gene is common essential in human cell lines",
        "common essential in DepMap 26Q1",
        "nonessential in DepMap 26Q1",
    ),
    "context": Task(
        "Context",
        "Whether a gene is essential in only a subset of contexts",
        "1 to 4 votes in DEG and strongly selective in DepMap, not common essential",
        "0 votes in DEG and nonessential in DepMap",
    ),
    "gene_effect": Task(
        "GeneEffect",
        "Whether genes can be ranked by continuous dependency strength",
        "continuous target: the mean gene effect in DepMap 26Q1, more negative meaning a "
        "stronger dependency; predictions are correlated with its negative",
        None,
    ),
    "blood": Task(
        "Blood",
        "Whether a gene is a blood-lineage dependency but not common essential",
        "depended on by more than 25% of measured blood cell lines, not common essential",
        "depended on by no blood cell lines",
    ),
    "mouse": Task(
        "Mouse",
        "Whether gene knockout is lethal in mice",
        "homozygous knockout lethal in the International Mouse Phenotyping Consortium (IMPC)",
        "viable after knockout",
    ),
    "depmap_23q4": Task(
        "DepMap 23Q4",
        "Whether a gene is common essential in the earlier DepMap release 23Q4; the "
        "predictors trained on it are those of the prospective validation against 26Q1",
        "common essential in DepMap 23Q4",
        "nonessential in DepMap 23Q4",
    ),
}

SCOPE_NOTES: dict[str, str] = {
    "crispr_screen_supported": (
        "scored on all negatives and on the positives supported by a CRISPR screen"
    ),
}


def number(value: object, digits: int = DIGITS) -> float | None:
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits)


def p_value(value: object) -> float | None:
    value = float(value)
    if not math.isfinite(value):
        return None
    return float(f"{value:.3g}")


def task_id(task: str) -> str:
    key = str(task).strip().lower().replace(" ", "_").replace("-", "_")
    if key in TASKS:
        return key
    for cohort_id, item in TASKS.items():
        if key == item.name.lower().replace(" ", "_"):
            return cohort_id
    raise ValueError(f"unknown task {task!r}; the tasks are {', '.join(TASKS)}")


def primary_metric(cohort_id: str) -> str:
    return PRIMARY_METRICS[cohort_target(cohort_id).kind]


@lru_cache(maxsize=1)
def symbols() -> pd.Series:
    frames = [
        pd.read_parquet(path, columns=[GENE_KEY, "gene_symbol"])
        for path in sorted((ROOT / "datasets").glob("*/data/*gene_universe.parquet"))
    ]
    table = pd.concat(frames, ignore_index=True).dropna().drop_duplicates(GENE_KEY)
    return table.set_index(GENE_KEY)["gene_symbol"]


def named(key: str) -> str:
    symbol = symbols().get(key)
    return key if symbol is None else f"{key} ({symbol})"


def examples(keys: list[str]) -> str:
    shown = ", ".join(named(key) for key in keys[:EXAMPLE_COUNT])
    return shown if len(keys) <= EXAMPLE_COUNT else f"{shown}, ..."


def resolve_gene(gene: str) -> str:
    text = str(gene).strip()
    if text.upper().startswith("HGNC:"):
        key = "HGNC:" + text.split(":", 1)[1]
        if key in symbols().index:
            return key
        raise ValueError(f"{gene!r} is not a gene of the benchmark")
    table = symbols()
    matches = table.index[table.str.upper() == text.upper()]
    if len(matches) == 0:
        raise ValueError(f"no gene of the benchmark has the symbol {gene!r}")
    return str(matches[0])


@lru_cache(maxsize=None)
def truth(cohort_id: str, outer_fold: int) -> pd.DataFrame:
    column = cohort_target(cohort_id).column
    table = read_evaluation_truth(cohort_id, outer_fold, split_strategy=SPLIT)
    return table[[GENE_KEY, column]].reset_index(drop=True)


@lru_cache(maxsize=None)
def test_folds(cohort_id: str) -> dict[str, int]:
    return {
        key: outer_fold
        for outer_fold in range(FOLD_COUNT)
        for key in truth(cohort_id, outer_fold)[GENE_KEY]
    }


@lru_cache(maxsize=None)
def published_predictions(cohort_id: str) -> pd.DataFrame:
    return pd.read_parquet(PREDICTIONS_DIRECTORY / f"{cohort_id}.parquet")


@lru_cache(maxsize=1)
def leaderboard_table() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIRECTORY / "primary_metric.csv")


@lru_cache(maxsize=1)
def significance_table() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIRECTORY / "significance.csv")


@lru_cache(maxsize=1)
def display_names() -> dict[str, str]:
    table = leaderboard_table()
    names = dict(zip(table["predictor"], table["display_name"]))
    return {method_id: names.get(method_id, method_id) for method_id in METHOD_IDS}


def predictor_id(name: str) -> str:
    text = name.strip().lower()
    for method_id, display in display_names().items():
        if text in (method_id, display.lower()):
            return method_id
    choices = ", ".join(f"{m} ({d})" for m, d in display_names().items())
    raise ValueError(f"unknown predictor {name!r}; the predictors are {choices}")


def resolve_path(path: str) -> Path:
    file = Path(path).expanduser()
    if not file.is_absolute():
        file = ROOT / file
    if not file.is_file():
        raise ValueError(f"{file}: no such file")
    return file


def read_table(path: str) -> tuple[Path, pd.DataFrame]:
    file = resolve_path(path)
    suffix = file.suffix.lower()
    if suffix == ".parquet":
        return file, pd.read_parquet(file)
    if suffix in (".tsv", ".txt"):
        return file, pd.read_csv(file, sep="\t")
    return file, pd.read_csv(file)


def submission_problems(frame: pd.DataFrame, cohort_id: str) -> tuple[list[str], dict[str, dict]]:
    absent = [column for column in SUBMISSION_COLUMNS if column not in frame.columns]
    if absent:
        return [
            f"missing columns {absent}; the table needs the columns gene_key, y_score and "
            f"outer_fold, it has {list(frame.columns)}"
        ], {}
    keys = frame[GENE_KEY].astype("string").fillna("")
    scores = pd.to_numeric(frame[SCORE], errors="coerce").to_numpy("float64")
    folds = pd.to_numeric(frame[FOLD], errors="coerce")
    problems = []
    unscored = sorted(set(keys[~np.isfinite(scores)]))
    if unscored:
        problems.append(f"{len(unscored)} genes have no finite y_score: {examples(unscored)}")
    unfolded = sorted(set(keys[~folds.isin(range(FOLD_COUNT))]))
    if unfolded:
        problems.append(
            f"{len(unfolded)} genes have an outer_fold other than 0 to 4: {examples(unfolded)}"
        )
    repeated = sorted(set(keys[keys.duplicated()]))
    if repeated:
        problems.append(f"{len(repeated)} gene_key values occur more than once: {examples(repeated)}")
    tolerated = _excluded_genes(cohort_id, SPLIT, None)
    home = test_folds(cohort_id)
    per_fold = {}
    for outer_fold in range(FOLD_COUNT):
        observed = set(keys[folds == outer_fold])
        expected = set(truth(cohort_id, outer_fold)[GENE_KEY])
        missing = sorted(expected - observed)
        stray = sorted(observed - expected - tolerated)
        per_fold[str(outer_fold)] = {
            "rows": int((folds == outer_fold).sum()),
            "test_genes": len(expected),
            "missing": len(missing),
            "not_test_genes": len(stray),
        }
        if missing:
            problems.append(
                f"fold {outer_fold}: {len(missing)} test genes have no prediction: "
                f"{examples(missing)}"
            )
        if stray:
            elsewhere = sum(1 for key in stray if key in home)
            note = f", {elsewhere} of them test genes of another fold" if elsewhere else ""
            problems.append(
                f"fold {outer_fold}: {len(stray)} genes are not test genes of this fold{note}: "
                f"{examples(stray)}"
            )
    return problems, per_fold


def submission_folds(frame: pd.DataFrame, cohort_id: str, source: Path) -> list[pd.DataFrame]:
    table = pd.DataFrame(
        {
            GENE_KEY: frame[GENE_KEY].astype("object"),
            SCORE: pd.to_numeric(frame[SCORE]).astype("float64"),
            FOLD: pd.to_numeric(frame[FOLD]).astype("int64"),
        }
    )
    return [
        join_targets(
            table.loc[table[FOLD] == outer_fold, [GENE_KEY, SCORE]],
            truth(cohort_id, outer_fold),
            cohort_id=cohort_id,
            split_strategy=SPLIT,
            source=f"{source} fold {outer_fold}",
        )
        for outer_fold in range(FOLD_COUNT)
    ]


def predictor_folds(method_id: str, cohort_id: str) -> list[pd.DataFrame]:
    table = published_predictions(prediction_cohort(cohort_id))
    return [
        join_targets(
            table.loc[table[FOLD] == outer_fold, [GENE_KEY, method_id]].rename(
                columns={method_id: SCORE}
            ),
            truth(cohort_id, outer_fold),
            cohort_id=cohort_id,
            split_strategy=SPLIT,
            source=f"{method_id} on {cohort_id} fold {outer_fold}",
        )
        for outer_fold in range(FOLD_COUNT)
    ]


def pooled(folds: list[pd.DataFrame], cohort_id: str) -> pd.DataFrame:
    scope = primary_scope(cohort_id)
    frames = [
        scope_frame(frame, cohort_id=cohort_id, scope=scope).assign(**{FOLD: outer_fold})
        for outer_fold, frame in enumerate(folds)
    ]
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values([FOLD, GENE_KEY], kind="stable")
        .reset_index(drop=True)
    )


def scope_note(cohort_id: str) -> dict[str, str]:
    scope = primary_scope(cohort_id)
    if scope == FULL_SCOPE:
        return {"name": scope, "meaning": "all genes of the task"}
    return {"name": scope, "meaning": SCOPE_NOTES.get(scope, scope)}


def ranked_points(cohort_id: str) -> pd.DataFrame:
    table = leaderboard_table()
    rows = table.loc[
        table["cohort_id"].eq(cohort_id) & table["predictor"].isin(METHOD_IDS)
    ]
    return rows.sort_values("point", ascending=False, kind="stable")
