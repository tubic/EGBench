"""Custom tasks on a user's own binary gene labels: protein-family folds, scoring, and the published predictors on the labels.

``make_task`` splits labelled genes into five folds by protein family with the procedure and the seed
that built the folds of the binary benchmark tasks, and writes one CSV per fold in the layout of
``get_fold``, so a model is trained on a custom task as on a benchmark task. ``check_task`` and
``evaluate_task`` check and score a submission table (gene_key, y_score, outer_fold) on a custom task
as ``check_predictions`` and ``evaluate`` do on a benchmark task. ``score_labels`` scores the test-fold
predictions of the 15 published predictors on one benchmark task against the user's labels. Genes
are resolved with the repository alone: the HGNC identifiers and symbols of the benchmark genes
(``datasets/*/data/*gene_universe.parquet``) and the protein families of
``datasets/homology/cluster_map.parquet``.

    python agent/labels.py check

rebuilds the folds of the six binary tasks from their labels and compares them with the published
folds, scores the DEG example predictor and the 15 predictors on the DEG labels and compares them with
the leaderboard, and writes the comparisons to ``agent/results/labels_check.csv``.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import multiprocessing
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd
from pydantic import Field

from benchmark import (
    ALPHA,
    EXAMPLE_COUNT,
    FOLD,
    GENE_KEY,
    ROOT,
    SCORE,
    SPLIT,
    SUBMISSION_COLUMNS,
    WORK_DIRECTORY,
    display_names,
    examples,
    leaderboard_table,
    number,
    p_value,
    published_predictions,
    read_table,
    significance_table,
    submission_folds,
    symbols,
    task_id,
    test_folds,
)
from egbench.contracts import COHORT_IDS, METHOD_IDS, SEED, TargetKind
from egbench.data import (
    FULL_SCOPE,
    TEST_ROLE,
    TRAIN_ROLE,
    VALIDATION_ROLE,
    cohort_tables_directory,
    cohort_target,
    fold_gene_keys,
    prediction_cohort,
    read_training_labels,
)
from egbench.evaluate import FOLD_COUNT, METRICS, PRIMARY_METRICS, score_frames
from egbench.stats import confidence_intervals, holm_adjusted, paired_difference

TASKS_DIRECTORY = WORK_DIRECTORY / "tasks"
FAMILY_TABLE = ROOT / "datasets" / "homology" / "cluster_map.parquet"
CHECK_OUTPUT = ROOT / "agent" / "results" / "labels_check.csv"
EXAMPLE_PREDICTOR = ROOT / "agent" / "examples" / "deg_logistic.csv"
LABEL_FILE = "labels.csv"
LEFT_OUT_FILE = "left_out.csv"
GENE_COLUMNS = ("gene_key", "hgnc_id", "gene", "gene_symbol", "symbol")
LABEL_COLUMNS = ("label", "essential", "y")
NAME = re.compile(r"[a-z][a-z0-9_]{0,39}")
ROLES = (TRAIN_ROLE, VALIDATION_ROLE, TEST_ROLE)
METRIC = PRIMARY_METRICS[TargetKind.BINARY]
BINARY_TASKS = tuple(c for c in COHORT_IDS if cohort_target(c).kind is TargetKind.BINARY)

# Constants of the fold procedure of the benchmark.
VALIDATION_FRACTION = 0.125
TOLERANCE = 1.0e-15
SEED_STRIDE = 10_000
CANDIDATES = 10_000
CANDIDATE_BLOCK = 256
MIXER = np.uint64(0xD2B74407B1CE6E93)
THRESHOLD = np.uint64(int(VALIDATION_FRACTION * (1 << 64)))
SPLITMIX = (
    np.uint64(0x9E3779B97F4A7C15),
    np.uint64(0xBF58476D1CE4E5B9),
    np.uint64(0x94D049BB133111EB),
)

EMPTY = "no gene identifier"
NOT_BINARY = "label is not 0 or 1"
UNKNOWN = "not an HGNC identifier or symbol of a benchmark gene"
CONFLICT = "the gene occurs with both labels"
NO_FAMILY = (
    "no protein family: the gene has no protein sequence in the benchmark, so it cannot be placed in a fold"
)
TOO_SMALL = (
    "the labels are too few to split into five folds: every test fold and every train and "
    "validation set needs positives and negatives, so give more of both from different protein "
    "families"
)


class Labels(NamedTuple):
    file: Path
    genes: pd.DataFrame
    left_out: pd.DataFrame
    merged_rows: int


class Family(NamedTuple):
    cluster_id: str
    size: int
    positives: int


def _column(frame: pd.DataFrame, choices: tuple[str, ...], what: str) -> str:
    named = {str(column).strip().lower(): column for column in frame.columns}
    for choice in choices:
        if choice in named:
            return named[choice]
    raise ValueError(
        f"the labels table needs a {what} column named {' or '.join(choices)}; its columns are "
        f"{list(frame.columns)}"
    )


@lru_cache(maxsize=1)
def _families() -> pd.Series:
    table = pd.read_parquet(FAMILY_TABLE, columns=[GENE_KEY, "cluster_id"])
    return table.set_index(GENE_KEY)["cluster_id"].astype("object")


@lru_cache(maxsize=1)
def _keys_by_symbol() -> dict[str, str]:
    """Every symbol a gene table of the benchmark gives a gene.

    Tables built from other HGNC releases give a few genes an older symbol, so a gene can carry two.
    """
    frames = [
        pd.read_parquet(path, columns=[GENE_KEY, "gene_symbol"])
        for path in sorted((ROOT / "datasets").glob("*/data/*gene_universe.parquet"))
    ]
    table = pd.concat(frames, ignore_index=True).dropna()
    table = table.assign(symbol=table["gene_symbol"].str.upper()).drop_duplicates([GENE_KEY, "symbol"])
    table = table.loc[~table["symbol"].duplicated(keep=False)]
    return dict(zip(table["symbol"], table[GENE_KEY].astype("object")))


def _key(text: str) -> str | None:
    upper = text.upper()
    if not upper.startswith("HGNC:"):
        return _keys_by_symbol().get(upper)
    digits = upper[5:].strip()
    key = f"HGNC:{int(digits)}" if digits.isdigit() else text
    return key if key in symbols().index or key in _families().index else None


def read_labels(path: str) -> Labels:
    """Read a labels table and resolve its genes; rows that cannot be used are set aside with a reason."""
    file, frame = read_table(path)
    gene_column = _column(frame, GENE_COLUMNS, "gene")
    label_column = _column(frame, LABEL_COLUMNS, "label")
    given = frame[gene_column].astype("string").str.strip().fillna("").astype("object")
    rows = pd.DataFrame(
        {
            "given": given.to_numpy(),
            GENE_KEY: [_key(text) if text else None for text in given],
            "label": pd.to_numeric(frame[label_column], errors="coerce").to_numpy("float64"),
            "reason": None,
        }
    )
    rows.loc[~rows["label"].isin((0.0, 1.0)), "reason"] = NOT_BINARY
    rows.loc[rows[GENE_KEY].isna(), "reason"] = UNKNOWN
    rows.loc[rows["given"].eq(""), "reason"] = EMPTY
    usable = rows["reason"].isna()
    counts = rows.loc[usable].groupby(GENE_KEY)["label"].nunique()
    rows.loc[usable & rows[GENE_KEY].isin(counts.index[counts > 1]), "reason"] = CONFLICT
    kept = rows.loc[rows["reason"].isna()]
    genes = kept.drop_duplicates(GENE_KEY).sort_values(GENE_KEY, kind="stable")
    genes = pd.DataFrame(
        {
            GENE_KEY: genes[GENE_KEY].to_numpy("object"),
            "gene_symbol": genes[GENE_KEY].map(symbols()).to_numpy("object"),
            "label": genes["label"].to_numpy("int64"),
            "given": genes["given"].to_numpy("object"),
        }
    )
    left_out = rows.loc[rows["reason"].notna(), ["given", GENE_KEY, "reason"]]
    return Labels(file, genes, left_out.reset_index(drop=True), int(len(kept) - len(genes)))


def _left_out_summary(left_out: pd.DataFrame) -> list[dict]:
    summary = []
    for reason, rows in left_out.groupby("reason", sort=False):
        shown = [str(value) for value in rows["given"]]
        listed = ", ".join(shown[:EXAMPLE_COUNT]) + (", ..." if len(shown) > EXAMPLE_COUNT else "")
        summary.append({"reason": reason, "rows": int(len(rows)), "examples": listed})
    return summary


def _fused(a: float, b: float, c: float) -> float:
    """a * b + c rounded once, as a fused multiply-add instruction computes it."""
    a_top, a_bottom = a.as_integer_ratio()
    b_top, b_bottom = b.as_integer_ratio()
    c_top, c_bottom = c.as_integer_ratio()
    return (a_top * b_top * c_bottom + c_top * a_bottom * b_bottom) / (a_bottom * b_bottom * c_bottom)


def _squares(values: list[float]) -> float:
    """Sum of squares of five numbers in the order and roundings of numpy.dot on Apple Accelerate.

    The published folds were built with that dot product. Ties between folds are broken by the
    rounding of this sum while the folds are nearly empty, so it is computed here the same way on
    every machine.
    """
    first, second, third, fourth, fifth = values
    return _fused(third, third, first * first) + (_fused(fourth, fourth, second * second) + fifth * fifth)


def _outer_objective(sizes: np.ndarray, positives: np.ndarray, total_size: int, total_positives: int) -> float:
    target_size = total_size / len(sizes)
    target_positives = total_positives / len(positives)
    size_error = [(int(size) - target_size) / target_size for size in sizes]
    positive_error = [(int(count) - target_positives) / max(target_positives, 1.0) for count in positives]
    return _squares(size_error) + _squares(positive_error)


def _better(improvement: float, key: tuple, best_improvement: float, best_key: tuple | None) -> bool:
    if improvement > best_improvement + TOLERANCE:
        return True
    return (
        improvement > TOLERANCE
        and math.isclose(improvement, best_improvement, rel_tol=0.0, abs_tol=TOLERANCE)
        and (best_key is None or key < best_key)
    )


def _outer_folds(families: list[Family]) -> dict[str, int]:
    """Greedy assignment of the families to five folds balancing genes and positives, then moves and swaps."""
    total_size = sum(family.size for family in families)
    total_positives = sum(family.positives for family in families)
    if (
        len(families) < FOLD_COUNT
        or total_positives < FOLD_COUNT
        or total_size - total_positives < FOLD_COUNT
    ):
        raise ValueError(TOO_SMALL)

    def objective(sizes: np.ndarray, positives: np.ndarray) -> float:
        return _outer_objective(sizes, positives, total_size, total_positives)

    sizes = np.zeros(FOLD_COUNT, dtype=np.int64)
    positives = np.zeros(FOLD_COUNT, dtype=np.int64)
    fold_of: dict[str, int] = {}
    for family in sorted(families, key=lambda f: (-f.size, -f.positives, f.cluster_id)):
        best_fold, best_score = 0, math.inf
        for fold in range(FOLD_COUNT):
            trial_sizes, trial_positives = sizes.copy(), positives.copy()
            trial_sizes[fold] += family.size
            trial_positives[fold] += family.positives
            score = objective(trial_sizes, trial_positives)
            if score < best_score - TOLERANCE:
                best_fold, best_score = fold, score
        fold_of[family.cluster_id] = best_fold
        sizes[best_fold] += family.size
        positives[best_fold] += family.positives

    ordered = sorted(families, key=lambda f: f.cluster_id)
    for _ in range(max(100, 20 * len(families))):
        sizes = np.zeros(FOLD_COUNT, dtype=np.int64)
        positives = np.zeros(FOLD_COUNT, dtype=np.int64)
        for family in families:
            sizes[fold_of[family.cluster_id]] += family.size
            positives[fold_of[family.cluster_id]] += family.positives
        current = objective(sizes, positives)
        best_improvement, best_key, best_action = 0.0, None, None
        for family in ordered:
            source = fold_of[family.cluster_id]
            for target in range(FOLD_COUNT):
                if target == source:
                    continue
                trial_sizes, trial_positives = sizes.copy(), positives.copy()
                trial_sizes[source] -= family.size
                trial_sizes[target] += family.size
                trial_positives[source] -= family.positives
                trial_positives[target] += family.positives
                improvement = current - objective(trial_sizes, trial_positives)
                key = ("move", family.cluster_id, source, target)
                if _better(improvement, key, best_improvement, best_key):
                    best_improvement, best_key, best_action = improvement, key, ("move", family, target)
        representatives: dict[tuple[int, int, int], Family] = {}
        for family in ordered:
            representatives.setdefault((fold_of[family.cluster_id], family.size, family.positives), family)
        chosen = sorted(representatives.values(), key=lambda f: f.cluster_id)
        for index, left in enumerate(chosen):
            left_fold = fold_of[left.cluster_id]
            for right in chosen[index + 1 :]:
                right_fold = fold_of[right.cluster_id]
                if left_fold == right_fold:
                    continue
                trial_sizes, trial_positives = sizes.copy(), positives.copy()
                trial_sizes[left_fold] += right.size - left.size
                trial_sizes[right_fold] += left.size - right.size
                trial_positives[left_fold] += right.positives - left.positives
                trial_positives[right_fold] += left.positives - right.positives
                improvement = current - objective(trial_sizes, trial_positives)
                first, second = sorted((left.cluster_id, right.cluster_id))
                key = ("swap", first, second, min(left_fold, right_fold), max(left_fold, right_fold))
                if _better(improvement, key, best_improvement, best_key):
                    best_improvement, best_key, best_action = improvement, key, ("swap", left, right)
        if best_action is None:
            for fold in range(FOLD_COUNT):
                if positives[fold] == 0 or positives[fold] == sizes[fold]:
                    raise ValueError(TOO_SMALL)
            return fold_of
        operation, first_family, second = best_action
        if operation == "move":
            fold_of[first_family.cluster_id] = second
        else:
            fold_of[first_family.cluster_id], fold_of[second.cluster_id] = (
                fold_of[second.cluster_id],
                fold_of[first_family.cluster_id],
            )
    raise RuntimeError("the assignment of families to folds did not converge")


def _validation_objective(size: int, positives: int, total_size: int, total_positives: int) -> float:
    target_size = total_size * VALIDATION_FRACTION
    target_positives = total_positives * VALIDATION_FRACTION
    return ((size - target_size) / target_size) ** 2 + (
        (positives - target_positives) / max(target_positives, 1.0)
    ) ** 2


def _both_classes(size: int, positives: int, total_size: int, total_positives: int) -> bool:
    train_size = total_size - size
    train_positives = total_positives - positives
    return min(positives, size - positives, train_positives, train_size - train_positives) > 0


def _splitmix64(state: np.ndarray) -> np.ndarray:
    values = state + SPLITMIX[0]
    values = (values ^ (values >> np.uint64(30))) * SPLITMIX[1]
    values = (values ^ (values >> np.uint64(27))) * SPLITMIX[2]
    return values ^ (values >> np.uint64(31))


def _representatives(families: list[Family]) -> list[Family]:
    chosen: dict[tuple[int, int], Family] = {}
    for family in families:
        key = (family.size, family.positives)
        if key not in chosen or family.cluster_id < chosen[key].cluster_id:
            chosen[key] = family
    return sorted(chosen.values(), key=lambda f: f.cluster_id)


def _validation(remainder: list[Family], fold: int) -> set[str]:
    """Validation families of one fold: a greedy pick and 9,999 hashed candidates, then adds, removes and swaps."""
    ordered = sorted(remainder, key=lambda f: f.cluster_id)
    total_size = sum(family.size for family in ordered)
    total_positives = sum(family.positives for family in ordered)
    if min(total_positives, total_size - total_positives) < 2:
        raise ValueError(TOO_SMALL)

    def objective(size: int, positives: int) -> float:
        return _validation_objective(size, positives, total_size, total_positives)

    def valid(size: int, positives: int) -> bool:
        return _both_classes(size, positives, total_size, total_positives)

    selected: set[str] = set()
    size = positives = 0
    for family in sorted(ordered, key=lambda f: (-f.size, -f.positives, f.cluster_id)):
        if objective(size + family.size, positives + family.positives) < objective(size, positives) - TOLERANCE:
            selected.add(family.cluster_id)
            size, positives = size + family.size, positives + family.positives
    first_seed = SEED + fold * SEED_STRIDE
    best: set[str] | None = None
    best_seed: int | None = None
    best_score = math.inf
    if valid(size, positives):
        best, best_seed, best_score = set(selected), first_seed, objective(size, positives)
    hashes = np.array(
        [int(hashlib.sha256(f.cluster_id.encode("utf-8")).hexdigest()[:16], 16) for f in ordered],
        dtype=np.uint64,
    )
    sizes = np.array([f.size for f in ordered], dtype=np.int64)
    counts = np.array([f.positives for f in ordered], dtype=np.int64)
    target_size = total_size * VALIDATION_FRACTION
    target_positives = total_positives * VALIDATION_FRACTION
    for start in range(1, CANDIDATES, CANDIDATE_BLOCK):
        stop = first_seed + min(CANDIDATES, start + CANDIDATE_BLOCK)
        seeds = np.arange(first_seed + start, stop, dtype=np.uint64)
        masks = _splitmix64(hashes[None, :] ^ (seeds[:, None] * MIXER)) < THRESHOLD
        chosen_sizes = masks @ sizes
        chosen_positives = masks @ counts
        train_sizes = total_size - chosen_sizes
        train_positives = total_positives - chosen_positives
        good = (
            (chosen_positives > 0)
            & (chosen_sizes - chosen_positives > 0)
            & (train_positives > 0)
            & (train_sizes - train_positives > 0)
        )
        scores = ((chosen_sizes - target_size) / target_size) ** 2 + (
            (chosen_positives - target_positives) / max(target_positives, 1.0)
        ) ** 2
        scores[~good] = np.inf
        for offset in np.flatnonzero(np.isfinite(scores)):
            seed, score = int(seeds[offset]), float(scores[offset])
            if score < best_score - TOLERANCE or (
                math.isclose(score, best_score, rel_tol=0.0, abs_tol=TOLERANCE)
                and (best_seed is None or seed < best_seed)
            ):
                best = {ordered[index].cluster_id for index in np.flatnonzero(masks[offset])}
                best_seed, best_score = seed, score
    if best is None:
        raise ValueError(TOO_SMALL)
    selected = best

    for _ in range(max(100, 20 * len(ordered))):
        inside = [f for f in ordered if f.cluster_id in selected]
        outside = [f for f in ordered if f.cluster_id not in selected]
        size = sum(f.size for f in inside)
        positives = sum(f.positives for f in inside)
        current = objective(size, positives)
        best_improvement, best_key, best_action = 0.0, None, None
        for family in ordered:
            if family.cluster_id in selected:
                operation, sign = "remove", -1
            else:
                operation, sign = "add", 1
            proposal = (size + sign * family.size, positives + sign * family.positives)
            if not valid(*proposal):
                continue
            improvement = current - objective(*proposal)
            key = (operation, family.cluster_id)
            if _better(improvement, key, best_improvement, best_key):
                best_improvement, best_key, best_action = improvement, key, (operation, family, None)
        for removed in _representatives(inside):
            for added in _representatives(outside):
                proposal = (size + added.size - removed.size, positives + added.positives - removed.positives)
                if not valid(*proposal):
                    continue
                improvement = current - objective(*proposal)
                key = ("swap", removed.cluster_id, added.cluster_id)
                if _better(improvement, key, best_improvement, best_key):
                    best_improvement, best_key, best_action = improvement, key, ("swap", removed, added)
        if best_action is None:
            return selected
        operation, family, added = best_action
        if operation == "add":
            selected.add(family.cluster_id)
        elif operation == "remove":
            selected.remove(family.cluster_id)
        else:
            selected.remove(family.cluster_id)
            selected.add(added.cluster_id)
    raise RuntimeError(f"the choice of validation families of fold {fold} did not converge")


def assign_folds(genes: pd.DataFrame) -> pd.DataFrame:
    """Role of every gene in every fold (gene_key, outer_fold, role) for genes with a cluster_id and a label."""
    families = [
        Family(str(cluster_id), int(len(part)), int(part["label"].sum()))
        for cluster_id, part in genes.groupby("cluster_id", sort=True)
    ]
    fold_of = _outer_folds(families)
    role_of: dict[tuple[str, int], str] = {}
    for fold in range(FOLD_COUNT):
        remainder = [family for family in families if fold_of[family.cluster_id] != fold]
        validation = _validation(remainder, fold)
        for family in families:
            if fold_of[family.cluster_id] == fold:
                role_of[family.cluster_id, fold] = TEST_ROLE
            elif family.cluster_id in validation:
                role_of[family.cluster_id, fold] = VALIDATION_ROLE
            else:
                role_of[family.cluster_id, fold] = TRAIN_ROLE
    keys = genes[GENE_KEY].to_numpy("object")
    clusters = genes["cluster_id"].to_numpy("object")
    return pd.DataFrame(
        {
            GENE_KEY: np.repeat(keys, FOLD_COUNT),
            FOLD: np.tile(np.arange(FOLD_COUNT, dtype=np.int64), len(keys)),
            "role": [role_of[cluster, fold] for cluster in clusters for fold in range(FOLD_COUNT)],
        }
    )


def _fold_table(genes: pd.DataFrame, roles: pd.Series) -> pd.DataFrame:
    table = genes.assign(role=genes[GENE_KEY].map(roles))
    parts = [
        table.loc[table["role"].eq(role), [GENE_KEY, "role", "label"]].sort_values(GENE_KEY)
        for role in (TRAIN_ROLE, VALIDATION_ROLE)
    ]
    parts.append(table.loc[table["role"].eq(TEST_ROLE), [GENE_KEY, "role"]].sort_values(GENE_KEY))
    frame = pd.concat(parts, ignore_index=True)
    frame.insert(1, "gene_symbol", frame[GENE_KEY].map(symbols()))
    return frame


def _new_name(name: str) -> str:
    text = str(name).strip()
    if not NAME.fullmatch(text):
        raise ValueError(
            f"name must be 1 to 40 lowercase letters, digits or underscores and start with a letter, "
            f"got {name!r}"
        )
    try:
        task_id(text)
    except ValueError:
        return text
    raise ValueError(f"{text!r} is the name of a benchmark task; choose another name")


def custom_tasks() -> list[str]:
    if not TASKS_DIRECTORY.is_dir():
        return []
    return sorted(path.name for path in TASKS_DIRECTORY.iterdir() if (path / LABEL_FILE).is_file())


def is_custom_task(task: str) -> bool:
    """Whether a task name passed to the tools names a task made by make_task."""
    text = str(task).strip()
    return bool(NAME.fullmatch(text)) and (TASKS_DIRECTORY / text / LABEL_FILE).is_file()


def task_directory(name: str) -> Path:
    text = str(name).strip()
    if is_custom_task(text):
        return TASKS_DIRECTORY / text
    known = ", ".join(custom_tasks()) or "none yet"
    raise ValueError(f"no custom task named {name!r}; make_task creates one (custom tasks: {known})")


def _make(labels: str, task: str, directory: Path) -> dict:
    read = read_labels(labels)
    genes = read.genes.assign(cluster_id=read.genes[GENE_KEY].map(_families()))
    homeless = genes["cluster_id"].isna()
    left_out = pd.concat(
        [
            read.left_out,
            genes.loc[homeless, ["given", GENE_KEY]].assign(reason=NO_FAMILY),
        ],
        ignore_index=True,
    )
    genes = genes.loc[~homeless].reset_index(drop=True)
    assignment = assign_folds(genes)
    tested = assignment.loc[assignment["role"].eq(TEST_ROLE)].set_index(GENE_KEY)[FOLD]
    truth = genes[[GENE_KEY, "gene_symbol", "label"]].assign(
        **{FOLD: genes[GENE_KEY].map(tested).to_numpy("int64")}
    )
    truth = truth.sort_values([FOLD, GENE_KEY], kind="stable").reset_index(drop=True)
    labels_path = directory / LABEL_FILE
    text = truth.to_csv(index=False)
    if labels_path.is_file() and labels_path.read_text() != text:
        raise ValueError(
            f"a custom task named {task!r} exists with other labels or folds ({directory}); give these "
            f"labels another name"
        )
    directory.mkdir(parents=True, exist_ok=True)
    folds = {}
    files = []
    for fold in range(FOLD_COUNT):
        roles = assignment.loc[assignment[FOLD].eq(fold)].set_index(GENE_KEY)["role"]
        table = _fold_table(genes, roles)
        path = directory / f"fold_{fold}.csv"
        table.to_csv(path, index=False)
        files.append(str(path))
        labelled = genes.assign(role=genes[GENE_KEY].map(roles))
        folds[str(fold)] = {
            role: {
                "genes": int(labelled["role"].eq(role).sum()),
                "positives": int(labelled.loc[labelled["role"].eq(role), "label"].sum()),
            }
            for role in ROLES
        }
    labels_path.write_text(text)
    left_out.assign(gene_symbol=left_out[GENE_KEY].map(symbols()))[
        ["given", GENE_KEY, "gene_symbol", "reason"]
    ].to_csv(directory / LEFT_OUT_FILE, index=False)
    return {
        "task": task,
        "labels": str(read.file),
        "genes": int(len(genes)),
        "positives": int(genes["label"].sum()),
        "protein_families": int(genes["cluster_id"].nunique()),
        "fold_files": files,
        "labels_file": str(labels_path),
        "folds": folds,
        "left_out": {
            "rows": int(len(left_out)),
            "file": str(directory / LEFT_OUT_FILE),
            "reasons": _left_out_summary(left_out),
        },
        "duplicate_rows_merged": read.merged_rows,
        "next": (
            f"for each fold_<k>.csv train on the train genes, tune on the validation genes and predict "
            f"the test genes; write the test predictions of the five folds to one table with the "
            f"columns gene_key, y_score, outer_fold and pass task={task} to check_predictions and "
            f"evaluate"
        ),
    }


def make_task(
    labels: Annotated[
        str,
        Field(
            description=(
                "CSV, TSV or parquet file with one row per gene: a gene column (gene_key, hgnc_id, "
                "gene, gene_symbol or symbol) holding HGNC identifiers such as HGNC:9187 or symbols "
                "such as POLR2A, and a label column (label, essential or y) holding 1 for positive "
                "and 0 for negative"
            )
        ),
    ],
    name: Annotated[
        str,
        Field(
            description=(
                "name of the new task: lowercase letters, digits and underscores, starting with a "
                "letter, for example my_screen; not a benchmark task name"
            )
        ),
    ],
) -> dict:
    """Make a custom task from your own binary gene labels, split into five folds by protein family.

    The labels file has one row per gene with a gene column and a 0/1 label column (see the
    parameter). Genes are split exactly as the benchmark tasks were: the genes of one protein
    family stay in one fold, the five folds balance genes and positives, and in each fold the
    genes outside the test fold are divided into train and validation genes. Writes
    agent/work/tasks/<name>/fold_<k>.csv under the data root for k = 0 to 4 with the columns gene_key, gene_symbol,
    role (train, validation or test) and label (empty for test genes), the same layout get_fold
    writes, and labels.csv with every gene's label and test fold. Returns the paths, the number of
    genes and positives per role in each fold, and the rows left out (unknown genes, labels other
    than 0 and 1, genes given with both labels, genes without a protein family) with reasons and
    examples; all of them are listed in left_out.csv. Afterwards pass task=<name> to
    check_predictions and evaluate as for a benchmark task.
    """
    task = _new_name(name)
    return _make(labels, task, TASKS_DIRECTORY / task)


def _truth(directory: Path) -> pd.DataFrame:
    table = pd.read_csv(directory / LABEL_FILE, dtype={GENE_KEY: "object"})
    return table[[GENE_KEY, "label", FOLD]].astype({"label": "int64", FOLD: "int64"})


def _problems(frame: pd.DataFrame, truth: pd.DataFrame) -> tuple[list[str], dict[str, dict]]:
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
        problems.append(f"{len(unfolded)} genes have an outer_fold other than 0 to 4: {examples(unfolded)}")
    repeated = sorted(set(keys[keys.duplicated()]))
    if repeated:
        problems.append(f"{len(repeated)} gene_key values occur more than once: {examples(repeated)}")
    home = dict(zip(truth[GENE_KEY], truth[FOLD]))
    per_fold = {}
    for outer_fold in range(FOLD_COUNT):
        observed = set(keys[folds == outer_fold])
        expected = set(truth.loc[truth[FOLD].eq(outer_fold), GENE_KEY])
        missing = sorted(expected - observed)
        stray = sorted(observed - expected)
        per_fold[str(outer_fold)] = {
            "rows": int((folds == outer_fold).sum()),
            "test_genes": len(expected),
            "missing": len(missing),
            "not_test_genes": len(stray),
        }
        if missing:
            problems.append(
                f"fold {outer_fold}: {len(missing)} test genes have no prediction: {examples(missing)}"
            )
        if stray:
            elsewhere = sum(1 for key in stray if key in home)
            note = f", {elsewhere} of them test genes of another fold" if elsewhere else ""
            problems.append(
                f"fold {outer_fold}: {len(stray)} genes are not test genes of this fold{note}: "
                f"{examples(stray)}"
            )
    return problems, per_fold


def _mean_of_defined(values: list[float]) -> float:
    defined = [value for value in values if not math.isnan(value)]
    return float(np.mean(defined)) if defined else math.nan


def _scores(job: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[dict[str, list[float]], dict]:
    y_true, y_score, fold = job
    per_fold = {
        name: [float(function(y_true[fold == k], y_score[fold == k])) for k in np.unique(fold)]
        for name, function in METRICS.items()
    }
    intervals = confidence_intervals(y_true, y_score, dict(METRICS), fold=fold, target_kind=TargetKind.BINARY)
    return per_fold, intervals


def _metrics(per_fold: dict[str, list[float]], intervals: dict) -> dict[str, dict]:
    return {
        name: {"point": number(_mean_of_defined(values)), "half_width": number(intervals[name].half_width)}
        for name, values in per_fold.items()
    }


def _evaluate(path: str, task: str, directory: Path) -> dict:
    file, frame = read_table(path)
    truth = _truth(directory)
    problems, _ = _problems(frame, truth)
    if problems:
        return {"task": task, "path": str(file), "ok": False, "problems": problems}
    submitted = pd.DataFrame(
        {GENE_KEY: frame[GENE_KEY].astype("object"), SCORE: pd.to_numeric(frame[SCORE]).astype("float64")}
    )
    scored = truth.merge(submitted, on=GENE_KEY, how="inner", validate="one_to_one")
    scored = scored.sort_values([FOLD, GENE_KEY], kind="stable")
    per_fold, intervals = _scores(
        (scored["label"].to_numpy("int64"), scored[SCORE].to_numpy("float64"), scored[FOLD].to_numpy("int64"))
    )
    return {
        "task": task,
        "path": str(file),
        "ok": True,
        "scope": {"name": FULL_SCOPE, "meaning": "all genes of the task"},
        "genes": int(len(scored)),
        "metrics": _metrics(per_fold, intervals),
        "per_fold": {name: [number(value) for value in values] for name, values in per_fold.items()},
    }


def check_task(
    path: Annotated[str, Field(description="CSV, TSV or parquet file of predictions")],
    task: Annotated[str, Field(description="name of a custom task made by make_task")],
) -> dict:
    """Check that a table of predictions can be scored on a custom task, without scoring it.

    The table has one row per gene and the columns gene_key (HGNC identifier such as HGNC:9187),
    y_score (higher means more likely positive) and outer_fold (0 to 4, the fold in which the gene
    is a test gene). Each fold must hold a prediction for every test gene of that fold of the
    custom task and for no other gene; the fold files of make_task list them. Returns ok, the rows
    per fold and every problem found, with example genes.
    """
    directory = task_directory(task)
    file, frame = read_table(path)
    problems, per_fold = _problems(frame, _truth(directory))
    return {
        "task": directory.name,
        "path": str(file),
        "ok": not problems,
        "rows": int(len(frame)),
        "folds": per_fold,
        "problems": problems,
    }


def evaluate_task(
    path: Annotated[str, Field(description="CSV, TSV or parquet file of predictions")],
    task: Annotated[str, Field(description="name of a custom task made by make_task")],
) -> dict:
    """Score a table of predictions on a custom task the way the benchmark tasks are scored.

    The table has the columns gene_key, y_score and outer_fold (see check_task). Each metric is the
    mean over the five folds, reported as point and half_width of the 95% gene-level bootstrap
    interval within folds. Returns AUPRC and AUROC with the value of each fold. A custom task has no
    published leaderboard, so no rank is given. A table with problems is not scored; the problems
    are returned instead.
    """
    directory = task_directory(task)
    return _evaluate(path, directory.name, directory)


def _paired(job: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    y_true, scores_a, scores_b, fold = job
    return paired_difference(
        y_true, scores_a, scores_b, METRICS[METRIC], fold=fold, target_kind=TargetKind.BINARY
    )


def _pool(function, jobs: list) -> list:
    if len(jobs) == 1:
        return [function(jobs[0])]
    # The bootstraps are Python loops, so they run in separate processes.
    workers = min(len(jobs), os.cpu_count() or 1)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return list(pool.map(function, jobs))


def score_labels(
    labels: Annotated[
        str,
        Field(
            description=(
                "CSV, TSV or parquet file with a gene column (gene_key, hgnc_id, gene, gene_symbol or "
                "symbol; HGNC identifiers or symbols) and a label column (label, essential or y; 1 "
                "positive, 0 negative)"
            )
        ),
    ],
    source_task: Annotated[
        str, Field(description="benchmark task whose published predictions are scored, for example deg")
    ],
) -> dict:
    """Score the 15 published predictors against your own gene labels, with their predictions on one task.

    For your genes that are test genes of the source task, each predictor's test-fold prediction is
    compared with your label. Returns, for each predictor, AUPRC and AUROC as point and half_width
    of the 95% gene-level bootstrap interval within the folds of the source task (the mean over the
    folds, as the leaderboard is computed), ranked by AUPRC; a paired bootstrap of the top predictor
    against each other predictor on AUPRC with Holm-adjusted p values and a verdict at the 5% level;
    the numbers of positives and negatives scored; and your genes that could not be scored, with
    examples. Folds in which your scored genes hold only one class are left out.
    """
    cohort_id = task_id(source_task)
    read = read_labels(labels)
    source = prediction_cohort(cohort_id)
    table = published_predictions(source)
    table = table.loc[table[GENE_KEY].isin(list(test_folds(cohort_id)))]
    merged = read.genes.merge(table, on=GENE_KEY, how="inner", validate="one_to_one")
    unscored = sorted(set(read.genes[GENE_KEY]) - set(merged[GENE_KEY]))
    classes = merged.groupby(FOLD)["label"].agg(["sum", "count"])
    folds = [int(k) for k, row in classes.iterrows() if 0 < row["sum"] < row["count"]]
    if not folds:
        raise ValueError(
            f"{len(merged)} of your genes are test genes of {cohort_id}, and no fold of {cohort_id} "
            f"holds both a positive (1) and a negative (0) of them; give more labelled genes of both "
            f"classes"
        )
    single = merged.loc[~merged[FOLD].isin(folds)]
    merged = merged.loc[merged[FOLD].isin(folds)].sort_values([FOLD, GENE_KEY], kind="stable")
    y_true = merged["label"].to_numpy("int64")
    fold = merged[FOLD].to_numpy("int64")
    names = display_names()
    outcomes = _pool(_scores, [(y_true, merged[m].to_numpy("float64"), fold) for m in METHOD_IDS])
    rows = []
    for method_id, (per_fold, intervals) in zip(METHOD_IDS, outcomes):
        rows.append((method_id, _mean_of_defined(per_fold[METRIC]), _metrics(per_fold, intervals)))
    rows.sort(key=lambda row: -row[1] if math.isfinite(row[1]) else math.inf)
    top = rows[0][0]
    others = [row[0] for row in rows[1:]]
    comparisons = _pool(
        _paired,
        [(y_true, merged[top].to_numpy("float64"), merged[m].to_numpy("float64"), fold) for m in others],
    )
    adjusted = holm_adjusted(np.array([item.p_value for item in comparisons], dtype="float64"))
    tests = []
    for method_id, item, holm in zip(others, comparisons, adjusted):
        if math.isfinite(holm) and holm <= ALPHA:
            verdict = "top predictor better" if item.difference > 0 else "this predictor better"
        else:
            verdict = "not distinguishable"
        tests.append(
            {
                "predictor": names[method_id],
                "difference_top_minus_predictor": number(item.difference),
                "half_width": number(item.half_width),
                "p": p_value(item.p_value),
                "holm_p": p_value(holm),
                "verdict": verdict,
            }
        )
    positives = int(y_true.sum())
    result = {
        "source_task": cohort_id,
        "path": str(read.file),
        "genes_scored": int(len(y_true)),
        "positives": positives,
        "negatives": int(len(y_true) - positives),
        "positive_fraction": number(positives / len(y_true), 3),
        "folds_scored": folds,
        "coverage": {
            "your_genes": int(len(read.genes)),
            "not_test_genes_of_task": len(unscored),
            "examples": examples(unscored),
            "in_folds_with_one_class": int(len(single)),
        },
        "left_out": _left_out_summary(read.left_out),
        "ranking": [
            {"rank": rank, "predictor": names[method_id], **metrics}
            for rank, (method_id, _, metrics) in enumerate(rows, start=1)
        ],
        "paired_tests": {"metric": METRIC, "top": names[top], "comparisons": tests},
        "note": (
            f"Each gene's score comes from the {source} fold model for which the gene was a test "
            f"gene, so no predictor saw its {source} label, but the predictors learned {source} "
            f"labels, which may correlate with yours, so this measures how well they transfer from "
            f"{source} to your labels."
        ),
    }
    if source != cohort_id:
        result["predictions_from"] = source
    return result


def _published_fold(cohort_id: str, outer_fold: int) -> pd.DataFrame:
    """The table get_fold writes for a fold of a benchmark task."""
    parts = [
        read_training_labels(cohort_id, outer_fold, role, split_strategy=SPLIT)[[GENE_KEY, "role", "label"]]
        for role in (TRAIN_ROLE, VALIDATION_ROLE)
    ]
    test = fold_gene_keys(cohort_id, outer_fold, TEST_ROLE, split_strategy=SPLIT)
    parts.append(pd.DataFrame({GENE_KEY: test.astype("object"), "role": TEST_ROLE}))
    table = pd.concat(parts, ignore_index=True)
    table.insert(1, "gene_symbol", table[GENE_KEY].map(symbols()))
    return table


def _check_folds(cohort_id: str, scratch: Path) -> dict:
    labels = cohort_tables_directory(cohort_id) / "common_binary.parquet"
    started = time.monotonic()
    made = _make(str(labels), f"check_{cohort_id}", scratch / cohort_id)
    seconds = time.monotonic() - started
    identical = True
    matching = compared = 0
    for outer_fold in range(FOLD_COUNT):
        published = _published_fold(cohort_id, outer_fold)
        mine = Path(made["fold_files"][outer_fold]).read_text()
        identical &= mine == published.to_csv(index=False)
        rebuilt = pd.read_csv(made["fold_files"][outer_fold], dtype={GENE_KEY: "object"})
        roles = rebuilt.set_index(GENE_KEY)["role"]
        expected = published.set_index(GENE_KEY)["role"]
        compared += len(expected)
        matching += int(roles.reindex(expected.index).eq(expected).sum())
    truth = _truth(scratch / cohort_id).set_index(GENE_KEY)[FOLD]
    homes = pd.Series(test_folds(cohort_id))
    identical &= truth.sort_index().equals(homes.sort_index().astype("int64").rename(FOLD).rename_axis(GENE_KEY))
    return {
        "check": "folds",
        "task": cohort_id,
        "subject": "role of each gene in folds 0 to 4",
        "compared": compared,
        "matching": matching,
        "largest_difference": math.nan,
        "identical": bool(identical and matching == compared),
        "seconds": round(seconds, 1),
    }


def _difference_row(check: str, subject: str, pairs: list[tuple[float, float]], seconds: float) -> dict:
    differences = [abs(a - b) for a, b in pairs]
    return {
        "check": check,
        "task": "deg",
        "subject": subject,
        "compared": len(pairs),
        "matching": sum(1 for value in differences if value == 0),
        "largest_difference": max(differences),
        "identical": all(value == 0 for value in differences),
        "seconds": round(seconds, 1),
    }


def _check_evaluate(scratch: Path) -> dict:
    started = time.monotonic()
    mine = _evaluate(str(EXAMPLE_PREDICTOR), "check_deg", scratch / "deg")
    file, frame = read_table(str(EXAMPLE_PREDICTOR))
    folds = submission_folds(frame, "deg", file)
    scored = score_frames(folds, method_id="example", cohort_id="deg", split_strategy=SPLIT)
    pairs = []
    for name, function in METRICS.items():
        pairs.append((mine["metrics"][name]["point"], number(scored.values[name])))
        pairs.append((mine["metrics"][name]["half_width"], number(scored.intervals[name].half_width)))
        for value, fold in zip(mine["per_fold"][name], folds):
            pairs.append((value, number(function(fold["label"].to_numpy(), fold[SCORE].to_numpy()))))
    return _difference_row(
        "evaluate_task", "example predictor deg_logistic: points, half-widths and fold values", pairs,
        time.monotonic() - started,
    )


def _check_score_labels() -> list[dict]:
    started = time.monotonic()
    result = score_labels(str(cohort_tables_directory("deg") / "common_binary.parquet"), "deg")
    seconds = time.monotonic() - started
    published = leaderboard_table()
    published = published.loc[published["cohort_id"].eq("deg") & published["predictor"].isin(METHOD_IDS)]
    by_name = published.set_index("display_name")
    pairs = []
    for row in result["ranking"]:
        reference = by_name.loc[row["predictor"]]
        pairs.append((row[METRIC]["point"], number(reference["point"])))
        pairs.append((row[METRIC]["half_width"], number(reference["half_width"])))
    rows = [_difference_row("score_labels", "AUPRC point and half-width of the 15 predictors", pairs, seconds)]
    names = display_names()
    top = next(m for m in METHOD_IDS if names[m] == result["paired_tests"]["top"])
    tests = significance_table()
    tests = tests.loc[
        tests["family"].eq("leaderboard_rank") & tests["cohort_id"].eq("deg") & tests["subject_a"].eq(top)
    ]
    found = {item["predictor"]: item for item in result["paired_tests"]["comparisons"]}
    pairs = []
    for _, test in tests.iterrows():
        item = found[names[test["subject_b"]]]
        pairs.append((item["difference_top_minus_predictor"], number(test["difference"])))
        pairs.append((item["half_width"], number(test["half_width"])))
        pairs.append((item["p"], p_value(test["p_value"])))
    if pairs:
        rows.append(
            _difference_row(
                "score_labels", "paired test of the top predictor: difference, half-width and p", pairs, seconds
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="reproduce the published folds and scores into agent/results/labels_check.csv")
    parser.parse_args(argv)
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        for number_done, cohort_id in enumerate(BINARY_TASKS, start=1):
            rows.append(_check_folds(cohort_id, scratch))
            row = rows[-1]
            print(
                f"{number_done}/{len(BINARY_TASKS)} {cohort_id}: {row['matching']} of {row['compared']} "
                f"roles match, identical files {row['identical']}, {row['seconds']} s",
                flush=True,
            )
        rows.append(_check_evaluate(scratch))
    rows.extend(_check_score_labels())
    table = pd.DataFrame(rows)
    CHECK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CHECK_OUTPUT, index=False)
    for row in rows[len(BINARY_TASKS) :]:
        print(
            f"{row['check']} {row['subject']}: {row['matching']} of {row['compared']} equal, largest "
            f"difference {row['largest_difference']:.2e}",
            flush=True,
        )
    print(f"written {CHECK_OUTPUT}")
    return 0 if table["identical"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
