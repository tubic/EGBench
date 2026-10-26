"""MCP server that exposes EGBench to AI agents, in two layers of tools.

Start it with ``python agent/server.py``, or with the ``egbench-mcp`` command of the installed
package (``launch.py``). It speaks the Model Context
Protocol over standard input and output, so any MCP client, whether a desktop assistant or a
script driving an open-weight model, can start it as a subprocess and call its tools.

The tools of layer 0 need only the repository: ``list_tasks``, ``get_fold``, ``leaderboard`` and
``gene_profile`` answer questions about the tasks, the folds, the published ranking and single
genes; ``check_predictions``, ``evaluate`` and ``compare`` check, score and compare a table of
predictions (``gene_key``, ``y_score``, ``outer_fold``) through the functions behind the published
leaderboard; ``make_task`` and ``score_labels`` work on a user's own labels. The tools of layer 1
(``annotation.py``, ``core_model.py``) read public data downloaded on first use. The predictions of
the 15 methods are read from ``agent/predictions/<cohort>.parquet``. The method cards and the
READMEs are served as resources, and two prompts walk an agent through evaluating a predictor and
explaining a gene.
"""
from __future__ import annotations

import math
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, wraps
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

import annotation
import core_model
import external
import labels
from benchmark import (
    ALPHA,
    FOLD,
    FUSION_ID,
    GENE_KEY,
    ROOT,
    SCORE,
    SPLIT,
    TASKS,
    WORK_DIRECTORY,
    display_names,
    leaderboard_table,
    named,
    number,
    p_value,
    pooled,
    predictor_folds,
    predictor_id,
    primary_metric,
    published_predictions,
    ranked_points,
    read_table,
    resolve_gene,
    scope_note,
    significance_table,
    submission_folds,
    submission_problems,
    symbols,
    task_id,
    test_folds,
    truth,
)
from egbench import __version__
from egbench.contracts import COHORT_IDS, LEADERBOARD_COHORT_IDS, METHOD_IDS, TargetKind
from egbench.data import (
    TEST_ROLE,
    cohort_target,
    fold_gene_keys,
    prediction_cohort,
    primary_scope,
    read_training_labels,
    scope_gene_keys,
)
from egbench.evaluate import FOLD_COUNT, metric_table, scope_frame, score_frames
from egbench.stats import CONFIDENCE_LEVEL, RESAMPLE_COUNT, holm_adjusted, paired_difference

AGENT = Path(__file__).resolve().parent
TOP_PERCENTILE = 90.0
DOWNLOAD_WAIT_SECONDS = 45.0

# Tools of layer 1 and the manifest items (agent/external.json) each of them reads; annotate_genes reads
# other items for its aspects dependency and mouse.
TOOL_INPUTS: dict[str, tuple[str, ...]] = {
    "annotate_genes": (annotation.ANNOTATION,),
    "annotate_genes:dependency": (annotation.DEPENDENCY_TABLE, "gene_reference"),
    "annotate_genes:mouse": (annotation.MOUSE_TABLE, "gene_reference"),
    "enrichment": annotation.ENRICHMENT_INPUTS,
    "evaluate_by_strata": (annotation.ANNOTATION,),
    "core_model": core_model.INPUTS,
}
LAYER_1 = "layer1"
LAYER_0_FILES: tuple[str, ...] = (
    "agent/predictions",
    "analysis/benchmark_report/results/primary_metric.csv",
    "analysis/benchmark_report/results/significance.csv",
    "datasets/homology/cluster_map.parquet",
)

FAMILY_NOTES: dict[str, str] = {
    "leaderboard_rank": "paired difference between two of the top three predictors",
    "fusion_gain": "the combination of the top five predictors minus its best member",
}

INSTRUCTIONS = """\
EGBench benchmarks 15 published predictors of human gene essentiality on six tasks
(DEG, DepMap, Context, GeneEffect, Blood, Mouse) plus the DepMap 23Q4 labels of the
prospective validation. Genes are split into five folds by protein family, so the genes
of one family never sit on both sides of a split; every gene is a test gene in exactly
one fold. Binary tasks are scored by AUPRC (AUROC is reported too), GeneEffect by the
Spearman correlation; intervals are point ± half-width of the 95% gene-level bootstrap
interval within folds. Start with list_tasks. To evaluate a new predictor: get_fold for
each fold, train outside this server, write the test predictions of the five folds to one
CSV with the columns gene_key, y_score, outer_fold, then check_predictions, evaluate and
compare. For labels of the user's own, make_task splits them into folds by protein family
and score_labels shows which published predictor fits them best. These tools need no
downloads. The tools of layer 1 read public data that is downloaded on first use:
annotate_genes and enrichment describe a list of genes as the manuscript does,
evaluate_by_strata scores predictions within groups of genes, and core_model trains the
manuscript's core feature set model on a task to predict other genes. A tool that lacks
its data returns what is missing instead of a result; call fetch_external with the items
it names, then call the tool again. layer_status shows what every tool needs. One step
belongs to the user: the three DepMap files, which only a browser can download. Pass on the
page and the path that fetch_external gives, and call it again once the user has saved them.
The first call of a layer-1 tool builds tables and the first call of core_model on a task trains
for minutes; tell the user before such a call, since a client may stop a call that outlasts
its time limit. Genes are named by HGNC identifiers such as HGNC:9187; the
tools also accept symbols. Report numbers as the tools return them."""

server = MCPServer("egbench", version=__version__, instructions=INSTRUCTIONS)


def _reported(function):
    """Hand the message of any failure to the client; MCPServer withholds it otherwise.

    A tool whose external data are not downloaded yet returns the report of what is missing.
    """

    @wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except external.Missing as missing:
            return missing.report()
        except ToolError:
            raise
        except Exception as error:
            raise ToolError(f"{type(error).__name__}: {error}") from error

    return call


def _custom(task: str) -> bool:
    """Whether a task name is one written by make_task rather than a benchmark task."""
    try:
        task_id(task)
    except ValueError:
        return labels.is_custom_task(task)
    return False


def _role_counts(table: pd.DataFrame, binary: bool) -> dict[str, dict[str, int]]:
    roles = {}
    for role, rows in table.groupby("role", sort=False):
        roles[role] = {"genes": int(len(rows))}
        if role != TEST_ROLE and binary:
            roles[role]["positives"] = int(rows["label"].astype("int64").sum())
    return roles


@lru_cache(maxsize=None)
def _percentiles(cohort_id: str) -> pd.DataFrame:
    table = published_predictions(cohort_id)
    ranks = table.groupby(FOLD)[list(METHOD_IDS)].rank(pct=True, method="average") * 100.0
    return pd.concat([table[[GENE_KEY, FOLD]], ranks], axis=1).set_index(GENE_KEY)


def _paired(job: tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    cohort_id, y_true, scores_a, scores_b, fold = job
    return paired_difference(
        y_true,
        scores_a,
        scores_b,
        metric_table(cohort_id)[primary_metric(cohort_id)],
        fold=fold,
        target_kind=cohort_target(cohort_id).kind,
    )


@server.tool()
@_reported
def list_tasks() -> dict:
    """List the tasks of EGBench with their labels, sizes and how they are scored.

    Returns, for each task: its identifier (pass it as `task` to the other tools), the question
    it asks, the definition of positives and negatives, the number of genes and positives that
    are scored, the primary metric, and whether it is on the published leaderboard. Call this
    first when you do not know the task identifiers.
    """
    tasks = []
    for cohort_id, item in TASKS.items():
        target = cohort_target(cohort_id)
        kept = scope_gene_keys(cohort_id, primary_scope(cohort_id))
        scored = pd.concat([truth(cohort_id, k) for k in range(FOLD_COUNT)], ignore_index=True)
        if kept is not None:
            scored = scored.loc[scored[GENE_KEY].isin(kept)]
        entry = {
            "task": cohort_id,
            "name": item.name,
            "question": item.question,
            "positives": item.positives,
            "negatives": item.negatives,
            "target": "binary" if target.kind is TargetKind.BINARY else "continuous",
            "primary_metric": primary_metric(cohort_id),
            "scope": scope_note(cohort_id),
            "genes": int(len(scored)),
            "on_leaderboard": cohort_id in LEADERBOARD_COHORT_IDS,
        }
        if target.kind is TargetKind.BINARY:
            positives = int(scored[target.column].astype("int64").sum())
            entry["positive_genes"] = positives
            entry["positive_fraction"] = number(positives / len(scored), 3)
        if prediction_cohort(cohort_id) != cohort_id:
            entry["note"] = (
                f"the published predictors are trained on {prediction_cohort(cohort_id)} "
                f"labels and their predictions are scored against this target"
            )
        tasks.append(entry)
    return {
        "folds": (
            f"{FOLD_COUNT} folds split by protein family ({SPLIT}); each gene is a test gene "
            f"in exactly one fold, the other genes of that fold are train or validation genes"
        ),
        "intervals": (
            f"point ± half_width of the {int(CONFIDENCE_LEVEL * 100)}% gene-level bootstrap "
            f"interval within folds, {RESAMPLE_COUNT} resamples"
        ),
        "tasks": tasks,
        "custom_tasks": labels.custom_tasks(),
    }


@server.tool()
@_reported
def get_fold(
    task: Annotated[str, Field(description="task identifier, for example deg or mouse")],
    fold: Annotated[int, Field(description="fold number, 0 to 4")],
    output_dir: Annotated[
        str, Field(description="directory to write the CSV to; empty for agent/work/ under the data root")
    ] = "",
) -> dict:
    """Write the genes of one fold of a task to a CSV file for training a model.

    The CSV has one row per gene with the columns gene_key, gene_symbol, role (train,
    validation or test) and label. Labels are given for train and validation genes and left
    empty for test genes. Train on the train genes, tune on the validation genes and predict
    the test genes; the test predictions of the five folds together form the table that
    check_predictions, evaluate and compare take. A task made by make_task has its fold files
    already; for it this returns the existing file. Returns the file path and the number of
    genes and positives per role.
    """
    outer_fold = int(fold)
    if outer_fold not in range(FOLD_COUNT):
        raise ValueError(f"fold must be 0 to {FOLD_COUNT - 1}, got {fold!r}")
    if _custom(task):
        path = labels.task_directory(task) / f"fold_{outer_fold}.csv"
        table = pd.read_csv(path)
        return {
            "task": task,
            "fold": outer_fold,
            "path": str(path),
            "columns": list(table.columns),
            "roles": _role_counts(table, binary=True),
        }
    cohort_id = task_id(task)
    column = cohort_target(cohort_id).column
    parts = [
        read_training_labels(cohort_id, outer_fold, role, split_strategy=SPLIT)[
            [GENE_KEY, "role", column]
        ]
        for role in ("train", "validation")
    ]
    test = fold_gene_keys(cohort_id, outer_fold, TEST_ROLE, split_strategy=SPLIT)
    parts.append(pd.DataFrame({GENE_KEY: test.astype("object"), "role": TEST_ROLE}))
    table = pd.concat(parts, ignore_index=True).rename(columns={column: "label"})
    table.insert(1, "gene_symbol", table[GENE_KEY].map(symbols()))
    directory = WORK_DIRECTORY if not output_dir else Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory.resolve() / f"{cohort_id}_fold{outer_fold}.csv"
    table.to_csv(path, index=False)
    return {
        "task": cohort_id,
        "fold": outer_fold,
        "path": str(path),
        "columns": list(table.columns),
        "roles": _role_counts(table, binary=cohort_target(cohort_id).kind is TargetKind.BINARY),
    }


@server.tool()
@_reported
def leaderboard(
    task: Annotated[str, Field(description="task identifier, for example deg or mouse")],
) -> dict:
    """Show the published leaderboard of one task.

    Returns the 15 predictors ranked by the primary metric of the task (AUPRC, or the Spearman
    correlation on GeneEffect) as point and half_width of the 95% interval, the combination of
    the top five predictors, and the paired tests between top predictors reported in the
    manuscript, Holm-adjusted within each family of tests.
    """
    cohort_id = task_id(task)
    if cohort_id not in LEADERBOARD_COHORT_IDS:
        raise ValueError(
            f"{TASKS[cohort_id].name} is not on the leaderboard; the leaderboard tasks are "
            f"{', '.join(LEADERBOARD_COHORT_IDS)}"
        )
    table = leaderboard_table()
    rows = table.loc[table["cohort_id"].eq(cohort_id)].sort_values("rank", kind="stable")
    ranking = [
        {
            "rank": int(row["rank"]),
            "predictor": row["predictor"],
            "name": row["display_name"],
            "point": number(row["point"]),
            "half_width": number(row["half_width"]),
        }
        for _, row in rows.iterrows()
        if row["predictor"] != FUSION_ID
    ]
    fusion = rows.loc[rows["predictor"].eq(FUSION_ID)]
    tests = significance_table()
    tests = tests.loc[tests["cohort_id"].eq(cohort_id)]
    return {
        "task": cohort_id,
        "metric": primary_metric(cohort_id),
        "scope": scope_note(cohort_id),
        "ranking": ranking,
        "top_five_combination": None
        if fusion.empty
        else {
            "point": number(fusion["point"].iat[0]),
            "half_width": number(fusion["half_width"].iat[0]),
        },
        "paired_tests": [
            {
                "family": FAMILY_NOTES.get(row["family"], row["family"]),
                "a": row["subject_a"],
                "b": row["subject_b"],
                "difference_a_minus_b": number(row["difference"]),
                "half_width": number(row["half_width"]),
                "holm_p": p_value(row["holm_p"]),
                "significant": bool(row["holds"]),
            }
            for _, row in tests.iterrows()
        ],
    }


@server.tool()
@_reported
def gene_profile(
    gene: Annotated[
        str, Field(description="HGNC symbol such as POLR2A, or HGNC identifier such as HGNC:9187")
    ],
) -> dict:
    """Look up one gene: its labels, protein family, folds and the predictions it received.

    For each task the gene belongs to, returns its label (1 essential, 0 nonessential; on
    GeneEffect the mean gene effect, more negative meaning a stronger dependency), the fold in
    which it is a test gene, and for each of the 15 published predictors the predicted score
    with its percentile among the test genes of that fold (100 means ranked first). Also
    returns the gene's protein
    family, used to keep family members in the same fold, and how many predictors rank the
    gene in the top 10% of each task.
    """
    key = resolve_gene(gene)
    families = pd.read_parquet(ROOT / "datasets" / "homology" / "cluster_map.parquet")
    family = families.loc[families[GENE_KEY].eq(key)]
    members = []
    if not family.empty:
        cluster_id = family["cluster_id"].iat[0]
        others = families.loc[families["cluster_id"].eq(cluster_id) & families[GENE_KEY].ne(key)]
        members = [named(member) for member in sorted(others[GENE_KEY])]
    names = display_names()
    tasks = {}
    for cohort_id in COHORT_IDS:
        outer_fold = test_folds(cohort_id).get(key)
        if outer_fold is None:
            tasks[cohort_id] = {"in_task": False}
            continue
        target = cohort_target(cohort_id)
        fold_truth = truth(cohort_id, outer_fold)
        label = fold_truth.loc[fold_truth[GENE_KEY].eq(key), target.column].iat[0]
        source = prediction_cohort(cohort_id)
        scores = published_predictions(source).set_index(GENE_KEY).loc[key]
        percentiles = _percentiles(source).loc[key]
        predictions = {
            names[method_id]: {
                "score": number(scores[method_id]),
                "percentile": number(percentiles[method_id], 1),
            }
            for method_id in METHOD_IDS
        }
        entry = (
            {"mean_gene_effect": number(label)}
            if target.kind is TargetKind.CONTINUOUS
            else {"label": int(label)}
        )
        tasks[cohort_id] = {
            "in_task": True,
            **entry,
            "test_fold": int(outer_fold),
            "predictors_in_top_10_percent": int(
                sum(value["percentile"] >= TOP_PERCENTILE for value in predictions.values())
            ),
            "predictions": predictions,
        }
    return {
        "gene_key": key,
        "symbol": symbols().get(key),
        "protein_family": {
            "size": int(family["cluster_size"].iat[0]) if not family.empty else None,
            "other_members": members[:20],
        },
        "tasks": tasks,
    }


@server.tool()
@_reported
def check_predictions(
    path: Annotated[str, Field(description="CSV, TSV or parquet file of predictions")],
    task: Annotated[str, Field(description="task identifier, for example deg or mouse")],
) -> dict:
    """Check that a table of predictions can be scored on one task, without scoring it.

    The table has one row per gene and the columns gene_key (HGNC identifier such as
    HGNC:9187), y_score (higher means more likely essential, or a stronger dependency on
    GeneEffect) and outer_fold (0 to 4, the fold in which the gene is a test gene). Each fold
    must hold a prediction for every test gene of that fold and for no other gene; get_fold
    lists them. A task made by make_task is checked against its own folds. Returns ok, the rows
    per fold and every problem found, with example genes.
    """
    if _custom(task):
        return labels.check_task(path, task)
    cohort_id = task_id(task)
    file, frame = read_table(path)
    problems, per_fold = submission_problems(frame, cohort_id)
    return {
        "task": cohort_id,
        "path": str(file),
        "ok": not problems,
        "rows": int(len(frame)),
        "folds": per_fold,
        "problems": problems,
    }


@server.tool()
@_reported
def evaluate(
    path: Annotated[str, Field(description="CSV, TSV or parquet file of predictions")],
    task: Annotated[str, Field(description="task identifier, for example deg or mouse")],
) -> dict:
    """Score a table of predictions on one task the way the published leaderboard is scored.

    The table has the columns gene_key, y_score and outer_fold (see check_predictions). Each
    metric is the mean over the five folds, reported as point and half_width of the 95%
    gene-level bootstrap interval within folds. Returns AUPRC and AUROC (the Spearman
    correlation on GeneEffect), the value of each fold, and the rank the table would take among
    the 15 published predictors on the primary metric. A task made by make_task is scored on
    its own labels, without the rank. A table with problems is not scored; the problems are
    returned instead.
    """
    if _custom(task):
        return labels.evaluate_task(path, task)
    cohort_id = task_id(task)
    file, frame = read_table(path)
    problems, _ = submission_problems(frame, cohort_id)
    if problems:
        return {"task": cohort_id, "path": str(file), "ok": False, "problems": problems}
    folds = submission_folds(frame, cohort_id, file)
    scored = score_frames(folds, method_id="submission", cohort_id=cohort_id, split_strategy=SPLIT)
    target = cohort_target(cohort_id)
    functions = metric_table(cohort_id)
    scoped = [
        scope_frame(fold, cohort_id=cohort_id, scope=primary_scope(cohort_id)) for fold in folds
    ]
    primary = primary_metric(cohort_id)
    point = scored.values[primary]
    result: dict[str, object] = {
        "task": cohort_id,
        "path": str(file),
        "ok": True,
        "scope": scope_note(cohort_id),
        "genes": int(sum(len(fold) for fold in scoped)),
        "metrics": {
            name: {
                "point": number(scored.values[name]),
                "half_width": number(scored.intervals[name].half_width),
            }
            for name in functions
        },
        "per_fold": {
            name: [
                number(function(fold[target.column].to_numpy(), fold[SCORE].to_numpy()))
                for fold in scoped
            ]
            for name, function in functions.items()
        },
    }
    if cohort_id in LEADERBOARD_COHORT_IDS:
        ranked = ranked_points(cohort_id)
        above = ranked.loc[ranked["point"] > point]
        result["rank_among_published"] = {
            "metric": primary,
            "rank": int(len(above)) + 1,
            "out_of": len(METHOD_IDS) + 1,
            "predictors_above": [
                f"{row['display_name']} {number(row['point'])}" for _, row in above.iterrows()
            ],
        }
    return result


@server.tool()
@_reported
def compare(
    path: Annotated[str, Field(description="CSV, TSV or parquet file of predictions")],
    task: Annotated[str, Field(description="task identifier, for example deg or mouse")],
    against: Annotated[
        str,
        Field(
            description="comma-separated predictor names such as HELP,Mahi; empty for all 15"
        ),
    ] = "",
) -> dict:
    """Compare a table of predictions with published predictors by a paired bootstrap.

    Uses the primary metric of the task (AUPRC, or the Spearman correlation on GeneEffect) on
    the same genes for both, resampling genes within folds. For each predictor returns the
    difference (table minus predictor) with the half_width of its 95% interval, the p value,
    the Holm-adjusted p value over the predictors compared in this call, and a verdict at the
    5% level. A table with problems is not compared; the problems are returned instead.
    """
    if _custom(task):
        raise ValueError(
            f"{task} is a task made by make_task, and the published predictors have no predictions "
            f"on its folds; score the table with evaluate, or score the published predictors on "
            f"these labels with score_labels"
        )
    cohort_id = task_id(task)
    file, frame = read_table(path)
    problems, _ = submission_problems(frame, cohort_id)
    if problems:
        return {"task": cohort_id, "path": str(file), "ok": False, "problems": problems}
    chosen = (
        list(METHOD_IDS)
        if not against.strip()
        else [predictor_id(name) for name in against.split(",") if name.strip()]
    )
    target = cohort_target(cohort_id)
    primary = primary_metric(cohort_id)
    submission = pooled(submission_folds(frame, cohort_id, file), cohort_id)
    keys = submission[GENE_KEY].to_numpy("object")
    y_true = submission[target.column].to_numpy("float64")
    fold = submission[FOLD].to_numpy("int64")
    names = display_names()
    jobs = []
    for method_id in chosen:
        other = pooled(predictor_folds(method_id, cohort_id), cohort_id)
        if not np.array_equal(keys, other[GENE_KEY].to_numpy("object")):
            raise ValueError(f"{method_id} is scored on other genes of {cohort_id}")
        scores = (submission[SCORE].to_numpy("float64"), other[SCORE].to_numpy("float64"))
        jobs.append((cohort_id, y_true, *scores, fold))
    if len(jobs) == 1:
        outcomes = [_paired(jobs[0])]
    else:
        # The bootstrap is a Python loop, so the comparisons run in separate processes.
        workers = min(len(jobs), os.cpu_count() or 1)
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            outcomes = list(pool.map(_paired, jobs))
    rows = list(zip(chosen, outcomes))
    adjusted = holm_adjusted(np.array([item.p_value for _, item in rows], dtype="float64"))
    results = []
    for (method_id, item), holm in zip(rows, adjusted):
        if math.isfinite(holm) and holm <= ALPHA:
            verdict = "table better" if item.difference > 0 else "predictor better"
        else:
            verdict = "not distinguishable"
        results.append(
            {
                "predictor": names[method_id],
                "difference": number(item.difference),
                "half_width": number(item.half_width),
                "p": p_value(item.p_value),
                "holm_p": p_value(holm),
                "verdict": verdict,
            }
        )
    results.sort(key=lambda entry: -(entry["difference"] or 0.0))
    return {
        "task": cohort_id,
        "path": str(file),
        "ok": True,
        "metric": primary,
        "scope": scope_note(cohort_id),
        "genes": int(len(keys)),
        "comparisons": results,
    }


# Layer 0 tools on labels of the user's own (labels.py).
for _function in (labels.make_task, labels.score_labels):
    server.tool()(_reported(_function))


def _megabytes(entry: dict) -> float | None:
    return None if entry["bytes"] is None else number(entry["bytes"] / 1e6, 1)


@dataclass
class _Download:
    item_id: str
    thread: threading.Thread
    started: bool = False
    done: int = 0
    result: dict | None = None
    error: str | None = None


_DOWNLOADS: dict[str, _Download] = {}
_DOWNLOAD_LOCK = threading.Lock()
_DOWNLOAD_SLOTS = threading.BoundedSemaphore(external.DOWNLOAD_WORKERS)


def _start_download(item_id: str) -> _Download:
    """Fetch one item in a thread of the server, or return the fetch of it already running."""
    with _DOWNLOAD_LOCK:
        running = _DOWNLOADS.get(item_id)
        if running is not None and running.thread.is_alive():
            return running

        def work() -> None:
            with _DOWNLOAD_SLOTS:
                state.started = True
                try:
                    state.result = external.fetch(item_id, progress=lambda done: setattr(state, "done", done))
                except Exception as error:
                    state.error = f"{type(error).__name__}: {error}"

        state = _Download(item_id, threading.Thread(target=work, daemon=True))
        _DOWNLOADS[item_id] = state
        state.thread.start()
        return state


def _requested_downloads(items: str) -> list[str]:
    names = [name.strip() for name in items.split(",") if name.strip()]
    wanted: list[str] = []
    for name in names:
        if name == LAYER_1:
            inputs = tuple(item for needed in TOOL_INPUTS.values() for item in needed)
        elif name in TOOL_INPUTS:
            inputs = TOOL_INPUTS[name]
        elif external.item(name)["kind"] == "build":
            inputs = (name,)
        else:
            wanted.append(name)
            continue
        wanted += external.plan(*inputs)[0]
    return list(dict.fromkeys(wanted))


@server.tool()
@_reported
def fetch_external(
    items: Annotated[
        str,
        Field(
            description=(
                "comma-separated item ids from layer_status, a layer-1 tool name such as "
                "core_model or annotate_genes:dependency to fetch everything it still needs, or "
                "layer1 for all of them; empty to list every item"
            )
        ),
    ] = "",
) -> dict:
    """Download the public data that the layer-1 tools need.

    Files go to the data root (EGBENCH_DATA_ROOT, or data/ in the repository). A few sites admit
    only browsers: for those the result says which page to download the file from and where to save
    it, which the user must do, and a second call confirms it is in place. Downloads run in a thread
    of the server, eight at a time, and stop if the client stops the server: a call waits up to 45
    seconds and then reports the progress; call again with the same items until every item is done.
    The tables derived from the downloads are built by the tool that needs them on its first call.
    """
    if not items.strip():
        return {"data_root": str(external.DATA_ROOT), "items": external.status()}
    wanted = _requested_downloads(items)
    if not wanted:
        return {
            "ok": True,
            "data_root": str(external.DATA_ROOT),
            "message": "every download these items need is already under the data root",
        }
    states = [_start_download(item_id) for item_id in wanted]
    deadline = time.monotonic() + DOWNLOAD_WAIT_SECONDS
    for state in states:
        state.thread.join(max(0.0, deadline - time.monotonic()))
    rows = []
    for state in states:
        entry = external.item(state.item_id)
        if state.error is not None:
            rows.append({"id": state.item_id, "ok": False, "status": "failed", "message": state.error})
        elif state.result is not None:
            rows.append(state.result)
        else:
            rows.append(
                {
                    "id": state.item_id,
                    "status": "downloading" if state.started else "waiting for a connection",
                    "megabytes_done": number(state.done / 1e6, 1),
                    "megabytes": _megabytes(entry),
                }
            )
    result = {"data_root": str(external.DATA_ROOT), "items": rows}
    if any(state.result is None and state.error is None for state in states):
        result["next"] = f"call fetch_external again with items={items} to see the progress"
    return result


@server.tool()
@_reported
def layer_status() -> dict:
    """Show which tools can run now and what the others still need.

    Layer 0 needs only the cloned repository: list_tasks, get_fold, leaderboard, gene_profile,
    check_predictions, evaluate, compare, make_task and score_labels. Layer 1 needs public
    downloads and runs on CPU: annotate_genes, enrichment, evaluate_by_strata and core_model;
    annotate_genes:dependency and annotate_genes:mouse stand for the aspects of annotate_genes
    that read DepMap and IMPC data. Returns the layer-1 downloads still missing with their size
    in megabytes (by_hand marks files a browser must download) and, for each layer-1 tool,
    whether it is ready, the ids and the total size in megabytes of the downloads it still
    needs, and the tables its first call will build with the minutes they took on a 96-core
    server. fetch_external downloads what is missing.
    """
    absent = [name for name in LAYER_0_FILES if not (ROOT / name).exists()]
    tools, missing = {}, []
    for tool, inputs in TOOL_INPUTS.items():
        downloads, builds = external.plan(*inputs)
        missing += [item_id for item_id in downloads if item_id not in missing]
        tools[tool] = {"ready": not downloads}
        if downloads:
            tools[tool]["needs_downloads"] = downloads
            needed = sum(external.item(item_id)["bytes"] or 0 for item_id in downloads)
            tools[tool]["download_megabytes"] = number(needed / 1e6, 1)
        if builds:
            tools[tool]["built_on_first_call"] = [
                {"id": item_id, "minutes": external.item(item_id)["minutes"]} for item_id in builds
            ]
    entries = [external.item(item_id) for item_id in missing]
    return {
        "data_root": str(external.DATA_ROOT),
        "layer_0": {
            "needs": "no downloads",
            "ready": not absent,
            **({"missing_files": absent} if absent else {}),
        },
        "layer_1": {
            "needs": "public downloads, CPU",
            "missing_downloads": [
                {
                    "id": entry["id"],
                    "title": entry["title"],
                    "megabytes": _megabytes(entry),
                    "by_hand": entry.get("url") is None,
                }
                for entry in entries
            ],
            "download_megabytes": number(sum(entry["bytes"] or 0 for entry in entries) / 1e6, 1),
            "tools": tools,
        },
    }


# Layer 1 tools (annotation.py, core_model.py).
for _function in (
    annotation.annotate_genes,
    annotation.enrichment,
    annotation.evaluate_by_strata,
    core_model.core_model,
):
    server.tool()(_reported(_function))


def _method_card(method_id: str):
    def read() -> str:
        return (ROOT / "methods" / method_id / "method.yaml").read_text()

    return read


for _method_id in METHOD_IDS:
    server.resource(
        f"egbench://methods/{_method_id}",
        name=f"method card of {_method_id}",
        description=(
            "publication, implementation, inputs (feature_contract), determinism and "
            f"parameter count of the method {_method_id}"
        ),
        mime_type="text/yaml",
    )(_method_card(_method_id))


@server.resource(
    "egbench://readme",
    name="README",
    description="layout of the repository, environments, external data and run commands",
    mime_type="text/markdown",
)
def readme() -> str:
    return (ROOT / "README.md").read_text()


@server.resource(
    "egbench://agent-readme",
    name="agent README",
    description="layers of the tools, their environment, the external data they download and their checks",
    mime_type="text/markdown",
)
def agent_readme() -> str:
    return (ROOT / "agent" / "README.md").read_text()


@server.prompt()
def evaluate_new_predictor(task: str, predictions_path: str) -> str:
    """Evaluate a new predictor on one task and place it among the published predictors."""
    return (
        f"I have a new gene essentiality predictor. Its test predictions for the EGBench task "
        f"'{task}' are in {predictions_path}. Check that the table can be scored and report "
        f"any problem with example genes. If it can be scored, evaluate it, report every metric "
        f"as point ± half-width with the rank it would take on the leaderboard, then compare it "
        f"with all 15 published predictors and say which ones it beats, which beat it and "
        f"which are not distinguishable after Holm adjustment."
    )


@server.prompt()
def explain_gene(gene: str) -> str:
    """Explain how the benchmark labels one gene and how the published predictors rank it."""
    return (
        f"Explain what EGBench knows about the gene {gene}: its label in each task, its protein "
        f"family, and how the 15 published predictors rank it. Point out tasks where the "
        f"predictors disagree with the label, and use the method cards to say what the "
        f"predictors that rank it highest have in common."
    )


if __name__ == "__main__":
    server.run()
