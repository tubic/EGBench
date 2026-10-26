"""Runs of the 15 published methods from the command line, in the environment of ``agent/requirements.txt``.

``train`` (``train_fold``) trains one fold of one method on one task as a background job: the method's
``features``, ``preflight`` and ``train`` steps (``methods/<method>/run.py``) in the method's
environment, on a GPU when the method needs one. ``check`` (``check_run``) scores a trained fold as
the leaderboard does and compares its test predictions gene by gene with the published ones in
``agent/predictions/``. ``genes`` (``predict_genes``) scores genes with the five folds of a method on a
task and averages the five scores. ``report`` (``rebuild_report``) reruns ``egbench report`` on every
trained fold and ``analysis`` (``run_analysis``) the scripts of one manuscript analysis, in the order
and with the arguments of the README; both keep a copy of the published tables before their first run
and compare the new tables with it. Each prints at once the job it started, or what is missing first;
``python agent/jobs.py status <job>`` follows the job.

    python agent/runs.py train <method> <task> <fold>
    python agent/runs.py check <method> <task> <fold>
    python agent/runs.py genes <method> <task> <genes>
    python agent/runs.py report
    python agent/runs.py analysis <analysis | all>

The jobs run this module as a script in the method's environment:

    python agent/runs.py fold <method> <cohort> <fold>
    python agent/runs.py predict <method> <cohort> <genes.parquet> <output dir>
    python agent/runs.py compare <published dir> <new dir> <record.json> [--since <epoch seconds>]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

AGENT = Path(__file__).resolve().parent
ROOT = AGENT.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
SPLIT = "family_5fold"
FOLD_COUNT = 5
STEPS = ("features", "preflight", "train")
SAVED = "agent/published"
PREDICTED = "agent/predicted"
SUMMARY = "summary.json"
BLIND = "blind_predictions.parquet"
GENE_KEY = "gene_key"
SCORE = "y_score"
README_ANALYSES = "## Analyses"
ANALYSES = AGENT / "analyses.json"
ALL = "all"
GPUS_OPTION = "--gpus"
WORKERS_OPTION = "--workers"
# Cores left idle for the jobs of other users when an analysis sets its threads.
RESERVED_CORES = 4
REPORT = "benchmark_report"
REPORT_WORKERS = 16
# Relative tolerance under which a rerun table counts as the published one: the reruns of the
# release check (STATUS 6.35) differ by at most 8.9e-9 outside the embedding probes.
TABLE_TOLERANCE = 1e-6


def _method_root(method: str) -> Path:
    return ROOT / "methods" / method


def model_directory(method: str, cohort_id: str, fold: int) -> Path:
    return _method_root(method) / "model" / cohort_id / SPLIT / f"fold_{fold}"


def _command(python: Path, *arguments: str) -> list[str]:
    return [str(python), str(Path(__file__)), *arguments]


def _environment() -> dict[str, str]:
    return {"PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'analysis'}"}


# ---------------------------------------------------------------- job side (method environments)


def _step(arguments: list[str], cwd: Path) -> None:
    started = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] {' '.join(arguments[1:])}", flush=True)
    subprocess.run(arguments, cwd=cwd, check=True)
    print(f"[{time.strftime('%H:%M:%S')}] done in {(time.time() - started) / 60:.1f} min", flush=True)


def run_fold(method: str, cohort_id: str, fold: int) -> None:
    """Features, preflight and train of one fold, in this interpreter; features already built are kept."""
    script = str(_method_root(method) / "run.py")
    options = ["--dataset", cohort_id, "--split-strategy", SPLIT, "--outer-fold", str(fold)]
    for step in STEPS:
        _step([sys.executable, script, step, *options], ROOT)


def run_predict(method: str, cohort_id: str, genes: Path, output: Path) -> None:
    """Score the genes of a gene_key parquet with each of the five folds and write their mean."""
    import pandas as pd

    script = str(_method_root(method) / "run.py")
    output.mkdir(parents=True, exist_ok=True)
    columns = {}
    for fold in range(FOLD_COUNT):
        fold_output = output / f"fold_{fold}"
        if not (fold_output / BLIND).is_file():
            _step([sys.executable, script, "predict", "--model-dir", str(model_directory(method, cohort_id, fold)),
                   "--genes", str(genes), "--output", str(fold_output)], ROOT)
        table = pd.read_parquet(fold_output / BLIND).set_index(GENE_KEY)[SCORE]
        columns[f"fold_{fold}"] = table
    frame = pd.DataFrame(columns)
    frame.insert(0, SCORE, frame.mean(axis=1))
    frame.index.name = GENE_KEY
    frame.reset_index().to_csv(output / "predictions.csv", index=False)
    print(f"wrote {output / 'predictions.csv'}: {len(frame)} genes", flush=True)


def _table(path: Path):
    import pandas as pd

    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def compare_tables(published: Path, new: Path, since: float | None = None) -> list[dict]:
    """Every table of the published directory against the table of the same name in the new one.

    The rerun writes into the directory that held the published tables, so a table last written
    before since, the start of the rerun, is the published table itself and counts as not rewritten.
    """
    import numpy as np

    rows = []
    for old_path in sorted(path for path in published.rglob("*") if path.suffix in (".csv", ".parquet")):
        name = str(old_path.relative_to(published))
        new_path = new / name
        row = {"table": name}
        if not new_path.is_file() or (since is not None and new_path.stat().st_mtime < since):
            rows.append({**row, "same": False, "note": "not written by the rerun"})
            continue
        old, current = _table(old_path), _table(new_path)
        if list(old.columns) != list(current.columns) or len(old) != len(current):
            rows.append({**row, "same": False, "note": f"shape {old.shape} published, {current.shape} rerun"})
            continue
        largest = 0.0
        same = True
        for column in old.columns:
            a, b = old[column], current[column]
            if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
                x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
                # Equal infinities, such as the open end of a bin, differ by nan.
                both = (np.isnan(x) & np.isnan(y)) | (x == y)
                with np.errstate(invalid="ignore"):
                    gap = np.where(both, 0.0, np.abs(x - y))
                finite = [np.abs(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in (x, y)]
                scale = np.maximum(1.0, np.maximum(*finite))
                if np.isnan(gap).any():
                    same = False
                else:
                    largest = max(largest, float(np.max(gap, initial=0.0)))
                    same = same and bool(np.all(gap <= TABLE_TOLERANCE * scale))
            else:
                same = same and bool((a.astype(str) == b.astype(str)).all())
        rows.append({**row, "same": same, "largest_difference": largest})
    return rows


def run_compare(published: Path, new: Path, record: Path, since: float | None = None) -> None:
    rows = compare_tables(published, new, since)
    record.write_text(json.dumps(rows, indent=1) + "\n")
    differing = [row["table"] for row in rows if not row["same"]]
    print(f"{len(rows) - len(differing)} of {len(rows)} tables agree with the published ones", flush=True)
    for table in differing:
        print(f"differs: {table}", flush=True)


# ---------------------------------------------------------------- command side (the environment of agent/requirements.txt)


def _not_ready(missing: list[dict], next_step: str) -> dict:
    return {"ok": False, "missing": missing, "next": next_step}


def _interpreter(name: str) -> tuple[Path | None, dict | None]:
    import environments

    try:
        return environments.python(name), None
    except Exception as error:
        entry = environments.status_of(name)
        return None, {"environment": name, "reason": str(error), **entry,
                      "next": f"python agent/environments.py install {name}"}


def _running(kind: str, **details) -> dict | None:
    import jobs

    for record in jobs.listing():
        if record["state"] == "running" and record.get("kind") == kind and all(
            record.get(key) == value for key, value in details.items()
        ):
            return record
    return None


def _cohort(task: str) -> tuple[str, str]:
    from benchmark import task_id
    from egbench.data import prediction_cohort

    task = task_id(task)
    return task, prediction_cohort(task)


def _method(method: str) -> str:
    from egbench.contracts import METHOD_IDS

    from benchmark import predictor_id

    key = predictor_id(method)
    if key not in METHOD_IDS:
        raise ValueError(f"unknown method {method!r}; the methods are {', '.join(METHOD_IDS)}")
    return key


def _fold(fold: int) -> int:
    if fold not in range(FOLD_COUNT):
        raise ValueError(f"fold is 0 to {FOLD_COUNT - 1}, got {fold}")
    return int(fold)


def _start(name: str, command: list[str], facts: dict, details: dict) -> dict:
    import jobs

    try:
        record = jobs.start(name, command, gpu_gb=facts["gpu_gb"], threads=facts["threads"],
                            environment=_environment(), details=details)
    except jobs.Unavailable as error:
        return _not_ready([{"gpu": str(error)}], "run again when a GPU has the memory; python agent/jobs.py list shows the jobs running")
    return {"ok": True, "job": record["id"], **details, "next": f"python agent/jobs.py status {record['id']}"}


def train_fold(method: str, task: str, fold: int) -> dict:
    """Start training one fold of a published method on a task, or report what it needs first."""
    import environments
    import inputs

    method, (task, cohort_id), fold = _method(method), _cohort(task), _fold(fold)
    details = {"kind": "train_fold", "method": method, "task": task, "cohort": cohort_id, "fold": fold}
    running = _running("train_fold", method=method, cohort=cohort_id, fold=fold)
    if running:
        return {"ok": True, "job": running["id"], **details, "note": "this fold is being trained already",
                "next": f"python agent/jobs.py status {running['id']}"}
    directory = model_directory(method, cohort_id, fold)
    if (directory / BLIND).is_file():
        return {"ok": True, **details, "trained": str(directory.relative_to(ROOT)),
                "next": f"python agent/runs.py check {method} {task} {fold}"}
    if directory.exists():
        return _not_ready([{"model_directory": str(directory), "reason": "holds an unfinished run"}],
                          "remove the directory, then run python agent/runs.py train again")
    python, missing_environment = _interpreter(environments.environment_of(method))
    missing = [missing_environment] if missing_environment else []
    absent = inputs.missing(method)
    if absent:
        missing.append(absent)
    facts = inputs.facts(method)
    if missing:
        return {**_not_ready(missing, "install the environment and build the inputs first"),
                "fold_minutes": facts["fold_minutes"], "fold_gpu_gb": facts["gpu_gb"]}
    details["expected_minutes"] = facts["fold_minutes"]
    return _start(f"train-{method}-{cohort_id}-{fold}", _command(python, "fold", method, cohort_id, str(fold)),
                  facts, details)


def check_run(method: str, task: str, fold: int) -> dict:
    """Score a trained fold as the leaderboard does and compare it gene by gene with the published fold."""
    import numpy as np
    import pandas as pd

    from benchmark import published_predictions
    from egbench.evaluate import evaluate_fold

    method, (task, cohort_id), fold = _method(method), _cohort(task), _fold(fold)
    directory = model_directory(method, cohort_id, fold)
    if not (directory / BLIND).is_file():
        return _not_ready([{"model_directory": str(directory.relative_to(ROOT)), "reason": "not trained"}],
                          f"python agent/runs.py train {method} {task} {fold}")
    record = evaluate_fold(directory)
    new = pd.read_parquet(directory / BLIND).set_index(GENE_KEY)[SCORE]
    published = published_predictions(cohort_id)
    published = published.loc[published["outer_fold"] == fold].set_index(GENE_KEY)[method]
    shared = new.index.intersection(published.index)
    gap = np.abs(new.loc[shared].to_numpy() - published.loc[shared].to_numpy())
    result = {
        "ok": True,
        "method": method,
        "task": task,
        "fold": fold,
        "evaluation": record,
        "published_comparison": {
            "genes": int(len(shared)),
            "only_in_new_run": int(len(new.index.difference(published.index))),
            "only_in_published": int(len(published.index.difference(new.index))),
            "largest_absolute_difference": float(gap.max()) if len(gap) else None,
            "mean_absolute_difference": float(gap.mean()) if len(gap) else None,
            "identical": bool(len(gap)) and float(gap.max()) == 0.0,
        },
    }
    if task != cohort_id:
        # GeneEffect has no folds of its own; the leaderboard scores the DepMap folds' predictions on it.
        result["note"] = (f"{task} is scored on the {cohort_id} folds, so the evaluation above is that of the "
                          f"{cohort_id} labels; evaluate scores the test predictions of the five folds on {task}")
    return result


def coverage(method: str, cohort_id: str) -> set[str]:
    """Genes the method can score on a task: its extractable genes in the gene universe of the task."""
    import pandas as pd

    availability = ROOT / "analysis" / "research_attention" / "data" / f"{cohort_id}_method_availability.parquet"
    if availability.is_file():
        table = pd.read_parquet(availability, columns=[GENE_KEY, f"extractable__{method}"])
        return set(table.loc[table[f"extractable__{method}"], GENE_KEY])
    # The other tasks draw their genes from the genes every method extracts on DepMap.
    candidates = AGENT / "candidates" / f"{cohort_id}_common_feature_available.parquet"
    return set(pd.read_parquet(candidates, columns=[GENE_KEY])[GENE_KEY])


def predict_genes(method: str, task: str, genes: str) -> dict:
    """Start scoring genes with the five folds of a method on a task, or report what it needs first."""
    import annotation
    import environments
    import inputs
    import pandas as pd

    from benchmark import symbols
    from egbench.data import TEST_ROLE, fold_gene_keys

    method, (task, cohort_id) = _method(method), _cohort(task)
    tokens, stem = annotation._gene_tokens(genes)
    keys, unresolved = annotation._resolve(tokens, symbols())
    covered = coverage(method, cohort_id)
    scored = [key for key in keys if key in covered]
    outside = [key for key in keys if key not in covered]
    absent = [fold for fold in range(FOLD_COUNT) if not (model_directory(method, cohort_id, fold) / BLIND).is_file()]
    if absent:
        return _not_ready([{"untrained_folds": absent}],
                          f"python agent/runs.py train {method} {task} <fold> for the folds {absent}")
    if not scored:
        return _not_ready([{"genes": "none of the genes can be scored", "outside_coverage": outside,
                            "unresolved": unresolved}], "pass genes of the task's gene universe")
    python, missing_environment = _interpreter(environments.environment_of(method))
    if missing_environment:
        return _not_ready([missing_environment], "install the environment first")
    # Every labelled gene is a test gene of exactly one fold.
    labelled = set()
    for fold in range(FOLD_COUNT):
        labelled.update(fold_gene_keys(cohort_id, fold, TEST_ROLE, split_strategy=SPLIT))
    from external import DATA_ROOT

    output = DATA_ROOT / PREDICTED / f"{time.strftime('%Y%m%d-%H%M%S')}-{method}-{cohort_id}-{stem}"
    output.mkdir(parents=True)
    pd.DataFrame({GENE_KEY: scored}).to_parquet(output / "genes.parquet", index=False)
    details = {"kind": "predict_genes", "method": method, "task": task, "cohort": cohort_id,
               "genes": len(scored), "output": str(output)}
    summary = {"outside_coverage": outside, "unresolved": unresolved,
               "labelled_in_task": sorted(set(scored) & labelled)}
    (output / SUMMARY).write_text(json.dumps(summary, indent=1) + "\n")
    result = _start(f"predict-{method}-{cohort_id}", _command(python, "predict", method, cohort_id,
                    str(output / "genes.parquet"), str(output)), inputs.facts(method), details)
    if result["ok"]:
        result["not_scored"] = {"outside_coverage": len(outside), "unresolved": len(unresolved)}
        if summary["labelled_in_task"]:
            result["note"] = (f"{len(summary['labelled_in_task'])} genes are labelled in {task}; four of the five "
                              f"folds were trained on each, so their test-fold predictions (gene_profile) are the "
                              f"unbiased scores")
    return result


def _published_copy(name: str, results: Path) -> Path:
    """The tables of an analysis as the repository holds them, copied once before the first rerun."""
    import shutil

    from external import DATA_ROOT

    copy = DATA_ROOT / SAVED / name
    if not copy.is_dir():
        shutil.copytree(results, copy)
    return copy


def analysis_commands() -> dict[str, list[list[str]]]:
    """The commands of each manuscript analysis, in order, from the Analyses section of the README."""
    text = (ROOT / "README.md").read_text()
    section = text.split(README_ANALYSES, 1)[1]
    block = section.split("```", 2)[1]
    commands: dict[str, list[list[str]]] = {}
    for line in block.splitlines():
        line = line.split("#", 1)[0].strip()
        match = re.match(r"python (analysis/([a-z_]+)/[a-z_0-9]+\.py)(.*)", line)
        if match:
            commands.setdefault(match.group(2), []).append([match.group(1), *match.group(3).split()])
    return commands


@lru_cache(maxsize=1)
def analyses() -> dict[str, dict]:
    """What each analysis needs before it runs (agent/analyses.json)."""
    return json.loads(ANALYSES.read_text())


def _untrained(models: dict[str, list[str]]) -> dict[str, dict[str, list[int]]]:
    from egbench.contracts import METHOD_IDS

    untrained: dict[str, dict[str, list[int]]] = {}
    for method, cohorts in models.items():
        for name in METHOD_IDS if method == ALL else (method,):
            for cohort_id in cohorts:
                absent = [k for k in range(FOLD_COUNT) if not (model_directory(name, cohort_id, k) / BLIND).is_file()]
                if absent:
                    untrained.setdefault(name, {})[cohort_id] = absent
    return untrained


def needs(name: str) -> list[dict]:
    """What the analysis still lacks: trained folds, method inputs, downloads, outputs of other analyses, environments."""
    import environments
    import external
    import inputs

    entry = analyses()[name]
    missing = []
    untrained = _untrained(entry.get("models", {}))
    if untrained:
        folds = sum(len(absent) for cohorts in untrained.values() for absent in cohorts.values())
        missing.append({"untrained_folds": folds, "by_method": untrained, "next": "train them with python agent/runs.py train"})
    for method in entry.get("inputs", []):
        if inputs.missing(method) is not None:
            missing.append({"inputs_of": method, "next": f"python agent/inputs.py build {method}"})
    downloads, _ = external.plan(*entry.get("external", []))
    if downloads:
        items = external.manifest()
        missing.append({"downloads": [external._listing(items[item_id]) for item_id in downloads],
                        "next": f"python agent/external.py fetch {' '.join(downloads)}"})
    for path, writer in entry.get("reads", {}).items():
        # A file that a running job is still writing is not complete yet.
        running = _running("run_analysis", analysis=writer) or _running("run_analysis", analysis=ALL)
        if running:
            missing.append({"output_of": writer, "file": path, "running": running["id"],
                            "next": f"wait for job {running['id']} to finish"})
        elif not (ROOT / path).exists():
            missing.append({"output_of": writer, "file": path, "next": f"python agent/runs.py analysis {writer}"})
    names = environments.analysis_environments(name)
    for environment in names:
        if not environments.status_of(environment).get("installed"):
            missing.append({"environment": environment, "next": f"python agent/environments.py install {environment}"})
    return missing


def _steps(name: str) -> list[list[str]]:
    """The commands of an analysis with the interpreters of their environments and the arguments filled in."""
    import environments

    steps = []
    for arguments in analysis_commands()[name]:
        script, options = arguments[0], arguments[1:]
        options = [str(environments.method_environment_root()) if option == "<env-root>" else option for option in options]
        if GPUS_OPTION in options:
            # A job sees only the card it was placed on, as device 0.
            options[options.index(GPUS_OPTION) + 1] = "0"
        python = environments.python(environments.environment_of_analysis(name, script))
        steps.append([str(python), script, *options])
    return steps


def rebuild_report() -> dict:
    """Start egbench report on the trained folds, or report what it still needs."""
    import environments
    import inputs
    import jobs

    running = _running("rebuild_report")
    if running:
        return {"ok": True, "job": running["id"], "note": "the report is being rebuilt already"}
    missing = needs(REPORT)
    if missing:
        return _not_ready(missing, "the report scores all 15 methods on the five folds of five tasks")
    inputs.link_data_root()
    python = environments.python(environments.environment_of_analysis(REPORT))
    results = ROOT / "analysis" / REPORT / "results"
    published = _published_copy(REPORT, results)
    comparison = published.parent / f"{REPORT}.json"
    steps = [[str(python), "-m", "egbench", "report", "--workers", str(REPORT_WORKERS)],
             [str(python), str(Path(__file__)), "compare", str(published), str(results), str(comparison),
              "--since", str(time.time())]]
    record = jobs.start("rebuild-report", _sequence(steps), threads=2, environment=_environment(),
                        details={"kind": "rebuild_report", "comparison": str(comparison),
                                 "expected_minutes": analyses()[REPORT]["minutes"]})
    return {"ok": True, "job": record["id"], "next": f"python agent/jobs.py status {record['id']}"}


def _sequence(steps: list[list[str]]) -> list[str]:
    """One command that runs the steps in order and stops at the first that fails."""
    return [sys.executable, "-c", "import subprocess, sys\nfor step in " + repr(steps) + ":\n"
            "    print(' '.join(step[1:]), flush=True)\n    subprocess.run(step, check=True)"]


def _threads(steps: list[list[str]]) -> int:
    """Threads per process that keep the most parallel step of a job within the idle cores."""
    import math
    import os

    import jobs

    workers = max((int(step[step.index(WORKERS_OPTION) + 1]) for step in steps if WORKERS_OPTION in step), default=1)
    idle = os.cpu_count() - math.ceil(os.getloadavg()[0]) - RESERVED_CORES
    return max(1, min(jobs.THREADS, idle // workers))


def run_analysis(name: str) -> dict:
    """Start the scripts of one manuscript analysis, or of all in the README's order, or report what they need."""
    import inputs
    import jobs

    commands = analysis_commands()
    names = list(commands) if name == ALL else [name]
    unknown = [item for item in names if item not in commands]
    if unknown:
        raise ValueError(f"unknown analysis {name!r}; the analyses are {', '.join(commands)}, or {ALL}")
    for record in jobs.listing():
        if record["state"] != "running" or record.get("kind") != "run_analysis":
            continue
        if record.get("analysis") == name:
            return {"ok": True, "job": record["id"], "note": "this analysis is running already"}
        # All the analyses and one of them write the same tables.
        if ALL in (name, record.get("analysis")):
            return _not_ready([{"running": record["id"], "analysis": record.get("analysis")}],
                              "wait for that job to end, or cancel it with python agent/jobs.py cancel <job>")
    produced = set(names)
    missing = [entry for item in names for entry in needs(item)
               if not (entry.get("output_of") in produced and names.index(entry["output_of"]) < names.index(item))]
    if missing:
        return _not_ready(missing, "provide what is listed, then run python agent/runs.py analysis again")
    inputs.link_data_root()
    steps, comparisons, started = [], [], time.time()
    for item in names:
        results = ROOT / "analysis" / item / "results"
        published = _published_copy(item, results)
        comparison = published.parent / f"{item}.json"
        steps += _steps(item) + [[sys.executable, str(Path(__file__)), "compare", str(published), str(results),
                                  str(comparison), "--since", str(started)]]
        comparisons.append(str(comparison))
    gpu = max((analyses()[item].get("gpu_gb") or 0 for item in names), default=0) or None
    try:
        record = jobs.start(f"analysis-{name}", _sequence(steps), gpu_gb=gpu, threads=_threads(steps),
                            environment=_environment(),
                            details={"kind": "run_analysis", "analysis": name, "comparison": comparisons[-1],
                                     "comparisons": comparisons,
                                     "expected_minutes": sum(analyses()[item]["minutes"] for item in names)})
    except jobs.Unavailable as error:
        return _not_ready([{"gpu": str(error)}], "run again when a GPU has the memory; python agent/jobs.py list shows the jobs running")
    return {"ok": True, "job": record["id"], "scripts": len(steps) - len(names), "next": f"python agent/jobs.py status {record['id']}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train, check and rerun the published methods and the manuscript's analyses.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, purpose in (("train", "train one fold of a method on a task as a background job"),
                          ("check", "score a trained fold and compare it with the published fold")):
        command = commands.add_parser(name, help=purpose)
        command.add_argument("method")
        command.add_argument("task")
        command.add_argument("fold", type=int)
    genes = commands.add_parser("genes", help="score genes with the five trained folds of a method on a task")
    genes.add_argument("method")
    genes.add_argument("task")
    genes.add_argument("genes", help="comma-separated symbols or HGNC identifiers, or the path of a gene table")
    commands.add_parser("report", help="rebuild the benchmark report from the trained folds and compare its tables")
    analysis = commands.add_parser("analysis", help="rerun one manuscript analysis, or all, and compare its tables")
    analysis.add_argument("name", help=f"analysis directory under analysis/, or {ALL}")
    fold = commands.add_parser("fold", help="job step: features, preflight and train of one fold")
    fold.add_argument("method")
    fold.add_argument("cohort")
    fold.add_argument("fold", type=int)
    predict = commands.add_parser("predict", help="job step: score genes with the five folds and average")
    predict.add_argument("method")
    predict.add_argument("cohort")
    predict.add_argument("genes", type=Path)
    predict.add_argument("output", type=Path)
    compare = commands.add_parser("compare", help="job step: compare rerun tables with the published ones")
    compare.add_argument("published", type=Path)
    compare.add_argument("new", type=Path)
    compare.add_argument("record", type=Path)
    compare.add_argument("--since", type=float, help="start of the rerun, in seconds since the epoch")
    arguments = parser.parse_args(argv)
    if arguments.command == "fold":
        run_fold(arguments.method, arguments.cohort, arguments.fold)
        return 0
    if arguments.command == "predict":
        run_predict(arguments.method, arguments.cohort, arguments.genes, arguments.output)
        return 0
    if arguments.command == "compare":
        run_compare(arguments.published, arguments.new, arguments.record, arguments.since)
        return 0
    if arguments.command == "train":
        result = train_fold(arguments.method, arguments.task, arguments.fold)
    elif arguments.command == "check":
        result = check_run(arguments.method, arguments.task, arguments.fold)
    elif arguments.command == "genes":
        result = predict_genes(arguments.method, arguments.task, arguments.genes)
    elif arguments.command == "report":
        result = rebuild_report()
    else:
        result = run_analysis(arguments.name)
    print(json.dumps(result, indent=1, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
