"""Checks that the MCP tools reproduce the published leaderboard and paired tests.

Each of the 15 predictors is written out as a submission table (gene_key, y_score,
outer_fold) from ``agent/predictions/`` and scored by the ``evaluate`` tool of
``agent/server.py`` on the six leaderboard tasks; the point and half-width of the primary
metric must equal those of ``analysis/benchmark_report/results/primary_metric.csv``. Each
leaderboard paired test of ``significance.csv`` is recomputed with the ``compare`` tool, the
first predictor as the submission and the second as the opponent; the difference, half-width
and p value before adjustment must equal the published ones. The tools report values to four
decimals and p values to three significant digits, so the published values are rounded the
same way before they are compared. The comparison table is written
to ``agent/results/reproduction.csv`` and the largest absolute differences are printed.

Run from the repository root: ``python agent/check_reproduction.py --workers 16``.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark  # noqa: E402
import server  # noqa: E402

OUTPUT = benchmark.ROOT / "agent" / "results" / "reproduction.csv"


def _submission(method_id: str, cohort_id: str, directory: Path) -> Path:
    folds = benchmark.predictor_folds(method_id, cohort_id)
    table = pd.concat(
        [
            frame[[benchmark.GENE_KEY, benchmark.SCORE]].assign(**{benchmark.FOLD: outer_fold})
            for outer_fold, frame in enumerate(folds)
        ],
        ignore_index=True,
    )
    path = directory / f"{cohort_id}_{method_id}.parquet"
    table.to_parquet(path, index=False)
    return path


def _evaluate_job(job: tuple[str, str, str]) -> dict[str, object]:
    method_id, cohort_id, directory = job
    path = _submission(method_id, cohort_id, Path(directory))
    result = server.evaluate(str(path), cohort_id)
    metric = benchmark.primary_metric(cohort_id)
    published = benchmark.leaderboard_table()
    row = published.loc[
        published["cohort_id"].eq(cohort_id) & published["predictor"].eq(method_id)
    ].iloc[0]
    return {
        "check": "evaluate",
        "task": cohort_id,
        "subject": method_id,
        "opponent": "",
        "metric": metric,
        "published_value": float(row["point"]),
        "tool_value": result["metrics"][metric]["point"],
        "published_half_width": float(row["half_width"]),
        "tool_half_width": result["metrics"][metric]["half_width"],
        "published_p": float("nan"),
        "tool_p": float("nan"),
    }


def _compare_job(job: tuple[str, str, str, str]) -> dict[str, object]:
    subject, opponent, cohort_id, directory = job
    path = _submission(subject, cohort_id, Path(directory))
    result = server.compare(str(path), cohort_id, benchmark.display_names()[opponent])
    item = result["comparisons"][0]
    published = benchmark.significance_table()
    row = published.loc[
        published["family"].eq("leaderboard_rank")
        & published["cohort_id"].eq(cohort_id)
        & published["subject_a"].eq(subject)
        & published["subject_b"].eq(opponent)
    ].iloc[0]
    return {
        "check": "compare",
        "task": cohort_id,
        "subject": subject,
        "opponent": opponent,
        "metric": result["metric"],
        "published_value": float(row["difference"]),
        "tool_value": item["difference"],
        "published_half_width": float(row["half_width"]),
        "tool_half_width": item["half_width"],
        "published_p": float(row["p_value"]),
        "tool_p": item["p"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=1, help="how many scorings run at once")
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        jobs: list[tuple] = [
            (method_id, cohort_id, directory)
            for cohort_id in server.LEADERBOARD_COHORT_IDS
            for method_id in server.METHOD_IDS
        ]
        tests = benchmark.significance_table()
        tests = tests.loc[tests["family"].eq("leaderboard_rank")]
        jobs += [
            (row["subject_a"], row["subject_b"], row["cohort_id"], directory)
            for _, row in tests.iterrows()
        ]
        started = time.monotonic()
        rows = []
        with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
            futures = [
                pool.submit(_evaluate_job if len(job) == 3 else _compare_job, job) for job in jobs
            ]
            for done, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                print(
                    f"{done}/{len(jobs)} scorings done, {time.monotonic() - started:.0f} s",
                    flush=True,
                )
    table = pd.DataFrame(rows).sort_values(["check", "task", "subject", "opponent"])
    table["value_difference"] = (
        table["tool_value"] - table["published_value"].map(benchmark.number)
    ).abs()
    table["half_width_difference"] = (
        table["tool_half_width"] - table["published_half_width"].map(benchmark.number)
    ).abs()
    table["p_matches"] = [
        tool == benchmark.p_value(published) if published == published else True
        for tool, published in zip(table["tool_p"], table["published_p"])
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    for check, group in table.groupby("check"):
        print(
            f"{check}: {len(group)} rows, largest value difference "
            f"{group['value_difference'].max():.2e}, largest half-width difference "
            f"{group['half_width_difference'].max():.2e}, p values matching "
            f"{int(group['p_matches'].sum())} of {len(group)}"
        )
    print(f"written {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
