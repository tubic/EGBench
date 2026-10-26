"""Runs the agent scenarios of ``agent/scenarios.json`` through ``agent/client.py`` and tabulates them.

Each scenario is one user request, given as free text or as a prompt of the server, and a list
of facts the final answer must contain; a fact is a list of accepted spellings, matched
without regard to case. A scenario marked ``"empty_data_root": true`` runs with an empty data
root, as right after cloning the repository; the others use the data root of the environment.
A scenario marked ``"python": true`` also offers the model a function that runs Python code in
``<data root>/agent/work/python/<id>``, emptied before the run, as a coding agent has when it trains a
model of its own. Every scenario runs once, with the sampling constants of the client, and writes its full log to
``agent/results/<id>.json``, with the interpreter path written as ``python``, the data root as
``<data root>``, the environment root as ``<environment root>`` and the repository path as ``<repository>``. The table
``agent/results/scenarios.csv`` holds, per scenario, the tools called, the number of model and tool calls, the time spent in
the model and in the tools, the tokens (prompt tokens of all calls, the part of them served from
the prompt cache, the part newly computed, and completion tokens including reasoning) and how
many of the expected facts the answer contains.

``--recount`` counts the expected facts again in the existing logs of the scenarios named by ``--only``, for a list of
facts corrected after a run.

Run from the repository root with the model API key in EGBENCH_LLM_API_KEY:
``python agent/run_scenarios.py --base-url URL --model NAME``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

AGENT = Path(__file__).resolve().parent
RESULTS = AGENT / "results"


def _found(answer: str | None, expect: list[list[str]]) -> list[bool]:
    text = (answer or "").lower()
    return [any(spelling.lower() in text for spelling in fact) for fact in expect]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True, help="endpoint base URL")
    parser.add_argument("--model", required=True, help="model name sent to the endpoint")
    parser.add_argument("--only", nargs="*", default=None, help="run only these scenario ids")
    parser.add_argument("--recount", action="store_true", help="count the expected facts again in the existing logs of --only, without running")
    arguments = parser.parse_args()
    scenarios = json.loads((AGENT / "scenarios.json").read_text())
    if arguments.only:
        scenarios = [item for item in scenarios if item["id"] in arguments.only]
    RESULTS.mkdir(parents=True, exist_ok=True)
    if arguments.recount:
        table = pd.read_csv(RESULTS / "scenarios.csv")
        table["missing_facts"] = table["missing_facts"].fillna("").astype(str)
        for scenario in scenarios:
            answer = json.loads((RESULTS / f"{scenario['id']}.json").read_text())["summary"]["final_answer"]
            found = _found(answer, scenario["expect"])
            row = table["scenario"] == scenario["id"]
            table.loc[row, "facts_found"] = sum(found)
            table.loc[row, "facts_expected"] = len(found)
            table.loc[row, "missing_facts"] = "; ".join("/".join(fact) for fact, hit in zip(scenario["expect"], found) if not hit)
        table.to_csv(RESULTS / "scenarios.csv", index=False)
        print(table[["scenario", "facts_found", "facts_expected", "missing_facts"]].to_string(index=False))
        return 0
    rows = []
    started = time.monotonic()
    for number, scenario in enumerate(scenarios, start=1):
        log = RESULTS / f"{scenario['id']}.json"
        command = [
            sys.executable,
            str(AGENT / "client.py"),
            "--server-command",
            f"{sys.executable} {AGENT / 'server.py'}",
            "--base-url",
            arguments.base_url,
            "--model",
            arguments.model,
            "--log",
            str(log),
        ]
        if "prompt" in scenario:
            command += ["--prompt", scenario["prompt"]]
            for key, value in scenario["arguments"].items():
                command += ["--arg", f"{key}={value}"]
        else:
            command += ["--task", scenario["task"]]
        print(f"scenario {number}/{len(scenarios)}: {scenario['id']}", flush=True)
        with tempfile.TemporaryDirectory() as empty:
            environment = dict(os.environ)
            if scenario.get("empty_data_root"):
                environment["EGBENCH_DATA_ROOT"] = empty
            if scenario.get("python"):
                root = Path(environment.get("EGBENCH_DATA_ROOT") or AGENT.parent / "data").expanduser().resolve()
                directory = root / "agent" / "work" / "python" / scenario["id"]
                shutil.rmtree(directory, ignore_errors=True)
                command += ["--python-dir", str(directory)]
            subprocess.run(command, check=False, cwd=AGENT.parent, env=environment)
            data_root = environment.get("EGBENCH_DATA_ROOT")
            # The logs are published, so the paths of this machine become placeholders.
            text = log.read_text().replace(sys.executable, "python")
            if data_root:
                text = text.replace(str(Path(data_root).expanduser().resolve()), "<data root>")
            if environment.get("EGBENCH_ENV_ROOT"):
                text = text.replace(str(Path(environment["EGBENCH_ENV_ROOT"]).expanduser().resolve()), "<environment root>")
            text = text.replace(str(AGENT.parent), "<repository>")
        log.write_text(text)
        record = json.loads(text)
        summary = record["summary"]
        calls = [call for step in record.get("steps", []) for call in step["tool_results"]]
        found = _found(summary["final_answer"], scenario["expect"])
        rows.append(
            {
                "scenario": scenario["id"],
                "tools": " ".join(call["name"] for call in calls),
                "model_calls": summary["model_calls"],
                "tool_calls": summary["tool_calls"],
                "tool_errors": sum(bool(call["is_error"]) for call in calls),
                "total_seconds": summary["total_seconds"],
                "model_seconds": summary["model_seconds"],
                "tool_seconds": summary["tool_seconds"],
                "prompt_tokens": summary["prompt_tokens"],
                "cached_prompt_tokens": summary["cache_n"],
                "computed_prompt_tokens": summary["prompt_n"],
                "completion_tokens": summary["completion_tokens"],
                "facts_found": sum(found),
                "facts_expected": len(found),
                "missing_facts": "; ".join(
                    "/".join(fact) for fact, hit in zip(scenario["expect"], found) if not hit
                ),
                "stop_reason": summary["stop_reason"],
            }
        )
        print(
            f"scenario {number}/{len(scenarios)} done: {sum(found)}/{len(found)} facts, "
            f"{summary['total_seconds']:.0f} s, {time.monotonic() - started:.0f} s in all",
            flush=True,
        )
    table = pd.DataFrame(rows)
    output = RESULTS / "scenarios.csv"
    if arguments.only and output.exists():
        kept = pd.read_csv(output)
        table = pd.concat([kept.loc[~kept["scenario"].isin(table["scenario"])], table])
        order = {item["id"]: number for number, item in enumerate(json.loads((AGENT / "scenarios.json").read_text()))}
        table = table.sort_values("scenario", key=lambda column: column.map(order), kind="stable")
    table.to_csv(output, index=False)
    print(table.drop(columns=["tools", "missing_facts"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
