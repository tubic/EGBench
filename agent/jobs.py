"""Background jobs of ``runs.py``: commands that outlive the command that started them.

A command that trains a fold, scores genes or reruns an analysis starts a job and returns at once;
``python agent/jobs.py status <id>`` then reports its state, the tail of its log and how long it has
run, and ``cancel`` stops it. Each job is a directory ``<data root>/agent/jobs/<id>/`` holding ``job.json``
and ``log.txt``. The job runs in a session of its own (``python agent/jobs.py run <id>``), which waits
for the command and writes its exit code, so a job survives a restart of the client or of the server
and its state is read back from disk.

A job that needs a GPU names the memory it needs; it is placed on the card with the most free memory,
counting the memory of the jobs started on that card in the last minutes, which may not be allocated
yet. Every job runs with a bounded number of threads for numpy, BLAS and OpenMP and with the variables
that make CUDA reductions deterministic.

    python agent/jobs.py list
    python agent/jobs.py status <id>
    python agent/jobs.py cancel <id>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import external

JOBS = external.DATA_ROOT / "agent" / "jobs"
RECORD = "job.json"
LOG = "log.txt"
THREADS = 8
# Memory a starting job may not have allocated yet; after this long, nvidia-smi shows what it holds.
RESERVATION_SECONDS = 600
# Memory left free on a card besides the job's own, for the CUDA context and other users.
GPU_MARGIN_GB = 1.0
TAIL_LINES = 20
CANCEL_SECONDS = 20
DETERMINISM = {"PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class Unavailable(Exception):
    """No GPU has the memory a job needs."""


def _directory(job_id: str) -> Path:
    directory = JOBS / job_id
    if not (directory / RECORD).is_file():
        raise ValueError(f"no job {job_id!r}; python agent/jobs.py list lists the jobs")
    return directory


def _read(job_id: str) -> dict:
    return json.loads((_directory(job_id) / RECORD).read_text())


def _write(record: dict) -> None:
    path = JOBS / record["id"] / RECORD
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(temporary, path)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def gpus() -> list[dict]:
    """Index, free and total memory in GB of every card nvidia-smi lists; empty without a GPU."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    cards = []
    for line in output.strip().splitlines():
        index, free, total = (field.strip() for field in line.split(","))
        cards.append({"index": int(index), "free_gb": int(free) / 1024, "total_gb": int(total) / 1024})
    return cards


def _reserved(index: int) -> float:
    now = time.time()
    total = 0.0
    for record in _records():
        if record.get("gpu") == index and record["state"] == "running" and now - record["started"] < RESERVATION_SECONDS:
            total += record["gpu_gb"]
    return total


def place(gpu_gb: float) -> int:
    """The card to run a job that needs gpu_gb of GPU memory on."""
    cards = gpus()
    if not cards:
        raise Unavailable("this machine has no GPU that nvidia-smi can see")
    room = {card["index"]: card["free_gb"] - _reserved(card["index"]) - GPU_MARGIN_GB for card in cards}
    index = max(room, key=room.get)
    if room[index] < gpu_gb:
        free = ", ".join(f"GPU {i}: {value + GPU_MARGIN_GB:.1f} GB" for i, value in sorted(room.items()))
        raise Unavailable(f"the job needs {gpu_gb:g} GB of GPU memory; free now {free}")
    return index


def start(name: str, command: list[str], *, gpu_gb: float | None = None, threads: int = THREADS,
          environment: dict[str, str] | None = None, cwd: Path = external.ROOT, details: dict | None = None) -> dict:
    """Start command as a background job and return its record; gpu_gb places it on a card."""
    gpu = None if gpu_gb is None else place(gpu_gb)
    JOBS.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
    suffix = 1
    while (JOBS / job_id).exists():
        suffix += 1
        job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{name}-{suffix}"
    (JOBS / job_id).mkdir()
    variables = {**DETERMINISM, **{variable: str(threads) for variable in THREAD_VARIABLES}, **(environment or {})}
    if gpu is not None:
        variables["CUDA_VISIBLE_DEVICES"] = str(gpu)
    record = {
        "id": job_id,
        "name": name,
        "command": command,
        "cwd": str(cwd),
        "environment": variables,
        "gpu": gpu,
        "gpu_gb": gpu_gb,
        "threads": threads,
        "details": details or {},
        "state": "running",
        "started": time.time(),
        "finished": None,
        "returncode": None,
        "pid": None,
    }
    _write(record)
    with open(JOBS / job_id / LOG, "ab") as log:
        runner = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "run", job_id],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    # Reaped here while this process lives; a runner that outlives it is adopted by init. The runner
    # writes its own pid and the end of the job, so the record is not written here again.
    threading.Thread(target=runner.wait, daemon=True).start()
    return {**record, "pid": runner.pid}


def _run(job_id: str) -> int:
    """Run the command of a job in this process's session and record how it ended."""
    record = _read(job_id)
    record["pid"] = os.getpid()
    _write(record)
    child = subprocess.Popen(record["command"], cwd=record["cwd"], env={**os.environ, **record["environment"]},
                             stdin=subprocess.DEVNULL)
    returncode = child.wait()
    record = _read(job_id)
    if record["state"] == "running":
        record["state"] = "finished" if returncode == 0 else "failed"
    record["returncode"] = returncode
    record["finished"] = time.time()
    _write(record)
    return returncode


def _records() -> list[dict]:
    if not JOBS.is_dir():
        return []
    records = []
    for directory in sorted(JOBS.iterdir()):
        if (directory / RECORD).is_file():
            record = json.loads((directory / RECORD).read_text())
            if record["state"] == "running" and record["pid"] and not _alive(record["pid"]):
                record["state"] = "lost"
            records.append(record)
    return records


def _tail(job_id: str, lines: int = TAIL_LINES) -> list[str]:
    path = JOBS / job_id / LOG
    if not path.is_file():
        return []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 64 * 1024))
        text = handle.read().decode(errors="replace").replace("\r", "\n")
    return [line for line in text.splitlines() if line.strip()][-lines:]


def _summary(record: dict) -> dict:
    end = record["finished"] or time.time()
    summary = {
        "id": record["id"],
        "name": record["name"],
        "state": record["state"],
        "minutes": round((end - record["started"]) / 60, 1),
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record["started"])),
    }
    if record["gpu"] is not None:
        summary["gpu"] = record["gpu"]
    if record["returncode"] is not None:
        summary["returncode"] = record["returncode"]
    # The details a tool adds never replace the state, the name or the minutes the job has run.
    return {**record["details"], **summary}


def listing() -> list[dict]:
    """Every job, oldest first, with its state and how long it has run or ran."""
    return [_summary(record) for record in _records()]


def status(job_id: str) -> dict:
    """State of one job and the last lines of its log."""
    _directory(job_id)
    record = next(record for record in _records() if record["id"] == job_id)
    return {**_summary(record), "log": str(JOBS / job_id / LOG), "log_tail": _tail(job_id)}


def cancel(job_id: str) -> dict:
    """Stop a running job and every process it started."""
    record = _read(job_id)
    if record["state"] != "running" or not record["pid"] or not _alive(record["pid"]):
        return {**status(job_id), "cancelled": False}
    record["state"] = "cancelled"
    _write(record)
    group = os.getpgid(record["pid"])
    os.killpg(group, signal.SIGTERM)
    deadline = time.time() + CANCEL_SECONDS
    while time.time() < deadline and _alive(record["pid"]):
        time.sleep(0.5)
    if _alive(record["pid"]):
        os.killpg(group, signal.SIGKILL)
    record = _read(job_id)
    record["finished"] = record["finished"] or time.time()
    _write(record)
    return {**status(job_id), "cancelled": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List, inspect or cancel the background jobs of runs.py.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="every job with its state")
    for name in ("status", "cancel", "run"):
        commands.add_parser(name).add_argument("job")
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        return _run(arguments.job)
    if arguments.command == "list":
        result = listing()
    elif arguments.command == "status":
        result = status(arguments.job)
    else:
        result = cancel(arguments.job)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
