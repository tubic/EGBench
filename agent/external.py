"""External data of the agent tools: the acquisition manifest, the data root, downloads and derived tables.

``agent/external.json`` lists every file the tools read that the repository does not hold. Its
``downloads`` are public files with their address, size and licence; a file whose site
admits only browsers has the page to download it from instead of an address. Its ``builds`` are the
tables the agent derives from them, each with its inputs, the function that writes it, and the size
and minutes it took on the 96-core server. A build may write a directory instead of a file, and a build
whose ``environment`` names one of ``agent/environments.json`` runs in that environment's interpreter. Files live under the data root, ``EGBENCH_DATA_ROOT`` or
``<repository>/data`` when it is unset: downloads at the paths of the External data table of the
README, so the methods and the analyses find them as well, derived tables under ``<data root>/agent/``.

A tool asks for its inputs with ``ensure``. Derived tables whose inputs are on disk are built on the
spot; when a download is missing, ``ensure`` raises ``Missing``, whose ``report`` names what is
missing, its size and licence, the tables that will be built after it and how long they take, and the
call to make next. ``fetch`` downloads one item.

    python agent/external.py status
    python agent/external.py fetch hgnc corum
"""
from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import lru_cache
from pathlib import Path

AGENT = Path(__file__).resolve().parent
ROOT = AGENT.parent
MANIFEST = AGENT / "external.json"
DATA_ROOT = Path(os.environ.get("EGBENCH_DATA_ROOT") or ROOT / "data").expanduser().resolve()
KINDS = {"downloads": "download", "builds": "build"}
CHUNK = 1 << 20
TIMEOUT_SECONDS = 120
PROGRESS_SECONDS = 10.0
# Single connections to EBI and Zenodo run at about 0.5 MB/s from the server; eight run at about 5 MB/s.
DOWNLOAD_WORKERS = 8
PART_SUFFIX = ".part"
LOCK_SUFFIX = ".lock"
USER_AGENT = "egbench-agent"
HUGGING_FACE = "https://huggingface.co"
MEGABYTE = 1e6


@lru_cache(maxsize=1)
def manifest() -> dict[str, dict]:
    table = json.loads(MANIFEST.read_text())
    items = {}
    for section, kind in KINDS.items():
        for entry in table[section]:
            if entry["id"] in items:
                raise ValueError(f"{MANIFEST}: the id {entry['id']!r} occurs twice")
            items[entry["id"]] = {**entry, "kind": kind}
    return items


def item(item_id: str) -> dict:
    try:
        return manifest()[item_id]
    except KeyError:
        raise ValueError(f"unknown item {item_id!r}; the items are {', '.join(manifest())}") from None


def path(item_id: str) -> Path:
    return DATA_ROOT / item(item_id)["path"]


def _megabytes(size: int | None) -> float | None:
    return None if size is None else round(size / MEGABYTE, 1)


def _listing(entry: dict) -> dict:
    listing = {
        "id": entry["id"],
        "title": entry["title"],
        "megabytes": _megabytes(entry["bytes"]),
        "licence": entry["licence"],
    }
    if entry.get("url") is None:
        listing["download_by_hand_from"] = entry["page"]
    return listing


class Missing(Exception):
    """Downloads a tool needs that are not under the data root, and the builds that follow them."""

    def __init__(self, downloads: list[str], builds: list[str]) -> None:
        self.downloads = downloads
        self.builds = builds
        super().__init__(f"not downloaded yet: {', '.join(downloads)}")

    def report(self) -> dict:
        items = manifest()
        total = sum(items[item_id]["bytes"] or 0 for item_id in self.downloads)
        return {
            "ok": False,
            "missing": [_listing(items[item_id]) for item_id in self.downloads],
            "download_megabytes": _megabytes(total),
            "built_after_download": [
                {"id": item_id, "title": items[item_id]["title"], "minutes": items[item_id]["minutes"]}
                for item_id in self.builds
            ],
            "hardware": "CPU",
            "data_root": str(DATA_ROOT),
            "next": f"call fetch_external with items={','.join(self.downloads)}",
        }


def _plan(item_id: str, downloads: list[str], builds: list[str]) -> None:
    if path(item_id).exists():
        return
    entry = item(item_id)
    if entry["kind"] == "download":
        if item_id not in downloads:
            downloads.append(item_id)
        return
    for source in entry["inputs"]:
        _plan(source, downloads, builds)
    if item_id not in builds:
        builds.append(item_id)


def plan(*item_ids: str) -> tuple[list[str], list[str]]:
    """The downloads still missing for the items, and the builds still to run, in order."""
    downloads: list[str] = []
    builds: list[str] = []
    for item_id in item_ids:
        _plan(item_id, downloads, builds)
    return downloads, builds


def _builder(entry: dict) -> Callable[[Path], None]:
    module_name, function_name = entry["builder"].split(":")
    if str(AGENT) not in sys.path:
        sys.path.insert(0, str(AGENT))
    return getattr(importlib.import_module(module_name), function_name)


def build(item_id: str) -> Path:
    """Write a derived table from its inputs, which must be on disk; returns its path."""
    entry = item(item_id)
    if entry["kind"] != "build":
        raise ValueError(f"{item_id} is downloaded, not built")
    if entry.get("environment"):
        import environments

        command = [str(environments.python(entry["environment"])), str(Path(__file__)), "build", item_id]
        # The server speaks MCP on standard output, so the builder's output goes to standard error.
        subprocess.run(command, check=True, stdout=sys.stderr, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        return path(item_id)
    return _write(entry)


def _write(entry: dict) -> Path:
    target = DATA_ROOT / entry["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + PART_SUFFIX)
    if temporary.is_dir():
        shutil.rmtree(temporary)
    _builder(entry)(temporary)
    os.replace(temporary, target)
    return target


def ensure(*item_ids: str) -> list[Path]:
    """Paths of the items, building the derived tables that are not on disk yet.

    Raises Missing, before anything is built, when a download any of them needs is absent.
    """
    downloads, builds = plan(*item_ids)
    if downloads:
        raise Missing(downloads, builds)
    for build_id in builds:
        # Jobs that build the inputs of several methods at once may need the same item.
        lock = path(build_id).with_name(path(build_id).name + LOCK_SUFFIX)
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            if not path(build_id).exists():
                build(build_id)
    return [path(item_id) for item_id in item_ids]


def _context(entry: dict) -> ssl.SSLContext | None:
    """The default TLS context, completed by the intermediate certificate a site fails to send.

    Browsers fetch such a certificate from the address in the site's certificate; it must
    still chain to a root of the system store, so the site is verified as usual.
    """
    if not entry.get("intermediate_certificate"):
        return None
    with urllib.request.urlopen(entry["intermediate_certificate"], timeout=TIMEOUT_SECONDS) as response:
        certificate = response.read()
    context = ssl.create_default_context()
    pem = certificate.lstrip().startswith(b"-----BEGIN")
    context.load_verify_locations(cadata=certificate.decode() if pem else certificate)
    return context


def _address(url: str) -> str:
    """The address to fetch, on the mirror named by HF_ENDPOINT for files of the Hugging Face Hub."""
    mirror = os.environ.get("HF_ENDPOINT")
    if mirror and url.startswith(HUGGING_FACE):
        return mirror.rstrip("/") + url[len(HUGGING_FACE):]
    return url


def _download(entry: dict, target: Path, progress: Callable[[int], None] | None) -> None:
    data = None
    if entry.get("query"):
        data = urllib.parse.urlencode({"query": entry["query"]}).encode()
    request = urllib.request.Request(_address(entry["url"]), data=data, headers={"User-Agent": USER_AGENT})
    size = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        opened = urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS, context=_context(entry))
    except urllib.error.HTTPError:
        raise
    except OSError as error:
        if entry["url"].startswith(HUGGING_FACE) and not os.environ.get("HF_ENDPOINT"):
            raise OSError(f"{error}; where huggingface.co cannot be reached, the user sets HF_ENDPOINT to a mirror "
                          f"of the Hugging Face Hub, such as https://hf-mirror.com, in the environment of the MCP "
                          f"server and restarts it") from error
        raise
    with opened as response, target.open("wb") as handle:
        while chunk := response.read(CHUNK):
            handle.write(chunk)
            size += len(chunk)
            if progress is not None:
                progress(size)


def fetch(item_id: str, *, progress: Callable[[int], None] | None = None) -> dict:
    """Download one item to the data root."""
    entry = item(item_id)
    if entry["kind"] != "download":
        raise ValueError(
            f"{item_id} is built from {', '.join(entry['inputs'])}; fetch those, the tool that "
            f"needs {item_id} builds it"
        )
    target = path(item_id)
    result = {"id": item_id, "title": entry["title"], "path": str(target), "megabytes": _megabytes(entry["bytes"])}
    if target.is_file():
        return {**result, "ok": True, "status": "already present"}
    if entry.get("url") is None:
        return {
            **result,
            "ok": False,
            "status": "download by hand",
            "message": (
                f"{entry['title']} cannot be downloaded by a program: open {entry['page']} in a browser, "
                f"download the file and save it as {target}. Then fetch {item_id} again, with fetch_external or "
                f"python agent/external.py fetch, to confirm it is in place."
            ),
        }
    temporary = target.with_name(target.name + PART_SUFFIX)
    _download(entry, temporary, progress)
    os.replace(temporary, target)
    return {**result, "ok": True, "status": "downloaded"}


def status() -> list[dict]:
    rows = []
    for item_id, entry in manifest().items():
        row = {"id": item_id, "kind": entry["kind"], "title": entry["title"], "present": path(item_id).exists()}
        row["megabytes"] = _megabytes(entry["bytes"])
        if entry["kind"] == "download":
            row["licence"] = entry["licence"]
        else:
            row["inputs"] = entry["inputs"]
            row["minutes"] = entry["minutes"]
        rows.append(row)
    return rows


def _attempt(item_id: str, progress: Callable[[int], None]) -> dict:
    try:
        return fetch(item_id, progress=progress)
    except Exception as error:
        return {"id": item_id, "ok": False, "status": "failed", "message": f"{type(error).__name__}: {error}"}


def fetch_all(item_ids: list[str]) -> list[dict]:
    """Fetch items on DOWNLOAD_WORKERS connections at once, printing the progress over all of them."""
    expected = sum(item(item_id)["bytes"] or 0 for item_id in item_ids)
    done: dict[str, int] = {}
    results: dict[str, dict] = {}
    started = last = time.monotonic()
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {
            pool.submit(_attempt, item_id, lambda size, key=item_id: done.__setitem__(key, size)): item_id
            for item_id in item_ids
        }
        pending = set(futures)
        while pending:
            finished, pending = wait(pending, timeout=PROGRESS_SECONDS, return_when=FIRST_COMPLETED)
            for future in finished:
                result = results[futures[future]] = future.result()
                print(f"{len(results)}/{len(item_ids)} {result['id']}: {result['status']}", flush=True)
                if not result["ok"]:
                    print(result["message"], flush=True)
            now = time.monotonic()
            if pending and expected and now - last >= PROGRESS_SECONDS:
                last = now
                share = 100 * sum(done.values()) / expected
                print(f"{share:.0f}% of {_megabytes(expected)} MB, {now - started:.0f} s", flush=True)
    return [results[item_id] for item_id in item_ids]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show or fetch the external data of the agent tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="list every item, whether it is under the data root and its size")
    fetching = commands.add_parser("fetch", help="download items")
    fetching.add_argument("items", nargs="+", help="item ids from agent/external.json")
    building = commands.add_parser("build", help="write derived items in this interpreter, their inputs first")
    building.add_argument("items", nargs="+", help="build ids from agent/external.json")
    arguments = parser.parse_args(argv)
    if arguments.command == "build":
        for item_id in arguments.items:
            entry = item(item_id)
            ensure(*entry["inputs"])
            _write(entry)
        return 0
    if arguments.command == "status":
        print(f"data root {DATA_ROOT}")
        for row in status():
            mark = "present" if row["present"] else "absent "
            size = "?" if row["megabytes"] is None else row["megabytes"]
            print(f"{mark} {row['kind']:8} {row['id']:28} {size:>8} MB  {row['title']}")
        return 0
    results = fetch_all(arguments.items)
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
