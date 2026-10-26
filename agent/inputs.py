"""Inputs of the 15 published methods: what each reads under ``methods/<method>/feature/source/`` and how it is made.

``agent/methods.json`` gives, for every method, the files it reads, each with the item of
``agent/external.json`` that makes it (a download, a build, or one file of a build that writes a
directory, ``item:path``) or a file of the repository (``repository:path``); the third-party code its
``acquire_assets.py`` fetches; the inputs the method reads only where they are, not through a link,
which are moved into the method directory instead (``in_place``); the GPU memory a fold needs, as nvidia-smi counts it, the threads it
runs with and the minutes a fold takes on the 96-core server. ``missing`` reports what a method still
lacks, with sizes, licences and minutes; ``build`` fetches the downloads, runs the builds, each in
its environment, and links the files into the method directory, so the data stay under the data root.

    python agent/inputs.py status
    python agent/inputs.py build <method>
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import external

AGENT = Path(__file__).resolve().parent
ROOT = AGENT.parent
MANIFEST = AGENT / "methods.json"
REPOSITORY = "repository:"
CANDIDATES = AGENT / "candidates"


@lru_cache(maxsize=1)
def methods() -> dict[str, dict]:
    return json.loads(MANIFEST.read_text())


def facts(method: str) -> dict:
    return methods()[method]


def _source(method: str) -> Path:
    return ROOT / "methods" / method / "feature" / "source"


def _item(reference: str) -> tuple[str, str]:
    item_id, _, inner = reference.partition(":")
    return item_id, inner


def _origin(reference: str) -> Path:
    """The file a reference names: under the repository, or under the data root."""
    if reference.startswith(REPOSITORY):
        return ROOT / reference[len(REPOSITORY):]
    item_id, inner = _item(reference)
    path = external.path(item_id)
    return path / inner if inner else path


def originals(item_id: str) -> list[tuple[str, tuple[str, ...]]]:
    """Each file of an item (the empty name for the item itself) and the method inputs it stands for, relative to methods/."""
    files: dict[str, list[str]] = {}
    for method, entry in methods().items():
        for target, reference in entry["inputs"].items():
            if not reference.startswith(REPOSITORY) and _item(reference)[0] == item_id:
                files.setdefault(_item(reference)[1], []).append(f"{method}/feature/source/{target}")
    return [(name, tuple(sorted(targets))) for name, targets in sorted(files.items())]


def original(item_id: str, name: str = "") -> str:
    """The first method input a file of an item stands for, relative to methods/."""
    return dict(originals(item_id))[name][0]


def built_by(module: str) -> list[str]:
    """The build items whose builder is in an agent module, in the order of the manifest."""
    return [item_id for item_id, entry in external.manifest().items() if entry.get("builder", "").startswith(f"{module}:")]


def _items(method: str) -> list[str]:
    references = facts(method)["inputs"].values()
    return sorted({_item(reference)[0] for reference in references if not reference.startswith(REPOSITORY)})


def _third_party(method: str) -> list[dict]:
    return [entry for entry in facts(method).get("third_party", []) if not (ROOT / "methods" / method / entry["creates"]).exists()]


def _linked(method: str) -> list[str]:
    """The targets not yet in place under the method's feature/source directory."""
    source = _source(method)
    return [target for target in facts(method)["inputs"] if not (source / target).exists()]


def missing(method: str) -> dict | None:
    """What the method lacks before a fold can run, or None when every input is in place."""
    if not _linked(method) and not _third_party(method):
        return None
    downloads, builds = external.plan(*_items(method))
    items = external.manifest()
    report = {
        "inputs": f"{len(_linked(method))} of {len(facts(method)['inputs'])} input files of {method} are not in place",
        "downloads": [external._listing(items[item_id]) for item_id in downloads],
        "download_megabytes": external._megabytes(sum(items[item_id]["bytes"] or 0 for item_id in downloads)),
        "builds": [
            {key: items[item_id][key] for key in ("id", "title", "minutes", "environment", "gpu_gb") if key in items[item_id]}
            for item_id in builds
        ],
        "third_party_code": [entry["source"] for entry in _third_party(method)],
        "next": f"python agent/inputs.py build {method}",
    }
    by_hand = [entry["id"] for entry in report["downloads"] if "download_by_hand_from" in entry]
    if by_hand:
        report["by_hand"] = f"download {', '.join(by_hand)} by hand first; python agent/external.py fetch names where to save them"
    return report


def gpu_gb(method: str) -> float | None:
    """GPU memory the builds the method still needs take, or None when they all run on CPU."""
    _, builds = external.plan(*_items(method))
    needs = [external.item(item_id).get("gpu_gb") for item_id in builds]
    needs = [need for need in needs if need]
    return max(needs) if needs else None


def link_data_root() -> None:
    """Make <repository>/data the data root, where the method and analysis code read external files."""
    data = ROOT / "data"
    if external.DATA_ROOT != data.resolve() and not data.exists():
        # Jobs that build the inputs of several methods start at the same moment.
        with contextlib.suppress(FileExistsError):
            data.symlink_to(external.DATA_ROOT, target_is_directory=True)


def _candidates() -> None:
    """The candidate gene tables, where the method code reads them (<repository>/data/candidates)."""
    link_data_root()
    target = ROOT / "data" / "candidates"
    target.mkdir(parents=True, exist_ok=True)
    for table in sorted(CANDIDATES.glob("*.parquet")):
        if not (target / table.name).is_file():
            temporary = target / f"{table.name}.{os.getpid()}{external.PART_SUFFIX}"
            shutil.copy2(table, temporary)
            os.replace(temporary, target / table.name)


def _place(origin: Path, target: Path, in_place: bool) -> None:
    """Link the input into the method directory, or move it there when the method refuses a link."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and not target.exists():
        target.unlink()
    if in_place:
        shutil.move(str(origin), str(target))
        origin.symlink_to(target, target_is_directory=target.is_dir())
        return
    target.symlink_to(origin)


def build(method: str) -> None:
    """Fetch, build and link every input of the method; runs in a job, in the server's environment."""
    _candidates()
    downloads, _ = external.plan(*_items(method))
    if downloads:
        fetched = external.fetch_all(downloads)
        failed = [result for result in fetched if not result["ok"]]
        if failed:
            raise SystemExit(json.dumps(failed, indent=1))
    external.ensure(*_items(method))
    for entry in _third_party(method):
        print(f"fetching {entry['source']}", file=sys.stderr, flush=True)
        subprocess.run([sys.executable, *entry["command"]], cwd=ROOT / "methods" / method, check=True)
    source = _source(method)
    in_place = set(facts(method).get("in_place", []))
    for target in _linked(method):
        _place(_origin(facts(method)["inputs"][target]), source / target, target in in_place)
    print(f"{method}: every input is in place", file=sys.stderr, flush=True)


def status() -> list[dict]:
    rows = []
    for method, entry in methods().items():
        lacking = missing(method)
        rows.append({"method": method, "ready": lacking is None, "files": len(entry["inputs"]),
                     "not_in_place": len(_linked(method))})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show or build the inputs of the published methods.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="every method and whether its inputs are in place")
    building = commands.add_parser("build", help="fetch, build and link the inputs of one method")
    building.add_argument("method")
    arguments = parser.parse_args(argv)
    if arguments.command == "status":
        for row in status():
            print(f"{'ready  ' if row['ready'] else 'missing'} {row['method']:16} {row['not_in_place']} of {row['files']} files not in place")
        return 0
    build(arguments.method)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
