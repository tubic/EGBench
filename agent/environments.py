"""Runtime environments of the 15 methods, the analyses and the two input builders with their own packages.

``agent/environments.json`` lists seven environments: their Python version, the lock file under
``agent/environments/`` that pins every package, whether they need a GPU, the methods, analyses and
input builders they serve, and the minutes and gigabytes their installation took on the 96-core
server. The analyses are named by their path under ``analysis/``; ``benchmark_report`` stands for
``python -m egbench report``. Environments live under the environment root, ``EGBENCH_ENV_ROOT`` or
``<data root>/envs`` when it is unset, one directory per environment. The ``egbench`` library is not
installed in them: methods, analyses and builders run with ``PYTHONPATH=<repository>/src``.

``install`` creates an environment with conda, ``conda create -p`` with the Python version and then
``pip install -r`` with the lock file, and checks that the installed packages are those of the lock
file. Run again on an environment whose installation stopped, it completes it.
``parameter_counts.py --env-root`` expects one interpreter per method at
``<env root>/egbench-<method>/bin/python``; ``method_environment_root`` writes these as symbolic
links to the environments under ``<environment root>/methods`` and returns that directory.

    python agent/environments.py status
    python agent/environments.py install torch
    python agent/environments.py link-methods
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import external

MANIFEST = external.AGENT / "environments.json"
ENV_ROOT = Path(os.environ.get("EGBENCH_ENV_ROOT") or external.DATA_ROOT / "envs").expanduser().resolve()
METHOD_LINKS = ENV_ROOT / "methods"
ANALYSIS_PREFIX = "analysis/"
CONDA_CHANNEL = "conda-forge"
# The prebuilt PyG wheels come from a page of links beside PyPI; uv by default takes a package only
# from the first source that lists it.
UV_INDEX_STRATEGY = "unsafe-best-match"
NOT_LOCKED = frozenset({"pip"})
# Scripts that count the parameters of every method in that method's environment.
METHOD_ENVIRONMENT_SCRIPTS = frozenset({"parameter_counts/parameter_counts.py"})
GIGABYTE = 1e9
NAME_SEPARATORS = re.compile(r"[-_.]+")
# The CUDA packages on PyPI and the flash-attn wheel of evo2 are built for Linux only.
LINUX_ONLY = re.compile(r"^nvidia-|linux_x86_64", re.MULTILINE)


@lru_cache(maxsize=1)
def manifest() -> dict[str, dict]:
    return {entry["name"]: entry for entry in json.loads(MANIFEST.read_text())["environments"]}


def entry(name: str) -> dict:
    try:
        return manifest()[name]
    except KeyError:
        raise ValueError(f"unknown environment {name!r}; the environments are {', '.join(manifest())}") from None


def prefix(name: str) -> Path:
    entry(name)
    return ENV_ROOT / name


def lock(name: str) -> Path:
    return external.AGENT / entry(name)["lock"]


class MissingEnvironment(FileNotFoundError):
    """An environment that is not installed under the environment root."""

    def __init__(self, name: str) -> None:
        self.name = name
        listing = entry(name)
        cost = ""
        if listing["minutes"] is not None:
            cost = f" (about {listing['minutes']:.0f} minutes and {listing['gigabytes']:.1f} GB on the server)"
        super().__init__(
            f"the {name} environment is not installed at {prefix(name)}; install it with "
            f"'python agent/environments.py install {name}'{cost}, or set EGBENCH_ENV_ROOT to the "
            f"directory that holds it"
        )


def python(name: str) -> Path:
    """The interpreter of an installed environment."""
    interpreter = prefix(name) / "bin" / "python"
    if not interpreter.exists():
        raise MissingEnvironment(name)
    return interpreter


def environment_of(user: str) -> str:
    """The environment of a method identifier, an analysis script or ``benchmark_report``.

    An analysis script is given by its path under ``analysis/``, with or without that prefix, such
    as ``same_input_mlp/same_input_mlp.py``.
    """
    key = user.removeprefix(ANALYSIS_PREFIX)
    for name, listing in manifest().items():
        if key in listing["methods"] or key in listing["analyses"]:
            return name
    raise ValueError(f"{user!r} is neither a method nor an analysis of {MANIFEST.name}")


def environment_of_analysis(analysis: str, script: str | None = None) -> str:
    """The environment of one script of an analysis, or of an analysis that is a single command."""
    if script is not None:
        return environment_of(script)
    names = analysis_environments(analysis)
    if len(names) != 1:
        raise ValueError(f"the scripts of {analysis} run in {', '.join(names)}; name the script")
    return names[0]


def analysis_environments(analysis: str) -> list[str]:
    """The environments the scripts of an analysis directory run in, with the method environments
    for a script that reads every method's weights through ``method_environment_root``."""
    names = []
    for name, listing in manifest().items():
        scripts = [key for key in listing["analyses"] if key == analysis or key.startswith(analysis + "/")]
        if scripts and name not in names:
            names.append(name)
        if any(script in METHOD_ENVIRONMENT_SCRIPTS for script in scripts):
            names += [method_name for method_name, method_listing in manifest().items()
                      if method_listing["methods"] and method_name not in names]
    if not names:
        raise ValueError(f"{analysis!r} is not an analysis of {MANIFEST.name}")
    return names


def status_of(name: str) -> dict:
    """Whether an environment is installed, where, what it serves and what its installation takes."""
    listing = entry(name)
    row = {key: listing[key] for key in ("name", "title", "python", "gpu", "methods", "analyses")}
    row["input_builders"] = listing.get("input_builders", [])
    row["path"] = str(prefix(name))
    row["installed"] = (prefix(name) / "bin" / "python").exists()
    row["minutes"] = listing["minutes"]
    row["gigabytes"] = listing["gigabytes"]
    return row


def status() -> list[dict]:
    return [status_of(name) for name in manifest()]


def _normal(package: str) -> str:
    return NAME_SEPARATORS.sub("-", package).lower()


def locked_packages(name: str) -> dict[str, str | None]:
    """Package names of a lock file with their pinned versions; None for a package given by its URL."""
    packages: dict[str, str | None] = {}
    for line in lock(name).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        if " @ " in line:
            packages[_normal(line.split(" @ ")[0])] = None
        else:
            package, version = line.split("==")
            packages[_normal(package)] = version
    return packages


def installed_packages(name: str) -> dict[str, str]:
    listing = subprocess.run(
        [str(python(name)), "-m", "pip", "list", "--format=freeze", "--disable-pip-version-check"],
        capture_output=True, text=True, check=True,
    ).stdout
    packages = {}
    for line in listing.splitlines():
        package, version = line.split("==")
        packages[_normal(package)] = version
    return packages


def differences(name: str) -> dict[str, list[str]]:
    """Packages of an installed environment that the lock file does not pin as installed."""
    locked = locked_packages(name)
    installed = {package: version for package, version in installed_packages(name).items() if package not in NOT_LOCKED}
    return {
        "missing": sorted(set(locked) - set(installed)),
        "other_version": sorted(
            f"{package} {installed[package]} instead of {version}"
            for package, version in locked.items()
            if package in installed and version is not None and installed[package] != version
        ),
        "not_locked": sorted(f"{package}=={installed[package]}" for package in set(installed) - set(locked)),
    }


def _gigabytes(directory: Path) -> float:
    used = 0
    for folder, _, files in os.walk(directory):
        for file in files:
            used += os.lstat(os.path.join(folder, file)).st_blocks * 512
    return round(used / GIGABYTE, 2)


def _without_conda(name: str) -> str:
    listing = entry(name)
    version, target = listing["python"], prefix(name)
    return (
        f"conda was not found; set CONDA_EXE to the conda executable or put it on PATH. Without conda, "
        f"create the environment with uv: 'uv venv --python {version} {target}' and then "
        f"'uv pip install --python {target}/bin/python --index-strategy {UV_INDEX_STRATEGY} -r {lock(name)}'. "
        f"With a Python {version} interpreter at hand, its own venv and pip do the same: "
        f"'python{version.rsplit('.', 1)[0]} -m venv {target}' and then "
        f"'{target}/bin/python -m pip install -r {lock(name)}'."
    )


def conda_executable() -> str | None:
    return os.environ.get("CONDA_EXE") or shutil.which("conda")


def unavailable(name: str) -> str | None:
    """Why install cannot create the environment here, or None when it can."""
    if (prefix(name) / "bin" / "python").exists():
        return None
    if not sys.platform.startswith("linux") and LINUX_ONLY.search(lock(name).read_text()):
        return (f"the lock file of {name} pins CUDA packages that are built for Linux only, so {name} installs "
                f"only on a Linux x86_64 machine with an NVIDIA GPU; on this machine only the environments cpu "
                f"and node2vec install")
    if conda_executable() is not None:
        return None
    return _without_conda(name)


def install(name: str) -> dict:
    """Create an environment from its lock file and check the installed packages against it.

    The output of conda and pip goes to standard error, which keeps standard output free for the
    MCP server.
    """
    listing = entry(name)
    target = prefix(name)
    started = time.monotonic()
    if not (target / "bin" / "python").exists():
        conda = conda_executable()
        if conda is None:
            raise RuntimeError(_without_conda(name))
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [conda, "create", "--yes", "--quiet", "--prefix", str(target), "--override-channels",
             "--channel", CONDA_CHANNEL, f"python={listing['python']}"],
            check=True, stdout=sys.stderr,
        )
    interpreter = str(target / "bin" / "python")
    subprocess.run(
        [interpreter, "-m", "pip", "install", "--no-input", "--disable-pip-version-check", "-r", str(lock(name))],
        check=True, stdout=sys.stderr,
    )
    check = subprocess.run([interpreter, "-m", "pip", "check"], capture_output=True, text=True)
    found = differences(name)
    return {
        "name": name,
        "path": str(target),
        "ok": check.returncode == 0 and not found["missing"] and not found["other_version"],
        "minutes": round((time.monotonic() - started) / 60, 1),
        "gigabytes": _gigabytes(target),
        "packages": len(installed_packages(name)),
        "pip_check": check.stdout.strip(),
        **found,
    }


def method_environment_root() -> Path:
    """The directory to pass as ``parameter_counts.py --env-root``, after linking
    ``<environment root>/methods/egbench-<method>`` to the environment of each method."""
    METHOD_LINKS.mkdir(parents=True, exist_ok=True)
    for name, listing in manifest().items():
        for method in listing["methods"]:
            link = METHOD_LINKS / f"egbench-{method}"
            if link.is_symlink():
                link.unlink()
            link.symlink_to(Path("..") / name, target_is_directory=True)
    return METHOD_LINKS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show, install or link the runtime environments.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="list every environment, whether it is installed and what it serves")
    installing = commands.add_parser("install", help="create environments from their lock files")
    installing.add_argument("names", nargs="+", choices=list(manifest()))
    commands.add_parser("link-methods", help="link egbench-<method> to each method's environment for parameter_counts.py")
    arguments = parser.parse_args(argv)
    if arguments.command == "status":
        print(f"environment root {ENV_ROOT}")
        for row in status():
            mark = "installed" if row["installed"] else "absent   "
            device = "GPU" if row["gpu"] else "CPU"
            print(f"{mark} {row['name']:11} Python {row['python']:8} {device}  {row['title']}")
        return 0
    if arguments.command == "link-methods":
        print(method_environment_root())
        return 0
    results = [install(name) for name in arguments.names]
    for result in results:
        print(json.dumps(result))
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
