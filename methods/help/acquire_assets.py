"""Fetch the svelearn package (pinned commit) that HELP imports into code/vendor.
Run from methods/help: python acquire_assets.py
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

METHOD_ROOT = Path(__file__).resolve().parent
UPSTREAM_REPOSITORY = "https://github.com/giordamaug/svelearn.git"
UPSTREAM_COMMIT = "c6652d4b27006045b9b0bd4c046241ead10506d3"
UPSTREAM_FILES = {
    "svelearn/models/__init__.py": "svelearn/models/__init__.py",
    "svelearn/models/splitvotingens.py": "svelearn/models/splitvotingens.py",
    "svelearn/validation/__init__.py": "svelearn/validation/__init__.py",
    "svelearn/validation/crossvalidate.py": "svelearn/validation/crossvalidate.py",
    "LICENSE": "svelearn_LICENSE",
}
DESTINATION = METHOD_ROOT / "code" / "vendor"


def fetch(*, force: bool = False) -> list[Path]:
    targets = [DESTINATION / name for name in UPSTREAM_FILES.values()]
    if all(path.is_file() for path in targets) and not force:
        return targets
    with tempfile.TemporaryDirectory() as scratch:
        clone = Path(scratch) / "svelearn"
        subprocess.run(["git", "clone", "--quiet", "--no-checkout", UPSTREAM_REPOSITORY, str(clone)], check=True)
        subprocess.run(["git", "-C", str(clone), "checkout", "--quiet", UPSTREAM_COMMIT, "--", *UPSTREAM_FILES], check=True)
        for source, target in zip(UPSTREAM_FILES, targets):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(clone / source, target)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace files that are already present")
    args = parser.parse_args()
    for path in fetch(force=args.force):
        print(path.relative_to(METHOD_ROOT))


if __name__ == "__main__":
    main()
