"""Fetch the EssTFNet source-release modules (pinned commit) that this method imports into code/upstream.
Run from methods/esstfnet: python acquire_assets.py
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

METHOD_ROOT = Path(__file__).resolve().parent
UPSTREAM_REPOSITORY = "https://github.com/QIANJINYDX/EssTFNet.git"
UPSTREAM_COMMIT = "4f9ee839bd35eccf00de5580a9ed3d9d1363922e"
UPSTREAM_MODULES = (
    "module/ATFNet.py",
    "module/initial.py",
    "module/lossfunc.py",
    "layers/__init__.py",
    "layers/Autoformer_EncDec.py",
    "layers/ComplexLayers.py",
    "layers/Embed.py",
    "layers/SelfAttention_Family.py",
    "layers/Transformer_EncDec.py",
    "utils/masking.py",
    "utils/params.py",
)
DESTINATION = METHOD_ROOT / "code" / "upstream"


def fetch(*, force: bool = False) -> list[Path]:
    targets = [DESTINATION / name for name in UPSTREAM_MODULES]
    if all(path.is_file() for path in targets) and not force:
        return targets
    with tempfile.TemporaryDirectory() as scratch:
        clone = Path(scratch) / "esstfnet"
        subprocess.run(["git", "clone", "--quiet", "--no-checkout", UPSTREAM_REPOSITORY, str(clone)], check=True)
        subprocess.run(["git", "-C", str(clone), "checkout", "--quiet", UPSTREAM_COMMIT, "--", *UPSTREAM_MODULES], check=True)
        for name, target in zip(UPSTREAM_MODULES, targets):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(clone / name, target)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace files that are already present")
    args = parser.parse_args()
    for path in fetch(force=args.force):
        print(path.relative_to(METHOD_ROOT))


if __name__ == "__main__":
    main()
