"""Download and verify the ESM-2 650M checkpoint and contact-regression head used by BiNGO.
Run from methods/bingo: python code/acquire_assets.py --asset-dir feature/source/model_assets
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

if __package__:
    from .contracts import require_method_path, sha256_file
    from .graph import BINGO_MODEL_NAME
else:
    from contracts import require_method_path, sha256_file
    from graph import BINGO_MODEL_NAME


MODEL_URL = "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt"
CONTACT_URL = (
    "https://dl.fbaipublicfiles.com/fair-esm/regression/"
    "esm2_t33_650M_UR50D-contact-regression.pt"
)
EXPECTED_SHA256 = {
    "esm2_t33_650M_UR50D.pt": (
        "ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c"
    ),
    "esm2_t33_650M_UR50D-contact-regression.pt": (
        "8ffe6edbd4173dc8d45c2cd5cb27d43aad77ec26b4c768200c58ae1f96693575"
    ),
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "egbench-bingo/1"})
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        temporary.open("wb") as output,
    ):
        while block := response.read(8 * 1024 * 1024):
            output.write(block)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(destination)


def acquire(asset_dir: Path) -> dict[str, object]:
    asset_dir = require_method_path(asset_dir, "asset_dir")
    assets = {
        "esm2_t33_650M_UR50D.pt": MODEL_URL,
        "esm2_t33_650M_UR50D-contact-regression.pt": CONTACT_URL,
    }
    rows: list[dict[str, object]] = []
    for filename, url in assets.items():
        path = asset_dir / filename
        if not path.is_file():
            _download(url, path)
        digest = sha256_file(path)
        expected = EXPECTED_SHA256[filename]
        if expected.startswith("PENDING_"):
            raise RuntimeError(
                f"{filename} was downloaded with sha256={digest}, but the adapter "
                "hash lock has not been finalized"
            )
        if digest != expected:
            raise RuntimeError(
                f"sha256 mismatch for {filename}: expected {expected}, observed {digest}"
            )
        rows.append(
            {
                "filename": filename,
                "url": url,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    manifest = {
        "model_name": BINGO_MODEL_NAME,
        "source": "official_fair_esm_file_host",
        "files": rows,
    }
    (asset_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.asset_dir.resolve())


if __name__ == "__main__":
    main()
