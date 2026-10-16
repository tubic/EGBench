"""Download the GOA, CORUM and Ensembl BioMart Pfam annotations used by DeEPsnap and record their checksums.
Run from methods/deepsnap: python code/acquire_assets.py --output-directory feature/source/annotations
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .contracts import (
        BIOMART_QUERY,
        CORUM_HUMAN_URL,
        CORUM_RELEASE_URL,
        ENSEMBL_BIOMART_URL,
        GOA_HUMAN_URL,
        sha256_file,
    )
except ImportError:
    from contracts import (
        BIOMART_QUERY,
        CORUM_HUMAN_URL,
        CORUM_RELEASE_URL,
        ENSEMBL_BIOMART_URL,
        GOA_HUMAN_URL,
        sha256_file,
    )


def _curl(url: str, *, output: Path | None = None, form: str | None = None) -> bytes:
    command = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--proto",
        "=https",
        "--max-time",
        "600",
        "--user-agent",
        "EG-Benchmarking/1.0",
    ]
    if form is not None:
        command.extend(["--data-urlencode", form])
    if output is not None:
        command.extend(["--output", str(output)])
    command.append(url)
    result = subprocess.run(command, check=True, capture_output=output is None)
    return result.stdout if output is None else b""


def _download(url: str, destination: Path, *, form: str | None = None) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    _curl(url, output=temporary, form=form)
    os.replace(temporary, destination)
    return {
        "url": url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def _json_url(url: str) -> Any:
    return json.loads(_curl(url).decode("utf-8"))


def acquire(output_directory: str | Path) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    corum_release = _json_url(CORUM_RELEASE_URL)
    registry_url = f"{ENSEMBL_BIOMART_URL}?type=registry"
    registry = _curl(registry_url).decode("utf-8")
    release_match = re.search(r'database="ensembl_mart_([0-9]+)"', registry)
    if release_match is None:
        raise RuntimeError("could not determine the Ensembl BioMart release")
    biomart_release = int(release_match.group(1))
    (output / "ensembl_biomart_registry.xml").write_text(registry, encoding="utf-8")

    goa = _download(GOA_HUMAN_URL, output / "goa_human.gaf.gz")
    corum = _download(CORUM_HUMAN_URL, output / "corum_human_complexes.tsv")
    pfam = _download(
        ENSEMBL_BIOMART_URL,
        output / f"ensembl_{biomart_release}_human_pfam.tsv",
        form=f"query={BIOMART_QUERY}",
    )
    manifest = {
        "schema_version": "1.0",
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "goa_human": goa,
            "corum_human": {**corum, "corum_release": corum_release},
            "ensembl_biomart_pfam": {
                **pfam,
                "ensembl_mart_release": biomart_release,
                "query": BIOMART_QUERY,
                "registry_sha256": sha256_file(output / "ensembl_biomart_registry.xml"),
            },
        },
    }
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    print(json.dumps(acquire(args.output_directory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
