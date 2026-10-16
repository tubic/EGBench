"""Download the Ensembl Compara and JensenLab COMPARTMENTS sources of the EPGAT static orthology and localisation table.
Run from methods/epgat: python code/acquire_assets.py --output-directory feature/source/annotations
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

try:
    from .contracts import COMPARA_RELEASE, COMPARA_URL, COMPARTMENTS_URL, sha256_file
except ImportError:
    from contracts import COMPARA_RELEASE, COMPARA_URL, COMPARTMENTS_URL, sha256_file


_CURL_COMMON = [
    "curl",
    "--fail",
    "--location",
    "--silent",
    "--show-error",
    "--proto",
    "=https",
    "--user-agent",
    "EG-Benchmarking/1.0",
]


def _remote_size(url: str) -> int:
    result = subprocess.run(
        [*_CURL_COMMON, "--max-time", "120", "--head", url],
        check=True,
        capture_output=True,
        text=True,
    )
    values = re.findall(r"^content-length:\s*(\d+)\s*$", result.stdout, re.I | re.M)
    if not values:
        raise RuntimeError(f"remote server did not provide Content-Length: {url}")
    return int(values[-1])


def _download_parallel_ranges(
    url: str,
    destination: Path,
    *,
    workers: int = 8,
) -> None:

    total_size = _remote_size(url)
    if total_size <= 0:
        raise RuntimeError(f"invalid remote Content-Length {total_size}: {url}")

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    parts_directory = destination.parent / f"{destination.name}.parts"
    parts_directory.mkdir(parents=True, exist_ok=True)
    for old_part in parts_directory.glob("part-*.bin"):
        old_part.unlink()

    chunk_size = (total_size + workers - 1) // workers
    ranges: list[tuple[int, int, int, Path]] = []
    for index in range(workers):
        start = index * chunk_size
        if start >= total_size:
            break
        end = min(total_size - 1, start + chunk_size - 1)
        ranges.append((index, start, end, parts_directory / f"part-{index:03d}.bin"))

    def fetch(item: tuple[int, int, int, Path]) -> Path:
        _, start, end, part = item
        subprocess.run(
            [
                *_CURL_COMMON,
                "--max-time",
                "1200",
                "--range",
                f"{start}-{end}",
                "--output",
                str(part),
                url,
            ],
            check=True,
        )
        expected_size = end - start + 1
        observed_size = part.stat().st_size
        if observed_size != expected_size:
            raise RuntimeError(
                f"range {start}-{end} returned {observed_size} bytes; "
                f"expected {expected_size}: {url}"
            )
        return part

    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        parts = list(executor.map(fetch, ranges))

    with temporary.open("wb") as output_handle:
        for part in parts:
            with part.open("rb") as input_handle:
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
    if temporary.stat().st_size != total_size:
        raise RuntimeError(
            f"merged download has {temporary.stat().st_size} bytes; "
            f"expected {total_size}: {url}"
        )
    os.replace(temporary, destination)
    for part in parts:
        part.unlink()
    parts_directory.rmdir()


def _download(
    url: str,
    destination: Path,
    *,
    parallel_ranges: bool = False,
) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return {
            "url": url,
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "reused_existing": True,
        }
    if parallel_ranges:
        _download_parallel_ranges(url, destination)
    else:
        temporary = destination.with_suffix(destination.suffix + ".part")
        command = [
            *_CURL_COMMON,
            "--max-time",
            "1200",
            "--output",
            str(temporary),
        ]
        if temporary.exists():
            command.extend(["--continue-at", "-"])
        command.append(url)
        subprocess.run(command, check=True)
        os.replace(temporary, destination)
    return {
        "url": url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def acquire(output_directory: str | Path) -> dict[str, object]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    records = {
        "ensembl_compara_115": _download(
            COMPARA_URL,
            output / "Compara.115.protein_default.homologies.tsv.gz",
        ),
        "jensenlab_human_compartments": _download(
            COMPARTMENTS_URL,
            output / "human_compartment_integrated_full.tsv",
            parallel_ranges=True,
        ),
    }
    manifest = {
        "schema_version": "1.0",
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "ensembl_compara_release": COMPARA_RELEASE,
        "sources": records,
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
