#!/usr/bin/env python3
"""Download the global-context Mahi gene embeddings from the authors' Hugging Face dataset.
Run from methods/mahi: python acquire_assets.py
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pyarrow.parquet as pq

MAHI_ROOT = Path(__file__).resolve().parent
HF_HOME = MAHI_ROOT / "feature" / "source" / "model_assets" / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
os.environ["HF_XET_CACHE"] = str(HF_HOME / "xet")
os.environ["HF_ASSETS_CACHE"] = str(HF_HOME / "assets")

from huggingface_hub import HfApi, hf_hub_download

from build_features import (
    FORMAL_EMBEDDING_DIMENSIONS,
    FORMAL_EMBEDDING_ROWS,
    validate_embedding_source,
    write_embedding_source,
)
from local_dataset import DEFAULT_EMBEDDING_SOURCE, method_path


REPOSITORY_ID = "anushaggs/Mahi_gene_embeddings"


def discover_core_float16_parquet(repo_files: list[str]) -> str:
    candidates = [
        name
        for name in repo_files
        if name.lower().endswith(".parquet")
        and "core" in name.lower()
        and re.search(r"(?:float16|fp16|f16)", name.lower())
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "could not uniquely discover the core float16 parquet in "
            f"{REPOSITORY_ID}; candidates={sorted(candidates)}"
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPOSITORY_ID)
    parser.add_argument("--source-file", help="A parquet path returned by the repository file listing")
    parser.add_argument(
        "--local-source",
        type=Path,
        help="Method-local copy of the official core float16 parquet; skips the Hub download",
    )
    parser.add_argument(
        "--stream-url",
        help="HTTP(S) URL for range-reading the official parquet without downloading non-global row groups",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EMBEDDING_SOURCE)
    args = parser.parse_args()
    output = method_path(args.output, label="embedding output")
    HF_HOME.mkdir(parents=True, exist_ok=True)
    HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)

    if args.local_source is not None and args.stream_url is not None:
        parser.error("--local-source and --stream-url are mutually exclusive")

    global_rows = None
    if args.stream_url is not None:
        import fsspec

        with fsspec.open(
            args.stream_url,
            mode="rb",
            block_size=4 * 1024 * 1024,
            cache_type="readahead",
        ) as remote_source:
            global_rows = pq.read_table(
                remote_source,
                columns=["entrez_id", "tissue", "embedding"],
                filters=[("tissue", "=", "global")],
            ).to_pandas()
    elif args.local_source is not None:
        cached_source = method_path(args.local_source, label="local official source")
        if not cached_source.is_file():
            raise FileNotFoundError(cached_source)
    else:
        api = HfApi()
        repo_files = list(api.list_repo_files(repo_id=args.repo_id, repo_type="dataset"))
        source_file = args.source_file or discover_core_float16_parquet(repo_files)
        if source_file not in repo_files:
            raise ValueError(f"--source-file is not present in the repository listing: {source_file}")
        if not source_file.lower().endswith(".parquet"):
            raise ValueError(f"--source-file must be a parquet file: {source_file}")
        if "core" not in source_file.lower() or not re.search(
            r"(?:float16|fp16|f16)", source_file.lower()
        ):
            raise ValueError("--source-file must identify the core float16 parquet asset")

        downloaded = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=source_file,
            cache_dir=HF_HUB_CACHE,
        )
        cached_source = method_path(Path(downloaded), label="downloaded source")
    if global_rows is None:
        global_rows = pq.read_table(
            cached_source,
            columns=["entrez_id", "tissue", "embedding"],
            filters=[("tissue", "=", "global")],
        ).to_pandas()
    if len(global_rows) != FORMAL_EMBEDDING_ROWS:
        raise ValueError(
            f"discovered asset has {len(global_rows)} global rows; expected {FORMAL_EMBEDDING_ROWS}"
        )
    write_embedding_source(output, global_rows)
    validate_embedding_source(
        output,
        expected_rows=FORMAL_EMBEDDING_ROWS,
        expected_dimensions=FORMAL_EMBEDDING_DIMENSIONS,
    )
    print(output.relative_to(MAHI_ROOT))


if __name__ == "__main__":
    main()
