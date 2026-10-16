"""Gene-level ESM-2 650M vectors: the residue mean of the layer 33 embeddings Bingo caches per gene.

Run from the release root in the Bingo environment: python analysis/foundation_model_probes/esm2_pool.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "methods" / "bingo" / "feature" / "source" / "esm2_graph_cache"
OUT = ROOT / "analysis" / "foundation_model_probes" / "work" / "esm2_650m_mean.parquet"


def main() -> None:
    index = pd.read_parquet(CACHE / "cache_index.parquet")
    n = len(index)
    vectors = np.empty((n, 1280), dtype="float32")
    started = time.time()
    step = max(1, n // 10)
    for i, row in enumerate(index.itertuples()):
        data = torch.load(CACHE / "graphs" / row.cache_file, map_location="cpu", weights_only=False)
        if data["gene_key"] != row.gene_key:
            raise ValueError(f"{row.cache_file}: holds {data['gene_key']}, index says {row.gene_key}")
        x = data["x"]
        if x.shape != (row.cached_length, 1280):
            raise ValueError(f"{row.cache_file}: x is {tuple(x.shape)}")
        vectors[i] = x.float().mean(dim=0).numpy()
        if (i + 1) % step == 0 or i + 1 == n:
            print(f"[esm2_pool] {i + 1}/{n} done, {time.time() - started:.0f} s", flush=True)
    frame = pd.DataFrame(vectors, columns=[f"e_{j}" for j in range(1280)])
    frame.insert(0, "truncated", index["truncated"].to_numpy())
    frame.insert(0, "cached_length", index["cached_length"].to_numpy())
    frame.insert(0, "gene_key", index["gene_key"].astype("object").to_numpy())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
