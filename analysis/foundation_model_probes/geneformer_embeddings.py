"""Geneformer-V1-10M gene token embeddings (256 dimensions) mapped to gene keys through the HGNC Ensembl identifiers.

Run from the release root: python analysis/foundation_model_probes/geneformer_embeddings.py
"""

from __future__ import annotations

import hashlib
import json
import pickle
import struct
from pathlib import Path

import numpy as np
import pandas as pd

from egbench.data import PROJECT_ROOT
from egbench.modules import symbol_to_key

PRETRAINED = PROJECT_ROOT / "data" / "pretrained" / "geneformer"
WEIGHTS = PRETRAINED / "model.safetensors"
TOKENS = PRETRAINED / "token_dictionary_gc30M.pkl"
CHECKSUMS = {
    WEIGHTS: "a5e33a757431643b3697de7ef6127950cdc49e06e58d4266b3a3ab191b683f14",
    TOKENS: "ab9dc40973fa5224d77b793e2fd114cacf3d08423ed9c4c49caf0ba9c7f218f1",
}
WORD_EMBEDDINGS = "bert.embeddings.word_embeddings.weight"
HGNC = PROJECT_ROOT / "data" / "hgnc" / "hgnc_complete_set_2026-07-07.txt"
OUT = PROJECT_ROOT / "analysis" / "foundation_model_probes" / "work" / "geneformer_gene_key.parquet"


def read_safetensor(path: Path, name: str) -> np.ndarray:
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
        entry = header[name]
        if entry["dtype"] != "F32":
            raise ValueError(f"{path.name}: {name} is {entry['dtype']}, not F32")
        start, end = entry["data_offsets"]
        handle.seek(8 + length + start)
        raw = handle.read(end - start)
    return np.frombuffer(raw, dtype="<f4").reshape(entry["shape"])


def ensembl_resolver() -> dict[str, str]:
    valid = set(symbol_to_key().values())
    hgnc = pd.read_csv(HGNC, sep="\t", dtype=str, low_memory=False,
                       usecols=["hgnc_id", "ensembl_gene_id"]).dropna(subset=["ensembl_gene_id"])
    hgnc = hgnc[hgnc["hgnc_id"].isin(valid)]
    counted = hgnc.groupby("ensembl_gene_id")["hgnc_id"].nunique()
    single = set(counted.index[counted == 1])
    return {str(e): str(k) for e, k in zip(hgnc["ensembl_gene_id"], hgnc["hgnc_id"]) if str(e) in single}


def main() -> None:
    for path, expected in CHECKSUMS.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"{path} has sha256 {digest}, expected {expected}")
    tokens = pickle.load(TOKENS.open("rb"))
    weights = read_safetensor(WEIGHTS, WORD_EMBEDDINGS)
    genes = {str(k): int(v) for k, v in tokens.items() if str(k).startswith("ENSG")}
    keys = list(genes)
    values = weights[[genes[k] for k in keys]].astype("float32")
    resolver = ensembl_resolver()
    mapped = np.asarray([resolver.get(k) for k in keys], dtype="object")
    keep = np.asarray([m is not None for m in mapped])
    frame = pd.DataFrame(values[keep], index=pd.Index(mapped[keep], name="gene_key"))
    frame.columns = [str(c) for c in frame.columns]
    frame = frame.groupby(level=0, sort=True).mean()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(frame):,} genes, {frame.shape[1]} dimensions", flush=True)


if __name__ == "__main__":
    main()
