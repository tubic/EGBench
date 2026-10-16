"""ESM-2 150M or 3B last-layer residue-mean embeddings of the 17,121 common extractable genes from the official weights (GPU).

Run from the release root in the Bingo environment:
python analysis/foundation_model_probes/esm2_extract.py <weights.pt> <esm2_150m|esm2_3b> <token budget>
"""

from __future__ import annotations

import gzip
import sys
import time
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
GENES = ROOT / "methods" / "proteomelm" / "feature" / "source" / "esmc_base_embeddings.parquet"
FASTA = ROOT / "methods" / "bingo" / "feature" / "source" / "deg_protein_sequences.fa.gz"
OUT_DIRECTORY = ROOT / "analysis" / "foundation_model_probes" / "work"
MAX_RESIDUES = 1022


def read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, list[str]] = {}
    current = None
    with gzip.open(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split("|")[0]
                sequences[current] = []
            else:
                sequences[current].append(line)
    return {key: "".join(parts) for key, parts in sequences.items()}


def main() -> None:
    weights, name, budget = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    torch.manual_seed(2026)
    wanted = pd.read_parquet(GENES, columns=["gene_key"])
    fasta = read_fasta(FASTA)
    missing = [g for g in wanted.gene_key if g not in fasta]
    if missing:
        raise ValueError(f"{len(missing)} genes have no sequence, for example {missing[:3]}")
    sequences = {g: fasta[g].rstrip("*") for g in wanted.gene_key}
    model_data = torch.load(str(weights), map_location="cpu", weights_only=False)
    model, alphabet = esm.pretrained.load_model_and_alphabet_core(weights.stem, model_data, None)
    model = model.eval().half().cuda()
    layer = model.num_layers
    dim = model.embed_dim
    converter = alphabet.get_batch_converter()
    order = sorted(sequences, key=lambda g: -len(sequences[g]))
    batches, current, tokens = [], [], 0
    for g in order:
        length = min(len(sequences[g]), MAX_RESIDUES) + 2
        if current and tokens + length > budget:
            batches.append(current)
            current, tokens = [], 0
        current.append(g)
        tokens += length
    if current:
        batches.append(current)
    vectors = np.empty((len(order), dim), dtype="float32")
    position = {g: i for i, g in enumerate(order)}
    started = time.time()
    step = max(1, len(batches) // 10)
    with torch.no_grad():
        for b, batch in enumerate(batches):
            data = [(g, sequences[g][:MAX_RESIDUES]) for g in batch]
            _, _, toks = converter(data)
            out = model(toks.cuda(), repr_layers=[layer])["representations"][layer]
            for i, (g, seq) in enumerate(data):
                vectors[position[g]] = out[i, 1: len(seq) + 1].float().mean(dim=0).cpu().numpy()
            if (b + 1) % step == 0 or b + 1 == len(batches):
                print(f"[{name}] {b + 1}/{len(batches)} batches done, {time.time() - started:.0f} s", flush=True)
    frame = pd.DataFrame(vectors, columns=[f"e_{j}" for j in range(dim)])
    frame.insert(0, "gene_key", np.array(order, dtype="object"))
    frame = frame.set_index("gene_key").reindex(wanted.gene_key).reset_index()
    OUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_DIRECTORY / f"{name}_mean.parquet", index=False)
    print(f"wrote {name}_mean.parquet, layer {layer}, {dim} dimensions", flush=True)


if __name__ == "__main__":
    main()
