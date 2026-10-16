#!/usr/bin/env python3
"""Write the DeepHE fold view of the 153-dimensional matrix, and build its node2vec block.
Rebuild the node2vec block from methods/deephe: python build_features.py build-node2vec
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_ID = "deephe"
METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from egbench.contracts import COHORT_IDS, SPLIT_STRATEGIES
from egbench.data import feature_directory, folds_directory

FEATURE_DIMENSIONS = 153

SEQUENCE_SOURCE = "feature/source/deephe_sequence_89.parquet"
NODE2VEC_OUTPUT = "feature/source/node2vec_64.parquet"
NODE2VEC_EDGES = "feature/source/biogrid_physical_edges.parquet"
NODE2VEC_GENE_UNIVERSE = "feature/source/gene_id_map.parquet"
REBUILD_COMMAND = "python build_features.py build-node2vec"


NODE2VEC_DIMENSIONS = 64
NODE2VEC_WALK_LENGTH = 80
NODE2VEC_WALKS_PER_NODE = 10
NODE2VEC_WINDOW = 10
NODE2VEC_NEGATIVE = 5
NODE2VEC_EPOCHS = 5
NODE2VEC_SEED = 2026


def _node2vec_token_hash(token: str) -> int:
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big") & 0x7FFFFFFF


def _node2vec_graph(edges: pd.DataFrame, universe: pd.Series) -> tuple:
    nodes = tuple(str(key).strip() for key in universe)
    if not all(nodes) or len(set(nodes)) != len(nodes):
        raise ValueError("the node2vec gene universe must be non-empty and free of duplicates")
    nodes = tuple(sorted(nodes))
    pairs: set[tuple[str, str]] = set()
    for left, right in edges.itertuples(index=False, name=None):
        left, right = str(left).strip(), str(right).strip()
        if left != right:
            pairs.add((left, right) if left < right else (right, left))
    outside = {node for pair in pairs for node in pair}.difference(nodes)
    if outside:
        raise ValueError(
            f"{len(outside)} edge endpoints fall outside the gene universe: {sorted(outside)[:5]}"
        )
    neighbours: dict[str, list[str]] = {node: [] for node in nodes}
    for left, right in pairs:
        neighbours[left].append(right)
        neighbours[right].append(left)
    return nodes, {node: tuple(sorted(values)) for node, values in neighbours.items()}


class _Node2VecWalks:

    def __init__(self, nodes: tuple, neighbours: dict) -> None:
        self.nodes = nodes
        self.neighbours = neighbours
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        total = len(self.nodes) * NODE2VEC_WALKS_PER_NODE
        report_every = max(total // 10, 1)
        generator = np.random.Generator(np.random.PCG64(NODE2VEC_SEED))
        walked = 0
        for _ in range(NODE2VEC_WALKS_PER_NODE):
            for start in self.nodes:
                walk = [start]
                while len(walk) < NODE2VEC_WALK_LENGTH:
                    candidates = self.neighbours[walk[-1]]
                    if not candidates:
                        break
                    walk.append(candidates[int(generator.integers(0, len(candidates)))])
                walked += 1
                if walked % report_every == 0:
                    print(
                        f"node2vec pass {self.passes}/{NODE2VEC_EPOCHS + 1}: "
                        f"{walked}/{total} walks",
                        flush=True,
                    )
                yield walk


def build_node2vec() -> Path:
    from gensim.models import Word2Vec

    edges_path = METHOD_ROOT / NODE2VEC_EDGES
    universe_path = METHOD_ROOT / NODE2VEC_GENE_UNIVERSE
    for path in (edges_path, universe_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    nodes, neighbours = _node2vec_graph(
        pd.read_parquet(edges_path, columns=["gene_key_a", "gene_key_b"]),
        pd.read_parquet(universe_path, columns=["gene_key"])["gene_key"],
    )
    print(
        f"node2vec: {len(nodes)} genes, "
        f"{sum(len(value) for value in neighbours.values()) // 2} edges",
        flush=True,
    )
    model = Word2Vec(
        sentences=_Node2VecWalks(nodes, neighbours),
        vector_size=NODE2VEC_DIMENSIONS,
        window=NODE2VEC_WINDOW,
        min_count=0,
        sample=0.0,
        workers=1,
        sg=1,
        hs=0,
        negative=NODE2VEC_NEGATIVE,
        epochs=NODE2VEC_EPOCHS,
        seed=NODE2VEC_SEED,
        sorted_vocab=1,
        hashfxn=_node2vec_token_hash,
    )
    matrix = np.vstack([model.wv[node] for node in nodes])
    if not np.isfinite(matrix).all():
        raise ValueError("node2vec produced a non-finite embedding")
    embedding = pd.DataFrame(
        matrix,
        columns=[f"node2vec_{index:03d}" for index in range(1, NODE2VEC_DIMENSIONS + 1)],
    )
    embedding.insert(0, "gene_key", list(nodes))
    destination = _node2vec_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    embedding.to_parquet(
        destination,
        index=False,
        engine="pyarrow",
        compression="zstd",
        compression_level=9,
    )
    return destination


def _node2vec_path() -> Path:
    return METHOD_ROOT / NODE2VEC_OUTPUT


def _source_paths() -> tuple[Path, Path]:
    return METHOD_ROOT / SEQUENCE_SOURCE, _node2vec_path()


def read_source_matrix() -> pd.DataFrame:
    sequence_path, node2vec_path = _source_paths()
    missing = [path for path in (sequence_path, node2vec_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"DeepHE source assets are absent: {[str(path) for path in missing]}. "
            f"Rebuild them with '{REBUILD_COMMAND}' first."
        )
    sequence = pd.read_parquet(sequence_path)
    node2vec = pd.read_parquet(node2vec_path)
    if sequence.shape[1] != 90 or node2vec.shape[1] != 65:
        raise ValueError(
            "DeepHE source components must contain 89 and 64 feature columns"
        )
    matrix = sequence.merge(node2vec, on="gene_key", how="inner", validate="one_to_one")
    if matrix.shape[1] != FEATURE_DIMENSIONS + 1:
        raise ValueError("DeepHE merged source must contain gene_key plus 153 features")
    return matrix.sort_values("gene_key", kind="mergesort").reset_index(drop=True)


def fold_feature_directory(
    cohort_id: str, split_strategy: str, outer_fold: int
) -> Path:
    return feature_directory(
        METHOD_ROOT,
        cohort_id,
        split_strategy=split_strategy,
        outer_fold=outer_fold,
    )


def build_fold_features(
    cohort_id: str, split_strategy: str, outer_fold: int
) -> tuple[Path, bool]:
    if cohort_id not in COHORT_IDS:
        raise ValueError(f"unsupported dataset: {cohort_id!r}")
    output_directory = fold_feature_directory(cohort_id, split_strategy, outer_fold)
    destination = output_directory / "features.parquet"
    if destination.is_file():
        return destination, True
    if output_directory.exists():
        raise FileExistsError(
            f"{output_directory} exists but does not hold features.parquet; "
            f"remove it before cutting this fold's view again"
        )
    authority_path = (
        folds_directory(cohort_id, split_strategy="family_5fold", root=REPO_ROOT)
        / "fold_genes"
        / f"fold_{outer_fold}_genes.parquet"
    )
    if not authority_path.is_file():
        raise FileNotFoundError(authority_path)
    authority = pd.read_parquet(authority_path, columns=["gene_key"])
    keys = sorted(authority["gene_key"].astype(str))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{authority_path} contains duplicate gene_key values")
    matrix = read_source_matrix().set_index("gene_key", drop=False)
    missing = sorted(set(keys) - set(matrix.index))
    if missing:
        raise ValueError(
            f"{len(missing)} authority genes lack DeepHE source features: "
            f"{missing[:5]}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    canonical = fold_feature_directory(cohort_id, SPLIT_STRATEGIES[0], 0) / "features.parquet"
    if destination != canonical and canonical.is_file():
        os.link(canonical, destination)
    else:
        subset = matrix.loc[keys].reset_index(drop=True)
        subset.to_parquet(destination, index=False)
    return destination, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build-node2vec",))
    parser.parse_args()
    print(build_node2vec(), flush=True)


if __name__ == "__main__":
    main()
