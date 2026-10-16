"""DepMap 26Q1 gene effect means and DEG vote counts."""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pv

from egbench.data import PROJECT_ROOT

GENE_KEY = "gene_key"

_GENE_EFFECT_MATRIX = "data/depmap_26q1/CRISPRGeneEffect.csv"
_DEPMAP_UNIVERSE = "datasets/DepMap/data/depmap26q1_gene_universe.parquet"
_GENE_LABEL = re.compile(r"^(.+) \((\d+)\)$")
MAXIMUM_MISSING_FRACTION = 0.10
GENE_EFFECT_COLUMN = "gene_effect"
MISSING_COLUMN = "missing_fraction"

_DEG_UNIVERSE = "datasets/DEG/data/gene_universe.parquet"
VOTE_COLUMN = "deg_vote_count"

_GENE_EFFECT: dict[str, pd.DataFrame] = {}


def _resolve_root(root: str | Path | None) -> Path:
    return PROJECT_ROOT if root is None else Path(root)


def summarise_gene_effect(matrix: pd.DataFrame, keys: Mapping[str, str]) -> pd.DataFrame:
    columns = [name for name in matrix.columns if name in keys]
    if not columns:
        raise ValueError("summarise_gene_effect: no column of the matrix maps to a gene key")
    values = matrix[columns].to_numpy("float64")
    measured = ~np.isnan(values)
    with np.errstate(invalid="ignore"):
        means = np.nansum(values, axis=0) / measured.sum(axis=0)
    frame = pd.DataFrame(
        {
            GENE_EFFECT_COLUMN: means,
            MISSING_COLUMN: 1.0 - measured.mean(axis=0),
        },
        index=pd.Index([keys[name] for name in columns], name=GENE_KEY),
    )
    if frame.index.has_duplicates:
        raise ValueError("summarise_gene_effect: two columns map to one gene key")
    return frame


def gene_effect_means(root: str | Path | None = None) -> pd.DataFrame:
    resolved = _resolve_root(root)
    key = str(resolved)
    cached = _GENE_EFFECT.get(key)
    if cached is not None:
        return cached
    universe = pd.read_parquet(resolved / _DEPMAP_UNIVERSE, columns=[GENE_KEY, "entrez_id"])
    by_entrez = dict(zip(universe["entrez_id"].astype("int64"), universe[GENE_KEY].astype("object")))
    table = pv.read_csv(resolved / _GENE_EFFECT_MATRIX)
    keys = {}
    for name in table.column_names[1:]:
        match = _GENE_LABEL.match(name)
        if match is None:
            raise ValueError(f"{_GENE_EFFECT_MATRIX}: column {name!r} is not 'SYMBOL (ENTREZ)'")
        gene_key = by_entrez.get(int(match.group(2)))
        if gene_key is not None:
            keys[name] = gene_key
    matrix = pd.DataFrame(
        {name: table.column(name).to_numpy(zero_copy_only=False) for name in keys}
    )
    summary = summarise_gene_effect(matrix, keys)
    _GENE_EFFECT[key] = summary
    return summary


def deg_vote_counts(root: str | Path | None = None) -> pd.Series:
    frame = pd.read_parquet(_resolve_root(root) / _DEG_UNIVERSE, columns=[GENE_KEY, VOTE_COLUMN])
    if frame[VOTE_COLUMN].isna().any():
        raise ValueError(f"{_DEG_UNIVERSE}: {VOTE_COLUMN!r} is missing on a gene")
    return pd.Series(
        frame[VOTE_COLUMN].to_numpy("float64"),
        index=pd.Index(frame[GENE_KEY].astype("object"), name=GENE_KEY),
        name=VOTE_COLUMN,
    )
