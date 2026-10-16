"""CORUM complexes and GO biological process annotations mapped to gene keys."""
from __future__ import annotations

import gzip
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from egbench.data import PROJECT_ROOT, read_cohort_labels

GENE_KEY = "gene_key"
TERM_ID = "term_id"
PAIR_COLUMNS: tuple[str, ...] = (TERM_ID, GENE_KEY)

COMPLEX = "complex"
PROCESS = "process"
GRANULARITIES: tuple[str, ...] = (COMPLEX, PROCESS)

TERM_MIN = 5
TERM_MAX = 200
COMPLEX_MIN = 2

EXCLUDED_EVIDENCE: frozenset[str] = frozenset({"IEA"})

_CORUM_FILE = "data/source/annotations/corum_human_complexes.tsv"
_GAF_FILE = "data/source/annotations/goa_human.gaf.gz"
_CORUM_ORGANISM = "Human"
_GAF_PROCESS_ASPECT = "P"
_GAF_NOT_QUALIFIER = "NOT"
_GAF_SYMBOL, _GAF_QUALIFIER, _GAF_TERM, _GAF_EVIDENCE, _GAF_ASPECT, _GAF_SYNONYMS = (
    2,
    3,
    4,
    6,
    8,
    10,
)

_DEG_UNIVERSE = "datasets/DEG/data/gene_universe.parquet"
_DEPMAP_UNIVERSE = "datasets/DepMap/data/depmap26q1_gene_universe.parquet"

_SYMBOL_MAPS: dict[str, dict[str, str]] = {}
_PAIRS: dict[tuple[str, str, frozenset[str]], pd.DataFrame] = {}
_NAMES: dict[str, pd.Series] = {}


def _resolve_root(root: str | Path | None) -> Path:
    return PROJECT_ROOT if root is None else Path(root)


def symbol_to_key(root: str | Path | None = None) -> dict[str, str]:
    resolved = _resolve_root(root)
    cached = _SYMBOL_MAPS.get(str(resolved))
    if cached is not None:
        return cached
    mapping: dict[str, str] = {}
    deg = pd.read_parquet(resolved / _DEG_UNIVERSE, columns=[GENE_KEY, "gene_symbol"])
    for symbol, key in zip(deg["gene_symbol"], deg[GENE_KEY]):
        mapping.setdefault(str(symbol), str(key))
    depmap = pd.read_parquet(
        resolved / _DEPMAP_UNIVERSE, columns=[GENE_KEY, "gene_symbol", "matrix_symbol"]
    )
    for column in ("gene_symbol", "matrix_symbol"):
        for symbol, key in zip(depmap[column], depmap[GENE_KEY]):
            mapping.setdefault(str(symbol), str(key))
    _SYMBOL_MAPS[str(resolved)] = mapping
    return mapping


def _pair_table(rows: Iterable[tuple[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=list(PAIR_COLUMNS))
    frame[TERM_ID] = frame[TERM_ID].astype("object")
    frame[GENE_KEY] = frame[GENE_KEY].astype("object")
    return frame.drop_duplicates().reset_index(drop=True)


def complex_pairs(root: str | Path | None = None) -> pd.DataFrame:
    resolved = _resolve_root(root)
    cache_key = (str(resolved), COMPLEX, frozenset())
    cached = _PAIRS.get(cache_key)
    if cached is not None:
        return cached
    mapping = symbol_to_key(root)
    table = pd.read_csv(
        resolved / _CORUM_FILE,
        sep="\t",
        usecols=["complex_id", "organism", "subunits_gene_name"],
        dtype="string",
    )
    table = table.loc[table["organism"] == _CORUM_ORGANISM]
    rows = []
    for complex_id, names in zip(table["complex_id"], table["subunits_gene_name"].fillna("")):
        for name in str(names).split(";"):
            key = mapping.get(name.strip())
            if key is not None:
                rows.append((f"CORUM:{complex_id}", key))
    pairs = _pair_table(rows)
    _PAIRS[cache_key] = pairs
    return pairs


def complex_names(root: str | Path | None = None) -> pd.Series:
    resolved = _resolve_root(root)
    cached = _NAMES.get(str(resolved))
    if cached is not None:
        return cached
    table = pd.read_csv(
        resolved / _CORUM_FILE,
        sep="\t",
        usecols=["complex_id", "complex_name", "organism"],
        dtype="string",
    )
    table = table.loc[table["organism"] == _CORUM_ORGANISM].drop_duplicates("complex_id")
    names = pd.Series(
        table["complex_name"].fillna("").astype("object").to_numpy(),
        index=pd.Index([f"CORUM:{item}" for item in table["complex_id"]], name=TERM_ID),
        name="term_name",
    )
    _NAMES[str(resolved)] = names
    return names


def process_pairs(
    root: str | Path | None = None,
    *,
    excluded_evidence: Iterable[str] = EXCLUDED_EVIDENCE,
) -> pd.DataFrame:
    resolved = _resolve_root(root)
    excluded = frozenset(excluded_evidence)
    cache_key = (str(resolved), PROCESS, excluded)
    cached = _PAIRS.get(cache_key)
    if cached is not None:
        return cached
    mapping = symbol_to_key(root)
    rows = []
    with gzip.open(resolved / _GAF_FILE, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= _GAF_SYNONYMS:
                continue
            if fields[_GAF_ASPECT] != _GAF_PROCESS_ASPECT:
                continue
            if fields[_GAF_EVIDENCE] in excluded:
                continue
            if _GAF_NOT_QUALIFIER in fields[_GAF_QUALIFIER].split("|"):
                continue
            key = mapping.get(fields[_GAF_SYMBOL])
            if key is None:
                for synonym in fields[_GAF_SYNONYMS].split("|"):
                    key = mapping.get(synonym.strip())
                    if key is not None:
                        break
            if key is not None:
                rows.append((fields[_GAF_TERM], key))
    pairs = _pair_table(rows)
    _PAIRS[cache_key] = pairs
    return pairs


def restrict_terms(
    pairs: pd.DataFrame,
    genes: Iterable[str],
    *,
    minimum: int,
    maximum: float,
) -> pd.DataFrame:
    if list(pairs.columns) != list(PAIR_COLUMNS):
        raise ValueError(
            f"restrict_terms: pairs must carry {list(PAIR_COLUMNS)}, got {list(pairs.columns)}"
        )
    if minimum < 1:
        raise ValueError(f"restrict_terms: minimum must be at least 1, got {minimum}")
    if maximum < minimum:
        raise ValueError(
            f"restrict_terms: maximum {maximum} is below minimum {minimum}"
        )
    kept = pairs.loc[pairs[GENE_KEY].isin(set(genes))]
    sizes = kept.groupby(TERM_ID)[GENE_KEY].size()
    terms = sizes.index[(sizes >= minimum) & (sizes <= maximum)]
    return kept.loc[kept[TERM_ID].isin(terms)].reset_index(drop=True)


def cohort_complexes(
    cohort_id: str, *, root: str | Path | None = None
) -> pd.DataFrame:
    genes = read_cohort_labels(cohort_id, root=root)[GENE_KEY]
    return restrict_terms(
        complex_pairs(root), genes, minimum=COMPLEX_MIN, maximum=float("inf")
    )


def cohort_processes(
    cohort_id: str,
    *,
    term_maximum: int = TERM_MAX,
    excluded_evidence: Iterable[str] = EXCLUDED_EVIDENCE,
    root: str | Path | None = None,
) -> pd.DataFrame:
    genes = read_cohort_labels(cohort_id, root=root)[GENE_KEY]
    return restrict_terms(
        process_pairs(root, excluded_evidence=excluded_evidence),
        genes,
        minimum=TERM_MIN,
        maximum=term_maximum,
    )


def cohort_modules(
    cohort_id: str, granularity: str, *, root: str | Path | None = None
) -> pd.DataFrame:
    if granularity == COMPLEX:
        return cohort_complexes(cohort_id, root=root)
    if granularity == PROCESS:
        return cohort_processes(cohort_id, root=root)
    raise ValueError(
        f"unknown granularity {granularity!r}, the granularities are {list(GRANULARITIES)}"
    )


def annotation_indicator(pairs: pd.DataFrame, keys: Iterable[str]) -> np.ndarray:
    annotated = set(pairs[GENE_KEY])
    return np.fromiter(
        (1.0 if key in annotated else 0.0 for key in keys), dtype="float64"
    )
