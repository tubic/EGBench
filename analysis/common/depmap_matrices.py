"""DepMap 26Q1 dependency probability and gene effect matrices keyed by gene, the cell line lineages, and the
variation of TCGA expression across cancer types.

Imported by hallmark_exceptions, prospective_validation, dependency_strength_context and mouse_lethal_stages.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pv
import pyarrow.parquet as pq

from egbench.data import PROJECT_ROOT
from egbench.reference import _DEPMAP_UNIVERSE, _GENE_EFFECT_MATRIX, _GENE_LABEL

DEPMAP = PROJECT_ROOT / "data" / "depmap_26q1"
DEPENDENCY = DEPMAP / "CRISPRGeneDependency.csv"
MODELS = DEPMAP / "Model.csv"
GENE_EFFECT = PROJECT_ROOT / _GENE_EFFECT_MATRIX
DEPMAP_UNIVERSE = PROJECT_ROOT / _DEPMAP_UNIVERSE
EXPRESSION = PROJECT_ROOT / "data" / "source" / "expression" / "tcga_gene_by_sample.parquet"
EXPRESSION_QA = PROJECT_ROOT / "data" / "source" / "expression" / "tcga_expression_qa.json"
DEPENDENT_ABOVE = 0.5


def _keyed_matrix(path: Path) -> pd.DataFrame:
    universe = pd.read_parquet(DEPMAP_UNIVERSE, columns=["gene_key", "entrez_id"])
    by_entrez = dict(zip(universe["entrez_id"].astype("int64"), universe["gene_key"].astype("object")))
    table = pv.read_csv(path)
    first = table.column_names[0]
    keys = {}
    for name in table.column_names[1:]:
        match = _GENE_LABEL.match(name)
        if match is None:
            raise ValueError(f"{path.name}: column {name!r} is not 'SYMBOL (ENTREZ)'")
        key = by_entrez.get(int(match.group(2)))
        if key is not None:
            keys[name] = key
    matrix = pd.DataFrame({keys[name]: table.column(name).to_numpy(zero_copy_only=False) for name in keys})
    matrix.index = pd.Index(table.column(first).to_pylist(), name="ModelID")
    if matrix.columns.has_duplicates:
        raise ValueError(f"{path.name}: two columns map to one gene key")
    return matrix


def dependency_matrix() -> pd.DataFrame:
    return _keyed_matrix(DEPENDENCY)


def gene_effect_matrix() -> pd.DataFrame:
    return _keyed_matrix(GENE_EFFECT)


def dependent_share(matrix: pd.DataFrame) -> pd.Series:
    return (matrix > DEPENDENT_ABOVE).sum(axis=0) / matrix.notna().sum(axis=0)


def tissue_variation() -> pd.Series:
    qa = json.loads(EXPRESSION_QA.read_text())
    counts = qa["cancer_type_counts"]
    schema = pq.read_schema(EXPRESSION)
    samples = [c for c in schema.names if c != "gene_key"]
    if sum(counts.values()) != len(samples):
        raise ValueError("the sample counts of the expression record do not match the matrix")
    order = [m.split("/")[0].upper() for m in qa["source_members"]]
    if sorted(order) != sorted(counts):
        raise ValueError("the cancer types of the expression record do not match")
    types = []
    for cancer in order:
        types.extend([cancer] * counts[cancer])
    tss = pd.Series([s.split("-")[1] for s in samples])
    by_type = pd.DataFrame({"type": types, "tss": tss})
    if (by_type.groupby("tss")["type"].nunique() > 1).any():
        raise ValueError("a tissue source site spans two cancer types; the sample to type map is not recoverable")
    frame = pd.read_parquet(EXPRESSION)
    keys = frame["gene_key"].astype("object").to_numpy()
    values = np.log1p(frame[samples].to_numpy("float32"))
    groups = pd.Series(types, index=samples)
    means = np.column_stack([values[:, (groups == t).to_numpy()].mean(axis=1) for t in counts])
    mean = means.mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = means.std(axis=1, ddof=1) / mean
    return pd.Series(cv, index=pd.Index(keys, name="gene_key"), name="tissue_cv")
