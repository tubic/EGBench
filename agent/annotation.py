"""Per-gene annotation of the agent tools and the three tools that read it: gene lists and predictions seen through
the manuscript's analyses of essential genes.

Four derived tables of ``agent/external.json`` are built here from public downloads. ``biogrid_edges`` holds the
BioGRID 5.0.259 physical interactions between genes of the gene reference. The three annotation tables hold one row
per gene of the gene reference, split by the downloads they need: ``annotation`` the CORUM complexes, yeast
orthologue and gene age, number of BioGRID interaction partners among the labelled genes of each task, median TCGA
expression, paralogues and protein family size and PubMed literature count; ``annotation_dependency`` the DepMap 26Q1
dependency; ``annotation_mouse`` the IMPC outcome of the mouse knockout. Every labelled gene of the seven tasks is a
gene of the gene reference, so no other gene is needed. The values are those the manuscript reads from the methods'
inputs.

The tools are ``annotate_genes`` (hallmarks, dependency or mouse knockout of a gene list), ``enrichment`` (GO
biological processes and CORUM complexes over-represented in a gene list) and ``evaluate_by_strata`` (a table of
test-fold predictions scored within groups of genes next to the 15 published predictors). Quantities relative to a
task, the medians of PPI degree and expression and the quartiles of the literature count, are taken over the
labelled genes of a reference task, as in the manuscript.

    python agent/annotation.py check [--manuscript DIR] [--biogrid-qa FILE]

recomputes tables of the manuscript from the annotation and the tools and writes agent/results/annotation_check.csv.
DIR is a copy of the repository holding the methods' inputs (methods/<method>/feature/) and the analysis caches
(analysis/<analysis>/work/); FILE is the provenance record of the BioGRID table. Comparisons that need them are
skipped without them.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd
import pyarrow.csv as pv
import pyarrow.parquet as pq
from pydantic import Field
from scipy.stats import hypergeom

import benchmark
import external

if str(benchmark.ROOT / "analysis") not in sys.path:
    sys.path.append(str(benchmark.ROOT / "analysis"))

from common.depmap_matrices import DEPENDENT_ABOVE  # noqa: E402
from common.hallmark_counts import HALLMARKS, RICH  # noqa: E402
from common.hallmarks import LAYER_BY_INDEX, LAYERS, SPECIES_PREFIX, YOUNGEST, share_interval, species_layers  # noqa: E402
from common.stats import Resampler, summarise  # noqa: E402
from egbench.contracts import METHOD_IDS, SEED, TargetKind  # noqa: E402
from egbench.data import (  # noqa: E402
    cohort_target,
    feature_directory,
    prediction_cohort,
    primary_scope,
    read_cohort_labels,
    scope_gene_keys,
)
from egbench.evaluate import FOLD_COUNT  # noqa: E402
from egbench.metrics.discrimination import average_ranks  # noqa: E402
from egbench.modules import EXCLUDED_EVIDENCE, TERM_MAX, TERM_MIN, restrict_terms, symbol_to_key  # noqa: E402
from egbench.reference import GENE_EFFECT_COLUMN, summarise_gene_effect  # noqa: E402
from egbench.stats import RESAMPLE_COUNT, benjamini_hochberg_adjusted  # noqa: E402
from egbench.strata import QUARTILE_BANDS, quartile_edges  # noqa: E402

ROOT = benchmark.ROOT
GENE_KEY = benchmark.GENE_KEY
FOLD = benchmark.FOLD
SCORE = benchmark.SCORE
TASK_IDS: tuple[str, ...] = tuple(benchmark.TASKS)
WORK = benchmark.WORK_DIRECTORY
CHECK_OUTPUT = ROOT / "agent" / "results" / "annotation_check.csv"
FAMILIES = ROOT / "datasets" / "homology" / "cluster_map.parquet"
PARALOGUE_EDGES = ROOT / "datasets" / "homology" / "filtered_edges.parquet"
DEPMAP_UNIVERSE = ROOT / "datasets" / "DepMap" / "data" / "depmap26q1_gene_universe.parquet"
MOUSE_UNIVERSE = ROOT / "datasets" / "Mouse" / "data" / "gene_universe.parquet"
RESULTS = ROOT / "analysis"
TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".parquet")
GENE_COLUMNS = ("gene_key", "gene_symbol", "symbol", "gene", "hgnc_id")
INLINE_GENES = 20
SHOWN_ROWS = 10
SHOWN_SYMBOLS = 10
TOP_TERMS = 15
Q_LEVEL = 0.05

ANNOTATION = "annotation"
DEPENDENCY_TABLE = "annotation_dependency"
MOUSE_TABLE = "annotation_mouse"
SYMBOL = "symbol"
COMPLEXES = "corum_complexes"
YEAST = "yeast_orthologue"
AGE = "age_layer"
DEGREE = "ppi_degree"
EXPRESSION = "tcga_median"
FAMILY_SIZE = "family_size"
PARALOGUES = "paralogues"
LITERATURE = "pubmed_count"
EFFECT = "gene_effect_mean"
DEPENDENT = "dependent_share"
MEASURED = "dependency_lines"
DEPMAP_CLASS = "depmap_class"
MOUSE_GENE = "mouse_gene"
MGI = "mgi_id"
VIABILITY = "impc_viability"
STAGE = "impc_lethal_stage"
COUNT = "hallmark_count"
SUBMISSION = "table"

BIOGRID_MEMBER = "Homo_sapiens"
BIOGRID_SUFFIX = ".tab3.txt"
BIOGRID_COLUMNS = {
    "Entrez Gene Interactor A": "entrez_a",
    "Entrez Gene Interactor B": "entrez_b",
    "Experimental System": "system",
    "Experimental System Type": "system_type",
    "Publication Source": "source",
    "Organism ID Interactor A": "organism_a",
    "Organism ID Interactor B": "organism_b",
}
HUMAN_TAXON = "9606"
PHYSICAL = "physical"
PUBMED_PREFIX = "PUBMED:"
SEPARATOR = "|"
EDGE_KEYS = ["gene_key_a", "gene_key_b"]
EDGE_COLUMNS = [*EDGE_KEYS, "experimental_systems", "pubmed_ids", "source_record_count"]
ORIGINAL_EDGES = Path("methods/epgat/feature/source/network/biogrid_physical_edges.parquet")

CORUM_ORGANISM = "Human"
TERM_ID = "term_id"
GAF_SYMBOL, GAF_QUALIFIER, GAF_TERM, GAF_EVIDENCE, GAF_ASPECT, GAF_SYNONYMS = 2, 3, 4, 6, 8, 10
GAF_PROCESS = "P"
GAF_NOT = "NOT"
COMPLEX_MINIMUM = 5
HUMAN_TAX_ID = 9606
GENE2PUBMED_CHUNK = 5_000_000
EXPRESSION_ROWS = 2048
DEPMAP_LABEL = re.compile(r"^(.+) \((\d+)\)$")
LINEAGE_MINIMUM = 30
LINEAGE_SHARE = 0.5
BREADTH_EDGES = (0.0, 0.01, 0.10, 0.50, 0.90, 1.0)
BREADTH_NAMES = ("0", "0-1%", "1-10%", "10-50%", "50-90%", ">90%")
TOP_LINEAGES = 3

ORTHOLOGY_TYPE = "ortholog_one2one"
ORTHOLOGUE_COLUMNS = {
    "Gene stable ID": "ensembl_gene_id",
    "Mouse gene stable ID": "mouse_gene_id",
    "Mouse gene name": MOUSE_GENE,
    "Mouse homology type": "homology_type",
}
VIABILITY_COLUMNS = {"Gene Accession Id": MGI, "Zygosity": "zygosity", "Viability Phenotype HOMs/HEMIs": "call"}
ZYGOSITY = "homozygote"
VIABILITY_CALLS = ("viable", "lethal", "subviable")
LETHAL = "lethal"
CONFLICTING = "conflicting"
PRE = "before_organogenesis"
ORGANOGENESIS = "during_organogenesis"
LATE = "perinatal_and_after"
EMBRYONIC_UNSTAGED = "embryonic_unstaged"
NO_ASSERTION = "no_lethal_assertion"
AFTER = "after_e12"
UNSTAGED = "unstaged"
STAGE_ROOTS: dict[str, tuple[str, ...]] = {
    PRE: ("MP:0013292",),
    ORGANOGENESIS: ("MP:0006207", "MP:0013293"),
    EMBRYONIC_UNSTAGED: ("MP:0008762",),
    LATE: ("MP:0002080", "MP:0002081", "MP:0002058", "MP:0002082", "MP:0010770"),
}
HEART_SEPTATION = "MP:0013294"
LATE_PROCEDURE = r"E14\.5|E15\.5|E18\.5"
STAGE_MEANINGS = {
    PRE: "embryonic lethal before organogenesis (MP:0013292)",
    ORGANOGENESIS: "embryonic lethal during organogenesis (MP:0006207 or MP:0013293)",
    AFTER: "lethal after E12.5: homozygous embryos recorded at E14.5 to E15.5 or E18.5, or MP:0013294",
    UNSTAGED: "a prenatal to preweaning lethality term but no homozygous embryo record; left out of stage comparisons",
    EMBRYONIC_UNSTAGED: "embryonic lethality (MP:0008762) without a stage",
    NO_ASSERTION: "lethal viability call without a lethality phenotype term",
}

STRATA: dict[str, str] = {
    "hallmark_count": "number of the four hallmarks, 0 to 4, relative to the task's medians",
    "literature_quartile": "quartiles Q1 (least studied) to Q4 of the PubMed count among the task's labelled genes",
    "family_size": "protein family of one gene (no paralogue) or of two genes or more",
    "ppi_degree": "quintiles 1 (fewest partners) to 5 of the BioGRID PPI degree among the task's labelled genes",
    "expression": "quintiles 1 (lowest) to 5 of the median TCGA expression among the task's labelled genes",
}
STRATA_ALIASES = {
    "hallmark": "hallmark_count", "hallmarks": "hallmark_count", "literature": "literature_quartile",
    "pubmed": "literature_quartile", "family": "family_size", "paralogue": "family_size", "ppi": "ppi_degree",
    "degree": "ppi_degree", "tcga": "expression", "tcga_median": "expression",
}
QUINTILES = 5
ASPECTS = ("hallmarks", "dependency", "mouse")
ENRICHMENT_INPUTS = ("gene_reference", "goa_human", "go_basic", "corum")

CHECK_COLUMNS = ["check", "task", "quantity", "compared", "mismatches", "max_abs_difference",
                 "max_relative_difference", "note"]
TOLERANCE = 1e-9


def task_degree(cohort_id: str) -> str:
    return f"{DEGREE}:{cohort_id}"


def lineage_column(name: str) -> str:
    return f"lineage:{name}"


def _note(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@lru_cache(maxsize=1)
def _reference() -> pd.DataFrame:
    frame = pd.read_parquet(
        external.path("gene_reference"), columns=[GENE_KEY, SYMBOL, "entrez_id", "ensembl_gene_id"]
    )
    frame[GENE_KEY] = frame[GENE_KEY].astype(str)
    return frame.reset_index(drop=True)


@lru_cache(maxsize=None)
def _task_genes(cohort_id: str) -> pd.Index:
    return pd.Index(read_cohort_labels(cohort_id)[GENE_KEY].astype(str), name=GENE_KEY)


# BioGRID physical interactions


def _biogrid_table(archive: Path, reference: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    with zipfile.ZipFile(archive) as bundle:
        members = [n for n in bundle.namelist() if BIOGRID_MEMBER in n and n.endswith(BIOGRID_SUFFIX)]
        if len(members) != 1:
            raise ValueError(f"{archive.name}: expected one {BIOGRID_MEMBER} TAB 3.0 file, found {members}")
        with bundle.open(members[0]) as handle:
            rows = pd.read_csv(handle, sep="\t", dtype=str, usecols=list(BIOGRID_COLUMNS), keep_default_na=False)
    rows = rows.rename(columns=BIOGRID_COLUMNS)
    counts = {"total_source_rows": len(rows)}
    rows = rows.loc[
        rows["organism_a"].eq(HUMAN_TAXON)
        & rows["organism_b"].eq(HUMAN_TAXON)
        & rows["system_type"].str.lower().eq(PHYSICAL)
    ]
    counts["human_physical_rows"] = len(rows)
    by_entrez = dict(zip(reference["entrez_id"].astype("int64").astype(str), reference[GENE_KEY]))
    left = rows["entrez_a"].map(by_entrez)
    right = rows["entrez_b"].map(by_entrez)
    mapped = (left.notna() & right.notna()).to_numpy()
    counts["mapped_rows_before_deduplication"] = int(mapped.sum())
    rows, left, right = rows.loc[mapped], left.loc[mapped], right.loc[mapped]
    loop = (left == right).to_numpy()
    counts["self_loops_removed"] = int(loop.sum())
    rows, left, right = rows.loc[~loop], left.loc[~loop].to_numpy(), right.loc[~loop].to_numpy()
    source = rows["source"].str.strip()
    pubmed = np.where(source.str.upper().str.startswith(PUBMED_PREFIX), source.str.split(":", n=1).str[1], source)
    records = pd.DataFrame({
        "gene_key_a": np.where(left < right, left, right),
        "gene_key_b": np.where(left < right, right, left),
        "system": rows["system"].str.strip().to_numpy(),
        "pubmed": pubmed,
    })
    edges = records.groupby(EDGE_KEYS, sort=True).size().rename("source_record_count").reset_index()
    for column, name in (("system", "experimental_systems"), ("pubmed", "pubmed_ids")):
        values = records.loc[records[column] != "", [*EDGE_KEYS, column]].drop_duplicates()
        joined = values.sort_values([*EDGE_KEYS, column]).groupby(EDGE_KEYS, sort=True)[column].agg(SEPARATOR.join)
        edges = edges.merge(joined.rename(name).reset_index(), on=EDGE_KEYS, how="left")
        edges[name] = edges[name].fillna("")
    counts["unique_undirected_edges"] = len(edges)
    edges["source_record_count"] = edges["source_record_count"].astype("int64")
    return edges[EDGE_COLUMNS], counts


def biogrid_edges(output: Path) -> None:
    """Write the BioGRID physical interactions between genes of the gene reference (build ``biogrid_edges``).

    Records of the human TAB 3.0 file with both interactors in Homo sapiens and the experimental system type
    physical, mapped from Entrez to HGNC identifiers through the gene reference, without self-interactions, merged into
    one row per unordered gene pair with its experimental systems, PubMed identifiers and number of records.
    """
    edges, _ = _biogrid_table(external.path("biogrid"), _reference())
    edges.to_parquet(output, index=False)


# Per-gene annotation


def _unique_symbols(reference: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    ambiguous: set[str] = set()
    for symbol, key in zip(reference[SYMBOL].astype(str).str.strip().str.upper(), reference[GENE_KEY]):
        if mapping.get(symbol, key) != key:
            ambiguous.add(symbol)
        mapping.setdefault(symbol, key)
    return {symbol: key for symbol, key in mapping.items() if symbol not in ambiguous}


def _corum_counts(reference: pd.DataFrame) -> pd.Series:
    symbols = _unique_symbols(reference)
    table = pd.read_csv(external.path("corum"), sep="\t", dtype=str, keep_default_na=False,
                        usecols=["complex_id", "organism", "subunits_gene_name"])
    human = table.loc[table["organism"].str.contains(CORUM_ORGANISM, case=False)]
    complexes: dict[str, set[str]] = {}
    for complex_id, names in zip(human["complex_id"], human["subunits_gene_name"]):
        tokens = [name.strip().upper() for name in names.split(";")]
        members = {symbols[token] for token in tokens if token in symbols}
        if members:
            complexes[complex_id] = members
    counts = pd.Series([key for members in complexes.values() for key in members], dtype="object").value_counts()
    return counts.reindex(reference[GENE_KEY]).fillna(0).astype("int64")


def _gene_age(keys: pd.Index) -> pd.DataFrame:
    table = pd.read_parquet(external.path("orthology"))
    table[GENE_KEY] = table[GENE_KEY].astype(str)
    table = table.set_index(GENE_KEY)
    absent = keys.difference(table.index)
    if len(absent):
        raise ValueError(f"{len(absent)} genes of the gene reference have no orthology row")
    layers = species_layers(list(table.columns))
    matrix = table.loc[keys, list(layers["column"])].astype("float64")
    if not np.isin(matrix.to_numpy(), (0.0, 1.0)).all():
        raise ValueError("an orthology column holds a value other than 0 and 1")
    age = np.full(len(keys), YOUNGEST, dtype="int64")
    for layer in sorted(LAYERS, key=lambda item: -item.index):
        if layer.index == YOUNGEST:
            continue
        present = matrix[[SPECIES_PREFIX + name for name in layer.species]].to_numpy().max(axis=1) == 1.0
        age[present] = layer.index
    yeast = matrix[SPECIES_PREFIX + LAYER_BY_INDEX[1].species[0]].to_numpy("int64")
    return pd.DataFrame({YEAST: yeast, AGE: age}, index=keys)


def _ppi_degrees(keys: pd.Index) -> pd.DataFrame:
    edges = pd.read_parquet(external.path("biogrid_edges"), columns=EDGE_KEYS).astype(str)
    edges = edges.loc[edges["gene_key_a"] != edges["gene_key_b"]]
    low = edges["gene_key_a"] < edges["gene_key_b"]
    pairs = pd.DataFrame({
        "a": np.where(low, edges["gene_key_a"], edges["gene_key_b"]),
        "b": np.where(low, edges["gene_key_b"], edges["gene_key_a"]),
    }).drop_duplicates()
    frame = pd.DataFrame(index=keys)
    for cohort_id in TASK_IDS:
        genes = set(_task_genes(cohort_id))
        partners = pd.concat([pairs.loc[pairs["b"].isin(genes), "a"], pairs.loc[pairs["a"].isin(genes), "b"]])
        frame[task_degree(cohort_id)] = partners.value_counts().reindex(keys, fill_value=0).astype("int64")
    return frame


def _expression_medians(keys: pd.Index) -> pd.Series:
    table = pq.read_table(external.path("tcga_expression"))
    genes = pd.Index([str(value) for value in table.column(GENE_KEY).to_pylist()])
    columns = [table.column(name).to_numpy() for name in table.column_names if name != GENE_KEY]
    medians = np.empty(len(genes), dtype="float64")
    for start in range(0, len(genes), EXPRESSION_ROWS):
        values = np.column_stack([column[start:start + EXPRESSION_ROWS] for column in columns]).astype("float64")
        if np.isnan(values).any() or (values < 0).any():
            raise ValueError("the TCGA matrix holds a missing or negative value")
        medians[start:start + len(values)] = np.median(np.log2(values + 1.0), axis=1)
    return pd.Series(medians, index=genes).reindex(keys)


def _families(keys: pd.Index) -> pd.DataFrame:
    families = pd.read_parquet(FAMILIES, columns=[GENE_KEY, "cluster_size"])
    size = families.astype({GENE_KEY: str}).set_index(GENE_KEY)["cluster_size"].reindex(keys)
    edges = pd.read_parquet(PARALOGUE_EDGES, columns=EDGE_KEYS).astype(str)
    low = edges["gene_key_a"] < edges["gene_key_b"]
    pairs = pd.DataFrame({
        "a": np.where(low, edges["gene_key_a"], edges["gene_key_b"]),
        "b": np.where(low, edges["gene_key_b"], edges["gene_key_a"]),
    }).drop_duplicates()
    partners = pd.concat([pairs["a"], pairs["b"]]).value_counts().reindex(keys, fill_value=0)
    return pd.DataFrame({
        FAMILY_SIZE: size.astype("Int64"),
        PARALOGUES: partners.astype("Int64").where(size.notna()),
    }, index=keys)


def _literature(reference: pd.DataFrame) -> pd.Series:
    parts = []
    reader = pd.read_csv(external.path("gene2pubmed"), sep="\t", header=0, names=["tax_id", "gene_id", "pubmed_id"],
                         usecols=[0, 1, 2], dtype="int64", chunksize=GENE2PUBMED_CHUNK)
    for chunk in reader:
        parts.append(chunk.loc[chunk["tax_id"] == HUMAN_TAX_ID, ["gene_id", "pubmed_id"]])
    human = pd.concat(parts, ignore_index=True).drop_duplicates()
    counts = human.groupby("gene_id").size()
    values = counts.reindex(reference["entrez_id"].astype("int64")).fillna(0).astype("int64").to_numpy()
    return pd.Series(values, index=pd.Index(reference[GENE_KEY]))


def _depmap_columns(names: list[str], by_entrez: dict[int, str]) -> dict[str, str]:
    columns = {}
    for name in names:
        match = DEPMAP_LABEL.match(name)
        if match is None:
            raise ValueError(f"DepMap column {name!r} is not 'SYMBOL (ENTREZ)'")
        key = by_entrez.get(int(match.group(2)))
        if key is not None:
            columns[name] = key
    return columns


def _by_entrez(reference: pd.DataFrame) -> dict[int, str]:
    return dict(zip(reference["entrez_id"].astype("int64"), reference[GENE_KEY]))


def _gene_effect(reference: pd.DataFrame) -> pd.DataFrame:
    keys = pd.Index(reference[GENE_KEY])
    table = pv.read_csv(external.path("depmap_gene_effect"))
    columns = _depmap_columns(table.column_names[1:], _by_entrez(reference))
    matrix = pd.DataFrame({name: table.column(name).to_numpy(zero_copy_only=False) for name in columns})
    effect = summarise_gene_effect(matrix, columns)[GENE_EFFECT_COLUMN]
    return effect.rename(EFFECT).reindex(keys).to_frame()


def _dependency(reference: pd.DataFrame) -> pd.DataFrame:
    keys = pd.Index(reference[GENE_KEY])
    table = pv.read_csv(external.path("depmap_gene_dependency"))
    columns = _depmap_columns(table.column_names[1:], _by_entrez(reference))
    dependency = pd.DataFrame({key: table.column(name).to_numpy(zero_copy_only=False) for name, key in columns.items()})
    dependency.index = pd.Index(table.column(table.column_names[0]).to_pylist(), name="ModelID")
    if dependency.columns.has_duplicates:
        raise ValueError("two DepMap dependency columns map to one gene")
    measured = dependency.notna().sum(axis=0)
    frame = pd.DataFrame(index=keys)
    frame[DEPENDENT] = ((dependency > DEPENDENT_ABOVE).sum(axis=0) / measured).reindex(keys)
    frame[MEASURED] = measured.reindex(keys).astype("Int64")
    models = pd.read_csv(external.path("depmap_models"), usecols=["ModelID", "OncotreeLineage"])
    lineage = models.set_index("ModelID")["OncotreeLineage"].reindex(dependency.index)
    if lineage.isna().any():
        raise ValueError(f"{int(lineage.isna().sum())} DepMap models have no lineage")
    sizes = lineage.value_counts()
    counted = list(sizes.index[sizes >= LINEAGE_MINIMUM])
    dependent = (dependency > DEPENDENT_ABOVE).astype("float64").where(dependency.notna())
    shares = dependent.groupby(lineage.to_numpy()).mean().T[counted]
    for name in counted:
        frame[lineage_column(name)] = shares[name].reindex(keys)
    return frame


def _ontology_children(path: Path) -> dict[str, set[str]]:
    parents: dict[str, list[str]] = {}
    current: dict[str, object] | None = None

    def close() -> None:
        if current is not None and "id" in current and not current.get("obsolete"):
            parents[str(current["id"])] = list(current.get("is_a", []))

    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line == "[Term]":
            close()
            current = {}
        elif line.startswith("[") and line.endswith("]"):
            close()
            current = None
        elif current is None:
            continue
        elif line.startswith("id: "):
            current["id"] = line[4:]
        elif line.startswith("is_a: "):
            current.setdefault("is_a", []).append(line[6:].split("!")[0].strip())
        elif line == "is_obsolete: true":
            current["obsolete"] = True
    close()
    children: dict[str, set[str]] = {}
    for term, above in parents.items():
        for parent in above:
            children.setdefault(parent, set()).add(term)
    return children


def _closure(roots: tuple[str, ...], children: dict[str, set[str]]) -> set[str]:
    found, stack = set(roots), list(roots)
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


def _lethal_stages(gene_of_mgi: dict[str, str]) -> pd.Series:
    children = _ontology_children(external.path("mp_ontology"))
    with gzip.open(external.path("impc_genotype_phenotype"), "rt", errors="replace") as handle:
        records = pd.read_csv(handle, usecols=["marker_accession_id", "zygosity", "procedure_name", "mp_term_id"],
                              low_memory=False)
    records = records.loc[records["zygosity"].astype("string").str.lower() == ZYGOSITY]
    records = records.assign(gene=records["marker_accession_id"].map(gene_of_mgi)).dropna(subset=["gene"])
    termed = records.dropna(subset=["mp_term_id"])
    flagged = {
        band: set(termed.loc[termed["mp_term_id"].isin(_closure(roots, children)), "gene"])
        for band, roots in STAGE_ROOTS.items()
    }
    late = records["procedure_name"].astype(str).str.contains(LATE_PROCEDURE, regex=True)
    embryos = set(records.loc[late | records["mp_term_id"].eq(HEART_SEPTATION), "gene"])
    stages = {}
    for gene in gene_of_mgi.values():
        band = next((name for name in (PRE, ORGANOGENESIS, EMBRYONIC_UNSTAGED, LATE) if gene in flagged[name]),
                    NO_ASSERTION)
        if band == LATE:
            band = AFTER if gene in embryos else UNSTAGED
        stages[gene] = band
    return pd.Series(stages, dtype="object")


def _mouse(reference: pd.DataFrame) -> pd.DataFrame:
    orthologues = pd.read_csv(external.path("ensembl_human_mouse"), sep="\t", dtype=str).fillna("")
    orthologues = orthologues.rename(columns=ORTHOLOGUE_COLUMNS)[list(ORTHOLOGUE_COLUMNS.values())]
    orthologues = orthologues.loc[orthologues["homology_type"].eq(ORTHOLOGY_TYPE)].drop(columns="homology_type")
    if not (orthologues["ensembl_gene_id"].is_unique and orthologues["mouse_gene_id"].is_unique):
        raise ValueError("the one-to-one human-mouse orthologues are not one to one")
    mgi = pd.read_csv(external.path("ensembl_mouse_mgi"), sep="\t", dtype=str).fillna("")
    mgi = mgi.rename(columns={"Gene stable ID": "mouse_gene_id", "MGI ID": MGI})[["mouse_gene_id", MGI]]
    mgi = mgi.loc[mgi[MGI].ne("")]
    if not mgi["mouse_gene_id"].is_unique:
        raise ValueError("a mouse gene carries more than one MGI accession")
    frame = reference[[GENE_KEY, "ensembl_gene_id"]].merge(orthologues, on="ensembl_gene_id", how="left",
                                                           validate="many_to_one")
    frame = frame.merge(mgi, on="mouse_gene_id", how="left", validate="many_to_one").set_index(GENE_KEY)
    calls = pd.read_csv(external.path("impc_viability"), usecols=list(VIABILITY_COLUMNS)).rename(columns=VIABILITY_COLUMNS)
    calls = calls.loc[calls["zygosity"].eq(ZYGOSITY)]
    unknown = set(calls["call"]) - set(VIABILITY_CALLS)
    if unknown:
        raise ValueError(f"the IMPC viability table carries unknown calls: {sorted(unknown)}")
    counts = calls.groupby([MGI, "call"]).size().unstack(fill_value=0).reindex(columns=list(VIABILITY_CALLS),
                                                                                 fill_value=0)
    verdict = counts.idxmax(axis=1)
    verdict[counts.eq(counts.max(axis=1), axis=0).sum(axis=1) > 1] = CONFLICTING
    named = frame[MGI].where(frame[MGI].notna() & frame["mouse_gene_id"].notna())
    result = pd.DataFrame(index=frame.index)
    result[MOUSE_GENE] = frame[MOUSE_GENE].where(frame["mouse_gene_id"].notna() & frame[MOUSE_GENE].ne(""))
    result[MGI] = named
    result[VIABILITY] = verdict.reindex(named).to_numpy()
    lethal = result.loc[result[VIABILITY].eq(LETHAL), MGI]
    stages = _lethal_stages(dict(zip(result[MGI].dropna(), result[MGI].dropna().index)))
    result[STAGE] = stages.reindex(lethal.index).reindex(result.index)
    return result


def _write_parts(output: Path, label: str, parts: tuple) -> None:
    started = time.monotonic()
    frame = pd.DataFrame(index=pd.Index(_reference()[GENE_KEY], name=GENE_KEY))
    for done, (name, part) in enumerate(parts, start=1):
        frame = frame.join(part(), how="left")
        _note(f"{label}: {done}/{len(parts)} parts done ({name}), {time.monotonic() - started:.0f} s")
    frame.reset_index().to_parquet(output, index=False)


def annotation_table(output: Path) -> None:
    """Write the gene properties of the hallmarks and strata, one row per gene of the gene reference (``annotation``)."""
    reference = _reference()
    keys = pd.Index(reference[GENE_KEY], name=GENE_KEY)
    _write_parts(output, ANNOTATION, (
        ("symbols", lambda: reference.set_index(GENE_KEY)[[SYMBOL]]),
        ("CORUM complexes", lambda: _corum_counts(reference).rename(COMPLEXES).to_frame()),
        ("gene age", lambda: _gene_age(keys)),
        ("PPI degree", lambda: _ppi_degrees(keys)),
        ("TCGA expression", lambda: _expression_medians(keys).rename(EXPRESSION).to_frame()),
        ("protein families", lambda: _families(keys)),
        ("literature", lambda: _literature(reference).rename(LITERATURE).to_frame()),
    ))


def dependency_table(output: Path) -> None:
    """Write the DepMap 26Q1 dependency of each gene of the gene reference (build ``annotation_dependency``)."""
    reference = _reference()
    _write_parts(output, DEPENDENCY_TABLE, (
        ("gene effect", lambda: _gene_effect(reference)),
        ("dependency probability", lambda: _dependency(reference)),
    ))


def mouse_table(output: Path) -> None:
    """Write the IMPC outcome of the mouse knockout of each gene of the gene reference (build ``annotation_mouse``)."""
    reference = _reference()
    _write_parts(output, MOUSE_TABLE, (("mouse knockout", lambda: _mouse(reference)),))


# Shared by the tools


@lru_cache(maxsize=None)
def _table(item_id: str) -> pd.DataFrame:
    (path,) = external.ensure(item_id)
    frame = pd.read_parquet(path)
    frame[GENE_KEY] = frame[GENE_KEY].astype(str)
    return frame.set_index(GENE_KEY)


def _annotation() -> pd.DataFrame:
    return _table(ANNOTATION)


@lru_cache(maxsize=1)
def _symbols() -> pd.Series:
    external.ensure("gene_reference")
    return _reference().set_index(GENE_KEY)[SYMBOL]


@lru_cache(maxsize=1)
def _depmap_classes() -> pd.Series:
    universe = pd.read_parquet(DEPMAP_UNIVERSE, columns=[GENE_KEY, "official_common_essential", "strongly_selective"])
    names = [
        "common essential" if common == 1 else "strongly selective" if selective == 1 else "neither"
        for common, selective in zip(universe["official_common_essential"], universe["strongly_selective"])
    ]
    return pd.Series(names, index=pd.Index(universe[GENE_KEY].astype(str), name=GENE_KEY), dtype="object")


@lru_cache(maxsize=None)
def _labels(cohort_id: str) -> pd.Series:
    frame = read_cohort_labels(cohort_id)
    target = cohort_target(cohort_id).column
    return pd.Series(frame[target].to_numpy("float64"), index=pd.Index(frame[GENE_KEY].astype(str), name=GENE_KEY))


def _binary(cohort_id: str) -> bool:
    return cohort_target(cohort_id).kind is TargetKind.BINARY


class TaskReference(NamedTuple):
    cohort_id: str
    labels: pd.Series
    ppi_median: float
    tcga_median: float
    literature_edges: np.ndarray


@lru_cache(maxsize=None)
def _task_reference(cohort_id: str) -> TaskReference:
    labels = _labels(cohort_id)
    rows = _annotation().loc[labels.index]
    expression = rows[EXPRESSION].astype("float64")
    if expression.isna().any():
        raise ValueError(f"{int(expression.isna().sum())} labelled genes of {cohort_id} have no TCGA expression")
    return TaskReference(
        cohort_id,
        labels,
        float(np.median(rows[task_degree(cohort_id)].astype("float64"))),
        float(np.median(expression)),
        quartile_edges(rows[LITERATURE].astype("float64")),
    )


def _hallmark_frame(rows: pd.DataFrame, reference: TaskReference) -> pd.DataFrame:
    flags = pd.DataFrame({
        "complex": rows[COMPLEXES].astype("float64") >= 1,
        "old": rows[YEAST].astype("float64") == 1,
        "connected": rows[task_degree(reference.cohort_id)].astype("float64") >= reference.ppi_median,
        "expressed": rows[EXPRESSION].astype("float64") >= reference.tcga_median,
    }, index=rows.index).astype("int64")
    flags[COUNT] = flags[list(HALLMARKS)].sum(axis=1)
    return flags


def _literature_bands(values: pd.Series, edges: np.ndarray) -> pd.Series:
    position = np.searchsorted(edges, values.astype("float64").to_numpy(), side="left")
    return pd.Series(np.asarray(QUARTILE_BANDS, dtype="object")[position], index=values.index).str.upper()


def _quintiles(values: pd.Series) -> tuple[pd.Series, np.ndarray]:
    edges = np.quantile(values.to_numpy("float64"), np.arange(1, QUINTILES) / QUINTILES)
    groups = np.searchsorted(edges, values.to_numpy("float64"), side="left") + 1
    return pd.Series(groups.astype(str), index=values.index), edges


def _plain(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return benchmark.number(value)
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [{column: _plain(value) for column, value in row.items()} for row in frame.to_dict("records")]


def _share(mask: pd.Series) -> float | None:
    return benchmark.number(mask.mean()) if len(mask) else None


def _gene_column(table: pd.DataFrame, file: Path) -> str:
    lowered = {str(column).lower(): column for column in table.columns}
    for name in GENE_COLUMNS:
        if name in lowered:
            return lowered[name]
    if len(table.columns) == 1:
        return table.columns[0]
    raise ValueError(f"{file}: no gene column; name one of {', '.join(GENE_COLUMNS)}, the columns are "
                     f"{list(table.columns)}")


def _gene_tokens(genes: str) -> tuple[list[str], str]:
    text = str(genes).strip()
    if not text:
        raise ValueError("no genes given; pass comma-separated symbols such as POLR2A,RPL3 or the path of a gene table")
    candidate = Path(text).expanduser()
    candidate = candidate if candidate.is_absolute() else ROOT / candidate
    if "," not in text and (candidate.is_file() or candidate.suffix.lower() in TABLE_SUFFIXES):
        file, table = benchmark.read_table(text)
        column = _gene_column(table, file)
        return [str(value) for value in table[column].dropna()], file.stem
    return [token for token in re.split(r"[,;\s]+", text) if token], "genes"


def _resolve(tokens: list[str], symbols: pd.Series) -> tuple[list[str], list[str]]:
    by_symbol = {str(symbol).upper(): key for key, symbol in symbols.items()}
    known = set(symbols.index)
    resolved: dict[str, None] = {}
    unresolved: dict[str, None] = {}
    for token in tokens:
        text = token.strip()
        if text.upper().startswith("HGNC:"):
            key = "HGNC:" + text.split(":", 1)[1].strip()
            key = key if key in known else None
        else:
            key = by_symbol.get(text.upper())
        if key is None:
            unresolved.setdefault(text)
        else:
            resolved.setdefault(key)
    return list(resolved), list(unresolved)


def _unresolved_note(unresolved: list[str]) -> dict:
    return {
        "count": len(unresolved),
        "genes": unresolved[:SHOWN_ROWS * 2],
        "meaning": "not an HGNC symbol or identifier of the 19,158 protein-coding genes of the gene reference",
    }


def _written(frame: pd.DataFrame, name: str) -> str:
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"{name}.csv"
    frame.to_csv(path, index=False)
    return str(path)


# annotate_genes


def _hallmark_view(rows: pd.DataFrame, reference: TaskReference) -> pd.DataFrame:
    flags = _hallmark_frame(rows, reference)
    view = pd.DataFrame(index=rows.index)
    view[SYMBOL] = rows[SYMBOL]
    view["in_task"] = rows.index.isin(reference.labels.index)
    label = reference.labels.reindex(rows.index)
    view["task_label"] = label.astype("Int64") if _binary(reference.cohort_id) else label
    view["complex"] = flags["complex"]
    view[COMPLEXES] = rows[COMPLEXES]
    view["old"] = flags["old"]
    view["connected"] = flags["connected"]
    view[DEGREE] = rows[task_degree(reference.cohort_id)]
    view["expressed"] = flags["expressed"]
    view[EXPRESSION] = rows[EXPRESSION]
    view[COUNT] = flags[COUNT]
    view[AGE] = rows[AGE]
    view["age_layer_name"] = rows[AGE].map(lambda index: LAYER_BY_INDEX[int(index)].name)
    view[PARALOGUES] = rows[PARALOGUES]
    view[FAMILY_SIZE] = rows[FAMILY_SIZE]
    view[LITERATURE] = rows[LITERATURE]
    view["literature_quartile"] = _literature_bands(rows[LITERATURE], reference.literature_edges)
    return view


def _hallmark_summary(view: pd.DataFrame, task_view: pd.DataFrame, reference: TaskReference) -> dict:
    labels = reference.labels
    counts = []
    for k in range(len(HALLMARKS) + 1):
        entry = {"hallmarks": k, "list_share": _share(view[COUNT] == k), "task_share": _share(task_view[COUNT] == k)}
        if _binary(reference.cohort_id):
            entry["task_essential_share"] = _share(labels[task_view[COUNT] == k] == 1)
        counts.append(entry)
    layers = [
        {"layer": layer.index, "name": layer.name, "list_share": _share(view[AGE] == layer.index),
         "task_share": _share(task_view[AGE] == layer.index)}
        for layer in LAYERS
    ]
    literature = task_view[LITERATURE].astype("float64")
    ranges = literature.groupby(task_view["literature_quartile"]).agg(["min", "max"])
    return {
        "thresholds": {
            "ppi_degree_median": benchmark.number(reference.ppi_median),
            "tcga_median_log2": benchmark.number(reference.tcga_median),
            "pubmed_count_of_task_quartiles": {band: f"{row['min']:.0f} to {row['max']:.0f}"
                                               for band, row in ranges.iterrows()},
        },
        "hallmark_count": counts,
        "three_or_more_hallmarks": {"list": _share(view[COUNT] >= RICH), "task": _share(task_view[COUNT] >= RICH)},
        "each_hallmark": {h: {"list": _share(view[h] == 1), "task": _share(task_view[h] == 1)} for h in HALLMARKS},
        "age_layers": layers,
        "with_paralogue": {"list": _share(view[PARALOGUES].astype("float64") >= 1),
                           "task": _share(task_view[PARALOGUES].astype("float64") >= 1)},
        "pubmed_count_median": {"list": benchmark.number(view[LITERATURE].astype("float64").median()),
                                "task": benchmark.number(task_view[LITERATURE].astype("float64").median())},
        "literature_quartile_list_share": {band.upper(): _share(view["literature_quartile"] == band.upper())
                                           for band in QUARTILE_BANDS},
    }


def _lineage_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in table.columns if column.startswith("lineage:")]


def _breadth(shares: pd.Series) -> pd.Series:
    values = shares.astype("float64")
    position = np.searchsorted(np.asarray(BREADTH_EDGES), values.fillna(0.0).to_numpy(), side="left")
    return pd.Series(np.asarray(BREADTH_NAMES, dtype="object")[position], index=shares.index).where(values.notna())


def _top_lineages(shares: pd.DataFrame) -> pd.Series:
    names = [column.split(":", 1)[1] for column in shares.columns]
    texts = []
    for values in shares.to_numpy("float64"):
        order = [i for i in np.argsort(-np.nan_to_num(values, nan=-1.0), kind="stable") if values[i] > 0]
        texts.append(", ".join(f"{names[i]} {values[i]:.2f}" for i in order[:TOP_LINEAGES]) or None)
    return pd.Series(texts, index=shares.index, dtype="object")


def _dependency_view(rows: pd.DataFrame, symbols: pd.Series) -> pd.DataFrame:
    shares = rows[_lineage_columns(rows)].astype("float64")
    view = pd.DataFrame(index=rows.index)
    view[SYMBOL] = symbols.reindex(rows.index)
    view[EFFECT] = rows[EFFECT]
    view[DEPENDENT] = rows[DEPENDENT]
    view[MEASURED] = rows[MEASURED]
    view["dependency_breadth"] = _breadth(rows[DEPENDENT])
    view[DEPMAP_CLASS] = _depmap_classes().reindex(rows.index)
    view["lineages_most_dependent"] = _top_lineages(shares)
    view["lineages_at_least_half"] = (shares >= LINEAGE_SHARE).sum(axis=1).astype("Int64").where(
        shares.notna().any(axis=1))
    return pd.concat([view, shares], axis=1)


def _dependency_summary(view: pd.DataFrame, task_view: pd.DataFrame) -> dict:
    lineages = [column for column in view.columns if column.startswith("lineage:")]
    measured, task_measured = view.dropna(subset=[DEPENDENT]), task_view.dropna(subset=[DEPENDENT])
    means = measured[lineages].mean().sort_values(ascending=False, kind="stable")
    task_means = task_measured[lineages].mean()
    return {
        "measured_in_depmap": int(len(measured)),
        "gene_effect_median": {"list": benchmark.number(measured[EFFECT].median()),
                               "task": benchmark.number(task_measured[EFFECT].median())},
        "dependency_breadth": [
            {"band": band, "list_share": _share(measured["dependency_breadth"] == band),
             "task_share": _share(task_measured["dependency_breadth"] == band)}
            for band in BREADTH_NAMES
        ],
        "depmap_class": {str(name): int(size) for name, size in view[DEPMAP_CLASS].value_counts().items()},
        "lineages_most_dependent": [
            {"lineage": column.split(":", 1)[1], "list_mean_share": benchmark.number(means[column]),
             "task_mean_share": benchmark.number(task_means[column])}
            for column in means.index[:5]
        ],
    }


def _mouse_view(rows: pd.DataFrame, symbols: pd.Series) -> pd.DataFrame:
    view = pd.DataFrame(index=rows.index)
    view[SYMBOL] = symbols.reindex(rows.index)
    view[MOUSE_GENE] = rows[MOUSE_GENE]
    view[MGI] = rows[MGI]
    call = rows[VIABILITY].astype("object")
    call = call.where(call.notna(), np.where(rows[MOUSE_GENE].notna(), "not phenotyped", "no one-to-one orthologue"))
    view[VIABILITY] = call
    view[STAGE] = rows[STAGE]
    return view


def _mouse_summary(view: pd.DataFrame, task_view: pd.DataFrame) -> dict:
    called = task_view.loc[task_view[VIABILITY].isin(["viable", LETHAL])]
    lethal = task_view.loc[task_view[VIABILITY].eq(LETHAL)]
    return {
        "impc_viability": {str(name): int(size) for name, size in view[VIABILITY].value_counts().items()},
        "lethal_stage": {str(name): int(size) for name, size in view[STAGE].value_counts().items()},
        "task_lethal_share_of_viable_or_lethal": _share(called[VIABILITY] == LETHAL),
        "task_lethal_stage_share": {str(name): benchmark.number(share)
                                    for name, share in lethal[STAGE].value_counts(normalize=True).items()},
        "stage_meanings": STAGE_MEANINGS,
    }


def annotate_genes(
    genes: Annotated[
        str,
        Field(description="comma-separated HGNC symbols or identifiers such as POLR2A,RPL3,HGNC:9187, or the path "
                          "of a CSV, TSV or parquet file with a column gene_key, gene_symbol, symbol or gene"),
    ],
    aspect: Annotated[
        str, Field(description="hallmarks (default), dependency or mouse; see the tool description")
    ] = "hallmarks",
    task: Annotated[
        str,
        Field(description="reference task whose labelled genes set the medians and quartiles and are the comparison "
                          "group, for example deg (default), depmap or mouse"),
    ] = "deg",
) -> dict:
    """Annotate a list of genes with the gene properties the manuscript relates to essentiality.

    aspect=hallmarks: for each gene the four hallmarks of essential genes, each 1 or 0: complex (member of at least
    one CORUM complex), old (has a Saccharomyces cerevisiae orthologue), connected (BioGRID physical interaction
    partners among the task's labelled genes at or above the task median) and expressed (median TCGA tumour
    expression, log2(x + 1), at or above the task median; genes absent from TCGA count as 0); hallmark_count, 0 to 4;
    gene age layer, 1 (Opisthokonta, oldest) to 9 (Eutheria, youngest); number of paralogues and protein family size;
    PubMed literature count and its quartile Q1 (least studied) to Q4 among the task's labelled genes; and the gene's
    label in the task (1 essential, 0 not; empty when not a task gene). The summary compares the list with all
    labelled genes of the task, including the share of essential genes among task genes with each hallmark count.
    aspect=dependency: DepMap 26Q1 mean gene effect (more negative is a stronger dependency), share of the measured
    cancer cell lines that depend on the gene (dependency probability above 0.5) and its band (0, 0-1%, 1-10%,
    10-50%, 50-90%, >90%), DepMap's class (common essential, strongly selective, neither), and the lineages with the
    highest share of dependent cell lines among the 18 lineages with at least 30 cell lines.
    aspect=mouse: the one-to-one mouse orthologue, the IMPC call for its homozygous knockout (viable, lethal,
    subviable, conflicting, not phenotyped, no one-to-one orthologue) and, for lethal knockouts, the stage of death.
    Returns one record per gene, a summary of the list next to the task's genes, and the genes not found. Lists of
    more than 20 genes are written in full to a CSV, whose path is returned with the first 10 records. Each aspect
    reads its own table: hallmarks needs no DepMap or IMPC file, dependency needs the DepMap 26Q1 files and mouse the
    IMPC and Ensembl files; when one is missing, the reply names what to download.
    """
    cohort_id = benchmark.task_id(task)
    choice = str(aspect).strip().lower()
    if choice not in ASPECTS:
        raise ValueError(f"unknown aspect {aspect!r}; the aspects are {', '.join(ASPECTS)}")
    tokens, stem = _gene_tokens(genes)
    if choice == "hallmarks":
        table = _annotation()
        symbols = table[SYMBOL]
    else:
        table = _table(DEPENDENCY_TABLE if choice == "dependency" else MOUSE_TABLE)
        symbols = _symbols()
    keys, unresolved = _resolve(tokens, symbols)
    if not keys:
        raise ValueError(f"none of the {len(tokens)} genes is a gene of the gene reference: "
                         f"{', '.join(unresolved[:SHOWN_ROWS])}")
    labels = _labels(cohort_id)
    rows, task_rows = table.loc[keys], table.loc[labels.index]
    if choice == "hallmarks":
        reference = _task_reference(cohort_id)
        view, task_view = _hallmark_view(rows, reference), _hallmark_view(task_rows, reference)
        summary = _hallmark_summary(view, task_view, reference)
    elif choice == "dependency":
        view, task_view = _dependency_view(rows, symbols), _dependency_view(task_rows, symbols)
        summary = _dependency_summary(view, task_view)
    else:
        view, task_view = _mouse_view(rows, symbols), _mouse_view(task_rows, symbols)
        summary = _mouse_summary(view, task_view)
    view = view.rename_axis(GENE_KEY).reset_index()
    shown = view.drop(columns=_lineage_columns(view))
    result: dict[str, object] = {
        "task": cohort_id,
        "aspect": choice,
        "genes": len(keys),
        "in_task": int(view[GENE_KEY].isin(labels.index).sum()),
        "summary": summary,
    }
    if len(keys) > INLINE_GENES:
        result["csv"] = _written(view, f"{stem}_{choice}_{cohort_id}")
        result["records_shown"] = f"the first {SHOWN_ROWS} of {len(keys)}; all are in the CSV"
        result["records"] = _records(shown.head(SHOWN_ROWS))
    else:
        result["records"] = _records(shown)
    if unresolved:
        result["unresolved"] = _unresolved_note(unresolved)
    return result


# enrichment


def _pair_table(rows: list[tuple[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=[TERM_ID, GENE_KEY]).astype("object")
    return frame.drop_duplicates().reset_index(drop=True)


@lru_cache(maxsize=1)
def _process_pairs() -> pd.DataFrame:
    (gaf,) = external.ensure("goa_human")
    mapping = symbol_to_key(ROOT)
    rows = []
    with gzip.open(gaf, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= GAF_SYNONYMS or fields[GAF_ASPECT] != GAF_PROCESS:
                continue
            if fields[GAF_EVIDENCE] in EXCLUDED_EVIDENCE or GAF_NOT in fields[GAF_QUALIFIER].split("|"):
                continue
            key = mapping.get(fields[GAF_SYMBOL])
            if key is None:
                key = next((mapping[s.strip()] for s in fields[GAF_SYNONYMS].split("|") if s.strip() in mapping), None)
            if key is not None:
                rows.append((fields[GAF_TERM], key))
    return _pair_table(rows)


@lru_cache(maxsize=1)
def _process_names() -> dict[str, str]:
    (obo,) = external.ensure("go_basic")
    names: dict[str, str] = {}
    alternatives: dict[str, str] = {}
    current = None
    for line in obo.read_text(encoding="utf-8").splitlines():
        if line == "[Term]":
            current = None
        elif line.startswith("id: GO:"):
            current = line[4:].strip()
        elif line.startswith("name: ") and current is not None:
            names[current] = line[6:].strip()
        elif line.startswith("alt_id: GO:") and current is not None:
            alternatives[line[8:].strip()] = current
    for alternative, primary in alternatives.items():
        names.setdefault(alternative, names.get(primary, ""))
    return names


@lru_cache(maxsize=1)
def _complexes() -> tuple[pd.DataFrame, dict[str, str]]:
    (path,) = external.ensure("corum")
    mapping = symbol_to_key(ROOT)
    table = pd.read_csv(path, sep="\t", usecols=["complex_id", "complex_name", "organism", "subunits_gene_name"],
                        dtype="string")
    table = table.loc[table["organism"] == CORUM_ORGANISM]
    rows = [
        (f"CORUM:{complex_id}", mapping[name.strip()])
        for complex_id, names in zip(table["complex_id"], table["subunits_gene_name"].fillna(""))
        for name in str(names).split(";")
        if name.strip() in mapping
    ]
    named = table.drop_duplicates("complex_id")
    names = {f"CORUM:{item}": str(name) for item, name in zip(named["complex_id"], named["complex_name"].fillna(""))}
    return _pair_table(rows), names


def _enrichment_table(pairs: pd.DataFrame, names: dict[str, str], inside: pd.Series, minimum: int,
                      maximum: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    terms = restrict_terms(pairs, inside.index, minimum=minimum, maximum=maximum)
    total, drawn = int(len(inside)), int(inside.sum())
    rows = []
    for term, members in terms.groupby(TERM_ID):
        member = inside.reindex(members[GENE_KEY]).to_numpy("float64")
        size, hits = int(len(member)), int(member.sum())
        rows.append({TERM_ID: term, "term_name": names.get(term, ""), "genes": size, "positives": hits,
                     "fold_enrichment": (hits / size) / (drawn / total),
                     "p": float(hypergeom.sf(hits - 1, total, drawn, size))})
    table = pd.DataFrame(rows, columns=[TERM_ID, "term_name", "genes", "positives", "fold_enrichment", "p"])
    if table.empty:
        return table.assign(q=[], rank=[]), terms
    table["q"] = benjamini_hochberg_adjusted(table["p"].to_numpy("float64"))
    table = table.sort_values(["q", "p", TERM_ID], kind="stable").reset_index(drop=True)
    table.insert(1, "rank", np.arange(1, len(table) + 1))
    return table, terms


def _term_summary(table: pd.DataFrame, terms: pd.DataFrame, inside: pd.Series, symbols: pd.Series) -> dict:
    hits = set(inside.index[inside.to_numpy()])
    top = []
    for row in table.head(TOP_TERMS).itertuples(index=False):
        members = terms.loc[terms[TERM_ID].eq(row.term_id), GENE_KEY]
        found = sorted(symbols.reindex([key for key in members if key in hits]).dropna().astype(str))
        top.append({
            "term_id": row.term_id, "name": row.term_name, "background_genes": int(row.genes),
            "list_genes": int(row.positives), "fold_enrichment": benchmark.number(row.fold_enrichment),
            "p": benchmark.p_value(row.p), "q": benchmark.p_value(row.q),
            "genes": ", ".join(found[:SHOWN_SYMBOLS]) + (", ..." if len(found) > SHOWN_SYMBOLS else ""),
        })
    return {"tested": int(len(table)), f"q_below_{Q_LEVEL}": int((table["q"] < Q_LEVEL).sum()) if len(table) else 0,
            "top": top}


def enrichment(
    genes: Annotated[
        str,
        Field(description="comma-separated HGNC symbols or identifiers, or the path of a CSV, TSV or parquet file "
                          "with a column gene_key, gene_symbol, symbol or gene"),
    ],
    background: Annotated[
        str,
        Field(description="background genes in the same forms; empty (default) for the labelled genes of the task"),
    ] = "",
    task: Annotated[
        str, Field(description="task whose labelled genes form the default background, for example deg or depmap")
    ] = "deg",
) -> dict:
    """Test which GO biological processes and CORUM protein complexes are over-represented in a gene list.

    As in the manuscript: GO biological process annotations of the GOA human file without IEA evidence or NOT
    qualifiers, direct annotations only, tested when the term has 5 to 200 genes in the background; CORUM human
    complexes tested when they have at least 5 genes in the background. Each term is tested by the upper tail of
    the hypergeometric distribution (list genes in the background drawn from the background), and q values are
    Benjamini-Hochberg adjusted over all terms tested, separately for GO and for CORUM. List genes outside the
    background are left out and listed. Returns for each source the number of terms tested, how many have q below
    0.05, and the 15 terms with the lowest q (background_genes, list_genes, fold_enrichment, p, q and the list genes
    in the term); the full tables are written to a CSV whose path is returned.
    """
    cohort_id = benchmark.task_id(task)
    external.ensure(*ENRICHMENT_INPUTS)
    symbols = _reference().set_index(GENE_KEY)[SYMBOL]
    tokens, stem = _gene_tokens(genes)
    keys, unresolved = _resolve(tokens, symbols)
    if str(background).strip():
        background_tokens, _ = _gene_tokens(background)
        universe, unresolved_background = _resolve(background_tokens, symbols)
        described = f"{len(universe)} genes given as background"
    else:
        universe, unresolved_background = list(_task_genes(cohort_id)), []
        described = f"the {len(_task_genes(cohort_id))} labelled genes of {benchmark.TASKS[cohort_id].name}"
    universe_index = pd.Index(universe, name=GENE_KEY)
    inside = pd.Series(universe_index.isin(keys), index=universe_index)
    outside = [key for key in keys if key not in universe_index]
    if not inside.any():
        raise ValueError(f"none of the {len(keys)} resolved genes is in the background ({described})")
    process, terms = _enrichment_table(_process_pairs(), _process_names(), inside, TERM_MIN, TERM_MAX)
    complexes_pairs, complex_names = _complexes()
    corum, members = _enrichment_table(complexes_pairs, complex_names, inside, COMPLEX_MINIMUM, float("inf"))
    full = pd.concat([process.assign(source="GO biological process"), corum.assign(source="CORUM complex")],
                     ignore_index=True)
    result: dict[str, object] = {
        "task": cohort_id,
        "genes": len(keys),
        "in_background": int(inside.sum()),
        "background": described,
        "go_biological_process": _term_summary(process, terms, inside, symbols),
        "corum_complexes": _term_summary(corum, members, inside, symbols),
        "csv": _written(full, f"{stem}_enrichment_{cohort_id}"),
    }
    if outside:
        result["outside_background"] = {"count": len(outside), "genes": [benchmark.named(k) for k in outside[:20]]}
    if unresolved or unresolved_background:
        result["unresolved"] = _unresolved_note(unresolved + unresolved_background)
    return result


# evaluate_by_strata


def _strata_name(by: str) -> str:
    key = str(by).strip().lower().replace(" ", "_").replace("-", "_")
    key = STRATA_ALIASES.get(key, key)
    if key not in STRATA:
        raise ValueError(f"unknown stratification {by!r}; choose one of {', '.join(STRATA)}")
    return key


def _strata(cohort_id: str, by: str) -> tuple[pd.Series, list[dict]]:
    reference = _task_reference(cohort_id)
    rows = _annotation().loc[reference.labels.index]
    if by == "hallmark_count":
        values = _hallmark_frame(rows, reference)[COUNT]
        groups = values.astype(str)
        names = [str(k) for k in range(len(HALLMARKS) + 1)]
    elif by == "literature_quartile":
        values = rows[LITERATURE].astype("float64")
        groups = _literature_bands(values, reference.literature_edges)
        names = [band.upper() for band in QUARTILE_BANDS]
    elif by == "family_size":
        values = rows[FAMILY_SIZE].astype("float64")
        groups = pd.Series(np.where(values >= 2, "2 or more", "1"), index=rows.index)
        names = ["1", "2 or more"]
    else:
        values = rows[task_degree(cohort_id) if by == "ppi_degree" else EXPRESSION].astype("float64")
        groups, _ = _quintiles(values)
        names = [str(k) for k in range(1, QUINTILES + 1)]
    definitions = [
        {"stratum": name, "lowest": benchmark.number(values[groups == name].min()),
         "highest": benchmark.number(values[groups == name].max())}
        for name in names
    ]
    return groups, definitions


def _strata_frame(cohort_id: str, submission: pd.DataFrame | None) -> pd.DataFrame:
    published = benchmark.published_predictions(prediction_cohort(cohort_id))
    target = cohort_target(cohort_id).column
    parts = []
    for outer_fold in range(FOLD_COUNT):
        truth = benchmark.truth(cohort_id, outer_fold)
        keys = pd.Index(truth[GENE_KEY].astype(str))
        scores = published.loc[published[FOLD] == outer_fold].astype({GENE_KEY: str}).set_index(GENE_KEY)
        scores = scores[list(METHOD_IDS)].reindex(keys)
        if submission is not None:
            own = submission.loc[submission[FOLD] == outer_fold].set_index(GENE_KEY)[SCORE]
            scores.insert(0, SUBMISSION, own.reindex(keys).to_numpy("float64"))
        if scores.isna().any().any():
            raise ValueError(f"fold {outer_fold}: a test gene of {cohort_id} has no score")
        part = pd.DataFrame({GENE_KEY: keys, FOLD: outer_fold, "label": truth[target].to_numpy("int64")})
        for column in scores.columns:
            part[column] = average_ranks(scores[column].to_numpy("float64")) / float(len(keys))
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    kept = scope_gene_keys(cohort_id, primary_scope(cohort_id))
    if kept is not None:
        frame = frame.loc[frame[GENE_KEY].isin(set(map(str, kept)))].reset_index(drop=True)
    return frame


class _Ranking(NamedTuple):
    order: np.ndarray
    labels: np.ndarray
    group: np.ndarray
    segment: np.ndarray
    starts: np.ndarray


def _ranking(scores: np.ndarray, labels: np.ndarray) -> _Ranking:
    order = np.argsort(scores.T, axis=1, kind="stable")
    ordered = np.take_along_axis(scores.T, order, axis=1)
    new = np.ones(ordered.shape, dtype=bool)
    new[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    group = (np.cumsum(new.ravel()) - 1).reshape(ordered.shape)
    segment = np.repeat(np.arange(ordered.shape[0]), new.sum(axis=1))
    return _Ranking(order, labels[order], group.ravel(), segment, group[:, 0])


def _discrimination(ranking: _Ranking, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUROC and AUPRC of each predictor when each gene counts weights[i] times; ties are scored as egbench does."""
    drawn = weights[ranking.order]
    count = ranking.segment.size
    positive = np.bincount(ranking.group, weights=(drawn * ranking.labels).ravel(), minlength=count)
    total = np.bincount(ranking.group, weights=drawn.ravel(), minlength=count)
    negative = total - positive
    positives = float((drawn[0] * ranking.labels[0]).sum())
    negatives = float(drawn[0].sum()) - positives
    width = len(ranking.starts)
    below = np.cumsum(negative) - negative
    below -= below[ranking.starts][ranking.segment]
    concordant = np.bincount(ranking.segment, weights=positive * (below + 0.5 * negative), minlength=width)
    auroc = concordant / (positives * negatives) if positives and negatives else np.full(width, np.nan)
    reached = np.cumsum(positive) - positive
    reached -= reached[ranking.starts][ranking.segment]
    taken = np.cumsum(total) - total
    taken -= taken[ranking.starts][ranking.segment]
    found, selected = positives - reached, float(drawn[0].sum()) - taken
    precision = np.divide(found, selected, out=np.zeros(count), where=selected > 0)
    auprc = np.bincount(ranking.segment, weights=positive * precision, minlength=width) / positives \
        if positives else np.full(width, np.nan)
    return auroc, auprc


def _stratum_metrics(frame: pd.DataFrame, groups: pd.Series, names: list[str], predictors: list[str]) -> pd.DataFrame:
    labels = frame["label"].to_numpy("int64")
    stratum = groups.reindex(frame[GENE_KEY]).to_numpy("object")
    if pd.isna(stratum).any():
        raise ValueError("a scored gene has no stratum")
    scores = frame[predictors].to_numpy("float64")
    members = {name: np.flatnonzero(stratum == name) for name in names}
    rankings = {name: _ranking(scores[rows], labels[rows].astype("float64")) for name, rows in members.items()}
    points = {name: _discrimination(rankings[name], np.ones(len(rows))) for name, rows in members.items()}
    replicates = {name: np.empty((2, len(predictors), RESAMPLE_COUNT)) for name in names}
    cells = pd.factorize(pd.MultiIndex.from_arrays([frame[FOLD], frame["label"]]))[0]
    resampler = Resampler(cells, seed=SEED)
    for r in range(RESAMPLE_COUNT):
        weights = np.bincount(resampler.draw(), minlength=len(frame)).astype("float64")
        for name, rows in members.items():
            replicates[name][:, :, r] = _discrimination(rankings[name], weights[rows])
    out = []
    for name, rows in members.items():
        for position, predictor in enumerate(predictors):
            entry = {"stratum": name, "predictor": predictor, "genes": int(len(rows)),
                     "positives": int(labels[rows].sum())}
            for index, metric in enumerate(("auroc", "auprc")):
                point = float(points[name][index][position])
                half_width, _ = summarise(point, replicates[name][index, position])
                entry[metric], entry[f"{metric}_half_width"] = point, half_width
            out.append(entry)
    return pd.DataFrame(out)


def evaluate_by_strata(
    path: Annotated[str, Field(description="CSV, TSV or parquet file of predictions with the columns gene_key, "
                                           "y_score and outer_fold")],
    task: Annotated[str, Field(description="binary task identifier: deg, depmap, context, blood, mouse or "
                                           "depmap_23q4")],
    by: Annotated[str, Field(description="hallmark_count, literature_quartile, family_size, ppi_degree or "
                                         "expression")],
) -> dict:
    """Score a table of test-fold predictions within groups of genes, next to the 15 published predictors.

    The groups are those of the manuscript, set on the labelled genes of the task: hallmark_count, 0 to 4 of the
    four hallmarks (see annotate_genes); literature_quartile, Q1 (least studied) to Q4 of the PubMed count;
    family_size, a protein family of one gene (no paralogue) or of two and more; ppi_degree and expression,
    quintiles 1 (lowest) to 5 of the BioGRID PPI degree among the task's genes and of the median TCGA expression.
    As in the manuscript, each predictor's scores are turned into percentiles among the test genes of their fold
    and the genes of a group are pooled over the five folds. For each group returns the genes and positives, the
    table's AUROC and AUPRC as point and half_width of the 95% gene-level bootstrap within folds and classes (4,000
    resamples, 15 to 30 seconds), its rank by AUROC among the 16, and the AUROC and AUPRC of the 15 published
    predictors on the same genes. AUPRC follows the share of positives, which differs between groups, so compare
    predictors within a group. The table must pass check_predictions; a table with problems is not scored.
    """
    cohort_id = benchmark.task_id(task)
    if not _binary(cohort_id):
        raise ValueError(f"{benchmark.TASKS[cohort_id].name} has a continuous target; the strata are scored on the "
                         f"binary tasks deg, depmap, context, blood, mouse and depmap_23q4")
    stratification = _strata_name(by)
    file, frame = benchmark.read_table(path)
    problems, _ = benchmark.submission_problems(frame, cohort_id)
    if problems:
        return {"task": cohort_id, "path": str(file), "ok": False, "problems": problems}
    submission = pd.DataFrame({
        GENE_KEY: frame[GENE_KEY].astype(str),
        SCORE: pd.to_numeric(frame[SCORE]).astype("float64"),
        FOLD: pd.to_numeric(frame[FOLD]).astype("int64"),
    })
    groups, definitions = _strata(cohort_id, stratification)
    scored = _strata_frame(cohort_id, submission)
    predictors = [SUBMISSION, *METHOD_IDS]
    table = _stratum_metrics(scored, groups, [item["stratum"] for item in definitions], predictors)
    names = benchmark.display_names()
    table["predictor"] = table["predictor"].map(lambda item: names.get(item, item))
    strata = []
    for definition in definitions:
        rows = table.loc[table["stratum"].eq(definition["stratum"])]
        own = rows.loc[rows["predictor"].eq(SUBMISSION)].iloc[0]
        others = rows.loc[rows["predictor"].ne(SUBMISSION)].sort_values("auroc", ascending=False, kind="stable")
        strata.append({
            **definition,
            "genes": int(own["genes"]),
            "positives": int(own["positives"]),
            "table": {metric: {"point": benchmark.number(own[metric]),
                               "half_width": benchmark.number(own[f"{metric}_half_width"])}
                      for metric in ("auroc", "auprc")},
            "auroc_rank": int((others["auroc"] > own["auroc"]).sum()) + 1,
            "out_of": len(predictors),
            "published": [
                {"predictor": row.predictor, "auroc": benchmark.number(row.auroc),
                 "auroc_half_width": benchmark.number(row.auroc_half_width), "auprc": benchmark.number(row.auprc),
                 "auprc_half_width": benchmark.number(row.auprc_half_width)}
                for row in others.itertuples(index=False)
            ],
        })
    return {
        "task": cohort_id,
        "path": str(file),
        "ok": True,
        "by": stratification,
        "definition": STRATA[stratification],
        "scope": benchmark.scope_note(cohort_id),
        "genes": int(len(scored)),
        "strata": strata,
        "csv": _written(table, f"{file.stem}_strata_{stratification}_{cohort_id}"),
    }


# check


def _row(check: str, task: str, quantity: str, compared: int, mismatches: int, difference: float = math.nan,
         relative: float = math.nan, note: str = "") -> dict:
    return {"check": check, "task": task, "quantity": quantity, "compared": int(compared),
            "mismatches": int(mismatches), "max_abs_difference": difference, "max_relative_difference": relative,
            "note": note}


def _same(left: pd.Series, right: pd.Series) -> tuple[int, int]:
    a = left.astype("object").where(left.notna(), None)
    b = right.astype("object").reindex(left.index)
    b = b.where(b.notna(), None)
    return len(a), int(sum(x != y for x, y in zip(a, b)))


def _close(left: pd.Series, right: pd.Series) -> tuple[int, int, float, float]:
    a = left.astype("float64")
    b = right.astype("float64").reindex(left.index)
    both = a.notna() & b.notna()
    difference = (a[both] - b[both]).abs()
    scale = np.maximum(a[both].abs(), b[both].abs())
    relative = (difference / scale.where(scale > 0)).fillna(0.0)
    missing = int((a.isna() != b.isna()).sum())
    largest = float(difference.max()) if len(difference) else math.nan
    return len(a), int((difference > TOLERANCE).sum()) + missing, largest, float(relative.max()) if len(relative) else math.nan


def _float_row(check: str, task: str, quantity: str, left: pd.Series, right: pd.Series, note: str = "") -> dict:
    compared, mismatches, difference, relative = _close(left, right)
    return _row(check, task, quantity, compared, mismatches, difference, relative, note)


def _exact_row(check: str, task: str, quantity: str, left: pd.Series, right: pd.Series, note: str = "") -> dict:
    compared, mismatches = _same(left, right)
    return _row(check, task, quantity, compared, mismatches, note=note)


def _check_biogrid(manuscript: Path | None, provenance: Path | None) -> list[dict]:
    (built,) = external.ensure("biogrid_edges")
    rebuilt = pd.read_parquet(built)
    rows = [_row("biogrid_edges", "", "rebuilt edges", len(rebuilt), 0, note=external.item("biogrid_edges")["path"])]
    if manuscript is None:
        return rows + [_row("biogrid_edges", "", "original edges", 0, 0, note="not compared: no --manuscript")]
    original = pd.read_parquet(manuscript / ORIGINAL_EDGES)
    merged = rebuilt.merge(original, on=EDGE_KEYS, how="outer", suffixes=("", "_original"), indicator=True)
    both = merged.loc[merged["_merge"].eq("both")].set_index(EDGE_KEYS)
    rows.append(_row("biogrid_edges", "", "gene pairs", len(merged), int((merged["_merge"] != "both").sum()),
                     note=f"original {len(original)} edges"))
    rows.append(_exact_row("biogrid_edges", "", "source_record_count", both["source_record_count"],
                           both["source_record_count_original"]))

    def sources(text: str) -> frozenset[str]:
        items = [item.split(":", 1)[1] if item.upper().startswith(PUBMED_PREFIX) else item
                 for item in text.split(SEPARATOR) if item]
        return frozenset(items)

    stored = both["experimental_systems_original"].map(sources)
    rows.append(_row("biogrid_edges", "", "publication sources", len(both),
                     int((stored != both["pubmed_ids"].map(sources)).sum()),
                     note="the original experimental_systems column holds the publication sources"))
    rows.append(_row("biogrid_edges", "", "original pubmed_ids holds the organism 9606", len(both),
                     int(both["pubmed_ids_original"].ne(HUMAN_TAXON).sum()),
                     note="the original lost the experimental systems; the rebuilt table records them"))
    if provenance is not None:
        _, counts = _biogrid_table(external.path("biogrid"), _reference())
        recorded = json.loads(provenance.read_text())
        for name, value in counts.items():
            rows.append(_row("biogrid_edges", "", name, 1, int(recorded.get(name) != value),
                             note=f"rebuilt {value}, recorded {recorded.get(name)}"))
    return rows


def _go_core(cohort_id: str) -> tuple[pd.DataFrame, set[str]]:
    labels = _labels(cohort_id)
    inside = pd.Series((labels == 1).to_numpy(), index=labels.index)
    table, terms = _enrichment_table(_process_pairs(), _process_names(), inside, TERM_MIN, TERM_MAX)
    top = set(table.loc[table["rank"] <= TOP_TERMS, TERM_ID])
    return table, set(terms.loc[terms[TERM_ID].isin(top), GENE_KEY])


def _check_hallmarks(cohort_id: str) -> list[dict]:
    hallmarks = RESULTS / "essential_gene_hallmarks" / "results"
    reference = _task_reference(cohort_id)
    labels = reference.labels
    rows = _annotation().loc[labels.index]
    flags = _hallmark_frame(rows, reference)
    out = []
    thresholds = pd.read_csv(hallmarks / "property_thresholds.csv").set_index("cohort_id").loc[cohort_id]
    for name, value in (("ppi_median", reference.ppi_median), ("tcga_median", reference.tcga_median)):
        out.append(_float_row("property_thresholds", cohort_id, name, pd.Series([value]), pd.Series([thresholds[name]])))
    go, core = _go_core(cohort_id)
    published = pd.read_csv(hallmarks / "go_enrichment.csv")
    published = published.loc[published["cohort_id"].eq(cohort_id)].set_index(TERM_ID)
    mine = go.set_index(TERM_ID)
    terms = mine.index.union(published.index)
    out.append(_row("go_enrichment", cohort_id, "terms", len(terms),
                    len(terms) - len(mine.index.intersection(published.index))))
    for column in ("genes", "positives", "rank"):
        out.append(_exact_row("go_enrichment", cohort_id, column, mine[column], published[column]))
    for column in ("fold_enrichment", "p", "q"):
        out.append(_float_row("go_enrichment", cohort_id, column, mine[column], published[column]))
    four = pd.read_csv(hallmarks / "four_properties.csv")
    for scope in ("all", "other"):
        inside = np.ones(len(labels), dtype=bool) if scope == "all" else ~labels.index.isin(core)
        values = labels.to_numpy()[inside]
        mine_rows = []
        for k in range(len(HALLMARKS) + 1):
            member = (flags[COUNT].to_numpy()[inside] == k)
            point, half = share_interval(values, member)
            mine_rows.append({"properties": k, "genes": int(member.sum()), "essentials": int(values[member].sum()),
                              "point": point, "half_width": half})
        mine_table = pd.DataFrame(mine_rows).set_index("properties")
        theirs = four.loc[four["cohort_id"].eq(cohort_id) & four["scope"].eq(scope)].set_index("properties")
        for column in ("genes", "essentials"):
            out.append(_exact_row("four_properties", cohort_id, f"{scope}: {column}", mine_table[column],
                                  theirs[column]))
        for column in ("point", "half_width"):
            out.append(_float_row("four_properties", cohort_id, f"{scope}: {column}", mine_table[column],
                                  theirs[column]))
    ages = pd.read_csv(hallmarks / "age_essential_share.csv")
    ages = ages.loc[ages["cohort_id"].eq(cohort_id) & ages["grouping"].eq("layer")].set_index("group")
    layer = rows[AGE].astype("int64")
    out.append(_exact_row("age_essential_share", cohort_id, "genes per layer", layer.value_counts().reindex(ages.index),
                          ages["genes"]))
    out.append(_exact_row("age_essential_share", cohort_id, "essentials per layer",
                          labels.groupby(layer).sum().reindex(ages.index), ages["essentials"]))
    bins = pd.read_csv(hallmarks / "degree_expression_bins.csv")
    for quantity, column in (("ppi_degree", task_degree(cohort_id)), ("tcga_median", EXPRESSION)):
        groups, edges = _quintiles(rows[column].astype("float64"))
        theirs = bins.loc[bins["cohort_id"].eq(cohort_id) & bins["quantity"].eq(quantity)].set_index("bin")
        cut = theirs["upper_inclusive"].iloc[:-1]
        out.append(_float_row("degree_expression_bins", cohort_id, f"{quantity} edges", pd.Series(edges, index=cut.index),
                              cut))
        out.append(_exact_row("degree_expression_bins", cohort_id, f"{quantity} genes",
                              groups.astype(int).value_counts().reindex(theirs.index), theirs["genes"]))
    corum_table, _ = _enrichment_table(*_complexes(), pd.Series((labels == 1).to_numpy(), index=labels.index),
                                       COMPLEX_MINIMUM, float("inf"))
    top = pd.read_csv(hallmarks / "corum_top.csv")
    top = top.loc[top["cohort_id"].eq(cohort_id)].set_index("complex_id")
    mine_corum = corum_table.set_index(TERM_ID)
    joined = mine_corum.index.union(top.index)
    out.append(_row("corum_top", cohort_id, "complexes", len(joined),
                    len(joined) - len(mine_corum.index.intersection(top.index))))
    for column in ("genes", "positives"):
        out.append(_exact_row("corum_top", cohort_id, column, mine_corum[column], top[column]))
    return out


def _check_cache(manuscript: Path | None, cohort_id: str) -> list[dict]:
    if manuscript is None:
        return []
    path = manuscript / "analysis" / "essential_gene_hallmarks" / "work" / f"gene_properties_{cohort_id}.csv"
    if not path.is_file():
        return [_row("gene_properties", cohort_id, "per gene", 0, 0, note=f"not compared: no {path.name}")]
    theirs = pd.read_csv(path).astype({GENE_KEY: str}).set_index(GENE_KEY)
    reference = _task_reference(cohort_id)
    rows = _annotation().loc[theirs.index]
    flags = _hallmark_frame(rows, reference)
    _, core = _go_core(cohort_id)
    out = [
        _float_row("gene_properties", cohort_id, "tcga_median", rows[EXPRESSION], theirs["tcga_median"]),
        _exact_row("gene_properties", cohort_id, "ppi_degree", rows[task_degree(cohort_id)].astype("int64"),
                   theirs["ppi_degree"].astype("int64")),
        _exact_row("gene_properties", cohort_id, "corum_any", flags["complex"], theirs["corum_any"].astype("int64")),
        _exact_row("gene_properties", cohort_id, "yeast_ortholog", rows[YEAST].astype("int64"),
                   theirs["yeast_ortholog"].astype("int64")),
        _exact_row("gene_properties", cohort_id, "age_layer", rows[AGE].astype("int64"), theirs["age_layer"]),
        _exact_row("gene_properties", cohort_id, "count_main", flags[COUNT], theirs["count_main"]),
        _exact_row("gene_properties", cohort_id, "core_function",
                   pd.Series(theirs.index.isin(core).astype("int64"), index=theirs.index), theirs["core_function"]),
    ]
    return out


def _check_tasks(cohort_id: str) -> list[dict]:
    reference = _task_reference(cohort_id)
    published = pd.read_csv(RESULTS / "dependency_strength_context" / "results" / "task_hallmarks.csv")
    published = published.loc[published["cohort_id"].eq(cohort_id)].set_index("quantity")["point"]
    frame = _strata_frame(cohort_id, None)[[GENE_KEY, FOLD, "label"]]
    flags = _hallmark_frame(_annotation().loc[frame[GENE_KEY]], reference).reset_index(drop=True)
    labels = frame["label"].to_numpy()
    mine = {}
    for name, values in (("rich", flags[COUNT] >= RICH), *((h, flags[h] == 1) for h in HALLMARKS)):
        mine[f"positive_{name}_share"] = float(values[labels == 1].mean())
        mine[f"negative_{name}_share"] = float(values[labels == 0].mean())
    ranking = [_ranking(flags.loc[(frame[FOLD] == k).to_numpy(), [COUNT]].to_numpy("float64"),
                        labels[(frame[FOLD] == k).to_numpy()].astype("float64")) for k in range(FOLD_COUNT)]
    mine["hallmark_count_auroc"] = float(np.mean([
        _discrimination(item, np.ones(item.order.shape[1]))[0][0] for item in ranking]))
    mine = pd.Series(mine)
    out = [_float_row("task_hallmarks", cohort_id, "hallmark shares and hallmark count AUROC", mine,
                      published.reindex(mine.index))]
    if cohort_id in ("deg", "depmap"):
        attention = pd.read_csv(RESULTS / "research_attention" / "results" / "attention_properties.csv")
        attention = attention.loc[attention["cohort_id"].eq(cohort_id)].set_index("quartile")
        rows = _annotation().loc[reference.labels.index]
        bands = _literature_bands(rows[LITERATURE], reference.literature_edges).str.lower()
        literature = rows[LITERATURE].astype("float64")
        out.append(_exact_row("attention_properties", cohort_id, "genes per quartile",
                              bands.value_counts().reindex(attention.index), attention["genes"]))
        out.append(_float_row("attention_properties", cohort_id, "lowest count per quartile",
                              literature.groupby(bands).min().reindex(attention.index), attention["pubmed_low"]))
        out.append(_float_row("attention_properties", cohort_id, "highest count per quartile",
                              literature.groupby(bands).max().reindex(attention.index), attention["pubmed_high"]))
    return out


def _check_dependency() -> list[dict]:
    table = _table(DEPENDENCY_TABLE)
    effect = pd.read_parquet(ROOT / "datasets" / "GeneEffect" / "folds" / "common_continuous.parquet")
    effect = effect.astype({GENE_KEY: str}).set_index(GENE_KEY)["gene_effect"]
    out = [_float_row("dependency", "gene_effect", "gene_effect_mean against the GeneEffect labels",
                      table.loc[effect.index, EFFECT], effect)]
    genes = pd.read_csv(RESULTS / "dependency_strength_context" / "results" / "lineage_genes.csv")
    genes = genes.astype({GENE_KEY: str}).set_index(GENE_KEY)
    shares = table.loc[genes.index, _lineage_columns(table)].astype("float64")
    shares.columns = [column.split(":", 1)[1] for column in shares.columns]
    out.append(_float_row("dependency", "context", "dependent_share against lineage_genes",
                          table.loc[genes.index, DEPENDENT], genes["dependent_share"]))
    out.append(_float_row("dependency", "context", "highest lineage share", shares.max(axis=1), genes["highest_share"]))
    out.append(_exact_row("dependency", "context", "highest lineage", shares.idxmax(axis=1), genes["highest_lineage"]))
    out.append(_exact_row("dependency", "context", "lineages with share at least 0.5",
                          (shares >= LINEAGE_SHARE).sum(axis=1), genes["dependent_lineages"]))
    labels = _labels("depmap")
    named = _depmap_classes().reindex(labels.index)
    expected = pd.Series(np.where(labels == 1, "common essential", "neither"), index=labels.index)
    out.append(_exact_row("dependency", "depmap", "depmap_class against the DepMap labels", named, expected))
    return out


def _check_mouse() -> list[dict]:
    table = _table(MOUSE_TABLE)
    universe = pd.read_parquet(MOUSE_UNIVERSE).astype({GENE_KEY: str}).set_index(GENE_KEY)
    out = [
        _exact_row("mouse", "mouse", "mgi_id against the Mouse gene universe", table.loc[universe.index, MGI],
                   universe["mgi_id"]),
        _exact_row("mouse", "mouse", "impc_viability against impc_viability_call", table.loc[universe.index, VIABILITY],
                   universe["impc_viability_call"]),
    ]
    stages = pd.read_csv(RESULTS / "mouse_lethal_stages" / "results" / "lethal_stage_genes.csv")
    stages = stages.astype({GENE_KEY: str}).set_index(GENE_KEY)
    stage = table.loc[stages.index, STAGE].where(stages["label"].eq(1), "viable")
    rows = _annotation().loc[stages.index]
    flags = _hallmark_frame(rows, _task_reference("mouse"))
    out += [
        _exact_row("mouse", "mouse", "stage against lethal_stage_genes", stage, stages["stage"]),
        _exact_row("mouse", "mouse", "hallmark count", flags[COUNT], stages["count"]),
        _exact_row("mouse", "mouse", "yeast orthologue", flags["old"], stages["old"]),
        _exact_row("mouse", "mouse", "paralogue", rows[PARALOGUES].astype("float64") >= 1, stages["paralogue"]),
        _exact_row("mouse", "mouse", "age layer", rows[AGE].astype("int64"), stages["age_layer"]),
        _float_row("mouse", "mouse", "gene_effect_mean", _table(DEPENDENCY_TABLE).loc[stages.index, EFFECT],
                   stages["gene_effect_mean"]),
    ]
    return out


def _check_corum(manuscript: Path | None) -> list[dict]:
    if manuscript is None:
        return [_row("corum_count", "", "against DeEPsnap", 0, 0, note="not compared: no --manuscript")]
    table = _annotation()
    out = []
    for cohort_id in TASK_IDS:
        if not _binary(cohort_id):
            continue
        directory = feature_directory(manuscript / "methods" / "deepsnap", cohort_id, split_strategy=benchmark.SPLIT,
                                      outer_fold=0)
        fold = pd.read_parquet(directory / "features.parquet", columns=[GENE_KEY, "corum_membership_count"])
        fold = fold.astype({GENE_KEY: str})
        fold = fold.set_index(GENE_KEY)["corum_membership_count"]
        out.append(_float_row("corum_count", cohort_id, "corum_complexes against DeEPsnap corum_membership_count",
                              table.loc[fold.index, COMPLEXES], fold))
    return out


def _check_strata(cohort_id: str) -> list[dict]:
    groups, definitions = _strata(cohort_id, "literature_quartile")
    frame = _strata_frame(cohort_id, None)
    table = _stratum_metrics(frame, groups, [item["stratum"] for item in definitions], list(METHOD_IDS))
    table["statistic"] = table["predictor"] + ":" + table["stratum"].str.lower()
    table = table.set_index("statistic")
    trend = pd.read_csv(RESULTS / "research_attention" / "results" / "attention_trend.csv")
    trend = trend.loc[trend["cohort_id"].eq(cohort_id) & trend["metric"].eq("auroc")].set_index("statistic")
    return [
        _float_row("attention_trend", cohort_id, "AUROC of the 15 predictors per literature quartile",
                   table["auroc"], trend["point"]),
        _float_row("attention_trend", cohort_id, "half-width", table["auroc_half_width"], trend["half_width"]),
    ]


def check(manuscript: Path | None, provenance: Path | None) -> pd.DataFrame:
    started = time.monotonic()
    binary = [cohort_id for cohort_id in TASK_IDS if _binary(cohort_id) and cohort_id != "depmap_23q4"]
    steps = [
        ("BioGRID edges", lambda: _check_biogrid(manuscript, provenance)),
        ("annotation tables", lambda: [_row(item_id, "", "genes", len(_table(item_id)), 0,
                                            note=external.item(item_id)["path"])
                                       for item_id in (ANNOTATION, DEPENDENCY_TABLE, MOUSE_TABLE)]),
        *((f"hallmarks of {c}", lambda c=c: _check_hallmarks(c) + _check_cache(manuscript, c)) for c in ("deg", "depmap")),
        *((f"hallmarks of the scored genes of {c}", lambda c=c: _check_tasks(c)) for c in binary),
        ("DepMap fields", _check_dependency),
        ("mouse fields", _check_mouse),
        ("CORUM counts", lambda: _check_corum(manuscript)),
        *((f"literature strata of {c}", lambda c=c: _check_strata(c)) for c in binary),
    ]
    rows: list[dict] = []
    for done, (name, step) in enumerate(steps, start=1):
        rows += step()
        print(f"[check] {done}/{len(steps)} {name} done, {time.monotonic() - started:.0f} s", flush=True)
    return pd.DataFrame(rows, columns=CHECK_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the agent's annotation against the manuscript's results.")
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check", help="recompute manuscript tables and write agent/results/annotation_check.csv")
    checking.add_argument("--manuscript", type=Path, help="repository copy with methods/<method>/feature/ and analysis caches")
    checking.add_argument("--biogrid-qa", type=Path, help="provenance record of the original BioGRID table")
    arguments = parser.parse_args(argv)
    manuscript = None if arguments.manuscript is None else arguments.manuscript.expanduser().resolve()
    provenance = None if arguments.biogrid_qa is None else arguments.biogrid_qa.expanduser().resolve()
    table = check(manuscript, provenance)
    CHECK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CHECK_OUTPUT, index=False)
    failing = table.loc[table["mismatches"] > 0]
    print(f"{len(table)} comparisons, {len(failing)} with mismatches; written {CHECK_OUTPUT}")
    for row in failing.itertuples(index=False):
        print(f"  {row.check} {row.task} {row.quantity}: {row.mismatches} of {row.compared}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
