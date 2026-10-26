"""Builders of the method inputs other than sequences: EPGAT, XGEP, DeEPsnap, DeepHE, HELP and Mahi.

Each builder writes one derived item of ``agent/external.json`` from public downloads with the code of the
method's own feature builder in ``methods/``, so the item is the file the method reads from its
``feature/source/``. ``epgat_static`` joins the orthology table with the JensenLab localisation confidences
of the 11 compartments that ``methods/epgat/code/build_features.py`` parses. ``xgep_150`` maps the three
pan-cancer embedding files of the XGEP authors to the gene reference. ``deepsnap_source_manifest`` records
the SHA-256 of the GOA, CORUM and Pfam files that DeEPsnap's feature step checks before it reads them.
``help_features`` assembles HELP's matrix from the authors' three tables. ``node2vec_64`` embeds the
BioGRID network for DeepHE and DeEPsnap. ``mahi_global_context`` keeps the global rows of the Mahi
embeddings, and ``mahi_features`` writes the matrix of each task: for DEG and DepMap the genes of the task's
gene universe that HGNC maps to an Entrez identifier with an embedding, for the other four tasks the rows
of their candidate genes (``agent/candidates/``) taken from the DepMap matrix. ``epgat_static`` runs in
EPGAT's ``torch`` environment, whose PyYAML EPGAT's modules import, ``xgep_150`` in the ``cpu`` environment,
whose Biopython XGEP's modules import, and ``node2vec_64`` in the ``node2vec`` environment, whose gensim
and pandas wrote the original embedding.

    python agent/method_inputs.py check --methods DIR [--compartments FILE]

compares the items under the data root, built beforehand with ``agent/external.py build``, and the
tables they share with other methods with the originals in DIR, a copy of the methods directory holding
``<method>/feature/source/``, and writes ``agent/results/method_inputs_check.csv``. FILE is a later
COMPARTMENTS download, whose localisation columns are compared with those of the original static table.
"""
from __future__ import annotations

import argparse
import contextlib
import filecmp
import hashlib
import importlib
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import external
import inputs

METHODS = external.ROOT / "methods"
CANDIDATES = external.AGENT / "candidates"
CHECK_OUTPUT = external.AGENT / "results" / "method_inputs_check.csv"
GENE_KEY = "gene_key"
FEATURES = "features.parquet"
XGEP_FILES = ("xgep_essential", "xgep_nonessential", "xgep_prediction")
DEEPSNAP_SOURCES = {"goa_human": "goa_human", "corum_human": "corum", "ensembl_biomart_pfam": "ensembl_pfam"}
HELP_BLOCKS = (("help_bio", "bio"), ("help_cccfs", "cc"), ("help_embn2v", "n2v"))
MAHI_COLUMNS = ["entrez_id", "tissue", "embedding"]
MAHI_VECTOR = pa.list_(pa.float16())
MAHI_GLOBAL = "global"
MAHI_UNIVERSES = {
    "deg": "datasets/DEG/data/gene_universe.parquet",
    "depmap": "datasets/DepMap/data/depmap26q1_gene_universe.parquet",
}
MAHI_DERIVED = ("context", "blood", "mouse", "depmap_23q4")
LOCALISATION = re.compile(r"compartment__|localization_missing$")
EDGE_COLUMNS = ["gene_key_a", "gene_key_b", "source_record_count"]
EPGAT_STATIC = "epgat/feature/source/generated/epgat_static.parquet"


def _method(file: str, *directories: str) -> ModuleType:
    """A file of ``methods/`` imported under a name of its own, with the directories its imports need on the path."""
    name = "methods_" + re.sub(r"\W", "_", file.removesuffix(".py"))
    if name not in sys.modules:
        for directory in directories:
            if str(METHODS / directory) not in sys.path:
                sys.path.insert(0, str(METHODS / directory))
        spec = importlib.util.spec_from_file_location(name, METHODS / file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _localisation(compartments: Path) -> pd.DataFrame:
    code = _method("epgat/code/build_features.py", "epgat/code")
    table, _ = code.parse_compartments(compartments, pd.read_parquet(external.path("gene_reference")))
    return table


def epgat_static(output: Path) -> None:
    """Builder of ``epgat_static``: orthologues in 194 species and localisation in 11 compartments.

    The localisation columns follow EPGAT's parser: COMPARTMENTS rows of the 11 overview terms, proteins
    matched by unversioned Ensembl protein identifier and then by symbol, the highest confidence of a gene
    in a term, 0 where it has none, and a column for genes with no localisation at all.
    """
    orthology = pd.read_parquet(external.path("orthology"))
    localisation = _localisation(external.path("compartments"))
    orthology.merge(localisation, on=GENE_KEY, validate="one_to_one").to_parquet(output, index=False)


def xgep_150(output: Path) -> None:
    """Builder of ``xgep_150``: the authors' 150-dimensional embedding of the genes the gene reference maps.

    Runs the reconstruction of ``methods/xgep/build_features.py rebuild-source``: rows of the three files
    matched by stable Ensembl gene identifier, their label columns discarded, sorted by gene.
    """
    code = _method("xgep/build_features.py")
    features = importlib.import_module("method_features.legacy_features")
    build = features.build_xgep_150(
        *(external.path(item_id) for item_id in XGEP_FILES),
        code._ensembl_mapping(external.path("gene_reference")),
    )
    features.write_deterministic_parquet(
        [row.as_record() for row in build.rows],
        output,
        sort_by=(GENE_KEY,),
        column_order=(GENE_KEY, *features.XGEP_150_COLUMNS),
    )


def deepsnap_source_manifest(output: Path) -> None:
    """Builder of ``deepsnap_source_manifest``: the SHA-256 of the three annotation files DeEPsnap reads."""
    sources = {key: {"sha256": _sha256(external.path(item_id))} for key, item_id in DEEPSNAP_SOURCES.items()}
    output.write_text(json.dumps({"sources": sources}, indent=2, sort_keys=True) + "\n")


def help_features(output: Path) -> None:
    """Builder of ``help_features``: HELP's matrix of the authors' BIO, CCcfs and node2vec tables.

    Follows ``methods/help/code/build_help_features.py``: each table filled with its column means, its
    constant columns dropped and standardised; symbols matched by approved symbol, then by unambiguous
    alias or previous symbol; rows of one gene averaged, genes the authors did not cover given the means.
    """
    code = _method("help/code/build_help_features.py")
    genes = pd.read_parquet(external.path("gene_reference"))
    with contextlib.redirect_stdout(sys.stderr):
        bio, *others = [code._load_block(external.path(item_id), prefix) for item_id, prefix in HELP_BLOCKS]
        resolver = code._symbol_to_gene_key(genes, external.path("hgnc"))
    merged = bio.join(others, how="outer")
    merged[GENE_KEY] = [resolver.get(str(symbol).upper()) for symbol in merged.index]
    resolved = merged.dropna(subset=[GENE_KEY]).groupby(GENE_KEY, sort=True).mean(numeric_only=True)
    features = resolved.reindex(genes[GENE_KEY].astype(str).drop_duplicates().sort_values())
    features = features.fillna(features.mean(numeric_only=True))
    features.rename_axis(GENE_KEY).reset_index().to_parquet(output, index=False)


def node2vec_64(output: Path) -> None:
    """Builder of ``node2vec_64``: the node2vec embedding of the BioGRID network over the gene reference.

    Runs ``build_node2vec`` of ``methods/deephe/build_features.py`` (walks of 80 genes, 10 from each gene,
    gensim Word2Vec with window 10, 5 negative samples, 5 epochs, one worker and seed 2026), its paths
    pointed at the data root.
    """
    code = _method("deephe/build_features.py")
    code.NODE2VEC_EDGES = external.path("biogrid_edges")
    code.NODE2VEC_GENE_UNIVERSE = external.path("gene_reference")
    code.NODE2VEC_OUTPUT = output
    with contextlib.redirect_stdout(sys.stderr):
        code.build_node2vec()


def mahi_global_context(output: Path) -> None:
    """Builder of ``mahi_global_context``: the global-context rows of the Mahi core-tissue embeddings.

    Writes the table of ``methods/mahi/acquire_assets.py``: Entrez identifier, tissue and float16 vector.
    """
    code = _method("mahi/build_features.py", "mahi")
    rows = pq.read_table(
        external.path("mahi_embeddings"), columns=MAHI_COLUMNS, filters=[("tissue", "=", MAHI_GLOBAL)]
    ).to_pandas()
    if len(rows) != code.FORMAL_EMBEDDING_ROWS:
        raise ValueError(f"the Mahi file has {len(rows)} global rows, the method expects {code.FORMAL_EMBEDDING_ROWS}")
    table = pa.table(
        {
            "entrez_id": pa.array(rows["entrez_id"].tolist(), type=pa.string()),
            "tissue": pa.array(rows["tissue"].tolist(), type=pa.string()),
            "embedding": pa.array([vector.tolist() for vector in rows["embedding"]], type=MAHI_VECTOR),
        }
    )
    pq.write_table(table, output)


def _entrez_by_hgnc() -> dict[str, str]:
    """HGNC identifier to Entrez identifier, as ``methods/mahi/local_dataset.read_hgnc_mapping`` reads them."""
    table = pd.read_csv(
        external.path("hgnc"), sep="\t", dtype="string", keep_default_na=False, na_filter=False,
        usecols=["hgnc_id", "entrez_id"],
    )
    hgnc, entrez = table["hgnc_id"].str.strip(), table["entrez_id"].str.strip()
    valid = entrez.str.fullmatch(r"[0-9]+", na=False)
    return dict(zip(hgnc[valid], entrez[valid]))


def mahi_features(output: Path) -> None:
    """Builder of ``mahi_features``: ``<task>/features.parquet``, the Mahi matrix of each task.

    DEG and DepMap hold the genes of their gene universe that HGNC maps to an Entrez identifier with a
    global embedding. The other tasks hold the rows of their candidate genes, taken from the DepMap matrix
    by ``methods/derive_cohort_source_assets.py``; the candidates are read from ``agent/candidates/``.
    """
    derive = _method("derive_cohort_source_assets.py")
    table = pq.read_table(external.path("mahi_global_context"), columns=["entrez_id", "embedding"])
    row_of = {entrez: row for row, entrez in enumerate(table["entrez_id"].to_pylist())}
    entrez = _entrez_by_hgnc()
    for task, universe in MAHI_UNIVERSES.items():
        genes = set(pd.read_parquet(external.ROOT / universe, columns=[GENE_KEY])[GENE_KEY].astype(str))
        keys = sorted(key for key in genes if entrez.get(key) in row_of)
        target = output / task / FEATURES
        target.parent.mkdir(parents=True)
        rows = table["embedding"].take([row_of[entrez[key]] for key in keys]).cast(MAHI_VECTOR)
        pq.write_table(pa.table({GENE_KEY: pa.array(keys, type=pa.string()), "embedding": rows}), target)
    donor = output / derive.DONOR / FEATURES
    available = set(pq.read_table(donor, columns=[GENE_KEY])[GENE_KEY].to_pylist())
    for task in MAHI_DERIVED:
        candidates = CANDIDATES / f"{task}_common_feature_available.parquet"
        keys = sorted(pd.read_parquet(candidates, columns=[GENE_KEY])[GENE_KEY].astype(str))
        target = output / task / FEATURES
        target.parent.mkdir(parents=True)
        if set(keys) == available:
            shutil.copyfile(donor, target)
        else:
            derive._select_rows(donor, keys, target)


# Comparison with the originals


def _counts(path: Path) -> dict:
    if path.suffix != ".parquet":
        return {}
    metadata = pq.ParquetFile(path).metadata
    return {"rows": metadata.num_rows, "columns": metadata.num_columns}


def _values(mine: pa.Table, theirs: pa.Table) -> dict:
    """Columns, rows and values that differ between two tables, and the largest absolute difference."""
    rows = np.zeros(theirs.num_rows, dtype=bool) if mine.num_rows == theirs.num_rows else None
    columns, values, largest = 0, 0, 0.0
    for name in theirs.column_names:
        if rows is None or name not in mine.column_names:
            columns += 1
            continue
        left, right = mine[name].combine_chunks(), theirs[name].combine_chunks()
        if left.equals(right):
            continue
        columns += 1
        shape = (len(right), -1)
        if pa.types.is_list(right.type) or pa.types.is_fixed_size_list(right.type):
            left, right = left.flatten(), right.flatten()
        a = left.to_numpy(zero_copy_only=False).reshape(shape)
        b = right.to_numpy(zero_copy_only=False).reshape(shape)
        if a.shape != b.shape:
            rows[:] = True
            continue
        if np.issubdtype(b.dtype, np.number):
            a, b = a.astype(np.float64), b.astype(np.float64)
            unequal = ~((a == b) | (np.isnan(a) & np.isnan(b)))
            largest = max(largest, float(np.nanmax(np.abs(a - b), initial=0.0)))
        else:
            unequal = a != b
        values += int(unequal.sum())
        rows |= unequal.any(axis=1)
    return {
        "rows": mine.num_rows,
        "original_rows": theirs.num_rows,
        "columns": mine.num_columns,
        "original_columns": theirs.num_columns,
        "differing_columns": columns + len(set(mine.column_names) - set(theirs.column_names)),
        "differing_rows": None if rows is None else int(rows.sum()),
        "differing_values": values,
        "largest_absolute_difference": largest,
        "identical": columns == 0 and mine.column_names == theirs.column_names,
    }


def _compare(item: str, mine: Path, methods: Path, original: str, columns: list[str] | None = None) -> dict:
    theirs = methods / original
    row = {"item": item, "original": original}
    if not theirs.is_file():
        return {**row, "comparison": "original absent", "identical": False}
    if columns is None and filecmp.cmp(mine, theirs, shallow=False):
        counts = _counts(theirs)
        if counts:
            counts.update(original_rows=counts["rows"], original_columns=counts["columns"])
        return {**row, "comparison": "bytes", **counts, "differing_columns": 0, "identical": True}
    if mine.suffix != ".parquet":
        return {**row, "comparison": "bytes", "identical": False}
    comparison = "values" if columns is None else "values of " + ", ".join(columns)
    return {**row, "comparison": comparison, **_values(pq.read_table(mine, columns=columns), pq.read_table(theirs, columns=columns))}


def _manifest_row(methods: Path) -> dict:
    original = "deepsnap/feature/source/annotations/source_manifest.json"
    mine = json.loads(external.path("deepsnap_source_manifest").read_text())["sources"]
    theirs = json.loads((methods / original).read_text())["sources"]
    same = [key for key in DEEPSNAP_SOURCES if mine[key]["sha256"] == theirs[key]["sha256"]]
    return {
        "item": "deepsnap_source_manifest",
        "original": original,
        "comparison": "sha256 of " + ", ".join(DEEPSNAP_SOURCES),
        "differing_values": len(DEEPSNAP_SOURCES) - len(same),
        "identical": len(same) == len(DEEPSNAP_SOURCES),
    }


def _compartments_row(current: Path, methods: Path) -> dict:
    """EPGAT's localisation columns from a later COMPARTMENTS file against those of the original static table."""
    new = _localisation(current)
    columns = [column for column in new.columns if LOCALISATION.match(column)]
    old = pd.read_parquet(methods / EPGAT_STATIC, columns=columns)
    for column in columns:
        change = new[column].astype(np.float64) - old[column].astype(np.float64)
        print(f"{column}: {int(change.ne(0).sum())} genes changed, {int(change.gt(0).sum())} up, "
              f"largest change {change.abs().max():g}", flush=True)
    return {
        "item": f"epgat_static from COMPARTMENTS of SHA-256 {_sha256(current)[:12]}",
        "original": EPGAT_STATIC,
        "comparison": "values of the localisation columns",
        **_values(pa.Table.from_pandas(new[columns]), pa.Table.from_pandas(old)),
    }


# The items compared with the methods' inputs, and the methods whose repository files are compared.
CHECKED = ("gene_reference", "tcga_expression", "biogrid_edges", "hgnc", "goa_human", "corum", "ensembl_pfam",
           *XGEP_FILES, "epgat_static", "xgep_150", "help_features", "node2vec_64", "mahi_global_context",
           "mahi_features")
REPOSITORY_METHODS = ("esssubgraph", "mahi")


def check(methods: Path, compartments: Path | None = None) -> pd.DataFrame:
    builds = [item_id for item_id in CHECKED if external.item(item_id)["kind"] == "build"]
    missing = [item_id for item_id in [*builds, "deepsnap_source_manifest"] if not external.path(item_id).exists()]
    if missing:
        raise SystemExit(f"build these items first with agent/external.py build: {', '.join(missing)}")
    rows = []
    for item_id in CHECKED:
        for name, targets in inputs.originals(item_id):
            mine = external.path(item_id) / name if name else external.path(item_id)
            columns = EDGE_COLUMNS if item_id == "biogrid_edges" else None
            rows.extend(_compare(item_id, mine, methods, target, columns) for target in targets)
    for method in REPOSITORY_METHODS:
        for target, reference in inputs.facts(method)["inputs"].items():
            if reference.startswith(inputs.REPOSITORY):
                path = reference[len(inputs.REPOSITORY):]
                rows.append(_compare(f"repository file {path}", external.ROOT / path, methods,
                                     f"{method}/feature/source/{target}"))
    rows.append(_manifest_row(methods))
    if compartments is not None:
        rows.append(_compartments_row(compartments, methods))
    columns = ["item", "original", "comparison", "rows", "original_rows", "columns", "original_columns",
               "differing_columns", "differing_rows", "differing_values", "largest_absolute_difference", "identical"]
    return pd.DataFrame(rows).reindex(columns=columns).convert_dtypes()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check", help="compare the built items with the methods' originals")
    checking.add_argument("--methods", type=Path, required=True, help="copy of the methods directory with feature/source/")
    checking.add_argument("--compartments", type=Path, help="a later COMPARTMENTS download to compare with the manuscript's")
    arguments = parser.parse_args(argv)
    compartments = None if arguments.compartments is None else arguments.compartments.expanduser()
    table = check(arguments.methods.expanduser(), compartments)
    CHECK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CHECK_OUTPUT, index=False)
    print(table.to_string(index=False))
    print(f"written {CHECK_OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
