"""The core feature set model of the manuscript as an agent tool, built from public downloads.

The core feature set holds four feature groups, laid out in the order of
``common.feature_blocks.CORE_BLOCKS``: the localisation confidences of 3,301 GO cellular component
terms (CCcfs) that the HELP authors published, 50 principal components of TCGA tumour expression
fitted on each fold's training genes, HELP's 18 multi-omics annotation columns, and whether each of
194 species has an orthologue of the gene in Ensembl Compara 115, with a missing indicator. The
learner of the manuscript (``common.feature_blocks.make_learner``, gradient boosting with early
stopping) is fitted on the training genes of each fold. The groups are rebuilt from the downloads by
the rules of the feature builders of HELP, EssSubgraph and EPGAT, so on the benchmark tasks the model
reproduces variant A of ``analysis/feature_groups/results/core_fixed.csv``.

``orthology`` and ``static_features`` write the derived tables ``orthology`` and ``core_static`` of
``agent/external.json``. ``core_model``, the tool, trains the five fold models of a benchmark task or
of a task made by ``make_task``, caches them under ``agent/work/core_model/<task>/`` of the data root, writes their
test-fold predictions as a submission table, scores it and predicts genes. A gene that is not in the
TCGA matrix gets the expression components of the fold's training median profile, as EssSubgraph
fills a missing value, and a test gene of a custom task that is not in the gene reference gets the
features of a gene without any annotation. ``check`` compares the feature groups and the fold models
trained so far with the manuscript's inputs and results, fits nothing, and writes
``agent/results/core_model_check.csv``; ``train all`` beforehand trains every fold of the benchmark
tasks.

    python agent/core_model.py train deg [--genes TP53,POLR2A]
    python agent/core_model.py train all --workers 5
    python agent/core_model.py check [--methods <dir>] [--cache <dir>] [--fixed <dir>] [--prospective <csv>]

With several workers the folds are fitted in separate processes, each with the BLAS and OpenMP
threads that OMP_NUM_THREADS allows, so set it to the cores divided by the workers.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
from pathlib import Path
from typing import Annotated, NamedTuple

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pydantic import Field

import external
from benchmark import (
    FOLD,
    GENE_KEY,
    ROOT,
    SCORE,
    SPLIT,
    TASKS,
    WORK_DIRECTORY,
    display_names,
    examples,
    number,
    pooled,
    predictor_folds,
    ranked_points,
    read_table,
    task_id,
)

if str(ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis"))

from common.feature_blocks import CORE_BLOCKS, make_learner
from egbench.contracts import LEADERBOARD_COHORT_IDS, METHOD_IDS, TargetKind
from egbench.data import (
    TEST_ROLE,
    TRAIN_ROLE,
    cohort_target,
    feature_directory,
    folds_directory,
    prediction_cohort,
    primary_scope,
    read_evaluation_truth,
    read_training_labels,
)
from egbench.evaluate import FOLD_COUNT, scope_frame
from egbench.metrics.discrimination import auprc, auroc
from egbench.metrics.regression import spearman
from egbench.stats import confidence_intervals
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

COMPONENTS = 50
COMPONENT_COLUMNS = [f"expression_pc_{index:03d}" for index in range(1, COMPONENTS + 1)]
PROJECTION_ROWS = 2048
CCCFS = "cc__"
BIO = "bio__"
ORTHOLOG = "ortholog__"
ORTHOLOGY_MISSING = "orthology_missing"
HUMAN = "homo_sapiens"
COMPARA_COLUMNS = ["gene_stable_id", "species", "homology_type", "homology_species"]
COMPARA_ROWS = 250_000
HGNC_COLUMNS = ["hgnc_id", "alias_symbol", "prev_symbol"]
INPUTS = ("core_static", "tcga_expression", "gene_reference", "hgnc")

MODELS = WORK_DIRECTORY / "core_model"
CUSTOM = WORK_DIRECTORY / "tasks"
CUSTOM_LABELS = "labels.csv"
CUSTOM_NAME = re.compile(r"[a-z][a-z0-9_]{0,39}")
PREDICTIONS = "predictions.csv"
SUMMARY = "summary.json"
GENES_FILE = "genes.csv"
PART_SUFFIX = ".part"
GENE_COLUMNS = ("gene_key", "hgnc_id", "gene", "gene_symbol", "symbol")
GENE_LIMIT = 50
PROGRESS_SECONDS = 30.0
TOLERANCE = 1e-9

CORE_FIXED = ROOT / "analysis" / "feature_groups" / "results" / "core_fixed.csv"
FEATURE_GROUPS_WORK = ROOT / "analysis" / "feature_groups" / "work"
CHECK_OUTPUT = ROOT / "agent" / "results" / "core_model_check.csv"
HELP_MATRIX = Path("help/feature/source/help_features_gene_map.parquet")
EPGAT_STATIC = Path("epgat/feature/source/generated/epgat_static.parquet")
CACHE_BLOCKS = ("cccfs", "help_bio", "orthology")
PROSPECTIVE = "depmap_23q4"

MODEL_NOTE = (
    "core feature set of the manuscript: localisation confidences of 3,301 GO cellular component "
    "terms and 18 multi-omics annotations published by the HELP authors, 50 principal components of "
    "TCGA tumour expression fitted on the fold's training genes, and orthologues in 194 species "
    "(Ensembl Compara 115); one gradient boosting classifier per fold, trained on its training genes"
)
NOT_FOUND = (
    "no gene of the gene reference has this HGNC identifier, symbol or previous symbol, so it has no "
    "features"
)
UNANNOTATED = "predicted as genes without any annotation, so their scores carry no information about them"
NO_EXPRESSION = "not in the TCGA matrix: its expression components are those of the training median profile"


class Spec(NamedTuple):
    name: str
    models: str
    kind: TargetKind
    custom: bool


class Static(NamedTuple):
    index: pd.Index
    blocks: dict[str, np.ndarray]


def _replace(path: Path, write: Callable[[Path], object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + PART_SUFFIX)
    write(temporary)
    os.replace(temporary, path)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _unversioned(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.split(".", n=1).str[0]


def _gene_reference(*columns: str) -> pd.DataFrame:
    return pd.read_parquet(external.path("gene_reference"), columns=[GENE_KEY, *columns])


def _orthology_table() -> pd.DataFrame:
    genes = _gene_reference("ensembl_gene_id")
    keys = sorted(genes[GENE_KEY].astype(str).unique())
    named = genes.loc[genes["ensembl_gene_id"].notna()]
    named = named.loc[named["ensembl_gene_id"].astype(str).str.strip().ne("")]
    mapping = pd.DataFrame(
        {
            "ensembl": _unversioned(named["ensembl_gene_id"]).to_numpy(),
            GENE_KEY: named[GENE_KEY].astype(str).to_numpy(),
        }
    ).drop_duplicates()
    mapping = mapping.drop_duplicates("ensembl", keep=False)
    gene_of = dict(zip(mapping["ensembl"], mapping[GENE_KEY]))
    found, species = [], set()
    with pd.read_csv(
        external.path("compara"), sep="\t", usecols=COMPARA_COLUMNS, dtype=str, chunksize=COMPARA_ROWS
    ) as reader:
        for chunk in reader:
            kept = chunk.loc[
                chunk["species"].eq(HUMAN)
                & chunk["homology_species"].ne(HUMAN)
                & chunk["homology_type"].str.contains("ortholog", case=False, na=False)
            ]
            names = kept["homology_species"].astype(str).str.strip()
            species.update(names[names.ne("")])
            owners = _unversioned(kept["gene_stable_id"]).map(gene_of)
            hit = names.ne("") & owners.notna()
            found.append(pd.DataFrame({GENE_KEY: owners[hit].to_numpy(), "species": names[hit].to_numpy()}))
    order = sorted(species)
    pairs = pd.concat(found, ignore_index=True).drop_duplicates()
    matrix = np.zeros((len(keys), len(order)), dtype=np.uint8)
    rows = pd.Index(keys).get_indexer(pairs[GENE_KEY])
    matrix[rows, pd.Index(order).get_indexer(pairs["species"])] = 1
    table = pd.DataFrame(matrix, columns=[ORTHOLOG + _slug(name) for name in order])
    table.insert(0, GENE_KEY, keys)
    table[ORTHOLOGY_MISSING] = (matrix.sum(axis=1) == 0).astype(np.uint8)
    return table


def orthology(output: Path) -> None:
    """Builder of ``orthology``: which species have an orthologue of each gene of the gene reference.

    Follows EPGAT's static feature builder: Compara rows of a human gene whose homology type names an
    orthologue in another species, genes matched by unversioned Ensembl gene identifiers that belong
    to one gene only, one column per species seen in those rows, and a column for genes with none.
    """
    _orthology_table().to_parquet(output, index=False)


def _help_block(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.map(str)
    frame = frame.fillna(frame.mean(numeric_only=True))
    frame = frame.drop(columns=frame.columns[frame.nunique() <= 1])
    values = StandardScaler().fit_transform(frame.to_numpy(dtype=np.float64))
    columns = [f"{prefix}{index}" for index in range(frame.shape[1])]
    return pd.DataFrame(values, index=frame.index, columns=columns)


def _symbol_resolver(genes: pd.DataFrame) -> dict[str, str]:
    symbols = zip(genes["symbol"].astype(str), genes[GENE_KEY].astype(str))
    resolver = {symbol.upper(): key for symbol, key in symbols}
    primary = set(resolver)
    valid = set(genes[GENE_KEY].astype(str))
    hgnc = pd.read_csv(external.path("hgnc"), sep="\t", dtype=str, low_memory=False, usecols=HGNC_COLUMNS)
    aliases: dict[str, str] = {}
    ambiguous: set[str] = set()
    for key, *fields in hgnc[HGNC_COLUMNS].itertuples(index=False, name=None):
        key = str(key)
        if key not in valid:
            continue
        for field in fields:
            if not isinstance(field, str) or not field:
                continue
            for alias in field.split("|"):
                alias = alias.strip().upper()
                if not alias or alias in primary:
                    continue
                if alias in aliases and aliases[alias] != key:
                    ambiguous.add(alias)
                else:
                    aliases[alias] = key
    for alias in ambiguous:
        aliases.pop(alias, None)
    for alias, key in aliases.items():
        resolver.setdefault(alias, key)
    return resolver


def static_features(output: Path) -> None:
    """Builder of ``core_static``: the CCcfs, HELP annotation and orthology groups of the gene reference.

    Follows HELP's feature builder: each table of the authors is filled with its column means, its
    constant columns are dropped and it is standardised; symbols are matched to genes by approved
    symbol, then by unambiguous HGNC alias or previous symbol; rows matched to one gene are averaged,
    and genes the authors did not cover get the column means.
    """
    genes = _gene_reference("symbol")
    resolver = _symbol_resolver(genes)
    bio = _help_block(external.path("help_bio"), BIO)
    merged = bio.join([_help_block(external.path("help_cccfs"), CCCFS)], how="outer")
    merged[GENE_KEY] = [resolver.get(str(symbol).upper()) for symbol in merged.index]
    resolved = merged.dropna(subset=[GENE_KEY]).groupby(GENE_KEY, sort=True).mean(numeric_only=True)
    keys = pd.Index(sorted(genes[GENE_KEY].astype(str).unique()), name=GENE_KEY)
    features = resolved.reindex(keys)
    features = features.fillna(features.mean(numeric_only=True))
    orthologues = pd.read_parquet(external.path("orthology"))
    orthologues = orthologues.set_index(orthologues[GENE_KEY].astype(str)).drop(columns=GENE_KEY)
    orthologues = orthologues.reindex(keys)
    if orthologues.isna().any().any():
        raise ValueError("the orthology table lacks genes of the gene reference; rebuild it")
    columns = [c for c in features.columns if c.startswith(CCCFS)]
    columns += [c for c in features.columns if c.startswith(BIO)]
    table = pd.concat([features[columns], orthologues], axis=1).rename_axis(GENE_KEY).reset_index()
    table.to_parquet(output, index=False)


@lru_cache(maxsize=1)
def _static() -> Static:
    (path,) = external.ensure("core_static")
    table = pd.read_parquet(path)
    columns = list(table.columns)
    groups = {
        "cccfs": [c for c in columns if c.startswith(CCCFS)],
        "help_bio": [c for c in columns if c.startswith(BIO)],
        "orthology": [c for c in columns if c.startswith(ORTHOLOG)] + [ORTHOLOGY_MISSING],
    }
    index = pd.Index(table[GENE_KEY].astype(str), name=GENE_KEY)
    return Static(index, {name: table[group].to_numpy(dtype=np.float64) for name, group in groups.items()})


@lru_cache(maxsize=1)
def _expression() -> tuple[pd.Index, np.ndarray]:
    (path,) = external.ensure("tcga_expression")
    table = pd.read_parquet(path)
    keys = pd.Index(table[GENE_KEY].astype(str), name=GENE_KEY)
    if keys.duplicated().any():
        raise ValueError(f"{path}: a gene appears twice")
    return keys, table.drop(columns=GENE_KEY).to_numpy(dtype=np.float64)


@lru_cache(maxsize=1)
def _expressed() -> frozenset[str]:
    (path,) = external.ensure("tcga_expression")
    keys = pq.read_table(path, columns=[GENE_KEY]).column(0).to_pylist()
    return frozenset(str(key) for key in keys)


@lru_cache(maxsize=1)
def _reference_symbols() -> pd.Series:
    genes = _gene_reference("symbol")
    return pd.Series(genes["symbol"].astype(str).to_numpy(), index=genes[GENE_KEY].astype(str).to_numpy())


@lru_cache(maxsize=1)
def _resolver() -> dict[str, str]:
    return _symbol_resolver(_gene_reference("symbol"))


@lru_cache(maxsize=1)
def _unmatched() -> str:
    """A gene of the gene reference that no symbol of the HELP tables matches; HELP gives it column means."""
    matched: set[str | None] = set()
    for item in ("help_bio", "help_cccfs"):
        symbols = pd.read_csv(external.path(item), usecols=[0]).iloc[:, 0].map(str)
        matched.update(_resolver().get(symbol.upper()) for symbol in symbols)
    keys = [key for key in _static().index if key not in matched]
    if not keys:
        raise ValueError("every gene of the gene reference has HELP features; no unannotated row exists")
    return keys[0]


@lru_cache(maxsize=1)
def _unexpressed() -> str:
    """A gene of the gene reference without a TCGA row; its components are those of the median profile."""
    keys = [key for key in _static().index if key not in _expressed()]
    if not keys:
        raise ValueError("every gene of the gene reference has a TCGA row, so no unannotated components exist")
    return keys[0]


def _logged(values: np.ndarray) -> np.ndarray:
    """log2(x + 1) in place, as EssSubgraph transforms the RSEM values."""
    values += 1.0
    return np.log2(values, out=values)


def _fill(values: np.ndarray, medians: np.ndarray) -> np.ndarray:
    missing = np.isnan(values)
    values[missing] = np.broadcast_to(medians, values.shape)[missing]
    return values


def _fit_components(keys: pd.Index, values: np.ndarray, genes: list[str]) -> dict:
    """EssSubgraph's expression PCA: log2(x + 1), per-sample median fill, standardisation, 50 components."""
    train = values[keys.get_indexer(genes)]
    if np.nanmin(train) < 0 or not np.isfinite(np.nanmax(train)):
        raise ValueError("TCGA expression values must be non-negative and finite, or missing")
    medians = np.nanmedian(_logged(train), axis=0)
    _fill(train, medians)
    means = train.mean(axis=0)
    scales = train.std(axis=0, ddof=0)
    scales[scales == 0.0] = 1.0
    train -= means
    train /= scales
    pca = PCA(n_components=COMPONENTS, svd_solver="full").fit(train)
    return {"medians": medians, "means": means, "scales": scales, "pca": pca}


def _project(fitted: dict, keys: pd.Index, values: np.ndarray, genes: pd.Index) -> np.ndarray:
    """The components of the genes; a gene without a TCGA row is filled entirely with the medians."""
    rows = keys.get_indexer(genes)
    projected = np.empty((len(genes), COMPONENTS), dtype=np.float64)
    for start in range(0, len(genes), PROJECTION_ROWS):
        part = rows[start : start + PROJECTION_ROWS]
        raw = np.full((len(part), values.shape[1]), np.nan)
        raw[part >= 0] = values[part[part >= 0]]
        _fill(_logged(raw), fitted["medians"])
        raw -= fitted["means"]
        raw /= fitted["scales"]
        projected[start : start + len(part)] = fitted["pca"].transform(raw)
    return projected


def _blank(static: Static, fitted: dict) -> dict[str, np.ndarray]:
    """The features of a gene without any annotation, given to test genes outside the gene reference."""
    orthology = np.zeros(static.blocks["orthology"].shape[1])
    orthology[-1] = 1.0
    row = static.index.get_loc(_unmatched())
    return {
        "cccfs": static.blocks["cccfs"][row],
        "help_bio": static.blocks["help_bio"][row],
        "orthology": orthology,
        "tcga_pca": fitted["components"][fitted["index"].get_loc(_unexpressed())],
    }


def _design(static: Static, fitted: dict, keys: Sequence[str]) -> np.ndarray:
    genes = pd.Index(np.asarray(keys, dtype=object))
    rows = static.index.get_indexer(genes)
    places = fitted["index"].get_indexer(genes)
    if ((rows < 0) != (places < 0)).any():
        raise ValueError("the fold model and the core feature table hold different genes; train the task again")
    outside = rows < 0
    blank = _blank(static, fitted) if outside.any() else {}
    sources = {name: (block, rows) for name, block in static.blocks.items()}
    sources["tcga_pca"] = (fitted["components"], places)
    parts = {}
    for name, (block, positions) in sources.items():
        part = block[np.maximum(positions, 0)]
        if outside.any():
            part[outside] = blank[name]
        parts[name] = part
    return np.ascontiguousarray(np.hstack([parts[key] for key in CORE_BLOCKS]))


def _spec(task: str) -> Spec:
    try:
        cohort_id = task_id(task)
    except ValueError:
        name = str(task).strip()
        if CUSTOM_NAME.fullmatch(name) and (CUSTOM / name / CUSTOM_LABELS).is_file():
            return Spec(name, name, TargetKind.BINARY, True)
        made = sorted(path.parent.name for path in CUSTOM.glob(f"*/{CUSTOM_LABELS}"))
        raise ValueError(
            f"unknown task {task!r}; the benchmark tasks are {', '.join(TASKS)}, the tasks made by "
            f"make_task are: {', '.join(made) or 'none yet'}"
        ) from None
    return Spec(cohort_id, prediction_cohort(cohort_id), cohort_target(cohort_id).kind, False)


def _labelled(keys: pd.Series, labels: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {GENE_KEY: keys.astype(str).to_numpy(object), "label": pd.to_numeric(labels).to_numpy(np.float64)}
    )


def _custom_file(name: str, file: str) -> Path:
    path = CUSTOM / name / file
    if not path.is_file():
        raise ValueError(f"{path} is missing; make the task again with make_task")
    return path


def _training(spec: Spec, outer_fold: int) -> tuple[pd.DataFrame, list[str]]:
    """The training genes and labels in the order the learner sees them, and the genes of the PCA fit."""
    if spec.custom:
        path = _custom_file(spec.name, f"fold_{outer_fold}.csv")
        table = pd.read_csv(path, dtype={GENE_KEY: object})
        train = table.loc[table["role"].eq(TRAIN_ROLE)]
        labels = pd.to_numeric(train["label"], errors="coerce")
        if not labels.isin((0.0, 1.0)).all() or labels.nunique() < 2:
            raise ValueError(
                f"fold {outer_fold} of {spec.name}: the train genes need labels 0 and 1, both present"
            )
        return _labelled(train[GENE_KEY], labels), sorted(train[GENE_KEY].astype(str))
    column = cohort_target(spec.name).column
    train = read_training_labels(spec.name, outer_fold, TRAIN_ROLE, split_strategy=SPLIT)
    directory = folds_directory(spec.name, split_strategy=SPLIT) / "fold_genes"
    roles = pd.read_parquet(directory / f"fold_{outer_fold}_genes.parquet", columns=[GENE_KEY, "role"])
    genes = sorted(roles.loc[roles["role"].eq(TRAIN_ROLE), GENE_KEY].astype(str))
    return _labelled(train[GENE_KEY], train[column]), genes


def _tests(spec: Spec, outer_fold: int) -> pd.DataFrame:
    """The test genes of one fold in evaluation order, their labels and whether they are scored."""
    if spec.custom:
        truth = pd.read_csv(_custom_file(spec.name, CUSTOM_LABELS), dtype={GENE_KEY: object})
        held = truth.loc[truth[FOLD].eq(outer_fold)]
        return _labelled(held[GENE_KEY], held["label"]).assign(scored=True)
    column = cohort_target(spec.name).column
    truth = read_evaluation_truth(spec.name, outer_fold, split_strategy=SPLIT)
    kept = scope_frame(truth, cohort_id=spec.name, scope=primary_scope(spec.name))
    frame = _labelled(truth[GENE_KEY], truth[column])
    return frame.assign(scored=frame[GENE_KEY].isin(set(kept[GENE_KEY].astype(str))).to_numpy())


def _fold_path(directory: Path, outer_fold: int) -> Path:
    return directory / f"fold_{outer_fold}.joblib"


def _fit_fold(job: tuple[str, int, str]) -> dict:
    """Fit the principal components and the learner of one fold and write them to the cache."""
    name, outer_fold, directory = job
    started = time.perf_counter()
    train, genes = _training(_spec(name), outer_fold)
    static = _static()
    keys, values = _expression()
    expressed = set(keys)
    fitted = _fit_components(keys, values, [gene for gene in genes if gene in expressed])
    fitted["index"] = static.index
    fitted["components"] = _project(fitted, keys, values, static.index)
    train = train.loc[train[GENE_KEY].isin(static.index)]
    model = make_learner()
    model.fit(_design(static, fitted, train[GENE_KEY]), train["label"].to_numpy(np.float64))
    stored = {
        "model": model,
        "genes": static.index.to_numpy(object),
        "components": fitted["components"],
        "rounds": int(model.n_iter_),
        "seconds": time.perf_counter() - started,
    }
    _replace(_fold_path(Path(directory), outer_fold), lambda path: joblib.dump(stored, path))
    return {"task": name, "fold": outer_fold, "rounds": stored["rounds"], "seconds": stored["seconds"]}


@lru_cache(maxsize=2 * FOLD_COUNT)
def _load(path: str, stamp: float) -> dict:
    stored = joblib.load(path)
    stored["index"] = pd.Index(stored["genes"], name=GENE_KEY)
    return stored


def _fitted(directory: Path, outer_fold: int) -> dict:
    path = _fold_path(directory, outer_fold)
    return _load(str(path), path.stat().st_mtime)


def _scores(static: Static, fitted: dict, keys: Sequence[str]) -> np.ndarray:
    return fitted["model"].predict_proba(_design(static, fitted, keys))[:, 1]


def _inputs_time(spec: Spec) -> float:
    if not spec.custom:
        return 0.0
    files = [CUSTOM_LABELS, *(f"fold_{k}.csv" for k in range(FOLD_COUNT))]
    return max(_custom_file(spec.name, file).stat().st_mtime for file in files)


def _untrained(spec: Spec) -> list[tuple[str, int, str]]:
    models = _spec(spec.models)
    directory = MODELS / models.name
    newest = _inputs_time(models)
    jobs = []
    for outer_fold in range(FOLD_COUNT):
        path = _fold_path(directory, outer_fold)
        if not (path.is_file() and path.stat().st_mtime >= newest):
            jobs.append((models.name, outer_fold, str(directory)))
    return jobs


def _fit_all(jobs: list[tuple[str, int, str]], workers: int) -> None:
    """Fit the folds, one after another or in spawned processes, printing progress to standard error."""
    if not jobs:
        return
    started = last = time.perf_counter()

    def report(done: int) -> None:
        seconds = time.perf_counter() - started
        print(f"[core model] {done}/{len(jobs)} fold models fitted, {seconds:.0f} s", file=sys.stderr, flush=True)

    if workers <= 1:
        for done, job in enumerate(jobs, start=1):
            _fit_fold(job)
            report(done)
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context) as pool:
        pending = {pool.submit(_fit_fold, job) for job in jobs}
        done = 0
        while pending:
            finished, pending = wait(pending, timeout=PROGRESS_SECONDS, return_when=FIRST_COMPLETED)
            for future in finished:
                future.result()
                done += 1
            if finished or time.perf_counter() - last >= PROGRESS_SECONDS:
                report(done)
                last = time.perf_counter()


def _intervals(frame: pd.DataFrame, kind: TargetKind) -> dict[str, dict[str, float]]:
    functions = {"auprc": auprc, "auroc": auroc} if kind is TargetKind.BINARY else {"spearman": spearman}
    intervals = confidence_intervals(
        frame["label"].to_numpy(np.float64),
        frame[SCORE].to_numpy(np.float64),
        functions,
        fold=frame[FOLD].to_numpy(np.int64),
        target_kind=kind,
    )
    return {name: {"point": item.point, "half_width": item.half_width} for name, item in intervals.items()}


def _trained_folds(spec: Spec) -> list[int]:
    return [k for k in range(FOLD_COUNT) if _fold_path(MODELS / spec.models, k).is_file()]


def _fold_scores(spec: Spec, outer_fold: int) -> pd.DataFrame:
    """The test genes of one fold in evaluation order, scored by that fold's model."""
    test = _tests(spec, outer_fold)
    test[SCORE] = _scores(_static(), _fitted(MODELS / spec.models, outer_fold), test[GENE_KEY])
    test[FOLD] = outer_fold
    return test


def _summary(spec: Spec) -> dict:
    """Score the test genes of each fold with that fold's model, write the submission table, score it."""
    directory = MODELS / spec.name
    models = MODELS / spec.models
    path = directory / SUMMARY
    newest = max(_fold_path(models, k).stat().st_mtime for k in range(FOLD_COUNT))
    if path.is_file() and path.stat().st_mtime >= newest:
        stored = json.loads(path.read_text())
        if "without_features" in stored:
            return stored
    static = _static()
    table = pd.concat([_fold_scores(spec, k) for k in range(FOLD_COUNT)], ignore_index=True)
    _replace(directory / PREDICTIONS, lambda file: table[[GENE_KEY, SCORE, FOLD]].to_csv(file, index=False))
    scored = table.loc[table["scored"]]
    fitted = [_fitted(models, k) for k in range(FOLD_COUNT)]
    summary = {
        "metrics": _intervals(scored, spec.kind),
        "scored_genes": int(len(scored)),
        "rounds": [item["rounds"] for item in fitted],
        "training_seconds": sum(item["seconds"] for item in fitted),
        "without_features": table.loc[~table[GENE_KEY].isin(static.index), GENE_KEY].tolist(),
    }
    _replace(path, lambda file: file.write_text(json.dumps(summary, indent=1) + "\n"))
    return summary


def _published_point(method_id: str, cohort_id: str) -> float:
    table = pooled(predictor_folds(method_id, cohort_id), cohort_id)
    column = cohort_target(cohort_id).column
    values = [auprc(part[column].to_numpy(), part[SCORE].to_numpy()) for _, part in table.groupby(FOLD)]
    return float(np.mean(values))


def _leader(spec: Spec) -> dict | None:
    if spec.custom:
        return None
    if spec.name in LEADERBOARD_COHORT_IDS:
        row = ranked_points(spec.name).iloc[0]
        return {
            "predictor": row["display_name"],
            "metric": row["metric"],
            "point": number(row["point"]),
            "half_width": number(row["half_width"]),
        }
    points = {method_id: _published_point(method_id, spec.name) for method_id in METHOD_IDS}
    best = max(points, key=points.get)
    return {
        "predictor": display_names()[best],
        "metric": "auprc",
        "point": number(points[best]),
        "half_width": None,
        "note": "the task is not on the leaderboard; best of the 15 published predictors by point AUPRC",
    }


def _exact_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={GENE_KEY: object}, float_precision="round_trip")


def _manuscript_row(cohort_id: str) -> pd.Series:
    table = _exact_csv(CORE_FIXED)
    return table.loc[table["first"].eq("A") & table["cohort_id"].eq(cohort_id)].iloc[0]


def _metric_pairs(row: pd.Series, metrics: dict) -> list[tuple[str, float, float]]:
    """(quantity, published, reproduced) for the primary metric and AUROC, point and half-width."""
    metric = row["metric"]
    pairs = [
        (f"{metric} point", row["first_point"], metrics[metric]["point"]),
        (f"{metric} half-width", row["first_half_width"], metrics[metric]["half_width"]),
    ]
    if "auroc" in metrics:
        pairs += [
            ("auroc point", row["first_auroc"], metrics["auroc"]["point"]),
            ("auroc half-width", row["first_auroc_half_width"], metrics["auroc"]["half_width"]),
        ]
    return [(quantity, float(published), float(reproduced)) for quantity, published, reproduced in pairs]


def _matches(cohort_id: str, summary: dict) -> bool:
    row = _manuscript_row(cohort_id)
    rounds = [int(value) for value in str(row["first_rounds"]).split(";")]
    pairs = _metric_pairs(row, summary["metrics"])
    return rounds == summary["rounds"] and all(abs(a - b) <= TOLERANCE for _, a, b in pairs)


def _is_file(text: str) -> bool:
    path = Path(text).expanduser()
    return (path if path.is_absolute() else ROOT / path).is_file()


def _requested(genes: str) -> list[str]:
    text = str(genes).strip()
    if not text:
        return []
    if "," not in text and _is_file(text):
        try:
            file, table = read_table(text)
        except (ValueError, OSError) as error:
            raise ValueError(
                f"{text} could not be read as a table ({error}); give a CSV, TSV or parquet file with "
                f"a gene column, or comma-separated genes"
            ) from None
        named = {str(column).strip().lower(): column for column in table.columns}
        only = table.columns[0] if table.shape[1] == 1 else None
        column = next((named[c] for c in GENE_COLUMNS if c in named), only)
        if column is None:
            raise ValueError(
                f"{file}: no gene column; name it one of {', '.join(GENE_COLUMNS)}, its columns are "
                f"{list(table.columns)}"
            )
        values = table[column].dropna().astype(str)
    else:
        values = text.split(",")
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _resolve(text: str) -> str | None:
    upper = text.strip().upper()
    if not upper.startswith("HGNC:"):
        return _resolver().get(upper)
    digits = upper[5:].strip()
    key = f"HGNC:{int(digits)}" if digits.isdigit() else text
    return key if key in _reference_symbols().index else None


def _genes(spec: Spec, requested: list[str]) -> dict:
    """Score the requested genes: a test gene by the model of its fold, any other by the five models' mean."""
    static = _static()
    predictions = _exact_csv(MODELS / spec.name / PREDICTIONS).set_index(GENE_KEY)
    ranked = np.sort(predictions[SCORE].to_numpy(np.float64))
    tests = pd.concat([_tests(spec, k) for k in range(FOLD_COUNT)], ignore_index=True)
    truth = tests.set_index(GENE_KEY)["label"]
    continuous = spec.kind is TargetKind.CONTINUOUS
    label = "mean_gene_effect" if continuous else "label"
    rows, without, others = [], [], []
    for text in requested:
        key = _resolve(text)
        if key is None:
            without.append({"gene": text, "reason": NOT_FOUND})
            continue
        row = {"gene": text, "gene_key": key, "symbol": _reference_symbols().get(key)}
        if key in predictions.index:
            row["score"] = float(predictions.at[key, SCORE])
            row["prediction"] = f"test prediction of fold {int(predictions.at[key, FOLD])}"
            row[label] = float(truth[key]) if continuous else int(truth[key])
        else:
            others.append(len(rows))
            row["score"] = None
            row["prediction"] = "mean of the five fold models; the gene is not in the task"
        if key not in _expressed():
            row["note"] = NO_EXPRESSION
        rows.append(row)
    if others:
        keys = [rows[place]["gene_key"] for place in others]
        scores = [_scores(static, _fitted(MODELS / spec.models, k), keys) for k in range(FOLD_COUNT)]
        for place, score in zip(others, np.mean(scores, axis=0)):
            rows[place]["score"] = float(score)
    for row in rows:
        row["percentile"] = number(100.0 * np.searchsorted(ranked, row["score"], side="right") / len(ranked), 1)
        row["score"] = number(row["score"])
    result: dict = {"genes": rows[:GENE_LIMIT], "genes_without_features": without}
    if len(rows) > GENE_LIMIT:
        path = MODELS / spec.name / GENES_FILE
        _replace(path, lambda file: pd.DataFrame(rows).to_csv(file, index=False))
        result["all_genes_path"] = str(path)
        result["genes_note"] = f"the first {GENE_LIMIT} of {len(rows)} genes; all of them are in all_genes_path"
    return result


def _run(task: str, genes: str = "", workers: int = 1) -> dict:
    started = time.perf_counter()
    spec = _spec(task)
    requested = _requested(genes)
    external.ensure(*INPUTS)
    # Calls on the same task, from one server or from two that share a data root, fit and write its folds in turn.
    lock = MODELS / f"{_spec(spec.models).name}{external.LOCK_SUFFIX}"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        jobs = _untrained(spec)
        _fit_all(jobs, workers)
        if jobs:
            _expression.cache_clear()
        summary = _summary(spec)
    result: dict = {
        "task": spec.name,
        "custom_task": spec.custom,
        "model": MODEL_NOTE,
        "primary_metric": "spearman" if spec.kind is TargetKind.CONTINUOUS else "auprc",
        "scored_genes": summary["scored_genes"],
        "metrics": {
            name: {"point": number(item["point"]), "half_width": number(item["half_width"])}
            for name, item in summary["metrics"].items()
        },
        "boosting_rounds": summary["rounds"],
        "published_leader": _leader(spec),
        "predictions_path": str(MODELS / spec.name / PREDICTIONS),
        "trained_in_this_call": bool(jobs),
        "training_seconds": number(summary["training_seconds"], 1),
    }
    if spec.models != spec.name:
        result["note"] = f"scored by the fold models of {spec.models}, as the manuscript scores {spec.name}"
    if spec.name in LEADERBOARD_COHORT_IDS:
        result["matches_manuscript"] = _matches(spec.name, summary)
    if summary["without_features"]:
        result["test_genes_without_features"] = {
            "genes": len(summary["without_features"]),
            "examples": examples(summary["without_features"]),
            "note": UNANNOTATED,
        }
    if requested:
        result.update(_genes(spec, requested))
    result["next"] = f"pass predictions_path with task={spec.name} to evaluate" + (
        "" if spec.custom else ", and to compare to test it against the 15 published predictors"
    )
    result["call_seconds"] = number(time.perf_counter() - started, 1)
    return result


def core_model(
    task: Annotated[
        str,
        Field(
            description=(
                "benchmark task (deg, depmap, context, gene_effect, blood, mouse or depmap_23q4) or the "
                "name of a task made by make_task"
            )
        ),
    ],
    genes: Annotated[
        str,
        Field(
            description=(
                "genes to predict: comma-separated HGNC symbols or identifiers such as TP53,HGNC:9187, "
                "or the path of a CSV, TSV or parquet file with a gene_key, gene or symbol column; "
                "empty for none"
            )
        ),
    ] = "",
) -> dict:
    """Train the manuscript's core feature set model on a task and predict gene essentiality with it.

    The model is gradient boosting on four feature groups from public data: localisation
    confidences of 3,301 GO cellular component terms, 50 principal components of TCGA tumour
    expression, 18 multi-omics annotations and orthologues in 194 species. One model is trained on
    the training genes of each of the task's five protein-family folds and predicts that fold's test
    genes, exactly as in the manuscript, so on the benchmark tasks the numbers are the manuscript's.

    Returns AUPRC and AUROC (the Spearman correlation on gene_effect) as point and half_width of the
    95% gene-level bootstrap interval within folds, the boosting rounds of each fold, the task's
    published leader, and predictions_path: a CSV of the test-fold predictions with the columns
    gene_key, y_score and outer_fold, which evaluate and compare accept. For each gene in genes it
    returns score (predicted probability of being essential, or of a strong dependency on
    gene_effect), percentile among the task's test genes (100 is the highest score), the label for
    genes of the task, and how the score was made: a gene of the task gets the prediction of the
    model of the fold in which it is a test gene, any other gene the mean of the five fold models.
    Genes outside the gene reference (the protein-coding genes with a protein sequence) have no
    features: in genes they are listed separately with no score, and the test genes of a custom
    task among them are predicted as genes without any annotation, so that every test gene has a
    prediction.

    The first call on a task trains five models: 7.5 minutes for deg on 32 CPU cores with 9.4 GB
    of memory, longer for larger tasks or fewer cores. The models are cached, so later calls on the
    task take seconds. gene_effect uses the models of depmap. When the public data are not
    downloaded yet, the call says what to fetch.
    """
    return _run(task, genes)


def _row(
    check: str,
    reference: str,
    task: str = "",
    fold: int | None = None,
    quantity: str = "all values",
    values: int = 0,
    published: float = np.nan,
    reproduced: float = np.nan,
    difference: float = np.nan,
    exact: bool | None = None,
) -> dict:
    return {
        "check": check,
        "reference": reference,
        "task": task,
        "fold": fold,
        "quantity": quantity,
        "values": values,
        "published": published,
        "reproduced": reproduced,
        "largest_absolute_difference": difference,
        "exact": exact,
    }


def _arrays(check: str, reference: str, ours: np.ndarray, theirs: np.ndarray, **fields) -> dict:
    ours, theirs = np.asarray(ours, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
    if ours.shape != theirs.shape:
        fields = {key: value for key, value in fields.items() if key != "quantity"}
        return _row(check, reference, quantity=f"shape {ours.shape} against {theirs.shape}", exact=False, **fields)
    difference = float(np.max(np.abs(ours - theirs))) if ours.size else 0.0
    exact = bool(np.array_equal(ours, theirs))
    return _row(check, reference, values=int(ours.size), difference=difference, exact=exact, **fields)


def _absent(check: str, reference: str, **fields) -> dict:
    return _row(check, reference, quantity="reference absent", **fields)


def _check_orthology(methods: Path) -> list[dict]:
    reference = f"methods/{EPGAT_STATIC.as_posix()}"
    built = _orthology_table()
    if not (methods / EPGAT_STATIC).is_file():
        return [_absent("orthology table", reference)]
    table = pd.read_parquet(methods / EPGAT_STATIC)
    columns = [GENE_KEY] + [c for c in table.columns if c.startswith(ORTHOLOG)] + [ORTHOLOGY_MISSING]
    theirs = table[columns].astype({GENE_KEY: object})
    same_columns = list(built.columns) == columns
    same_genes = built[GENE_KEY].tolist() == theirs[GENE_KEY].tolist()
    rows = [
        _row("orthology table", reference, quantity="columns", values=len(columns) - 1,
             published=len(columns) - 1, reproduced=built.shape[1] - 1, exact=same_columns),
        _row("orthology table", reference, quantity="genes", values=len(theirs),
             published=len(theirs), reproduced=len(built), exact=same_genes),
    ]
    if same_columns and len(built) == len(theirs):
        row = _arrays("orthology table", reference, built[columns[1:]].to_numpy(), theirs[columns[1:]].to_numpy())
        row["exact"] = bool(built.equals(theirs))
        rows.append(row)
    return rows


def _check_static(methods: Path, cache: Path) -> list[dict]:
    static = _static()
    rows = []
    help_reference = f"methods/{HELP_MATRIX.as_posix()}"
    if (methods / HELP_MATRIX).is_file():
        matrix = pd.read_parquet(methods / HELP_MATRIX)
        matrix = matrix.set_index(matrix[GENE_KEY].astype(str)).reindex(static.index)
        for block, prefix in (("cccfs", CCCFS), ("help_bio", BIO)):
            theirs = matrix[[c for c in matrix.columns if str(c).startswith(prefix)]].to_numpy(np.float64)
            rows.append(_arrays("HELP matrix", help_reference, static.blocks[block], theirs, quantity=block))
    else:
        rows.append(_absent("HELP matrix", help_reference))
    epgat_reference = f"methods/{EPGAT_STATIC.as_posix()}"
    if (methods / EPGAT_STATIC).is_file():
        table = pd.read_parquet(methods / EPGAT_STATIC)
        table = table.set_index(table[GENE_KEY].astype(str)).reindex(static.index)
        columns = [c for c in table.columns if c.startswith(ORTHOLOG)] + [ORTHOLOGY_MISSING]
        theirs = table[columns].to_numpy()
        rows.append(_arrays("EPGAT static table", epgat_reference, static.blocks["orthology"], theirs,
                            quantity="orthology"))
    else:
        rows.append(_absent("EPGAT static table", epgat_reference))
    for cohort_id in LEADERBOARD_COHORT_IDS:
        directory = cache / cohort_id
        if not (directory / "genes.txt").is_file():
            continue
        places = static.index.get_indexer((directory / "genes.txt").read_text().split())
        if (places < 0).any():
            raise ValueError(f"{directory}: {int((places < 0).sum())} genes are not in the core feature table")
        for block in CACHE_BLOCKS:
            reference = f"analysis/feature_groups/work/cache/{cohort_id}/{block}.npy"
            theirs = np.load(directory / f"{block}.npy")
            rows.append(_arrays("analysis cache", reference, static.blocks[block][places], theirs,
                                task=cohort_id, quantity=block))
    if not any(row["check"] == "analysis cache" for row in rows):
        rows.append(_absent("analysis cache", "analysis/feature_groups/work/cache"))
    return rows


def _check_components(methods: Path) -> list[dict]:
    rows = []
    for cohort_id in dict.fromkeys(_spec(name).models for name in TASKS):
        for outer_fold in _trained_folds(_spec(cohort_id)):
            path = feature_directory(
                methods / "esssubgraph", cohort_id, split_strategy=SPLIT, outer_fold=outer_fold
            )
            reference = f"methods/{path.relative_to(methods).as_posix()}/features.parquet"
            if not (path / "features.parquet").is_file():
                rows.append(_absent("principal components", reference, task=cohort_id, fold=outer_fold))
                continue
            table = pd.read_parquet(path / "features.parquet")
            fitted = _fitted(MODELS / cohort_id, outer_fold)
            ours = fitted["components"][fitted["index"].get_indexer(table[GENE_KEY].astype(str))]
            theirs = table[COMPONENT_COLUMNS].to_numpy()
            rows.append(_arrays("principal components", reference, ours, theirs, task=cohort_id, fold=outer_fold))
    return rows


def _published(
    cohort_id: str, fixed: Path, prospective: Path | None
) -> tuple[pd.DataFrame | None, list[int] | None, str, str]:
    """The manuscript's test scores of the core feature set on a task, its rounds per fold, their sources."""
    if cohort_id == PROSPECTIVE:
        reference = "core feature set scores of analysis/prospective_validation/prospective_core.py"
        if prospective is None or not prospective.is_file():
            return None, None, reference, reference
        table = _exact_csv(prospective).rename(columns={SCORE: "published", "fold": FOLD})
        return table, table.groupby(FOLD)["rounds"].first().astype(int).tolist(), reference, reference
    rounds = [int(value) for value in str(_manuscript_row(cohort_id)["first_rounds"]).split(";")]
    rounds_reference = CORE_FIXED.relative_to(ROOT).as_posix()
    reference = f"analysis/feature_groups/work/fixed/{cohort_id}_A.csv"
    if not (fixed / f"{cohort_id}_A.csv").is_file():
        return None, rounds, reference, rounds_reference
    table = _exact_csv(fixed / f"{cohort_id}_A.csv")
    table = table.loc[table["role"].eq(TEST_ROLE)].rename(columns={SCORE: "published", "fold": FOLD})
    return table, rounds, reference, rounds_reference


def _check_folds(fixed: Path, prospective: Path | None) -> list[dict]:
    rows = []
    for name in TASKS:
        spec = _spec(name)
        trained = _trained_folds(spec)
        table, rounds, reference, rounds_reference = _published(name, fixed, prospective)
        for outer_fold in trained:
            if table is None:
                rows.append(_absent("test predictions", reference, task=name, fold=outer_fold))
            else:
                theirs = table.loc[table[FOLD].eq(outer_fold), [GENE_KEY, "published"]]
                ours = _fold_scores(spec, outer_fold)[[GENE_KEY, SCORE]]
                merged = theirs.merge(ours, on=GENE_KEY, how="left", validate="one_to_one")
                rows.append(_arrays("test predictions", reference, merged[SCORE].to_numpy(),
                                    merged["published"].to_numpy(), task=name, fold=outer_fold))
            if spec.models == name and rounds is not None:
                published, reproduced = rounds[outer_fold], _fitted(MODELS / name, outer_fold)["rounds"]
                rows.append(_row("boosting rounds", rounds_reference, task=name, fold=outer_fold,
                                 quantity="rounds", values=1, published=published, reproduced=reproduced,
                                 difference=abs(published - reproduced), exact=published == reproduced))
        missing = [str(k) for k in range(FOLD_COUNT) if k not in trained]
        if missing and spec.models == name:
            rows.append(_row("fold models", (MODELS / name).relative_to(ROOT).as_posix(), task=name,
                             quantity=f"folds not trained: {', '.join(missing)}"))
    return rows


def _check_core_fixed(summaries: dict[str, dict]) -> list[dict]:
    reference = CORE_FIXED.relative_to(ROOT).as_posix()
    rows = []
    for cohort_id, summary in summaries.items():
        if cohort_id not in LEADERBOARD_COHORT_IDS:
            continue
        for quantity, published, reproduced in _metric_pairs(_manuscript_row(cohort_id), summary["metrics"]):
            rows.append(_row("core feature set", reference, task=cohort_id, quantity=quantity, values=1,
                             published=published, reproduced=reproduced,
                             difference=abs(published - reproduced), exact=published == reproduced))
    return rows


def check(methods: Path, cache: Path, fixed: Path, prospective: Path | None = None) -> pd.DataFrame:
    """Compare the feature groups and the trained fold models with the manuscript's; fits nothing."""
    print("[check] orthology table", file=sys.stderr, flush=True)
    rows = _check_orthology(methods)
    external.ensure(*INPUTS)
    print("[check] static feature groups", file=sys.stderr, flush=True)
    rows += _check_static(methods, cache)
    print("[check] principal components and test predictions of the trained folds", file=sys.stderr, flush=True)
    rows += _check_components(methods)
    rows += _check_folds(fixed, prospective)
    complete = [spec for spec in map(_spec, TASKS) if len(_trained_folds(spec)) == FOLD_COUNT]
    summaries = {}
    for done, spec in enumerate(complete, start=1):
        summaries[spec.name] = _summary(spec)
        print(f"[check] {done}/{len(complete)} fully trained tasks scored", file=sys.stderr, flush=True)
    rows += _check_core_fixed(summaries)
    table = pd.DataFrame(rows).astype({"fold": "Int64"})
    _replace(CHECK_OUTPUT, lambda file: table.to_csv(file, index=False))
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the core feature set model, or check it against the manuscript."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="train and score a task, or the seven benchmark tasks with 'all'")
    training.add_argument("task")
    training.add_argument("--genes", default="", help="genes to predict, as the tool takes them")
    training.add_argument("--workers", type=int, default=1, help="processes fitting folds in parallel")
    checking = commands.add_parser(
        "check", help=f"compare the trained folds with the manuscript, write {CHECK_OUTPUT.relative_to(ROOT)}"
    )
    checking.add_argument("--methods", type=Path, default=ROOT / "methods",
                          help="directory holding the feature/ directories of help, epgat and esssubgraph")
    checking.add_argument("--cache", type=Path, default=FEATURE_GROUPS_WORK / "cache",
                          help="array cache written by analysis/feature_groups/blocks.py")
    checking.add_argument("--fixed", type=Path, default=FEATURE_GROUPS_WORK / "fixed",
                          help="scores written by analysis/feature_groups/core_fixed.py")
    checking.add_argument("--prospective", type=Path, default=None,
                          help="DepMap 23Q4 test scores of the core feature set (gene_key, fold, y_score, rounds)")
    arguments = parser.parse_args(argv)
    if arguments.command == "train" and arguments.task != "all":
        print(json.dumps(_run(arguments.task, arguments.genes, arguments.workers), indent=1), flush=True)
        return 0
    if arguments.command == "train":
        external.ensure(*INPUTS)
        _fit_all(sorted({job for name in TASKS for job in _untrained(_spec(name))}), arguments.workers)
        for name in TASKS:
            print(json.dumps(_run(name), indent=1), flush=True)
        return 0
    prospective = None if arguments.prospective is None else arguments.prospective.expanduser()
    table = check(
        arguments.methods.expanduser(), arguments.cache.expanduser(), arguments.fixed.expanduser(), prospective
    )
    with pd.option_context("display.width", 200, "display.max_rows", None, "display.max_columns", None):
        print(table.drop(columns="reference").to_string(index=False))
    print(f"wrote {CHECK_OUTPUT}")
    decisive = table.loc[table["check"].isin(("orthology table", "core feature set", "test predictions"))]
    return int(bool((decisive["largest_absolute_difference"].fillna(0.0) > TOLERANCE).any()))


if __name__ == "__main__":
    raise SystemExit(main())
