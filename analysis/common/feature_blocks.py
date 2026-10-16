"""The 22 feature groups read from the 15 methods' inputs, their array cache, the gradient boosting learner and the process pool.

Imported by feature_groups, essential_gene_hallmarks and research_attention; it runs nothing by itself.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from egbench.contracts import SEED, TargetKind
from egbench.data import (
    PROJECT_ROOT,
    TEST_ROLE,
    TRAIN_ROLE,
    VALIDATION_ROLE,
    cohort_target,
    primary_scope,
    read_evaluation_truth,
    read_training_labels,
)
from egbench.evaluate import scope_frame
from egbench.metrics.discrimination import auprc, auroc
from egbench.stats import confidence_intervals, paired_difference

from common.stats import METHODS_DIRECTORY, SPLIT, FOLDS, pooled, process_pool

FEATURE_GROUPS = PROJECT_ROOT / "analysis" / "feature_groups"
WORK = FEATURE_GROUPS / "work"
RESULTS = FEATURE_GROUPS / "results"
CACHE = WORK / "cache"
CCCFS_TERMS = WORK / "cccfs_terms.csv"
ANNOTATIONS = PROJECT_ROOT / "data" / "source" / "annotations"
GO_OBO = ANNOTATIONS / "go-basic.obo"
GO_OBO_SHA256 = "b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc"
EPGAT_SOURCE = METHODS_DIRECTORY / "epgat" / "feature" / "source"
EPGAT_EDGES = EPGAT_SOURCE / "network" / "biogrid_physical_edges.parquet"

ALONE_COHORTS: tuple[str, ...] = ("deg", "depmap", "context", "blood", "mouse")
FORWARD_COHORTS: tuple[str, ...] = ("deg", "depmap")
COHORT_NAMES: dict[str, str] = {
    "deg": "DEG", "depmap": "DepMap", "context": "Context", "blood": "Blood", "mouse": "Mouse",
}
LEADERS: dict[str, str] = {
    "deg": "mahi", "depmap": "help", "context": "help", "blood": "help", "mouse": "mahi",
}
METHOD_NAMES: dict[str, str] = {
    "bingo": "Bingo", "deepcelless": "DeepCellEss", "deephe": "DeepHE",
    "deeplyessential": "DeeplyEssential", "deepsnap": "DeEPsnap", "ednn_eg": "eDNN-EG",
    "epgat": "EPGAT", "esssubgraph": "EssSubgraph", "esstfnet": "EssTFNet",
    "gcnn_sfm": "GCNN-SFM", "help": "HELP", "mahi": "Mahi", "pheg": "Pheg",
    "proteomelm": "ProteomeLM-Ess", "xgep": "XGEP-XGB",
}

MAX_ROUNDS = 500
VALIDATION_FRACTION = 0.1
PERMUTATIONS = 5
FORWARD_STEPS = 8
CORE_TOLERANCE = 0.005
WORKERS = 12
CORE_BLOCKS: tuple[str, ...] = ("cccfs", "tcga_pca", "help_bio", "orthology")

GENE_KEY = "gene_key"
FOLD = "fold"
ROLE = "role"
SCORE = "y_score"
LABEL = "label"

CATEGORIES: dict[str, str] = {
    "sequence": "sequence composition",
    "pretrained": "pretrained embedding",
    "network": "network",
    "expression": "expression",
    "annotation": "annotation and localisation",
    "evolution": "evolution",
    "chromatin": "chromatin",
}


@dataclass(frozen=True)
class Block:
    key: str
    name: str
    category: str
    readers: tuple[str, ...]
    fold_dependent: bool
    per_column: bool
    source: str


BLOCKS: tuple[Block, ...] = (
    Block("seq_composition", "Codon and amino acid composition", "sequence",
          ("deephe", "deeplyessential", "deepsnap"), False, True,
          "All 89 columns of the DeeplyEssential fold table, identical to the first 89 columns of the DeepHE and "
          "DeEPsnap fold tables: 64 codon frequencies, 20 amino acid frequencies, GC content, coding sequence "
          "length, codon adaptation index, maximum relative synonymous codon usage and protein length"),
    Block("zcurve", "Z curve (800 columns selected per fold)", "sequence",
          ("pheg",), True, True,
          "The 800 columns of the Pheg fold table, selected per fold from 4,545 Z curve columns by support vector "
          "machine recursive feature elimination on the fold's training labels"),
    Block("fasttext", "Supervised fastText sequence representation", "sequence",
          ("ednn_eg",), True, False,
          "Columns fasttext_001 to fasttext_100 of the eDNN-EG fold table, trained on the fold's training genes"),
    Block("physchem", "Physicochemical properties and composition (4,250 columns selected per fold)", "sequence",
          ("esstfnet",), True, True,
          "The 4,250 columns of the EssTFNet fold table, selected per fold from 86,151 candidates by recursive "
          "feature elimination on the fold's training labels"),
    Block("esmc", "ESM-C 600M embedding", "pretrained",
          ("mahi",), False, False,
          "Dimensions 0 to 1,151 of the 3,666-dimensional Mahi vector after APPNP propagation on the MAGE global "
          "network"),
    Block("proteomelm", "ProteomeLM-L embedding", "pretrained",
          ("proteomelm",), False, False,
          "Columns plm_0 to plm_1151 of the ProteomeLM-Ess fold table, the layer 8 proteome representation of "
          "ProteomeLM-L"),
    Block("evo2", "Evo 2 7B embedding", "pretrained",
          ("esstfnet",), False, False,
          "The 4,096-dimensional Evo 2 embedding of the EssTFNet fold table, a weighted mean over coding sequence "
          "windows"),
    Block("esm2", "ESM-2 650M embedding", "pretrained",
          ("bingo",), False, False,
          "The residue mean of the ESM-2 650M residue embeddings Bingo reads (1,280 dimensions); Bingo itself reads "
          "the residue embeddings and contact maps"),
    Block("biogrid_n2v", "BioGRID node2vec", "network",
          ("deephe", "deepsnap"), False, False,
          "Columns node2vec_001 to node2vec_064 of the DeepHE fold table, identical to the same columns of the "
          "DeEPsnap fold table; BioGRID 5.0.259 physical interactions"),
    Block("string_n2v", "STRING node2vec", "network",
          ("help",), False, False,
          "Columns n2v__0 to n2v__127 of the HELP fold table, computed by the authors from the human STRING network "
          "with combined score at least 0.5"),
    Block("mahi_gnn", "Multiplex network GNN embedding", "network",
          ("mahi",), False, False,
          "Dimensions 3,154 to 3,665 of the Mahi vector: multiplex graph pretraining on 35 MAGE tissue networks, "
          "propagated by APPNP"),
    Block("ppi_degree", "PPI degree", "network",
          ("epgat",), False, True,
          "EPGAT's ppi_induced_degree: the degree in the BioGRID physical interaction graph induced on the task's "
          "genes, counted on the whole network as EPGAT counts it when scoring"),
    Block("xgep", "XGEP TCGA expression embedding", "expression",
          ("xgep",), False, False,
          "Columns xgep_000 to xgep_149 of the XGEP-XGB fold table, the authors' co-embedding of the TCGA pan-cancer "
          "RNA-seq matrix"),
    Block("tcga_pca", "TCGA expression principal components (50 fitted per fold)", "expression",
          ("esssubgraph", "epgat"), True, False,
          "Columns expression_pc_001 to expression_pc_050 of the EssSubgraph fold table, fitted on the fold's "
          "training genes; EPGAT reads the same components"),
    Block("cccfs", "GO cellular component localisation confidence", "annotation",
          ("help",), False, True,
          "Columns cc__0 to cc__3300 of the HELP fold table, the JensenLab COMPARTMENTS confidence of 3,301 GO "
          "cellular component terms"),
    Block("go_enrichment", "GO cellular component neighbourhood enrichment (100 terms selected per fold)",
          "annotation", ("deepsnap",), True, True,
          "Columns go_enrichment_001 to go_enrichment_100 of the DeEPsnap fold table: hypergeometric -log10 p of "
          "the gene and its BioGRID neighbours on the 100 GO cellular component terms most frequent among the "
          "fold's training positives"),
    Block("corum", "CORUM complex counts", "annotation",
          ("deepsnap",), False, True,
          "Columns corum_membership_count and corum_neighbor_complex_ratio_sum of the DeEPsnap fold table"),
    Block("pfam", "Pfam domain counts", "annotation",
          ("deepsnap",), False, True,
          "Columns pfam_domain_instance_count, pfam_unique_domain_type_count and pfam_sidf of the DeEPsnap fold "
          "table"),
    Block("compartments", "11 compartment confidences and missing indicator", "annotation",
          ("epgat",), False, True,
          "The 11 compartment__* columns and localization_missing of EPGAT's static feature table, JensenLab "
          "COMPARTMENTS integrated confidence"),
    Block("help_bio", "HELP multi-omics annotation (18 columns)", "annotation",
          ("help",), False, True,
          "Columns bio__0 to bio__17 of the HELP fold table, named in the header order of the authors' Human_BIO.csv"),
    Block("orthology", "Orthologues in 194 species and missing indicator", "evolution",
          ("epgat",), False, True,
          "The 194 ortholog__* columns and orthology_missing of EPGAT's static feature table, Ensembl Compara 115"),
    Block("deepsea", "DeepSEA chromatin features", "chromatin",
          ("mahi",), False, False,
          "Dimensions 1,152 to 3,153 of the Mahi vector: 2,002 DeepSEA chromatin features weighted by distance to "
          "the transcription start site, propagated by APPNP"),
)
BLOCK_BY_KEY: dict[str, Block] = {block.key: block for block in BLOCKS}
BLOCK_KEYS: tuple[str, ...] = tuple(block.key for block in BLOCKS)

HELP_BIO_NAMES: tuple[str, ...] = (
    "Gene_length", "Transcripts_count", "GC_content", "Gene_Disease_association",
    "Gtex_median_expr", "HPA_median_expr", "GO_MF", "GO_BP", "GO_CC", "BIOGRID",
    "KEGG", "REACTOME", "UCSC_TFBS", "UP_TISSUE", "orth_count",
    "Driver_genes_MUT_count", "Driver_genes_MET_count", "Driver_genes_CNV_count",
)


def cohort_cache(cohort: str) -> Path:
    return CACHE / cohort


def block_path(cohort: str, fold: int, key: str) -> Path:
    if BLOCK_BY_KEY[key].fold_dependent:
        return cohort_cache(cohort) / f"fold_{fold}" / f"{key}.npy"
    return cohort_cache(cohort) / f"{key}.npy"


def names_path(cohort: str, fold: int, key: str) -> Path:
    return block_path(cohort, fold, key).with_suffix(".names.json")


def write_table(frame: pd.DataFrame, name: str, directory: Path = RESULTS) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


_GENES: dict[str, pd.Index] = {}


def cohort_genes(cohort: str) -> pd.Index:
    if cohort not in _GENES:
        keys = (cohort_cache(cohort) / "genes.txt").read_text().split()
        _GENES[cohort] = pd.Index(keys, dtype="object", name=GENE_KEY)
    return _GENES[cohort]


def block_array(cohort: str, fold: int, key: str) -> np.ndarray:
    return np.load(block_path(cohort, fold, key), mmap_mode="r")


def block_names(cohort: str, fold: int, key: str) -> list[str]:
    return json.loads(names_path(cohort, fold, key).read_text())


def block_width(cohort: str, fold: int, key: str) -> int:
    return int(block_array(cohort, fold, key).shape[1])


def read_indexed(path, columns=None) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=None if columns is None else [GENE_KEY, *columns])
    frame[GENE_KEY] = frame[GENE_KEY].astype(str)
    if frame[GENE_KEY].duplicated().any():
        raise ValueError(f"{path}: a gene appears twice")
    return frame.set_index(GENE_KEY)


def read_vector(path, column: str, width: int) -> pd.DataFrame:
    table = pq.read_table(path, columns=[GENE_KEY, column])
    keys = [str(v) for v in table[GENE_KEY].to_pylist()]
    flat = np.asarray(table[column].combine_chunks().values.to_numpy(zero_copy_only=False), dtype="float64")
    if flat.size != len(keys) * width:
        raise ValueError(f"{path}: {column} is not {width} wide")
    return pd.DataFrame(flat.reshape(len(keys), width), index=pd.Index(keys, name=GENE_KEY))


def go_names() -> dict[str, str]:
    digest = hashlib.sha256(GO_OBO.read_bytes()).hexdigest()
    if digest != GO_OBO_SHA256:
        raise ValueError(f"{GO_OBO} has sha256 {digest}, expected {GO_OBO_SHA256}")
    names, alternatives, current = {}, {}, None
    for line in GO_OBO.read_text(encoding="utf-8").splitlines():
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


def induced_degree(genes: pd.Index) -> np.ndarray:
    edges = pd.read_parquet(EPGAT_EDGES, columns=["gene_key_a", "gene_key_b"]).astype(str)
    universe = set(genes)
    kept = edges[edges.gene_key_a.isin(universe) & edges.gene_key_b.isin(universe)
                 & (edges.gene_key_a != edges.gene_key_b)]
    left = np.where(kept.gene_key_a < kept.gene_key_b, kept.gene_key_a, kept.gene_key_b)
    right = np.where(kept.gene_key_a < kept.gene_key_b, kept.gene_key_b, kept.gene_key_a)
    pairs = pd.DataFrame({"a": left, "b": right}).drop_duplicates()
    counts = pd.concat([pairs.a, pairs.b]).value_counts()
    return counts.reindex(genes).fillna(0).to_numpy("float64")


@dataclass(frozen=True)
class Role:
    keys: np.ndarray
    rows: np.ndarray
    labels: np.ndarray


_ROLES: dict[tuple[str, int], dict[str, Role]] = {}


def fold_roles(cohort: str, fold: int) -> dict[str, Role]:
    if (cohort, fold) in _ROLES:
        return _ROLES[(cohort, fold)]
    target = cohort_target(cohort).column
    frames = {
        TRAIN_ROLE: read_training_labels(cohort, fold, TRAIN_ROLE, split_strategy=SPLIT),
        VALIDATION_ROLE: read_training_labels(cohort, fold, VALIDATION_ROLE, split_strategy=SPLIT),
        TEST_ROLE: scope_frame(
            read_evaluation_truth(cohort, fold, split_strategy=SPLIT),
            cohort_id=cohort, scope=primary_scope(cohort),
        ),
    }
    genes = cohort_genes(cohort)
    roles = {}
    for role, frame in frames.items():
        keys = frame[GENE_KEY].astype(str).to_numpy("object")
        rows = genes.get_indexer(keys)
        if (rows < 0).any():
            raise ValueError(f"{cohort} fold {fold}: {int((rows < 0).sum())} {role} genes are not in the cache")
        roles[role] = Role(keys=keys, rows=rows, labels=frame[target].to_numpy("float64"))
    _ROLES[(cohort, fold)] = roles
    return roles


def design(cohort: str, fold: int, keys: tuple[str, ...], rows: np.ndarray,
           extra: np.ndarray | None = None) -> np.ndarray:
    arrays = [block_array(cohort, fold, key) for key in keys]
    if extra is not None:
        arrays.append(extra)
    out = np.empty((len(rows), sum(int(a.shape[1]) for a in arrays)), dtype="float64")
    start = 0
    for array in arrays:
        stop = start + int(array.shape[1])
        out[:, start:stop] = array[rows]
        start = stop
    return out


def make_learner():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=MAX_ROUNDS,
        early_stopping=True,
        validation_fraction=VALIDATION_FRACTION,
        random_state=SEED,
    )


def fit_set(cohort: str, fold: int, keys: tuple[str, ...], extra: np.ndarray | None = None,
            keep_model: bool = False) -> dict:
    roles = fold_roles(cohort, fold)
    x_train = design(cohort, fold, keys, roles[TRAIN_ROLE].rows, extra)
    model = make_learner()
    model.fit(x_train, roles[TRAIN_ROLE].labels)
    del x_train
    result = {"n_iter": int(model.n_iter_), "columns": None}
    for role in (VALIDATION_ROLE, TEST_ROLE):
        x = design(cohort, fold, keys, roles[role].rows, extra)
        result["columns"] = int(x.shape[1])
        result[role] = model.predict_proba(x)[:, 1]
    if keep_model:
        result["model"] = model
    return result


def score_frame(cohort: str, fits: dict[int, dict]) -> pd.DataFrame:
    frames = []
    for fold in FOLDS:
        roles = fold_roles(cohort, fold)
        for role in (VALIDATION_ROLE, TEST_ROLE):
            frames.append(pd.DataFrame({
                GENE_KEY: roles[role].keys, FOLD: fold, ROLE: role,
                LABEL: roles[role].labels, SCORE: fits[fold][role],
            }))
    return pd.concat(frames, ignore_index=True)


def pooled_validation_auprc(frame: pd.DataFrame) -> float:
    rows = frame[frame[ROLE] == VALIDATION_ROLE]
    return float(auprc(rows[LABEL].to_numpy("float64"), rows[SCORE].to_numpy("float64")))


def fold_validation_auprc(frame: pd.DataFrame) -> float:
    rows = frame[frame[ROLE] == VALIDATION_ROLE]
    return float(np.mean([
        auprc(part[LABEL].to_numpy("float64"), part[SCORE].to_numpy("float64"))
        for _, part in rows.groupby(FOLD)
    ]))


def test_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame[ROLE] == TEST_ROLE].reset_index(drop=True)


def test_intervals(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    rows = test_rows(frame)
    result = confidence_intervals(
        rows[LABEL].to_numpy("float64"), rows[SCORE].to_numpy("float64"),
        {"auprc": auprc, "auroc": auroc}, fold=rows[FOLD].to_numpy(),
        target_kind=TargetKind.BINARY,
    )
    return {name: (float(v.point), float(v.half_width)) for name, v in result.items()}


def paired_test(frame_a: pd.DataFrame, scores_b: np.ndarray, metric=auprc) -> tuple[float, float, float]:
    rows = test_rows(frame_a)
    result = paired_difference(
        rows[LABEL].to_numpy("float64"), rows[SCORE].to_numpy("float64"),
        np.asarray(scores_b, dtype="float64"), metric, fold=rows[FOLD].to_numpy(),
        target_kind=TargetKind.BINARY,
    )
    return float(result.difference), float(result.half_width), float(result.p_value)


def leader_scores(cohort: str, keys: np.ndarray) -> np.ndarray:
    frame = pooled(LEADERS[cohort], cohort)
    frame[GENE_KEY] = frame[GENE_KEY].astype(str)
    scores = frame.set_index(GENE_KEY)[SCORE].reindex(pd.Index(keys, dtype="object")).to_numpy("float64")
    if np.isnan(scores).any():
        raise ValueError(f"{LEADERS[cohort]} does not score every test gene of {cohort}")
    return scores


def leader_frame(cohort: str, template: pd.DataFrame) -> pd.DataFrame:
    rows = test_rows(template).copy()
    rows[SCORE] = leader_scores(cohort, rows[GENE_KEY].to_numpy())
    return rows


def run_pool(function, jobs: list, workers: int, label: str, interval: float = 60.0) -> dict:
    started = time.time()
    results: dict = {}
    workers = max(1, min(workers, len(jobs)))
    print(f"[{label}] {len(jobs)} jobs on {workers} processes", flush=True)
    with process_pool(workers) as pool:
        futures = {pool.submit(function, job): job for job in jobs}
        pending = set(futures)
        last = started
        while pending:
            done, pending = wait(pending, timeout=interval, return_when=FIRST_COMPLETED)
            for future in done:
                results[futures[future]] = future.result()
            now = time.time()
            if now - last >= interval or not pending:
                print(f"[{label}] {len(results)}/{len(jobs)} done, {now - started:.0f} s", flush=True)
                last = now
    return results
