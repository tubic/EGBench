"""Builders of the method inputs that need a GPU or a pretrained model: BiNGO, ProteomeLM-Ess and EssTFNet.

Each builder runs the code of the method's own feature step in ``methods/`` on the task FASTA of
``agent/sequences.py`` and the candidate genes of ``agent/candidates/``, with the method's paths pointed
at the data root. ``bingo_graph_cache`` holds the ESM-2 650M residue embeddings and contact graphs of
the 17,121 DEG and DepMap candidate genes (``methods/bingo/code/cache_features.py``, proteins cut at
1,000 residues, the top 20% of contacts kept, batches of at most two proteins and 1,024 tokens); paths in
its manifest are relative to the cache. ``esmc_embeddings`` holds the mean ESM-C 600M embedding of every
candidate protein and ``proteomelm_features`` the layer-8 ProteomeLM-L matrix of each task
(``methods/proteomelm/code/embed_proteome.py``: the candidate genes in sorted order, encoded jointly in
blocks of 512, so a gene's values depend on its block). ``esstfnet_raw_candidates`` holds EssTFNet's
86,151 physicochemical candidate features of each task with their column table
(``methods/esstfnet/build_raw_candidates.py``, 16 genes a row group) and ``evo2_embeddings`` the Evo 2
7B base embedding of each candidate CDS (``methods/esstfnet/code/embed_dna.py``: windows of 8,192 nt,
the token mean of ``blocks.28.mlp.l3``, float16). A gene's protein and CDS are the same in every task, so
each gene is embedded once. The other four tasks have the 16,978 candidate genes of DepMap, so their files
are hard links to DepMap's, as the methods' inputs are. The GPU builders run on ``cuda:0``; set
``CUDA_VISIBLE_DEVICES`` to choose the card. ``agent/methods.json`` maps every file to the method inputs it stands for.

    python agent/embeddings.py check {inputs,bingo,evo2} --methods ORIGINAL

compares the items under the data root, built beforehand with ``agent/external.py build``, with the
inputs under ORIGINAL, the ``methods`` directory of the manuscript's working copy, and writes the rows of
that part to ``agent/results/embeddings_check.csv``. ``inputs`` compares the downloads byte by byte, the
tables value by value and the ProteomeLM archive file by file; it needs pandas and pyarrow only.
``bingo`` rebuilds a sample of 100 genes in whole batches of the full build (BiNGO's environment, a GPU)
and ``evo2`` embeds a sample of 20 genes, 6 of them longer than one window (the evo2 environment, a GPU).
"""
from __future__ import annotations

import argparse
import contextlib
import filecmp
import os
import shutil
import sys
import tarfile
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import external
import inputs
from method_inputs import _method

if str(external.ROOT / "src") not in sys.path:
    sys.path.insert(0, str(external.ROOT / "src"))
from egbench.contracts import SEED  # noqa: E402

CANDIDATES = external.AGENT / "candidates"
CHECK_OUTPUT = external.AGENT / "results" / "embeddings_check.csv"
GENE_KEY = "gene_key"
COMPUTED = ("deg", "depmap")
DONOR = "depmap"
DERIVED = ("context", "blood", "mouse", "depmap_23q4")
TASKS = (*COMPUTED, *DERIVED)
DEVICE = "cuda:0"
PROGRESS_SECONDS = 60.0
BINGO_TOKEN_BUDGET = 1024
BINGO_BATCH_SIZE = 2
BINGO_SAMPLE = {"longer than the cut": 10, "single": 20, "pairs": 35}
PROTEOMELM_LAYER = 8
RAW = "raw_candidates_86151.parquet"
RAW_COLUMNS = "raw_candidate_columns.parquet"
RAW_ROWS_PER_GROUP = 16
EVO2 = "evo2_7b_base_embeddings.parquet"
EVO2_SAMPLE = 20
EVO2_LONG = 6
COMPARED_ROWS = 512
# The items compared with the methods' inputs, in the order of the check record.
CHECKED = (
    "esm2_650m", "esm2_650m_contact", "esmc_600m", "proteomelm_l_config", "proteomelm_l_weights",
    "esstfnet_feature_columns", "bingo_graph_cache", "esmc_embeddings", "proteomelm_features",
    "esstfnet_raw_candidates", "evo2_embeddings",
)
SAMPLED = ("bingo_graph_cache", "evo2_embeddings")
UPSTREAM = "proteomelm/code/upstream"


class _Progress:
    """Prints how many of a known number of genes are done, at most every PROGRESS_SECONDS."""

    def __init__(self, label: str, total: int) -> None:
        self.label, self.total = label, total
        self.started = self.last = time.monotonic()

    def update(self, done: int) -> None:
        now = time.monotonic()
        if now - self.last >= PROGRESS_SECONDS or done == self.total:
            self.last = now
            minutes = (now - self.started) / 60
            print(f"{self.label}: {done}/{self.total} genes, {minutes:.1f} min", file=sys.stderr, flush=True)


def _candidates(task: str) -> list[str]:
    path = CANDIDATES / f"{task}_common_feature_available.parquet"
    return sorted(pd.read_parquet(path, columns=[GENE_KEY])[GENE_KEY].astype(str))


def _proteins(task: str) -> Path:
    return external.path("protein_sequences") / f"{task}.fa.gz"


def _cds(task: str) -> Path:
    return external.path("cds_sequences") / f"{task}.fa.gz"


def _link_derived(output: Path, name: str) -> None:
    """Hard-link DepMap's file to the four other tasks; ``name`` holds ``{task}``."""
    donor = output / name.format(task=DONOR)
    for task in DERIVED:
        target = output / name.format(task=task)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(donor, target)


# BiNGO


def _bingo():
    """BiNGO's cache code and its gene-merging step, the cache code's paths freed from the method directory."""
    cache = _method("bingo/code/cache_features.py", "bingo/code")
    extract = _method("bingo/code/extract_features.py", "bingo/code")
    cache.require_method_path = lambda path, label: Path(path).resolve()
    return cache, extract


def _bingo_genes(cache, extract) -> tuple[dict[str, list[str]], tuple[str, ...], dict[str, str], tuple[str, ...]]:
    """Candidate genes of DEG and DepMap, their union, the union's proteins and the genes the two share."""
    genes = {task: _candidates(task) for task in COMPUTED}
    sequences = {task: cache.read_protein_fasta(_proteins(task)) for task in COMPUTED}
    union, merged, overlap = extract.merge_gene_sequences(
        deg_genes=genes["deg"],
        depmap_genes=genes["depmap"],
        deg_sequences=sequences["deg"],
        depmap_sequences=sequences["depmap"],
    )
    return genes, union, merged, overlap


def _bingo_build(cache, output: Path, genes, merged: dict[str, str], gene_sources: tuple[dict, ...]) -> dict:
    import torch

    cache._method_relative = lambda path: os.path.relpath(path, output)
    progress = _Progress("BiNGO graph cache", len(genes))
    write_progress = cache._write_progress

    def reporting(path: Path, **record) -> None:
        write_progress(path, **record)
        progress.update(record["completed"])

    cache._write_progress = reporting
    return cache.build_cache_from_sequences(
        genes=genes,
        sequences=merged,
        asset_dir=external.path("esm2_650m").parent,
        cache_dir=output,
        device=torch.device(DEVICE),
        token_budget=BINGO_TOKEN_BUDGET,
        max_batch_size=BINGO_BATCH_SIZE,
        source_paths=tuple(_proteins(task) for task in COMPUTED),
        gene_sources=gene_sources,
    )


def bingo_graph_cache(output: Path) -> None:
    """Builder of ``bingo_graph_cache``: graphs/, cache_index.parquet and cache_manifest.json.

    Runs ``build_cache_from_sequences`` over the union of the DEG and DepMap candidate genes, whose
    shared proteins must agree, and adds the gene counts that ``extract_features.py`` adds.
    """
    cache, extract = _bingo()
    genes, union, merged, overlap = _bingo_genes(cache, extract)
    sources = tuple(
        {
            "path": f"agent/candidates/{task}_common_feature_available.parquet",
            "gene_count": len(genes[task]),
            "gene_key_sha256": cache.gene_key_sha256(genes[task]),
        }
        for task in COMPUTED
    )
    manifest = _bingo_build(cache, output, union, merged, sources)
    manifest.update(
        dataset_gene_counts={task: len(genes[task]) for task in COMPUTED},
        union_gene_count=len(union),
        overlap_gene_count=len(overlap),
    )
    cache._write_json_atomic(output / "cache_manifest.json", manifest)


# ProteomeLM-Ess


def _proteomelm():
    """ProteomeLM's embedding code, pointed at the ESM-C download, and its build constants."""
    code = _method("proteomelm/code/embed_proteome.py")
    weights = external.path("esmc_600m")
    code.ESMC_WEIGHTS = weights
    code.ESMC_RUNTIME_ROOT = weights.parents[2]
    return code, _method("proteomelm/build_features.py")


def _proteomelm_genes(code, task: str) -> tuple[list[str], list[str]]:
    """The candidate genes of a task that its FASTA holds, in candidate order, and their proteins."""
    requested = code.read_gene_list(CANDIDATES / f"{task}_common_feature_available.parquet")
    labels, sequences = code.read_fasta(_proteins(task))
    protein = dict(zip(labels, sequences))
    keys = code.select_common_genes(requested, labels)
    return keys, [protein[key] for key in keys]


def esmc_embeddings(output: Path) -> None:
    """Builder of ``esmc_embeddings``: the mean ESM-C 600M embedding of the DEG and DepMap candidate proteins.

    Runs ``load_or_compute_esm`` for DEG and then DepMap, as ``build_features.py`` does: bfloat16,
    chunks of 512 genes, batches of 2,048 residues, proteins cut at 4,096 residues.
    """
    code, features = _proteomelm()
    with contextlib.redirect_stdout(sys.stderr):
        for task in COMPUTED:
            labels, sequences = _proteomelm_genes(code, task)
            code.load_or_compute_esm(
                labels, sequences, output, DEVICE, features.TOKEN_BUDGET, features.ESM_CACHE_CHUNK_GENES
            )


def _upstream(archive: Path, directory: Path) -> Path:
    """Unpack the ProteomeLM release into ``directory`` and return the folder that holds the package."""
    with tarfile.open(archive) as release:
        release.extractall(directory, filter="data")
    return next(directory.iterdir())


def proteomelm_features(output: Path) -> None:
    """Builder of ``proteomelm_features``: ``{task}_features.parquet``, ProteomeLM-L layer 8 of each task.

    Reads the ESM-C embeddings of ``esmc_embeddings`` and contextualises them with ProteomeLM-L in
    bfloat16, blocks of 512 genes in candidate order.
    """
    import torch

    code, features = _proteomelm()
    cache = code._read_embedding_cache(external.path("esmc_embeddings"), "esm")
    output.mkdir()
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(sys.stderr):
        sys.path.insert(0, str(_upstream(external.path("proteomelm_source"), Path(directory) / "upstream")))
        from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM

        model_directory = external.path("proteomelm_l_weights").parent
        model = ProteomeLMForMaskedLM.from_pretrained(str(model_directory), local_files_only=True)
        model = model.to(DEVICE, dtype=torch.bfloat16).eval()
        contextualize = code._contextualizer(model, DEVICE, PROTEOMELM_LAYER)
        for task in COMPUTED:
            labels, sequences = _proteomelm_genes(code, task)
            values, missing = code._cached_embeddings(cache, labels, sequences, "esm")
            if missing:
                raise ValueError(f"esmc_embeddings lacks {len(missing)} genes of {task}; rebuild it")
            matrix = code.contextualize_in_blocks(
                labels, sequences, values, features.ESM_BLOCK, Path(directory) / task, contextualize,
                cache_prefix=f"layer{PROTEOMELM_LAYER}_",
            )
            code._write_final_features(output / f"{task}_features.parquet", labels, matrix)
    _link_derived(output, "{task}_features.parquet")


# EssTFNet


def esstfnet_raw_candidates(output: Path) -> None:
    """Builder of ``esstfnet_raw_candidates``: ``{task}/raw_candidates_86151.parquet`` and the column table.

    Runs ``build_raw_candidates.py`` over the candidate genes of DEG and DepMap instead of every gene of
    the FASTA, on the method's number of worker processes.
    """
    code = _method("esstfnet/build_raw_candidates.py", "esstfnet")
    output.mkdir()
    code._write_raw_candidate_contract(output / RAW_COLUMNS)
    with ProcessPoolExecutor(max_workers=code.WORKERS) as pool:
        for task in COMPUTED:
            cds = code.read_fasta(_cds(task))
            rows = [(key, cds[key]) for key in _candidates(task)]
            chunks = output / task / code.CHUNK_DIRECTORY_NAME
            chunks.mkdir(parents=True)
            jobs = [
                (index, str(code._chunk_path(chunks, index)), tuple(rows[start : start + RAW_ROWS_PER_GROUP]))
                for index, start in enumerate(range(0, len(rows), RAW_ROWS_PER_GROUP))
            ]
            progress = _Progress(f"EssTFNet candidates of {task}", len(rows))
            done = 0
            for _, count in pool.map(code._write_raw_chunk_task, jobs):
                done += count
                progress.update(done)
            code._merge_raw_chunks([Path(job[1]) for job in jobs], output / task / RAW)
            shutil.rmtree(chunks)
    _link_derived(output, f"{{task}}/{RAW}")


def _evo2(directory: Path):
    """EssTFNet's Evo 2 code and the model, loaded from the download with ``directory`` as HF_HOME."""
    code = _method("esstfnet/code/embed_dna.py")
    code.MODEL_CHECKPOINT_PATH = external.path("evo2_7b_base")
    code.DEFAULT_MODEL_CACHE_DIR = directory
    with contextlib.redirect_stdout(sys.stderr):
        model = code.load_evo2_model(DEVICE)
    return code, model


def _evo2_sequences(code) -> tuple[dict[str, list[str]], dict[str, str]]:
    genes = {task: _candidates(task) for task in COMPUTED}
    sequences: dict[str, str] = {}
    for task in COMPUTED:
        cds = code.read_fasta(_cds(task))
        sequences.update((key, cds[key]) for key in genes[task])
    return genes, sequences


def evo2_embeddings(output: Path) -> None:
    """Builder of ``evo2_embeddings``: ``{task}/evo2_7b_base_embeddings.parquet``, one float16 row per gene."""
    with tempfile.TemporaryDirectory() as directory:
        code, model = _evo2(Path(directory))
        genes, sequences = _evo2_sequences(code)
        progress = _Progress("Evo 2 embeddings", len(sequences))
        vectors = {}
        with contextlib.redirect_stdout(sys.stderr):
            for done, key in enumerate(sorted(sequences), 1):
                vectors[key] = code.embed_gene(model, sequences[key], DEVICE, code.MODEL_CONTEXT_NT)
                progress.update(done)
    for task in COMPUTED:
        target = output / task / EVO2
        target.parent.mkdir(parents=True)
        code._write_output(target, genes[task], np.stack([vectors[key] for key in genes[task]]))
    _link_derived(output, f"{{task}}/{EVO2}")


# Comparison with the originals


def _files(item_id: str) -> Iterator[tuple[str | None, str, str]]:
    """Task, file name and method input, with ``{task}`` unfilled, of every file of an item."""
    for name, targets in inputs.originals(item_id):
        # Longest first, so depmap_23q4 is not read as depmap.
        task = next((t for t in sorted(TASKS, key=len, reverse=True) if name.startswith((f"{t}/", f"{t}_"))), None)
        template = name if task is None else "{task}" + name[len(task):]
        for target in targets:
            yield task, template, target if task is None else target.replace(f"source/{task}", "source/{task}", 1)


def _tables(path: Path) -> Iterator[pa.Table]:
    """The rows of a Parquet file in tables of COMPARED_ROWS, whatever its row groups."""
    pending: list[pa.RecordBatch] = []
    count = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=COMPARED_ROWS):
        pending.append(batch)
        count += batch.num_rows
        while count >= COMPARED_ROWS:
            table = pa.Table.from_batches(pending)
            yield table.slice(0, COMPARED_ROWS)
            rest = table.slice(COMPARED_ROWS)
            pending, count = rest.to_batches(), rest.num_rows
    if count:
        yield pa.Table.from_batches(pending)


def _matrix(table: pa.Table) -> tuple[list[list], np.ndarray]:
    """The text columns of a table and its numbers, one row per gene, list columns spread out."""
    texts, blocks = [], []
    for column in table.columns:
        values = column.combine_chunks()
        if pa.types.is_string(values.type) or pa.types.is_large_string(values.type):
            texts.append(values.to_pylist())
        elif pa.types.is_fixed_size_list(values.type):
            blocks.append(values.flatten().to_numpy(zero_copy_only=False).reshape(len(values), -1))
        else:
            blocks.append(values.to_numpy(zero_copy_only=False).reshape(-1, 1))
    return texts, np.hstack(blocks).astype(np.float64)


def _compare_tables(mine: Path, original: Path) -> dict:
    left, right = pq.ParquetFile(mine), pq.ParquetFile(original)
    row = {
        "comparison": "values",
        "rows": left.metadata.num_rows,
        "original_rows": right.metadata.num_rows,
        "file_identical": filecmp.cmp(mine, original, shallow=False),
    }
    if not left.schema_arrow.equals(right.schema_arrow) or row["rows"] != row["original_rows"]:
        return {**row, "identical": False}
    texts_equal, rows, values, largest = True, 0, 0, 0.0
    for a, b in zip(_tables(mine), _tables(original)):
        (text_a, numbers_a), (text_b, numbers_b) = _matrix(a), _matrix(b)
        texts_equal &= text_a == text_b
        unequal = (numbers_a != numbers_b) & ~(np.isnan(numbers_a) & np.isnan(numbers_b))
        rows += int(unequal.any(axis=1).sum())
        values += int(unequal.sum())
        largest = max(largest, float(np.nanmax(np.abs(numbers_a - numbers_b), initial=0.0)))
    return {
        **row,
        "differing_rows": rows,
        "differing_values": values,
        "largest_absolute_difference": largest,
        "identical": texts_equal and values == 0,
    }


def _same_file(first: Path, second: Path) -> bool:
    return first.is_file() and second.is_file() and os.path.samefile(first, second)


def _upstream_row(methods: Path) -> dict:
    """The files of the ProteomeLM archive against the method's copy of the upstream code."""
    original = methods / UPSTREAM
    theirs = {
        path.relative_to(original) for path in original.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    with tempfile.TemporaryDirectory() as directory:
        release = _upstream(external.path("proteomelm_source"), Path(directory))
        files = {path.relative_to(release) for path in release.rglob("*") if path.is_file()}
        differing = sum(
            name not in theirs or not filecmp.cmp(release / name, original / name, shallow=False) for name in files
        )
    return {
        "item": "proteomelm_source",
        "file": "",
        "original": UPSTREAM,
        "comparison": "files of the archive",
        "rows": len(files),
        "original_rows": len(theirs),
        "differing_rows": differing + len(theirs - files),
        "identical": differing == 0 and theirs == files,
    }


def check_inputs(methods: Path) -> list[dict]:
    """Downloads byte by byte, the fully built items value by value, and the ProteomeLM archive."""
    rows = []
    for item_id in CHECKED:
        if item_id in SAMPLED:
            continue
        for task, name, target in _files(item_id):
            file, original = name.format(task=task), methods / target.format(task=task)
            mine = external.path(item_id) / file if file else external.path(item_id)
            row = {"item": item_id, "file": file, "original": target.format(task=task)}
            if not original.exists():
                continue
            if (
                task in DERIVED
                and _same_file(mine, external.path(item_id) / name.format(task=DONOR))
                and _same_file(original, methods / target.format(task=DONOR))
            ):
                row.update(comparison="the DepMap file, as is the original", identical=True)
            elif mine.suffix == ".parquet":
                row.update(_compare_tables(mine, original))
            else:
                same = filecmp.cmp(mine, original, shallow=False)
                row.update(comparison="bytes", identical=same, file_identical=same)
            rows.append(row)
            print({key: row.get(key) for key in ("item", "file", "comparison", "identical")}, flush=True)
    rows.append(_upstream_row(methods))
    return rows


def _bingo_sample(cache, union: tuple[str, ...], merged: dict[str, str]) -> list[str]:
    """Whole batches of the full build, so the sample is batched as the full build batches it.

    Proteins longer than the cut and single proteins longer than half a batch are never paired, and a
    pair is never joined by a third protein, so these batches stay as they are among the sample.
    """
    cut = cache.MAX_PROTEIN_LENGTH
    pending = [(key, Path(), Path(), merged[key][:cut]) for key in union]
    batches = cache._group_pending_batches(
        pending, token_budget=BINGO_TOKEN_BUDGET, max_batch_size=BINGO_BATCH_SIZE
    )
    half = BINGO_TOKEN_BUDGET // BINGO_BATCH_SIZE - 2
    kinds = {
        "longer than the cut": [batch for batch in batches if len(merged[batch[0][0]]) > cut],
        "single": [batch for batch in batches if len(batch) == 1 and half < len(merged[batch[0][0]]) <= cut],
        "pairs": [batch for batch in batches if len(batch) == 2],
    }
    random = np.random.default_rng(SEED)
    sample = []
    for kind, count in BINGO_SAMPLE.items():
        for index in random.choice(len(kinds[kind]), size=count, replace=False):
            sample.extend(item[0] for item in kinds[kind][index])
    return sorted(sample)


def check_bingo(methods: Path) -> list[dict]:
    """Rebuild a sample of the graph cache and compare embeddings, edges and the other fields of each graph."""
    import torch

    cache, extract = _bingo()
    _, union, merged, _ = _bingo_genes(cache, extract)
    sample = _bingo_sample(cache, union, merged)
    original = methods / inputs.original("bingo_graph_cache")
    theirs_index = pd.read_parquet(original / "cache_index.parquet").set_index(GENE_KEY)["cache_file"]
    fields = ("original_length", "cached_length", "truncated", "protein_sequence_sha256")
    x_rows = x_values = edge_rows = edge_values = field_rows = 0
    x_largest = threshold_largest = 0.0
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "cache"
        started = time.monotonic()
        _bingo_build(cache, output, sample, merged, ())
        minutes = round((time.monotonic() - started) / 60, 2)
        mine_index = pd.read_parquet(output / "cache_index.parquet").set_index(GENE_KEY)["cache_file"]
        for key in sample:
            mine = torch.load(output / "graphs" / mine_index[key], weights_only=False)
            theirs = torch.load(original / "graphs" / theirs_index[key], weights_only=False)
            gap = (mine["x"].double() - theirs["x"].double()).abs()
            x_rows += bool((gap > 0).any())
            x_values += int((gap > 0).sum())
            x_largest = max(x_largest, float(gap.max()))
            edges = [set(map(tuple, graph["edge_index"].T.tolist())) for graph in (mine, theirs)]
            edge_rows += edges[0] != edges[1]
            edge_values += len(edges[0] ^ edges[1])
            if mine["cached_length"] > 1:
                threshold_largest = max(threshold_largest, abs(mine["contact_threshold"] - theirs["contact_threshold"]))
            field_rows += any(mine[field] != theirs[field] for field in fields)
    residues = sum(min(len(merged[key]), cache.MAX_PROTEIN_LENGTH) for key in sample)
    total = sum(min(len(sequence), cache.MAX_PROTEIN_LENGTH) for sequence in merged.values())
    print(f"{len(sample)} genes, {residues} residues in {minutes} min; the {len(union)} genes would take "
          f"about {minutes * total / residues:.0f} min", flush=True)
    common = {"item": "bingo_graph_cache", "rows": len(sample), "original_rows": len(sample), "minutes": minutes}
    described = f"{len(sample)} genes in whole batches of the full build"
    return [
        {**common, "file": "graphs/*.pt x", "original": f"{inputs.original('bingo_graph_cache')}/graphs",
         "comparison": f"residue embeddings of {described}", "differing_rows": x_rows,
         "differing_values": x_values, "largest_absolute_difference": x_largest, "identical": x_values == 0},
        {**common, "file": "graphs/*.pt edge_index", "original": f"{inputs.original('bingo_graph_cache')}/graphs",
         "comparison": "contact edges, directed edges in one graph only; largest difference of the contact threshold",
         "differing_rows": edge_rows, "differing_values": edge_values,
         "largest_absolute_difference": threshold_largest, "identical": edge_values == 0},
        {**common, "file": "graphs/*.pt lengths", "original": f"{inputs.original('bingo_graph_cache')}/graphs",
         "comparison": "protein length, cached length, truncation and protein SHA-256",
         "differing_rows": field_rows, "identical": field_rows == 0},
    ]


def check_evo2(methods: Path) -> list[dict]:
    """Embed a sample of genes, some longer than one window, and compare them with the original rows."""
    with tempfile.TemporaryDirectory() as directory:
        code, model = _evo2(Path(directory))
        genes, sequences = _evo2_sequences(code)
        window = code.MODEL_CONTEXT_NT
        keys = sorted(sequences)
        random = np.random.default_rng(SEED)
        long = [key for key in keys if len(sequences[key]) > window]
        short = [key for key in keys if len(sequences[key]) <= window]
        sample = sorted([*random.choice(long, EVO2_LONG, replace=False),
                         *random.choice(short, EVO2_SAMPLE - EVO2_LONG, replace=False)])
        started = time.monotonic()
        with contextlib.redirect_stdout(sys.stderr):
            vectors = {key: code.embed_gene(model, sequences[key], DEVICE, window) for key in sample}
        minutes = round((time.monotonic() - started) / 60, 2)
    nucleotides = sum(len(sequences[key]) for key in sample)
    total = sum(len(sequence) for sequence in sequences.values())
    print(f"{len(sample)} genes, {nucleotides} nt in {minutes} min; the {len(sequences)} genes would take "
          f"about {minutes * total / nucleotides:.0f} min", flush=True)
    rows = []
    for task in COMPUTED:
        chosen = [key for key in sample if key in set(genes[task])]
        target = inputs.original("evo2_embeddings", f"{task}/{EVO2}")
        table = pq.read_table(methods / target, filters=[(GENE_KEY, "in", chosen)])
        theirs = dict(zip(table[GENE_KEY].to_pylist(), np.asarray(table["embedding"].to_pylist(), dtype=np.float64)))
        gap = np.abs(np.stack([vectors[key].astype(np.float64) - theirs[key] for key in chosen]))
        largest_value = max(float(np.abs(theirs[key]).max()) for key in chosen)
        print(f"{task}: largest absolute value of the sampled original rows {largest_value:g}", flush=True)
        rows.append({
            "item": "evo2_embeddings", "file": f"{task}/{EVO2}", "original": target,
            "comparison": f"{len(chosen)} genes, {sum(len(sequences[key]) > window for key in chosen)} longer than "
                          f"{window} nt; largest absolute value {largest_value:g}",
            "rows": len(chosen), "original_rows": len(theirs), "differing_rows": int((gap > 0).any(axis=1).sum()),
            "differing_values": int((gap > 0).sum()), "largest_absolute_difference": float(gap.max()),
            "identical": not gap.any(), "minutes": minutes,
        })
    return rows


PARTS = {"inputs": check_inputs, "bingo": check_bingo, "evo2": check_evo2}
COLUMNS = ["item", "file", "original", "comparison", "rows", "original_rows", "differing_rows", "differing_values",
           "largest_absolute_difference", "identical", "file_identical", "minutes"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check", help="compare the built items with the method inputs")
    checking.add_argument("part", choices=list(PARTS))
    checking.add_argument("--methods", type=Path, required=True, help="methods directory holding the original inputs")
    arguments = parser.parse_args(argv)
    rows = pd.DataFrame(PARTS[arguments.part](arguments.methods.expanduser())).reindex(columns=COLUMNS)
    rows["file"] = rows["file"].fillna("")
    if CHECK_OUTPUT.is_file():
        earlier = pd.read_csv(CHECK_OUTPUT, keep_default_na=False, na_values=[""])
        earlier["file"] = earlier["file"].fillna("")
        replaced = set(zip(rows["item"], rows["file"], rows["original"]))
        earlier = earlier[[key not in replaced for key in zip(earlier["item"], earlier["file"], earlier["original"])]]
        rows = pd.concat([earlier, rows], ignore_index=True)
    order = {item_id: index for index, item_id in enumerate([*CHECKED, "proteomelm_source"])}
    rows = rows.sort_values("item", key=lambda column: column.map(order), kind="stable").convert_dtypes()
    CHECK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(CHECK_OUTPUT, index=False)
    print(rows.to_string(index=False))
    print(f"written {CHECK_OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
