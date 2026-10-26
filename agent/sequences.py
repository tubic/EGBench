"""Builders of the sequence inputs of the sequence methods: representative sequences, task FASTA, DeepHE and Pheg features.

``ensembl97_sequences`` gives every HGNC gene of Ensembl 97 one coding sequence and its protein, the
longest CDS, then the longest protein, then the first transcript identifier, as the DEG task was
built; ``gencode50_sequences`` gives every gene of the gene reference the CDS and protein of its
GENCODE v50 transcript. A method takes for each gene the Ensembl 97 pair when its condition accepts
it, else the GENCODE v50 pair (``select``). The three FASTA items hold, for the six tasks, the proteins
of the general condition, the proteins of DeepCellEss's 21-letter alphabet and the CDS of A, C, G and
T only: DEG and DepMap restricted to their gene universes, Context, Blood, Mouse and DepMap 23Q4 to
their candidate genes in ``agent/candidates``, as ``methods/derive_cohort_source_assets.py`` selects
them from the DepMap file. ``deephe_sequence_89`` computes the 89 sequence features of DeepHE,
DeeplyEssential and DeEPsnap and ``pheg_fv45`` Pheg's 4,545 Z-curve features with their QA table.
``agent/methods.json`` maps every file to the method inputs it stands for.

    python agent/sequences.py check --methods ORIGINAL

rebuilds every item in a temporary directory, compares it with the inputs under ORIGINAL, the
``methods`` directory of the manuscript's working copy, and writes the comparison with the size and
the minutes of each build to ``agent/results/sequences_check.csv``. FASTA are compared as text and as
gzip bytes, tables value by value.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

import external
import inputs
import pheg_features
import sources

LINE_WIDTH = 60
ENSEMBL_LABEL = "Ensembl97"
GENCODE_LABEL = "GENCODE50"
SOURCES = ("ensembl97", "gencode50")
CHECK_OUTPUT = external.AGENT / "results" / "sequences_check.csv"
UNIVERSES = {
    "deg": external.ROOT / "datasets" / "DEG" / "data" / "gene_universe.parquet",
    "depmap": external.ROOT / "datasets" / "DepMap" / "data" / "depmap26q1_gene_universe.parquet",
}
DONOR = "depmap"
DERIVED = ("context", "blood", "mouse", "depmap_23q4")
TASKS = (*UNIVERSES, *DERIVED)
CANDIDATES = external.AGENT / "candidates"
STANDARD_PROTEIN = frozenset("ACDEFGHIKLMNPQRSTVWY")
CONDITIONS: dict[str, Callable[[str, str], bool]] = {
    "general": lambda cds, protein: bool(cds) and bool(protein),
    "acgt_cds": lambda cds, protein: bool(cds) and set(cds) <= set("ACGT"),
    "deepcelless": lambda cds, protein: bool(protein) and set(protein) <= set("RSPFICETYKGANUMLWDVHQ"),
    "deephe": lambda cds, protein: bool(cds)
    and len(cds) % 3 == 0
    and set(cds) <= set("ACGTN")
    and bool(protein)
    and "*" not in protein
    and set(protein) <= STANDARD_PROTEIN | {"U", "X"},
    "pheg": lambda cds, protein: bool(cds) and set(cds) <= set("ACGTN"),
}
XGEP_CODE = external.ROOT / "methods" / "xgep" / "code"
# DeepHE's codon adaptation index takes logarithms and an exponential, whose last bit differs between
# the C libraries of macOS and Linux.
TOLERANCE = 1e-15
PHEG_MATRIX = "pheg_fv45_4545.parquet"
PHEG_QA = "pheg_fv45_qa.parquet"



def _stable(identifier: str) -> str:
    return identifier.strip().split(".", 1)[0]


def _unique(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """The keys, upper-cased, that map to one HGNC identifier only."""
    targets: dict[str, set[str]] = defaultdict(set)
    for key, hgnc_id in pairs:
        if key.strip() and hgnc_id:
            targets[key.strip().upper()].add(hgnc_id)
    return {key: next(iter(found)) for key, found in targets.items() if len(found) == 1}


def _hgnc_resolver() -> tuple[Callable[[str, str], str | None], dict[str, str]]:
    """Resolve an Ensembl gene by its stable identifier, else by approved, previous or alias symbol."""
    with external.path("hgnc").open(encoding="utf-8", newline="") as handle:
        approved = [row for row in csv.DictReader(handle, delimiter="\t") if row.get("status") == "Approved"]

    def symbols(field: str) -> Iterator[tuple[str, str]]:
        return ((item, row["hgnc_id"]) for row in approved for item in row.get(field, "").split("|") if item)

    by_ensembl = _unique(
        (_stable(row["ensembl_gene_id"]), row["hgnc_id"]) for row in approved if row.get("ensembl_gene_id")
    )
    by_symbol = (
        _unique((row["symbol"], row["hgnc_id"]) for row in approved),
        _unique(symbols("prev_symbol")),
        _unique(symbols("alias_symbol")),
    )

    def resolve(ensembl_gene_id: str, symbol: str) -> str | None:
        ensembl, name = _stable(ensembl_gene_id).upper(), symbol.strip().upper()
        if ensembl in by_ensembl:
            return by_ensembl[ensembl]
        return next((table[name] for table in by_symbol if name in table), None)

    return resolve, {row["hgnc_id"]: row["symbol"] for row in approved}


def _records(path: Path) -> Iterator[tuple[str, str]]:
    header, chunks = None, []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:].strip(), []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def _tags(header: str) -> dict[str, str]:
    return dict(token.split(":", 1) for token in header.split()[1:] if ":" in token)


def _write_fasta(path: Path, records: Iterable[tuple[str, str]]) -> None:
    """Gzipped FASTA, 60 letters a line, with neither file name nor time in the gzip header."""
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as handle:
            for header, sequence in records:
                handle.write(f">{header}\n")
                for start in range(0, len(sequence), LINE_WIDTH):
                    handle.write(sequence[start : start + LINE_WIDTH] + "\n")


def ensembl97_sequences(output: Path) -> None:
    """Write ensembl97.{cds,pep}.fa.gz: one protein-coding transcript of Ensembl 97 per HGNC gene."""
    resolve, symbol_of = _hgnc_resolver()
    cds_of: dict[str, tuple[str, str, str]] = {}
    for header, sequence in _records(external.path("ensembl97_cds")):
        tags = _tags(header)
        transcript = header.split()[0]
        if tags.get("gene_biotype") != "protein_coding":
            continue
        if transcript not in cds_of or len(sequence) > len(cds_of[transcript][2]):
            cds_of[transcript] = (_stable(tags.get("gene", "")), tags.get("gene_symbol", ""), sequence.upper())
    candidates: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for header, protein in _records(external.path("ensembl97_pep")):
        tags = _tags(header)
        transcript = tags.get("transcript", "")
        if tags.get("gene_biotype") != "protein_coding" or transcript not in cds_of:
            continue
        cds_gene, cds_symbol, cds = cds_of[transcript]
        gene = _stable(tags.get("gene", cds_gene))
        gene_key = resolve(gene, tags.get("gene_symbol", cds_symbol)) if gene == cds_gene else None
        if gene_key:
            candidates[gene_key].append((gene, transcript, cds, protein.upper().rstrip("*")))
    chosen = {
        gene_key: min(found, key=lambda item: (-len(item[2]), -len(item[3]), item[1]))
        for gene_key, found in candidates.items()
    }
    headers = {key: f"{key}|{symbol_of[key]}|{chosen[key][0]}|{chosen[key][1]}|{ENSEMBL_LABEL}" for key in chosen}
    output.mkdir()
    for index, kind in ((2, "cds"), (3, "pep")):
        _write_fasta(output / f"ensembl97.{kind}.fa.gz", ((headers[key], chosen[key][index]) for key in sorted(chosen)))


def _cds_ranges(path: Path) -> dict[str, tuple[int, str]]:
    """Start of the CDS in each protein-coding transcript of GENCODE, with the transcript sequence."""
    found = {}
    for header, sequence in _records(path):
        fields = header.split("|")
        start = next(int(field[4:].split("-")[0]) for field in fields if field.startswith("CDS:"))
        found[fields[0]] = (start, sequence.upper())
    return found


def _transcript(row) -> str:
    identifier, version = row.ensembl_transcript_id, row.ensembl_transcript_version
    if identifier.endswith("_PAR_Y"):
        return f"{identifier.removesuffix('_PAR_Y')}.{version}_PAR_Y"
    return f"{identifier}.{version}"


def gencode50_sequences(output: Path) -> None:
    """Write gencode50.{cds,pep}.fa.gz: the CDS without stop codon and the protein of each gene reference transcript."""
    reference = pd.read_parquet(
        external.path("gene_reference"),
        columns=["gene_key", "ensembl_transcript_id", "ensembl_transcript_version", "cds_length_nt"],
    ).sort_values("gene_key")
    translations = sources._translations(external.path("gencode_translations"))
    transcripts = _cds_ranges(external.path("gencode_pc_transcripts"))
    pairs = []
    for row in reference.itertuples(index=False):
        start, sequence = transcripts[_transcript(row)]
        pairs.append((row.gene_key, sequence[start - 1 : start - 1 + row.cds_length_nt], translations[_transcript(row)][2]))
    output.mkdir()
    for index, kind in ((1, "cds"), (2, "pep")):
        _write_fasta(output / f"gencode50.{kind}.fa.gz", ((f"{pair[0]}|{GENCODE_LABEL}", pair[index]) for pair in pairs))


def _read(path: Path) -> dict[str, str]:
    return {header.split("|", 1)[0].split()[0]: sequence.upper() for header, sequence in _records(path)}


def select(condition: str) -> dict[str, tuple[str, str, str]]:
    """CDS, protein and source of every gene: Ensembl 97 when the condition accepts it, else GENCODE v50."""
    accepts = CONDITIONS[condition]
    loaded = []
    for name in SOURCES:
        directory = external.path(f"{name}_sequences")
        loaded.append((name, _read(directory / f"{name}.cds.fa.gz"), _read(directory / f"{name}.pep.fa.gz")))
    genes = sorted(set().union(*(set(cds) & set(protein) for _, cds, protein in loaded)))
    selected = {}
    for gene_key in genes:
        for name, cds, protein in loaded:
            if gene_key in cds and gene_key in protein and accepts(cds[gene_key], protein[gene_key].rstrip("*")):
                selected[gene_key] = (cds[gene_key], protein[gene_key].rstrip("*"), name)
                break
    return selected


def _task_genes(selected: dict[str, tuple[str, str, str]]) -> dict[str, list[str]]:
    genes = {
        task: sorted(set(pd.read_parquet(path, columns=["gene_key"])["gene_key"].astype(str)) & set(selected))
        for task, path in UNIVERSES.items()
    }
    for task in DERIVED:
        keys = sorted(pd.read_parquet(CANDIDATES / f"{task}_common_feature_available.parquet")["gene_key"].astype(str))
        missing = sorted(set(keys) - set(genes[DONOR]))
        if missing:
            raise ValueError(f"the {DONOR} selection lacks {len(missing)} candidate genes of {task}: {missing[:5]}")
        genes[task] = keys
    return genes


def _task_fastas(output: Path, condition: str, field: int) -> None:
    selected = select(condition)
    output.mkdir()
    for task, keys in _task_genes(selected).items():
        _write_fasta(output / f"{task}.fa.gz", ((f"{key}|{selected[key][2]}", selected[key][field]) for key in keys))


def protein_sequences(output: Path) -> None:
    """Proteins of the general condition for the six tasks (BiNGO and ProteomeLM)."""
    _task_fastas(output, "general", 1)


def deepcelless_proteins(output: Path) -> None:
    """Proteins written in DeepCellEss's 21 letters for the six tasks."""
    _task_fastas(output, "deepcelless", 1)


def cds_sequences(output: Path) -> None:
    """CDS of A, C, G and T only for the six tasks (GCNN-SFM, eDNN-EG and EssTFNet)."""
    _task_fastas(output, "acgt_cds", 0)


def deephe_sequence_89(output: Path) -> None:
    """Write DeepHE's 89 sequence features: codon and amino-acid frequencies, GC content, lengths, CAI and RSCU."""
    if str(XGEP_CODE) not in sys.path:
        sys.path.insert(0, str(XGEP_CODE))
    from method_features.gene_mapping import SequenceAsset, write_deterministic_parquet
    from method_features.legacy_features import DEEPHE_SEQUENCE_89_COLUMNS, build_deephe_sequence_89

    assets = (
        SequenceAsset(
            gene_key=key,
            symbol=key,
            transcript_id=source,
            protein_id=source,
            cds_sequence=cds,
            protein_sequence=protein,
            start_codon=cds[:3],
            stop_codon=cds[-3:],
            has_start_codon_feature=False,
            has_stop_codon_feature=False,
            cds_length_divisible_by_three=True,
            protein_has_terminal_stop=False,
            protein_has_internal_stop=False,
            illegal_protein_characters="",
            selenocysteine_positions=tuple(index for index, residue in enumerate(protein) if residue == "U"),
            selenocysteine_supported=True,
            translation_length_matches_cds=None,
            sequence_qc_pass=True,
        )
        for key, (cds, protein, source) in sorted(select("deephe").items())
    )
    rows = build_deephe_sequence_89(assets).rows
    write_deterministic_parquet(
        [row.as_record() for row in rows], output, column_order=("gene_key", *DEEPHE_SEQUENCE_89_COLUMNS)
    )


def pheg_fv45(output: Path) -> None:
    """Write Pheg's gene by 4,545 FV4,5 matrix and its per-gene QA table."""
    output.mkdir()
    cds = {key: record[0] for key, record in select("pheg").items()}
    pheg_features.write(cds, output / PHEG_MATRIX, output / PHEG_QA)


def _fasta_text(path: Path) -> bytes:
    with gzip.open(path, "rb") as handle:
        return handle.read()


def _compare_fasta(rebuilt: Path, original: Path) -> dict:
    mine, theirs = _read(rebuilt), _read(original)
    text = _fasta_text(rebuilt) == _fasta_text(original)
    differing = sum(mine.get(key) != sequence for key, sequence in theirs.items()) + len(set(mine) - set(theirs))
    return {
        "comparison": "decompressed text",
        "records": len(mine),
        "original_records": len(theirs),
        "identical": text,
        "file_identical": rebuilt.read_bytes() == original.read_bytes(),
        "genes_differing": differing,
    }


def _compare_table(rebuilt: Path, original: Path) -> dict:
    mine, theirs = pd.read_parquet(rebuilt), pd.read_parquet(original)
    row = {
        "comparison": f"values, tolerance {TOLERANCE:g}",
        "records": len(mine),
        "original_records": len(theirs),
        "identical": mine.equals(theirs),
        "file_identical": rebuilt.read_bytes() == original.read_bytes(),
    }
    if list(mine.columns) != list(theirs.columns) or not mine["gene_key"].equals(theirs["gene_key"]):
        return {**row, "within_tolerance": False}
    floats = [column for column in theirs.columns if pd.api.types.is_float_dtype(theirs[column])]
    others = [column for column in theirs.columns if column not in floats]
    gap = np.abs(mine[floats].to_numpy("float64") - theirs[floats].to_numpy("float64"))
    largest = float(gap.max(initial=0.0))
    return {
        **row,
        "largest_absolute_difference": largest,
        "genes_differing": int((gap > 0).any(axis=1).sum()),
        "columns_differing": " ".join(floats[index] for index in np.flatnonzero((gap > 0).any(axis=0))),
        "within_tolerance": bool(mine[others].equals(theirs[others]) and largest <= TOLERANCE),
    }


def _files(item_id: str) -> list[tuple[str, tuple[str, ...]]]:
    return inputs.originals(item_id)


def check(methods: Path) -> list[dict]:
    rows = []
    for item_id in inputs.built_by("sequences"):
        entry = external.item(item_id)
        external.ensure(*entry["inputs"])
        builder = globals()[entry["builder"].split(":")[1]]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / Path(entry["path"]).name
            started = time.monotonic()
            builder(output)
            minutes = round((time.monotonic() - started) / 60, 2)
            size = sum(file.stat().st_size for file in output.rglob("*")) if output.is_dir() else output.stat().st_size
            print(f"{item_id}: built in {minutes} min, {size} bytes", flush=True)
            for name, targets in _files(item_id):
                rebuilt = output / name if name else output
                for target in targets:
                    original = methods / target
                    if name.endswith(".fa.gz"):
                        row = _compare_fasta(rebuilt, original)
                    else:
                        row = _compare_table(rebuilt, original)
                    rows.append({"asset": f"{item_id}/{name}".rstrip("/"), "original": target, **row,
                                 "bytes": rebuilt.stat().st_size, "minutes": minutes})
                    print({key: rows[-1][key] for key in ("asset", "original", "identical")}, flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check", help="rebuild the items and compare them with the method inputs")
    checking.add_argument("--methods", type=Path, required=True, help="methods directory holding the original inputs")
    arguments = parser.parse_args(argv)
    try:
        rows = check(arguments.methods.expanduser())
    except external.Missing as missing:
        print(missing.report())
        return 1
    table = pd.DataFrame(rows).convert_dtypes()
    CHECK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CHECK_OUTPUT, index=False)
    print(table.to_string(index=False))
    print(f"written {CHECK_OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
