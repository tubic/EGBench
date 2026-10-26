"""Builders of two derived tables of the agent tools from public downloads: gene reference and TCGA expression matrix.

``gene_reference`` picks one protein-coding transcript of GENCODE v50 for every approved HGNC
protein-coding gene and writes the gene identifier table of the previous benchmark: HGNC, Entrez and
Ensembl gene, transcript and protein identifiers, the rule that picked the transcript and the
evidence that tied it to the gene. ``tcga_expression`` converts the 33 RSEM archives of Zenodo
record 7885656 into a genes by primary tumour samples matrix of upper-quartile normalised counts
keyed by HGNC identifier, with the record ``tcga_expression_qa.json`` beside it. Both are the
builders of the items of the same names in ``agent/external.json``; ``agent/external.py`` calls them
once their inputs are under the data root.

    python agent/sources.py check --gene-reference ORIGINAL --tcga-expression ORIGINAL

rebuilds both tables in a temporary directory, compares them column by column with the originals
(the QA record is read beside the original matrix) and writes the comparison, with the size and the
minutes of each build, to ``agent/results/sources_check.csv``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import external

GENCODE = "GENCODE v50"
CHECK_OUTPUT = external.AGENT / "results" / "sources_check.csv"
ENSEMBL_ID = re.compile(r"(?P<stable>ENS[A-Z]*\d+)(?:\.(?P<version>\d+))?(?P<par>_PAR_Y)?")
HGNC_ID = re.compile(r"HGNC:\d+")
GTF_ATTRIBUTE = re.compile(r'(\S+) "([^"]*)"')
GTF_FEATURES = frozenset({"transcript", "CDS", "start_codon", "stop_codon"})
PROTEIN_CODING = ('gene_type "protein_coding"', 'transcript_type "protein_coding"')
INCOMPLETE_TAGS = frozenset({"cds_start_NF", "cds_end_NF"})
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYUX")
UNRANKED = 2**31 - 1
REFERENCE_SCHEMA = pa.schema(
    [
        ("gene_key", pa.string()),
        ("hgnc_id", pa.string()),
        ("symbol", pa.string()),
        ("entrez_id", pa.string()),
        ("ensembl_gene_id", pa.string()),
        ("ensembl_gene_version", pa.int64()),
        ("ensembl_transcript_id", pa.string()),
        ("ensembl_transcript_version", pa.int64()),
        ("ensembl_protein_id", pa.string()),
        ("ensembl_protein_version", pa.int64()),
        ("selection_reason", pa.string()),
        ("transcript_rank", pa.int64()),
        ("cds_length_nt", pa.int64()),
        ("complete_cds", pa.bool_()),
        ("mapping_status", pa.string()),
        ("mapping_evidence", pa.string()),
        ("source_release", pa.string()),
    ]
)
TCGA_SAMPLE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-(?P<type>[0-9]{2})[A-Z]-")
PRIMARY_TUMOUR_CODES = ("01", "03", "09")
SAMPLE_KEY_LENGTH = 16
EXPRESSION_MEMBER = re.compile(r"(?:^|/)[A-Z0-9]+__geneExpT\.txt$")
ARCHIVE_SUFFIX = "rnaSeq.tar"
QA_FILE = "tcga_expression_qa.json"


class Gene(NamedTuple):
    hgnc_id: str
    symbol: str
    entrez_id: str | None
    ensembl_gene_id: str | None


def _identifier(text: str) -> tuple[str, int | None]:
    """Stable identifier, with the _PAR_Y suffix of genes on the Y pseudo-autosomal region, and version."""
    match = ENSEMBL_ID.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"not an Ensembl identifier: {text!r}")
    version = match.group("version")
    return match.group("stable") + (match.group("par") or ""), None if version is None else int(version)


def _full(text: str) -> str:
    stable, version = _identifier(text)
    if version is None:
        return stable
    if stable.endswith("_PAR_Y"):
        return f"{stable.removesuffix('_PAR_Y')}.{version}_PAR_Y"
    return f"{stable}.{version}"


def _protein_coding_genes(path: Path) -> list[Gene]:
    columns = ["hgnc_id", "symbol", "locus_group", "status", "entrez_id", "ensembl_gene_id"]
    table = pd.read_csv(path, sep="\t", usecols=columns, dtype=str, keep_default_na=False)
    table = table.apply(lambda column: column.str.strip())
    table = table.loc[
        table["status"].str.casefold().eq("approved")
        & table["locus_group"].str.casefold().eq("protein-coding gene")
    ].drop_duplicates("hgnc_id", keep=False)
    return [
        Gene(
            row.hgnc_id,
            row.symbol,
            row.entrez_id or None,
            _identifier(row.ensembl_gene_id)[0] if row.ensembl_gene_id else None,
        )
        for row in table.itertuples(index=False)
    ]


def _merged_into(path: Path) -> dict[str, tuple[str, ...]]:
    """The HGNC identifiers each withdrawn identifier was merged into, one for each report of a merge."""
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    named = {str(column).strip().upper(): column for column in table.columns}
    merged = next(column for name, column in named.items() if name.startswith("MERGED_INTO"))
    return {
        hgnc_id.strip(): tuple(
            report.split("|")[0].strip()
            for report in reports.strip().strip('"').split(",")
            if HGNC_ID.fullmatch(report.split("|")[0].strip())
        )
        for hgnc_id, reports in zip(table[named["HGNC_ID"]], table[merged])
    }


def _transcripts(path: Path) -> dict[str, dict]:
    """Protein-coding transcripts of the GTF: gene, protein, HGNC identifier, tags, CDS length and codons."""
    found: dict[str, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 9 or columns[2] not in GTF_FEATURES:
                continue
            raw = columns[8]
            if not all(text in raw for text in PROTEIN_CODING):
                continue
            attributes: dict[str, list[str]] = defaultdict(list)
            for key, value in GTF_ATTRIBUTE.findall(raw):
                attributes[key].append(value)
            entry = found.setdefault(
                attributes["transcript_id"][0],
                {
                    "gene_id": attributes["gene_id"][0],
                    "protein_id": None,
                    "hgnc_id": None,
                    "tags": set(),
                    "cds": 0,
                    "start": False,
                    "stop": False,
                },
            )
            for key in ("protein_id", "hgnc_id"):
                if entry[key] is None and attributes.get(key):
                    entry[key] = attributes[key][0]
            entry["tags"].update(attributes.get("tag", ()))
            if columns[2] == "CDS":
                entry["cds"] += int(columns[4]) - int(columns[3]) + 1
            elif columns[2] == "start_codon":
                entry["start"] = True
            elif columns[2] == "stop_codon":
                entry["stop"] = True
    return found


def _rankings(path: Path) -> dict[str, int]:
    ranks = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            ranks[_full(columns[4])] = int(columns[3])
    return ranks


def _fasta(path: Path):
    header, parts = None, []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text.startswith(">"):
                if header is not None:
                    yield header, "".join(parts).upper()
                header, parts = text[1:], []
            elif text:
                parts.append("".join(text.split()))
    if header is not None:
        yield header, "".join(parts).upper()


def _translations(path: Path) -> dict[str, tuple[str, str, str]]:
    """Protein identifier, gene identifier and sequence of each translated transcript."""
    found = {}
    for header, sequence in _fasta(path):
        fields = header.split("|")
        ids = {
            prefix: next(f for f in fields if f.startswith(prefix) and _identifier(f)[1] is not None)
            for prefix in ("ENSP", "ENST", "ENSG")
        }
        found[_full(ids["ENST"])] = (_full(ids["ENSP"]), _full(ids["ENSG"]), sequence)
    return found


def _selenocysteine(path: Path) -> dict[str, tuple[int, ...]]:
    positions: dict[str, set[int]] = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            if len(columns) == 2:
                positions[_full(columns[0])].add(int(columns[1]))
    return {transcript: tuple(sorted(values)) for transcript, values in positions.items()}


def _pick(candidates: list[str], transcripts: dict[str, dict], ranks: dict[str, int]) -> tuple[str, str] | None:
    """The transcript of a gene: MANE Select, else best GENCODE rank, else Ensembl canonical, else longest CDS."""
    eligible = [t for t in candidates if transcripts[t]["protein_id"] is not None and transcripts[t]["cds"] > 0]
    if not eligible:
        return None
    mane = [t for t in eligible if "MANE_Select" in transcripts[t]["tags"]]
    if mane:
        return min(mane, key=lambda t: (ranks.get(t, UNRANKED), t)), "mane_select"
    ranked = [t for t in eligible if t in ranks]
    if ranked:
        best = min(ranks[t] for t in ranked)
        return min(t for t in ranked if ranks[t] == best), "transcript_ranking"
    canonical = [t for t in eligible if "Ensembl_canonical" in transcripts[t]["tags"]]
    if canonical:
        return min(canonical, key=lambda t: (-transcripts[t]["cds"], t)), "ensembl_canonical"
    longest = max(transcripts[t]["cds"] for t in eligible)
    return min(t for t in eligible if transcripts[t]["cds"] == longest), "longest_cds"


def _passes(sequence: str, cds: int, selenocysteine: tuple[int, ...]) -> bool:
    """Sequence checks of the translation: no internal stop, legal residues, selenocysteines and length."""
    protein = sequence[:-1] if sequence.endswith("*") else sequence
    observed = tuple(index + 1 for index, residue in enumerate(protein) if residue == "U")
    return (
        bool(protein)
        and cds > 0
        and cds % 3 == 0
        and "*" not in protein
        and set(protein) <= AMINO_ACIDS
        and observed == selenocysteine
        and len(protein) == cds // 3
    )


def gene_reference(output: Path) -> None:
    """Write the gene identifier table: one GENCODE v50 transcript per approved HGNC protein-coding gene."""
    genes = _protein_coding_genes(external.path("hgnc"))
    by_hgnc = {gene.hgnc_id: gene for gene in genes}
    shared: dict[str, list[Gene]] = defaultdict(list)
    for gene in genes:
        if gene.ensembl_gene_id:
            shared[gene.ensembl_gene_id].append(gene)
    by_ensembl = {}
    for ensembl_id, owners in shared.items():
        if len(owners) == 1:
            by_ensembl[ensembl_id] = owners[0]
        else:
            for gene in owners:
                by_hgnc.pop(gene.hgnc_id, None)
    merged_into = _merged_into(external.path("hgnc_withdrawn"))
    transcripts = _transcripts(external.path("gencode_gtf"))
    ranks = _rankings(external.path("gencode_rankings"))
    translations = _translations(external.path("gencode_translations"))
    selenocysteine = _selenocysteine(external.path("gencode_selenocysteine"))

    grouped: dict[str, list[str]] = defaultdict(list)
    blocked: set[str] = set()
    evidence: dict[str, str] = {}
    for transcript_id, transcript in transcripts.items():
        gene_stable = _identifier(transcript["gene_id"])[0]
        given = resolved = transcript["hgnc_id"]
        notes = [f"{GENCODE} GTF"]
        by_id = by_hgnc.get(given or "")
        if given is not None and by_id is None and given in merged_into:
            targets = [target for target in merged_into[given] if target in by_hgnc]
            if len(targets) != 1:
                if gene_stable in by_ensembl:
                    blocked.add(by_ensembl[gene_stable].hgnc_id)
                continue
            resolved, by_id = targets[0], by_hgnc[targets[0]]
            notes.append(f"HGNC withdrawn {given}->{resolved}")
        by_stable = by_ensembl.get(gene_stable)
        if by_stable is not None and (
            (resolved is not None and resolved != by_stable.hgnc_id)
            or (by_id is not None and by_id.hgnc_id != by_stable.hgnc_id)
        ):
            blocked.update(gene.hgnc_id for gene in (by_id, by_stable) if gene is not None)
            continue
        match = by_id or by_stable
        if match is None:
            continue
        grouped[match.hgnc_id].append(transcript_id)
        if by_id is not None:
            notes.append("approved HGNC hgnc_id")
            if by_id.ensembl_gene_id and by_id.ensembl_gene_id != gene_stable:
                notes.append("GENCODE/HGNC Ensembl stable-ID discrepancy retained")
        else:
            notes.append("approved HGNC Ensembl stable ID")
        evidence[transcript_id] = "; ".join(notes)

    rows = []
    for gene_key in sorted(by_hgnc):
        if gene_key in blocked or gene_key not in grouped:
            continue
        picked = _pick(grouped[gene_key], transcripts, ranks)
        if picked is None:
            continue
        transcript_id, reason = picked
        transcript = transcripts[transcript_id]
        translation = translations.get(transcript_id)
        if translation is None or translation[:2] != (transcript["protein_id"], transcript["gene_id"]):
            continue
        if not _passes(translation[2], transcript["cds"], selenocysteine.get(transcript_id, ())):
            continue
        gene = by_hgnc[gene_key]
        gene_id, gene_version = _identifier(transcript["gene_id"])
        stable_transcript, transcript_version = _identifier(transcript_id)
        protein_id, protein_version = _identifier(transcript["protein_id"])
        rows.append(
            {
                "gene_key": gene_key,
                "hgnc_id": gene_key,
                "symbol": gene.symbol,
                "entrez_id": gene.entrez_id,
                "ensembl_gene_id": gene_id,
                "ensembl_gene_version": gene_version,
                "ensembl_transcript_id": stable_transcript,
                "ensembl_transcript_version": transcript_version,
                "ensembl_protein_id": protein_id,
                "ensembl_protein_version": protein_version,
                "selection_reason": reason,
                "transcript_rank": ranks.get(transcript_id),
                "cds_length_nt": transcript["cds"],
                "complete_cds": (
                    transcript["start"] and transcript["stop"] and not INCOMPLETE_TAGS & transcript["tags"]
                ),
                "mapping_status": "exact",
                "mapping_evidence": evidence[transcript_id],
                "source_release": GENCODE,
            }
        )
    table = pa.Table.from_pylist(rows, schema=REFERENCE_SCHEMA)
    pq.write_table(
        table,
        output,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
        version="2.6",
    )


def _archive_member(archive: Path) -> tuple[str, pd.DataFrame]:
    with tarfile.open(archive, "r:*") as bundle:
        members = sorted(
            (m for m in bundle.getmembers() if m.isfile() and EXPRESSION_MEMBER.search(m.name)),
            key=lambda m: m.name,
        )
        if len(members) != 1:
            raise ValueError(f"{archive.name}: expected one tumour expression table, found {len(members)}")
        frame = pd.read_csv(
            bundle.extractfile(members[0]), sep="\t", dtype={"GeneSymbol": str, "EntrezID": str}
        )
    return members[0].name, frame


def _write_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + external.PART_SUFFIX)
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def tcga_expression(output: Path, progress: Callable[[str], None] | None = None) -> None:
    """Write the TCGA matrix of RSEM upper-quartile normalised counts, genes by primary tumour samples."""
    hgnc = pd.read_csv(
        external.path("hgnc"), sep="\t", usecols=["hgnc_id", "locus_group", "status", "entrez_id"], dtype=str,
        keep_default_na=False,
    )
    coding = hgnc.loc[
        hgnc["status"].eq("Approved") & hgnc["locus_group"].eq("protein-coding gene") & hgnc["entrez_id"].ne("")
    ]
    gene_of = dict(zip(coding["entrez_id"].str.strip(), coding["hgnc_id"].str.strip()))
    inputs = external.item("tcga_expression")["inputs"]
    archives = sorted(
        (external.path(item_id) for item_id in inputs if item_id.startswith("tcga_")), key=lambda path: path.name
    )
    blocks, members, counts, excluded = [], [], {}, {}
    aliquots: dict[str, list[str]] = defaultdict(list)
    for index, archive in enumerate(archives, start=1):
        member, frame = _archive_member(archive)
        members.append(member)
        samples = [str(column) for column in frame.columns if column not in ("GeneSymbol", "EntrezID")]
        kinds = {sample: TCGA_SAMPLE.match(sample) for sample in samples}
        malformed = [sample for sample, match in kinds.items() if match is None]
        if malformed:
            raise ValueError(f"{archive.name}: malformed TCGA sample identifiers {malformed[:3]}")
        primary = sorted(s for s in samples if kinds[s].group("type") in PRIMARY_TUMOUR_CODES)
        excluded[archive.name] = sorted(set(samples) - set(primary))
        first: dict[str, str] = {}
        for sample in primary:
            aliquots[sample[:SAMPLE_KEY_LENGTH]].append(sample)
            first.setdefault(sample[:SAMPLE_KEY_LENGTH], sample)
        kept = sorted(first.values())
        counts[archive.name.removesuffix(ARCHIVE_SUFFIX).upper()] = len(kept)
        values = frame[["EntrezID", *kept]].assign(gene_key=frame["EntrezID"].map(gene_of))
        values = values.dropna(subset=["gene_key"]).drop(columns="EntrezID")
        values[kept] = values[kept].apply(pd.to_numeric, errors="coerce")
        blocks.append(values.groupby("gene_key", sort=True).mean())
        if progress is not None:
            progress(f"{index}/{len(archives)} archives read")
    matrix = pd.concat(blocks, axis=1, join="outer", verify_integrity=True).sort_index()
    matrix.index.name = "gene_key"
    matrix = matrix.reset_index()
    _write_json(
        {
            "expression_unit": "RSEM normalized count (upper quartile)",
            "log_transformed": False,
            "orientation": "gene_by_sample",
            "source_members": members,
            "genes": int(len(matrix)),
            "samples": int(sum(counts.values())),
            "cancer_type_counts": dict(sorted(counts.items())),
            "duplicate_aliquots_dropped": {
                key: sorted(names)[1:] for key, names in aliquots.items() if len(names) > 1
            },
            "accepted_tcga_sample_type_codes": list(PRIMARY_TUMOUR_CODES),
            "non_primary_samples_excluded": excluded,
        },
        output.parent / QA_FILE,
    )
    matrix.to_parquet(output, index=False, engine="pyarrow", compression="zstd", compression_level=9)


def _compare(name: str, rebuilt: pd.DataFrame, original: pd.DataFrame) -> dict:
    differing, largest = [], 0.0
    for column in original.columns:
        if column not in rebuilt.columns:
            differing.append(column)
            continue
        mine, theirs = rebuilt[column], original[column]
        if len(mine) != len(theirs) or mine.dtype != theirs.dtype:
            differing.append(column)
        elif pd.api.types.is_float_dtype(theirs):
            a, b = mine.to_numpy("float64"), theirs.to_numpy("float64")
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                differing.append(column)
                continue
            gap = float(np.nanmax(np.abs(a - b), initial=0.0))
            largest = max(largest, gap)
            if gap > 0:
                differing.append(column)
        elif not mine.equals(theirs):
            differing.append(column)
    if differing:
        print(f"{name}: {len(differing)} columns differ, for example {differing[:5]}", flush=True)
    return {
        "table": name,
        "rows": int(len(rebuilt)),
        "original_rows": int(len(original)),
        "columns": int(len(rebuilt.columns)),
        "original_columns": int(len(original.columns)),
        "differing_columns": len(differing) + len(set(rebuilt.columns) - set(original.columns)),
        "largest_absolute_difference": largest,
        "identical": bool(list(rebuilt.columns) == list(original.columns) and rebuilt.equals(original)),
    }


def _check(item_id: str, original: Path) -> list[dict]:
    entry = external.item(item_id)
    external.ensure(*entry["inputs"])
    builder = globals()[entry["builder"].split(":")[1]]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / Path(entry["path"]).name
        started = time.monotonic()

        def report(state: str) -> None:
            print(f"{item_id}: {state}, {time.monotonic() - started:.0f} s", flush=True)

        if item_id == "tcga_expression":
            builder(output, report)
        else:
            builder(output)
        minutes = (time.monotonic() - started) / 60
        size = output.stat().st_size
        print(f"{item_id}: built in {minutes:.1f} min, {size} bytes", flush=True)
        rows = [_compare(item_id, pd.read_parquet(output), pd.read_parquet(original))]
        rows[0].update(bytes=size, minutes=round(minutes, 1))
        if item_id == "tcga_expression":
            mine = (output.parent / QA_FILE).read_text()
            theirs = (original.parent / QA_FILE).read_text()
            rows.append(
                {
                    "table": QA_FILE,
                    "rows": len(json.loads(mine)),
                    "original_rows": len(json.loads(theirs)),
                    "identical": mine == theirs,
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="rebuild the tables and compare them with originals")
    check.add_argument("--gene-reference", type=Path, help="original gene identifier table, gene_id_map.parquet")
    check.add_argument(
        "--tcga-expression", type=Path, help="original tcga_gene_by_sample.parquet, with its QA record beside it"
    )
    arguments = parser.parse_args(argv)
    rows = []
    try:
        for item_id, original in (
            ("gene_reference", arguments.gene_reference),
            ("tcga_expression", arguments.tcga_expression),
        ):
            if original is not None:
                rows.extend(_check(item_id, original.expanduser()))
    except external.Missing as missing:
        print(json.dumps(missing.report(), indent=1))
        return 1
    table = pd.DataFrame(rows).convert_dtypes()
    CHECK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CHECK_OUTPUT, index=False)
    print(table.to_string(index=False))
    print(f"written {CHECK_OUTPUT}")
    return 0 if table["identical"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
