"""HGNC and GENCODE gene reference and sequence asset construction.
Imported by the XGEP source rebuild.
"""
from __future__ import annotations

import csv
import gzip
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal, TextIO

import pyarrow as pa
import pyarrow.parquet as pq
from Bio.Seq import Seq


GENCODE_RELEASE = "GENCODE v50"
_ENSEMBL_ID = re.compile(r"^(?P<stable>ENS[A-Z]*\d+)(?:\.(?P<version>\d+))?(?P<par>_PAR_Y)?$")
_HGNC_ID = re.compile(r"^HGNC:\d+$")
_STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_LEGAL_AMINO_ACIDS = _STANDARD_AMINO_ACIDS | {"U", "X"}
_LEGAL_NUCLEOTIDES = frozenset("ACGTN")
_STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})


class GeneReferenceError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class EnsemblIdentifier:

    stable_id: str
    version: int | None

    @property
    def full_id(self) -> str:
        if self.version is None:
            return self.stable_id
        if self.stable_id.endswith("_PAR_Y"):
            base = self.stable_id.removesuffix("_PAR_Y")
            return f"{base}.{self.version}_PAR_Y"
        return f"{self.stable_id}.{self.version}"


def parse_ensembl_identifier(value: str) -> EnsemblIdentifier:

    cleaned = value.strip()
    match = _ENSEMBL_ID.fullmatch(cleaned)
    if match is None:
        raise GeneReferenceError(f"invalid Ensembl identifier: {value!r}")
    stable_id = match.group("stable") + (match.group("par") or "")
    version_text = match.group("version")
    return EnsemblIdentifier(stable_id, int(version_text) if version_text else None)


@dataclass(frozen=True)
class HGNCRecord:
    gene_key: str
    symbol: str
    name: str
    locus_group: str
    locus_type: str
    status: str
    entrez_id: str | None
    ensembl_gene_id: str | None
    aliases: tuple[str, ...]
    previous_symbols: tuple[str, ...]
    mane_select_transcript: str | None
    raw_row_number: int

    @property
    def is_approved_protein_coding(self) -> bool:
        return self.status.casefold() == "approved" and self.locus_group.casefold() == (
            "protein-coding gene"
        )


@dataclass(frozen=True)
class WithdrawnHGNCRecord:
    hgnc_id: str
    status: str
    withdrawn_symbol: str
    merged_into: tuple[str, ...]
    raw_row_number: int


def _split_pipe_field(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    cleaned = value.strip().strip('"')
    if not cleaned:
        return ()
    return tuple(item.strip() for item in cleaned.split("|") if item.strip())


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def parse_hgnc_complete(path: Path) -> tuple[HGNCRecord, ...]:

    required = {"hgnc_id", "symbol", "locus_group", "status", "ensembl_gene_id"}
    records: list[HGNCRecord] = []
    with _open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = set(reader.fieldnames or ())
        missing = sorted(required - columns)
        if missing:
            raise GeneReferenceError(f"HGNC complete set lacks columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            gene_key = (row.get("hgnc_id") or "").strip()
            if not _HGNC_ID.fullmatch(gene_key):
                raise GeneReferenceError(
                    f"invalid HGNC ID on complete-set line {row_number}: {gene_key!r}"
                )
            ensembl = (row.get("ensembl_gene_id") or "").strip() or None
            if ensembl is not None:
                ensembl = parse_ensembl_identifier(ensembl).stable_id
            mane_field = (row.get("mane_select") or "").strip().strip('"')
            mane_transcript = mane_field.split("|", maxsplit=1)[0] if mane_field else None
            if mane_transcript is not None:
                mane_transcript = parse_ensembl_identifier(mane_transcript).full_id
            records.append(
                HGNCRecord(
                    gene_key=gene_key,
                    symbol=(row.get("symbol") or "").strip(),
                    name=(row.get("name") or "").strip(),
                    locus_group=(row.get("locus_group") or "").strip(),
                    locus_type=(row.get("locus_type") or "").strip(),
                    status=(row.get("status") or "").strip(),
                    entrez_id=(row.get("entrez_id") or "").strip() or None,
                    ensembl_gene_id=ensembl,
                    aliases=_split_pipe_field(row.get("alias_symbol")),
                    previous_symbols=_split_pipe_field(row.get("prev_symbol")),
                    mane_select_transcript=mane_transcript,
                    raw_row_number=row_number,
                )
            )
    return tuple(records)


def parse_hgnc_withdrawn(path: Path) -> tuple[WithdrawnHGNCRecord, ...]:

    records: list[WithdrawnHGNCRecord] = []
    with _open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        normalized = {str(name).strip().upper(): str(name) for name in reader.fieldnames or ()}
        required = {"HGNC_ID", "STATUS", "WITHDRAWN_SYMBOL"}
        missing = sorted(required - set(normalized))
        if missing:
            raise GeneReferenceError(f"HGNC withdrawn set lacks columns: {missing}")
        merged_column = next(
            (original for key, original in normalized.items() if key.startswith("MERGED_INTO")),
            None,
        )
        for row_number, row in enumerate(reader, start=2):
            hgnc_id = (row.get(normalized["HGNC_ID"]) or "").strip()
            if not _HGNC_ID.fullmatch(hgnc_id):
                raise GeneReferenceError(
                    f"invalid HGNC ID on withdrawn-set line {row_number}: {hgnc_id!r}"
                )
            merged_reports = _split_pipe_field(row.get(merged_column) if merged_column else None)
            merged_ids = tuple(item for item in merged_reports if _HGNC_ID.fullmatch(item))
            records.append(
                WithdrawnHGNCRecord(
                    hgnc_id=hgnc_id,
                    status=(row.get(normalized["STATUS"]) or "").strip(),
                    withdrawn_symbol=(row.get(normalized["WITHDRAWN_SYMBOL"]) or "").strip(),
                    merged_into=merged_ids,
                    raw_row_number=row_number,
                )
            )
    return tuple(records)


def parse_gtf_attributes(text: str) -> dict[str, tuple[str, ...]]:

    values: dict[str, list[str]] = defaultdict(list)
    for part in text.strip().strip(";").split(";"):
        item = part.strip()
        if not item:
            continue
        key_value = item.split(maxsplit=1)
        if len(key_value) != 2:
            raise GeneReferenceError(f"malformed GTF attribute: {item!r}")
        key, value = key_value
        values[key].append(value.strip().strip('"'))
    return {key: tuple(items) for key, items in values.items()}


@dataclass(frozen=True, order=True)
class GenomicSegment:
    chromosome: str
    start: int
    end: int
    strand: Literal["+", "-"]

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class TranscriptRecord:
    gene_id: str
    transcript_id: str
    protein_id: str | None
    hgnc_id: str | None
    gene_name: str
    gene_type: str
    transcript_type: str
    chromosome: str
    strand: Literal["+", "-"]
    tags: tuple[str, ...]
    cds_segments: tuple[GenomicSegment, ...]
    stop_codon_segments: tuple[GenomicSegment, ...]
    has_start_codon: bool
    has_stop_codon: bool

    @property
    def cds_length_nt(self) -> int:
        return sum(segment.length for segment in self.cds_segments)

    @property
    def is_protein_coding(self) -> bool:
        return self.gene_type == "protein_coding" and self.transcript_type == "protein_coding"

    @property
    def is_mane_select(self) -> bool:
        return "MANE_Select" in self.tags

    @property
    def is_ensembl_canonical(self) -> bool:
        return "Ensembl_canonical" in self.tags

    @property
    def is_complete(self) -> bool:
        incomplete_tags = {"cds_start_NF", "cds_end_NF"}
        return (
            self.has_start_codon
            and self.has_stop_codon
            and not incomplete_tags.intersection(self.tags)
        )


@dataclass
class _TranscriptAccumulator:
    gene_id: str
    transcript_id: str
    protein_id: str | None
    hgnc_id: str | None
    gene_name: str
    gene_type: str
    transcript_type: str
    chromosome: str
    strand: Literal["+", "-"]
    tags: set[str] = field(default_factory=set)
    cds_segments: list[GenomicSegment] = field(default_factory=list)
    stop_codon_segments: list[GenomicSegment] = field(default_factory=list)
    has_start_codon: bool = False
    has_stop_codon: bool = False


def _one_attribute(
    attributes: Mapping[str, tuple[str, ...]], key: str, *, required: bool = False
) -> str | None:
    values = attributes.get(key, ())
    unique = tuple(dict.fromkeys(values))
    if len(unique) > 1:
        raise GeneReferenceError(f"GTF attribute {key!r} has conflicting values: {unique}")
    if not unique:
        if required:
            raise GeneReferenceError(f"GTF feature lacks required attribute {key!r}")
        return None
    return unique[0]


def parse_gencode_gtf(path: Path) -> tuple[TranscriptRecord, ...]:

    accumulators: dict[str, _TranscriptAccumulator] = {}
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 9:
                raise GeneReferenceError(f"GTF line {line_number} does not have 9 columns")
            chromosome, _source, feature, start_text, end_text, _score, strand, _frame, raw = (
                columns
            )
            if feature not in {"transcript", "CDS", "start_codon", "stop_codon"}:
                continue
            if strand not in {"+", "-"}:
                raise GeneReferenceError(f"invalid GTF strand on line {line_number}: {strand!r}")
            attributes = parse_gtf_attributes(raw)
            transcript_id = _one_attribute(attributes, "transcript_id", required=True)
            gene_id = _one_attribute(attributes, "gene_id", required=True)
            assert transcript_id is not None and gene_id is not None
            parse_ensembl_identifier(transcript_id)
            parse_ensembl_identifier(gene_id)
            start = int(start_text)
            end = int(end_text)
            if start <= 0 or end < start:
                raise GeneReferenceError(f"invalid GTF interval on line {line_number}")
            candidate = _TranscriptAccumulator(
                gene_id=gene_id,
                transcript_id=transcript_id,
                protein_id=_one_attribute(attributes, "protein_id"),
                hgnc_id=_one_attribute(attributes, "hgnc_id"),
                gene_name=_one_attribute(attributes, "gene_name") or "",
                gene_type=_one_attribute(attributes, "gene_type") or "",
                transcript_type=_one_attribute(attributes, "transcript_type") or "",
                chromosome=chromosome,
                strand=strand,
            )
            if candidate.hgnc_id is not None and not _HGNC_ID.fullmatch(candidate.hgnc_id):
                raise GeneReferenceError(
                    f"invalid GTF hgnc_id on line {line_number}: {candidate.hgnc_id!r}"
                )
            existing = accumulators.get(transcript_id)
            if existing is None:
                existing = candidate
                accumulators[transcript_id] = existing
            else:
                identity_fields = (
                    "gene_id",
                    "chromosome",
                    "strand",
                    "gene_type",
                    "transcript_type",
                )
                for identity_field in identity_fields:
                    old = getattr(existing, identity_field)
                    new = getattr(candidate, identity_field)
                    if old and new and old != new:
                        raise GeneReferenceError(
                            f"transcript {transcript_id} has conflicting {identity_field}"
                        )
                for optional_field in ("protein_id", "hgnc_id", "gene_name"):
                    old = getattr(existing, optional_field)
                    new = getattr(candidate, optional_field)
                    if old and new and old != new:
                        raise GeneReferenceError(
                            f"transcript {transcript_id} has conflicting {optional_field}"
                        )
                    if not old and new:
                        setattr(existing, optional_field, new)
            existing.tags.update(attributes.get("tag", ()))
            segment = GenomicSegment(chromosome, start, end, strand)
            if feature == "CDS":
                existing.cds_segments.append(segment)
            elif feature == "start_codon":
                existing.has_start_codon = True
            elif feature == "stop_codon":
                existing.has_stop_codon = True
                existing.stop_codon_segments.append(segment)

    records: list[TranscriptRecord] = []
    for transcript_id in sorted(accumulators):
        item = accumulators[transcript_id]
        cds_segments = tuple(sorted(item.cds_segments, key=lambda value: (value.start, value.end)))
        stop_segments = tuple(
            sorted(item.stop_codon_segments, key=lambda value: (value.start, value.end))
        )
        records.append(
            TranscriptRecord(
                gene_id=item.gene_id,
                transcript_id=item.transcript_id,
                protein_id=item.protein_id,
                hgnc_id=item.hgnc_id,
                gene_name=item.gene_name,
                gene_type=item.gene_type,
                transcript_type=item.transcript_type,
                chromosome=item.chromosome,
                strand=item.strand,
                tags=tuple(sorted(item.tags)),
                cds_segments=cds_segments,
                stop_codon_segments=stop_segments,
                has_start_codon=item.has_start_codon,
                has_stop_codon=item.has_stop_codon,
            )
        )
    return tuple(records)


def parse_transcript_rankings(path: Path) -> dict[str, int]:

    rankings: dict[str, int] = {}
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 5:
                raise GeneReferenceError(
                    f"transcript-ranking line {line_number} has fewer than 5 columns"
                )
            transcript_id = parse_ensembl_identifier(columns[4]).full_id
            try:
                rank = int(columns[3])
            except ValueError as exc:
                raise GeneReferenceError(
                    f"invalid transcript rank on line {line_number}: {columns[3]!r}"
                ) from exc
            if rank <= 0:
                raise GeneReferenceError(f"non-positive transcript rank on line {line_number}")
            previous = rankings.get(transcript_id)
            if previous is not None and previous != rank:
                raise GeneReferenceError(
                    f"conflicting ranks for transcript {transcript_id}: {previous}, {rank}"
                )
            rankings[transcript_id] = rank
    return rankings


@dataclass(frozen=True)
class TranslationRecord:
    protein_id: str
    transcript_id: str
    gene_id: str
    sequence: str
    raw_header: str


def _iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    with _open_text(path) as handle:
        header: str | None = None
        sequence: list[str] = []
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if header is not None:
                    yield header, "".join(sequence).upper()
                header = text[1:]
                if not header:
                    raise GeneReferenceError(f"empty FASTA header on line {line_number}")
                sequence = []
            else:
                if header is None:
                    raise GeneReferenceError(f"FASTA sequence before header on line {line_number}")
                sequence.append("".join(text.split()))
        if header is not None:
            yield header, "".join(sequence).upper()


def parse_pc_translations(path: Path) -> tuple[TranslationRecord, ...]:

    records: list[TranslationRecord] = []
    seen: dict[str, TranslationRecord] = {}
    for header, sequence in _iter_fasta(path):
        identifiers = header.split("|")
        protein_ids = [
            item
            for item in identifiers
            if item.startswith("ENSP") and parse_ensembl_identifier(item).version is not None
        ]
        transcript_ids = [
            item
            for item in identifiers
            if item.startswith("ENST") and parse_ensembl_identifier(item).version is not None
        ]
        gene_ids = [
            item
            for item in identifiers
            if item.startswith("ENSG") and parse_ensembl_identifier(item).version is not None
        ]
        if len(protein_ids) != 1 or len(transcript_ids) != 1 or len(gene_ids) != 1:
            raise GeneReferenceError(
                f"translation FASTA header has ambiguous Ensembl IDs: {header}"
            )
        protein_id = parse_ensembl_identifier(protein_ids[0]).full_id
        transcript_id = parse_ensembl_identifier(transcript_ids[0]).full_id
        gene_id = parse_ensembl_identifier(gene_ids[0]).full_id
        record = TranslationRecord(protein_id, transcript_id, gene_id, sequence, header)
        previous = seen.get(transcript_id)
        if previous is not None and previous != record:
            raise GeneReferenceError(f"conflicting translations for transcript {transcript_id}")
        seen[transcript_id] = record
        records.append(record)
    return tuple(records)


def parse_selenocysteine_metadata(path: Path) -> dict[str, tuple[int, ...]]:

    positions: dict[str, set[int]] = defaultdict(set)
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 2:
                raise GeneReferenceError(
                    f"selenocysteine line {line_number} does not have 2 columns"
                )
            transcript_id = parse_ensembl_identifier(columns[0]).full_id
            try:
                position = int(columns[1])
            except ValueError as exc:
                raise GeneReferenceError(
                    f"invalid selenocysteine position on line {line_number}"
                ) from exc
            if position <= 0:
                raise GeneReferenceError(
                    f"non-positive selenocysteine position on line {line_number}"
                )
            positions[transcript_id].add(position)
    return {key: tuple(sorted(value)) for key, value in sorted(positions.items())}


SelectionReason = Literal["mane_select", "transcript_ranking", "ensembl_canonical", "longest_cds"]


@dataclass(frozen=True)
class SelectedTranscript:
    gene_key: str
    hgnc_id: str
    symbol: str
    entrez_id: str | None
    ensembl_gene_id: str
    ensembl_gene_version: int | None
    ensembl_transcript_id: str
    ensembl_transcript_version: int | None
    ensembl_protein_id: str
    ensembl_protein_version: int | None
    selection_reason: SelectionReason
    transcript_rank: int | None
    cds_length_nt: int
    complete_cds: bool
    mapping_status: Literal["exact"] = "exact"
    mapping_evidence: str = "HGNC complete; GENCODE v50 GTF"
    source_release: str = GENCODE_RELEASE


def _selection_pool(
    candidates: Sequence[TranscriptRecord], rankings: Mapping[str, int]
) -> tuple[SelectionReason, list[TranscriptRecord]]:
    mane = [item for item in candidates if item.is_mane_select]
    if mane:
        return "mane_select", mane
    ranked = [item for item in candidates if item.transcript_id in rankings]
    if ranked:
        best_rank = min(rankings[item.transcript_id] for item in ranked)
        return "transcript_ranking", [
            item for item in ranked if rankings[item.transcript_id] == best_rank
        ]
    canonical = [item for item in candidates if item.is_ensembl_canonical]
    if canonical:
        return "ensembl_canonical", canonical
    longest = max(item.cds_length_nt for item in candidates)
    return "longest_cds", [item for item in candidates if item.cds_length_nt == longest]


def choose_primary_transcript(
    candidates: Sequence[TranscriptRecord], rankings: Mapping[str, int]
) -> tuple[TranscriptRecord, SelectionReason]:

    eligible = [
        item
        for item in candidates
        if item.is_protein_coding and item.protein_id is not None and item.cds_length_nt > 0
    ]
    if not eligible:
        raise GeneReferenceError("no protein-coding transcript with a positive-length CDS")
    reason, pool = _selection_pool(eligible, rankings)
    if reason == "mane_select":
        pool.sort(
            key=lambda item: (
                rankings.get(item.transcript_id, 2**31 - 1),
                item.transcript_id,
            )
        )
    elif reason == "ensembl_canonical":
        pool.sort(key=lambda item: (-item.cds_length_nt, item.transcript_id))
    else:
        pool.sort(key=lambda item: item.transcript_id)
    return pool[0], reason


@dataclass(frozen=True)
class MappingIssue:
    gene_key: str | None
    source_identifier: str
    mapping_status: Literal[
        "ambiguous", "retired_unresolved", "readthrough", "complex_locus", "missing"
    ]
    reason: str
    candidates: tuple[str, ...]
    mapping_evidence: str


@dataclass(frozen=True)
class SequenceAsset:
    gene_key: str
    symbol: str
    transcript_id: str
    protein_id: str
    cds_sequence: str | None
    protein_sequence: str
    start_codon: str | None
    stop_codon: str | None
    has_start_codon_feature: bool
    has_stop_codon_feature: bool
    cds_length_divisible_by_three: bool
    protein_has_terminal_stop: bool
    protein_has_internal_stop: bool
    illegal_protein_characters: str
    selenocysteine_positions: tuple[int, ...]
    selenocysteine_supported: bool
    translation_length_matches_cds: bool | None
    sequence_qc_pass: bool


@dataclass(frozen=True)
class ProteinUniverseRow:
    gene_key: str
    symbol: str
    ensembl_transcript_id: str
    ensembl_protein_id: str
    protein_length_aa: int
    contains_x: bool
    x_fraction: float
    contains_selenocysteine: bool
    mapping_status: Literal["exact"] = "exact"
    sequence_qc_status: Literal["pass"] = "pass"


@dataclass(frozen=True)
class GeneReferenceBuild:
    selected_transcripts: tuple[SelectedTranscript, ...]
    sequence_assets: tuple[SequenceAsset, ...]
    protein_universe: tuple[ProteinUniverseRow, ...]
    issues: tuple[MappingIssue, ...]


def _index_hgnc(
    records: Sequence[HGNCRecord],
) -> tuple[dict[str, HGNCRecord], dict[str, HGNCRecord], list[MappingIssue]]:
    by_key_raw: dict[str, list[HGNCRecord]] = defaultdict(list)
    for record in records:
        if record.is_approved_protein_coding:
            by_key_raw[record.gene_key].append(record)
    issues: list[MappingIssue] = []
    by_key: dict[str, HGNCRecord] = {}
    for key, grouped in sorted(by_key_raw.items()):
        unique_records = list(dict.fromkeys(grouped))
        if len(unique_records) != 1:
            issues.append(
                MappingIssue(
                    gene_key=key,
                    source_identifier=key,
                    mapping_status="ambiguous",
                    reason="duplicate conflicting HGNC complete-set rows",
                    candidates=tuple(str(item.raw_row_number) for item in unique_records),
                    mapping_evidence="HGNC complete",
                )
            )
            continue
        by_key[key] = unique_records[0]

    by_ensembl_raw: dict[str, list[HGNCRecord]] = defaultdict(list)
    for record in by_key.values():
        if record.ensembl_gene_id:
            by_ensembl_raw[record.ensembl_gene_id].append(record)
    by_ensembl: dict[str, HGNCRecord] = {}
    for ensembl_id, grouped in sorted(by_ensembl_raw.items()):
        if len(grouped) == 1:
            by_ensembl[ensembl_id] = grouped[0]
            continue
        candidates = tuple(sorted(item.gene_key for item in grouped))
        for item in grouped:
            by_key.pop(item.gene_key, None)
            issues.append(
                MappingIssue(
                    gene_key=item.gene_key,
                    source_identifier=ensembl_id,
                    mapping_status="ambiguous",
                    reason="one Ensembl gene maps to multiple approved HGNC IDs",
                    candidates=candidates,
                    mapping_evidence="HGNC complete",
                )
            )
    return by_key, by_ensembl, issues


def _extract_genome_sequences(
    path: Path, transcripts: Iterable[TranscriptRecord]
) -> dict[tuple[str, int, int], str]:
    wanted: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for transcript in transcripts:
        for segment in transcript.cds_segments + transcript.stop_codon_segments:
            wanted[segment.chromosome].add((segment.start, segment.end))
    if not wanted:
        return {}
    extracted: dict[tuple[str, int, int], str] = {}
    remaining_chromosomes = set(wanted)
    for header, sequence in _iter_fasta(path):
        chromosome = header.split()[0]
        intervals = wanted.get(chromosome)
        if not intervals:
            continue
        upper = sequence.upper()
        illegal = sorted(set(upper) - _LEGAL_NUCLEOTIDES)
        if illegal:
            raise GeneReferenceError(
                f"genome record {chromosome} contains unsupported characters: {illegal}"
            )
        for start, end in intervals:
            if end > len(upper):
                raise GeneReferenceError(
                    f"genomic interval {chromosome}:{start}-{end} exceeds FASTA length"
                )
            extracted[(chromosome, start, end)] = upper[start - 1 : end]
        remaining_chromosomes.discard(chromosome)
        if not remaining_chromosomes:
            break
    missing = [
        (chromosome, start, end)
        for chromosome, intervals in wanted.items()
        for start, end in intervals
        if (chromosome, start, end) not in extracted
    ]
    if missing:
        preview = ", ".join(f"{chrom}:{start}-{end}" for chrom, start, end in missing[:5])
        raise GeneReferenceError(f"genome FASTA lacks required intervals: {preview}")
    return extracted


def _assemble_segments(
    segments: Sequence[GenomicSegment], extracted: Mapping[tuple[str, int, int], str]
) -> str:
    if not segments:
        return ""
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    sequence = "".join(extracted[(item.chromosome, item.start, item.end)] for item in ordered)
    if ordered[0].strand == "-":
        sequence = str(Seq(sequence).reverse_complement())
    return sequence


def _validate_translation(
    transcript: TranscriptRecord,
    translation: TranslationRecord,
    cds_sequence: str | None,
    stop_codon: str | None,
    selenocysteine_positions: tuple[int, ...],
) -> SequenceAsset:
    raw_protein = translation.sequence.upper()
    terminal_stop = raw_protein.endswith("*")
    protein = raw_protein[:-1] if terminal_stop else raw_protein
    internal_stop = "*" in protein
    illegal = "".join(sorted(set(protein) - _LEGAL_AMINO_ACIDS))
    observed_u = tuple(index + 1 for index, residue in enumerate(protein) if residue == "U")
    seleno_supported = observed_u == selenocysteine_positions
    divisible = transcript.cds_length_nt > 0 and transcript.cds_length_nt % 3 == 0
    length_matches = None if cds_sequence is None else len(protein) == len(cds_sequence) // 3
    sequence_qc_pass = (
        bool(protein)
        and divisible
        and not internal_stop
        and not illegal
        and seleno_supported
        and length_matches is not False
    )
    return SequenceAsset(
        gene_key="",
        symbol="",
        transcript_id=transcript.transcript_id,
        protein_id=translation.protein_id,
        cds_sequence=cds_sequence,
        protein_sequence=protein,
        start_codon=cds_sequence[:3] if cds_sequence else None,
        stop_codon=stop_codon,
        has_start_codon_feature=transcript.has_start_codon,
        has_stop_codon_feature=transcript.has_stop_codon,
        cds_length_divisible_by_three=divisible,
        protein_has_terminal_stop=terminal_stop,
        protein_has_internal_stop=internal_stop,
        illegal_protein_characters=illegal,
        selenocysteine_positions=selenocysteine_positions,
        selenocysteine_supported=seleno_supported,
        translation_length_matches_cds=length_matches,
        sequence_qc_pass=sequence_qc_pass,
    )


def build_gene_reference(
    hgnc_records: Sequence[HGNCRecord],
    transcripts: Sequence[TranscriptRecord],
    rankings: Mapping[str, int],
    translations: Sequence[TranslationRecord],
    *,
    withdrawn_records: Sequence[WithdrawnHGNCRecord] = (),
    genome_fasta: Path | None = None,
    selenocysteine: Mapping[str, Sequence[int]] | None = None,
) -> GeneReferenceBuild:

    by_hgnc, by_ensembl, issues = _index_hgnc(hgnc_records)
    translation_index: dict[str, TranslationRecord] = {}
    ambiguous_translations: set[str] = set()
    for translation in translations:
        previous = translation_index.get(translation.transcript_id)
        if previous is not None and previous != translation:
            ambiguous_translations.add(translation.transcript_id)
        else:
            translation_index[translation.transcript_id] = translation

    withdrawn_index: dict[str, tuple[str, ...]] = {}
    for record in withdrawn_records:
        previous = withdrawn_index.get(record.hgnc_id)
        if previous is not None and previous != record.merged_into:
            raise GeneReferenceError(f"conflicting HGNC withdrawal records for {record.hgnc_id}")
        withdrawn_index[record.hgnc_id] = record.merged_into

    grouped: dict[str, list[TranscriptRecord]] = defaultdict(list)
    blocked_gene_keys: set[str] = set()
    mapping_evidence_by_transcript: dict[str, str] = {}
    for transcript in transcripts:
        if not transcript.is_protein_coding:
            continue
        gene_stable = parse_ensembl_identifier(transcript.gene_id).stable_id
        raw_hgnc_id = transcript.hgnc_id
        resolved_hgnc_id = raw_hgnc_id
        evidence_parts = ["GENCODE v50 GTF"]
        hgnc_match = by_hgnc.get(raw_hgnc_id or "")
        if raw_hgnc_id is not None and hgnc_match is None and raw_hgnc_id in withdrawn_index:
            merged_into = withdrawn_index[raw_hgnc_id]
            approved_targets = tuple(item for item in merged_into if item in by_hgnc)
            if len(approved_targets) == 1:
                resolved_hgnc_id = approved_targets[0]
                hgnc_match = by_hgnc[resolved_hgnc_id]
                evidence_parts.append(f"HGNC withdrawn {raw_hgnc_id}->{resolved_hgnc_id}")
            else:
                ensembl_candidate = by_ensembl.get(gene_stable)
                if ensembl_candidate is not None:
                    blocked_gene_keys.add(ensembl_candidate.gene_key)
                issues.append(
                    MappingIssue(
                        gene_key=(
                            ensembl_candidate.gene_key if ensembl_candidate is not None else None
                        ),
                        source_identifier=raw_hgnc_id,
                        mapping_status=(
                            "ambiguous" if len(approved_targets) > 1 else "retired_unresolved"
                        ),
                        reason="GENCODE uses an HGNC ID without one approved merge target",
                        candidates=approved_targets,
                        mapping_evidence="HGNC withdrawn; GENCODE v50 GTF",
                    )
                )
                continue
        ensembl_match = by_ensembl.get(gene_stable)
        hgnc_conflict = (
            resolved_hgnc_id is not None
            and ensembl_match is not None
            and resolved_hgnc_id != ensembl_match.gene_key
        )
        if hgnc_conflict or (
            hgnc_match is not None and ensembl_match is not None and hgnc_match != ensembl_match
        ):
            affected = {
                match.gene_key for match in (hgnc_match, ensembl_match) if match is not None
            }
            blocked_gene_keys.update(affected)
            issues.append(
                MappingIssue(
                    gene_key=min(affected) if affected else transcript.hgnc_id,
                    source_identifier=transcript.transcript_id,
                    mapping_status="ambiguous",
                    reason="GENCODE hgnc_id conflicts with HGNC Ensembl mapping",
                    candidates=tuple(
                        sorted(
                            affected
                            | ({resolved_hgnc_id} if resolved_hgnc_id is not None else set())
                        )
                    ),
                    mapping_evidence="HGNC complete; GENCODE v50 GTF",
                )
            )
            continue
        match = hgnc_match or ensembl_match
        if match is not None:
            grouped[match.gene_key].append(transcript)
            if hgnc_match is not None:
                evidence_parts.append("approved HGNC hgnc_id")
                if (
                    hgnc_match.ensembl_gene_id is not None
                    and hgnc_match.ensembl_gene_id != gene_stable
                ):
                    evidence_parts.append("GENCODE/HGNC Ensembl stable-ID discrepancy retained")
            else:
                evidence_parts.append("approved HGNC Ensembl stable ID")
            mapping_evidence_by_transcript[transcript.transcript_id] = "; ".join(evidence_parts)

    selected_pairs: list[tuple[HGNCRecord, TranscriptRecord, SelectionReason]] = []
    for gene_key, hgnc in sorted(by_hgnc.items()):
        if gene_key in blocked_gene_keys:
            continue
        candidates = grouped.get(gene_key, [])
        if not candidates:
            issues.append(
                MappingIssue(
                    gene_key=gene_key,
                    source_identifier=hgnc.ensembl_gene_id or gene_key,
                    mapping_status="missing",
                    reason="no matching protein-coding GENCODE transcript",
                    candidates=(),
                    mapping_evidence="HGNC complete; GENCODE v50 GTF",
                )
            )
            continue
        try:
            transcript, reason = choose_primary_transcript(candidates, rankings)
        except GeneReferenceError as exc:
            issues.append(
                MappingIssue(
                    gene_key=gene_key,
                    source_identifier=hgnc.ensembl_gene_id or gene_key,
                    mapping_status="missing",
                    reason=str(exc),
                    candidates=tuple(sorted(item.transcript_id for item in candidates)),
                    mapping_evidence="GENCODE v50 GTF",
                )
            )
            continue
        if transcript.transcript_id in ambiguous_translations:
            issues.append(
                MappingIssue(
                    gene_key=gene_key,
                    source_identifier=transcript.transcript_id,
                    mapping_status="ambiguous",
                    reason="selected transcript has conflicting translation FASTA records",
                    candidates=(transcript.transcript_id,),
                    mapping_evidence="GENCODE v50 pc_translations",
                )
            )
            continue
        translation = translation_index.get(transcript.transcript_id)
        if translation is None or transcript.protein_id is None:
            issues.append(
                MappingIssue(
                    gene_key=gene_key,
                    source_identifier=transcript.transcript_id,
                    mapping_status="missing",
                    reason="selected transcript has no translation",
                    candidates=(),
                    mapping_evidence="GENCODE v50 pc_translations",
                )
            )
            continue
        if (
            translation.protein_id != transcript.protein_id
            or translation.gene_id != transcript.gene_id
        ):
            issues.append(
                MappingIssue(
                    gene_key=gene_key,
                    source_identifier=transcript.transcript_id,
                    mapping_status="ambiguous",
                    reason="GTF and translation FASTA identifiers disagree",
                    candidates=(transcript.protein_id, translation.protein_id),
                    mapping_evidence="GENCODE v50 GTF; pc_translations",
                )
            )
            continue
        selected_pairs.append((hgnc, transcript, reason))

    extracted: dict[tuple[str, int, int], str] = {}
    if genome_fasta is not None:
        extracted = _extract_genome_sequences(genome_fasta, (item[1] for item in selected_pairs))

    selected_rows: list[SelectedTranscript] = []
    assets: list[SequenceAsset] = []
    universe: list[ProteinUniverseRow] = []
    selenocysteine = selenocysteine or {}
    for hgnc, transcript, reason in selected_pairs:
        translation = translation_index[transcript.transcript_id]
        gene_id = parse_ensembl_identifier(transcript.gene_id)
        transcript_id = parse_ensembl_identifier(transcript.transcript_id)
        assert transcript.protein_id is not None
        protein_id = parse_ensembl_identifier(transcript.protein_id)
        cds_sequence = (
            _assemble_segments(transcript.cds_segments, extracted) if genome_fasta else None
        )
        stop_codon = (
            _assemble_segments(transcript.stop_codon_segments, extracted)
            if genome_fasta and transcript.stop_codon_segments
            else None
        )
        positions = tuple(
            sorted(set(int(value) for value in selenocysteine.get(transcript.transcript_id, ())))
        )
        asset = _validate_translation(transcript, translation, cds_sequence, stop_codon, positions)
        asset = SequenceAsset(
            **{
                **asdict(asset),
                "gene_key": hgnc.gene_key,
                "symbol": hgnc.symbol,
            }
        )
        if not asset.sequence_qc_pass:
            qc_failures = []
            if not asset.protein_sequence:
                qc_failures.append("empty protein")
            if not asset.cds_length_divisible_by_three:
                qc_failures.append("CDS length is not positive and divisible by three")
            if asset.protein_has_internal_stop:
                qc_failures.append("internal protein stop")
            if asset.illegal_protein_characters:
                qc_failures.append(f"illegal protein characters={asset.illegal_protein_characters}")
            if not asset.selenocysteine_supported:
                qc_failures.append("selenocysteine metadata mismatch")
            if asset.translation_length_matches_cds is False:
                qc_failures.append("protein/CDS length mismatch")
            evidence = "GENCODE v50 GTF; pc_translations; Selenocysteine"
            if genome_fasta is not None:
                evidence += "; genome FASTA"
            issues.append(
                MappingIssue(
                    gene_key=hgnc.gene_key,
                    source_identifier=transcript.transcript_id,
                    mapping_status="missing",
                    reason=f"selected transcript failed sequence QC: {', '.join(qc_failures)}",
                    candidates=(transcript.transcript_id, translation.protein_id),
                    mapping_evidence=evidence,
                )
            )
            continue
        selected_rows.append(
            SelectedTranscript(
                gene_key=hgnc.gene_key,
                hgnc_id=hgnc.gene_key,
                symbol=hgnc.symbol,
                entrez_id=hgnc.entrez_id,
                ensembl_gene_id=gene_id.stable_id,
                ensembl_gene_version=gene_id.version,
                ensembl_transcript_id=transcript_id.stable_id,
                ensembl_transcript_version=transcript_id.version,
                ensembl_protein_id=protein_id.stable_id,
                ensembl_protein_version=protein_id.version,
                selection_reason=reason,
                transcript_rank=rankings.get(transcript.transcript_id),
                cds_length_nt=transcript.cds_length_nt,
                complete_cds=transcript.is_complete,
                mapping_evidence=mapping_evidence_by_transcript[transcript.transcript_id],
            )
        )
        assets.append(asset)
        protein_length = len(asset.protein_sequence)
        x_count = asset.protein_sequence.count("X")
        universe.append(
            ProteinUniverseRow(
                gene_key=hgnc.gene_key,
                symbol=hgnc.symbol,
                ensembl_transcript_id=transcript.transcript_id,
                ensembl_protein_id=translation.protein_id,
                protein_length_aa=protein_length,
                contains_x=x_count > 0,
                x_fraction=x_count / protein_length,
                contains_selenocysteine="U" in asset.protein_sequence,
            )
        )

    return GeneReferenceBuild(
        selected_transcripts=tuple(sorted(selected_rows, key=lambda item: item.gene_key)),
        sequence_assets=tuple(sorted(assets, key=lambda item: item.gene_key)),
        protein_universe=tuple(sorted(universe, key=lambda item: item.gene_key)),
        issues=tuple(
            sorted(
                issues,
                key=lambda item: (
                    item.gene_key or "",
                    item.mapping_status,
                    item.source_identifier,
                    item.reason,
                ),
            )
        ),
    )


def _row_mapping(row: object) -> dict[str, Any]:
    if is_dataclass(row):
        return asdict(row)
    if isinstance(row, Mapping):
        return dict(row)
    raise TypeError(f"row must be a dataclass or mapping, found {type(row).__name__}")


def write_deterministic_parquet(
    rows: Sequence[object],
    path: Path,
    *,
    sort_by: Sequence[str] = ("gene_key",),
    column_order: Sequence[str] | None = None,
) -> None:

    mapped = [_row_mapping(row) for row in rows]
    if not mapped:
        raise GeneReferenceError("cannot infer a Parquet schema from zero rows")
    columns = list(column_order or mapped[0].keys())
    expected = set(columns)
    for index, row in enumerate(mapped):
        if set(row) != expected:
            raise GeneReferenceError(f"Parquet row {index} has inconsistent columns")
    unknown_sort = set(sort_by) - expected
    if unknown_sort:
        raise GeneReferenceError(f"unknown Parquet sort columns: {sorted(unknown_sort)}")
    mapped.sort(key=lambda row: tuple(row[key] for key in sort_by))
    table = pa.Table.from_pylist([{key: row[key] for key in columns} for row in mapped])
    table = table.replace_schema_metadata(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=9,
            use_dictionary=False,
            write_statistics=True,
            data_page_version="1.0",
            version="2.6",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_deterministic_fasta(
    records: Iterable[tuple[str, str]], path: Path, *, line_width: int = 60
) -> None:

    if line_width <= 0:
        raise ValueError("line_width must be positive")
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for header, sequence in records:
        clean_header = header.strip()
        clean_sequence = "".join(sequence.split()).upper()
        if not clean_header or clean_header.startswith(">") or "\n" in clean_header:
            raise GeneReferenceError(f"invalid FASTA header: {header!r}")
        if clean_header in seen:
            raise GeneReferenceError(f"duplicate FASTA header: {clean_header}")
        if not clean_sequence:
            raise GeneReferenceError(f"empty FASTA sequence for {clean_header}")
        seen.add(clean_header)
        normalized.append((clean_header, clean_sequence))
    normalized.sort(key=lambda item: item[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for header, sequence in normalized:
                handle.write(f">{header}\n")
                for offset in range(0, len(sequence), line_width):
                    handle.write(sequence[offset : offset + line_width] + "\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_mapping_issues_tsv(issues: Sequence[MappingIssue], path: Path) -> None:

    columns = (
        "gene_key",
        "source_identifier",
        "issue_class",
        "mapping_status",
        "reason",
        "candidates",
        "mapping_evidence",
    )
    ordered = sorted(
        issues,
        key=lambda item: (
            item.gene_key or "",
            item.mapping_status,
            item.source_identifier,
            item.reason,
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for issue in ordered:
                writer.writerow(
                    {
                        "gene_key": issue.gene_key or "",
                        "source_identifier": issue.source_identifier,
                        "issue_class": (
                            "ambiguous" if issue.mapping_status == "ambiguous" else "rejected"
                        ),
                        "mapping_status": issue.mapping_status,
                        "reason": issue.reason,
                        "candidates": "|".join(issue.candidates),
                        "mapping_evidence": issue.mapping_evidence,
                    }
                )
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_gene_reference_assets(build: GeneReferenceBuild, output_dir: Path) -> dict[str, Path]:

    if not build.selected_transcripts or not build.sequence_assets:
        raise GeneReferenceError("gene-reference build has no exact sequence assets")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "gene_id_map": output_dir / "gene_id_map.parquet",
        "transcript_selection": output_dir / "transcript_selection.parquet",
        "protein_universe": output_dir / "protein_universe.parquet",
        "selected_cds": output_dir / "selected_cds.fa",
        "selected_proteins": output_dir / "selected_proteins.fa",
        "mapping_exclusions": output_dir / "mapping_exclusions.tsv",
    }
    write_deterministic_parquet(build.selected_transcripts, outputs["gene_id_map"])
    write_deterministic_parquet(build.selected_transcripts, outputs["transcript_selection"])
    write_deterministic_parquet(build.protein_universe, outputs["protein_universe"])
    write_mapping_issues_tsv(build.issues, outputs["mapping_exclusions"])

    asset_by_key = {item.gene_key: item for item in build.sequence_assets}
    cds_records: list[tuple[str, str]] = []
    protein_records: list[tuple[str, str]] = []
    for selected in build.selected_transcripts:
        asset = asset_by_key[selected.gene_key]
        header = "|".join(
            (
                selected.gene_key,
                selected.symbol,
                asset.transcript_id,
                asset.protein_id,
            )
        )
        if asset.cds_sequence is None:
            raise GeneReferenceError("CDS FASTA requires a genome FASTA during construction")
        cds_records.append((header, asset.cds_sequence))
        protein_records.append((header, asset.protein_sequence))
    write_deterministic_fasta(cds_records, outputs["selected_cds"])
    write_deterministic_fasta(protein_records, outputs["selected_proteins"])
    return outputs


def build_gene_reference_from_files(
    *,
    hgnc_complete: Path,
    hgnc_withdrawn: Path,
    gtf: Path,
    transcript_rankings: Path,
    translations_fasta: Path,
    selenocysteine_metadata: Path,
    genome_fasta: Path | None = None,
) -> GeneReferenceBuild:

    withdrawn_records = parse_hgnc_withdrawn(hgnc_withdrawn)
    return build_gene_reference(
        parse_hgnc_complete(hgnc_complete),
        parse_gencode_gtf(gtf),
        parse_transcript_rankings(transcript_rankings),
        parse_pc_translations(translations_fasta),
        withdrawn_records=withdrawn_records,
        genome_fasta=genome_fasta,
        selenocysteine=parse_selenocysteine_metadata(selenocysteine_metadata),
    )
