"""Sequence features and XGEP pan-cancer embedding matrix construction.
Imported by the XGEP source rebuild.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal

from Bio.Data import CodonTable

from method_features.gene_mapping import SequenceAsset, write_deterministic_parquet
from method_features.multiomics import write_json_atomic


CODON_ORDER = tuple("".join(bases) for bases in product("ACGT", repeat=3))
AMINO_ACID_ORDER = tuple("ACDEFGHIKLMNPQRSTVWY")
DEEPHE_SEQUENCE_89_COLUMNS = (
    *(f"codon_frequency_{codon}" for codon in CODON_ORDER),
    "gc_content",
    "cds_length_nt",
    "cai_within_gene",
    "rscu_max_gene",
    *(f"amino_acid_frequency_{residue}" for residue in AMINO_ACID_ORDER),
    "protein_length_aa",
)
XGEP_150_COLUMNS = tuple(f"xgep_{index:03d}" for index in range(150))

_HGNC_ID = re.compile(r"^HGNC:[1-9]\d*$")
_ENSEMBL_GENE_ID = re.compile(
    r"^(?P<stable>ENSG\d+)(?:\.\d+)?(?P<par>_PAR_Y)?$",
    flags=re.IGNORECASE,
)
_LEGAL_CDS = frozenset("ACGTN")
_LEGAL_PROTEIN = frozenset(AMINO_ACID_ORDER) | {"U", "X"}

_STANDARD_CODE = CodonTable.unambiguous_dna_by_id[1]
_CODON_TO_RESIDUE = {
    **{codon: residue for codon, residue in _STANDARD_CODE.forward_table.items()},
    **{codon: "*" for codon in _STANDARD_CODE.stop_codons},
}
_SYNONYMOUS_CODONS: dict[str, tuple[str, ...]] = {
    residue: tuple(codon for codon in CODON_ORDER if _CODON_TO_RESIDUE[codon] == residue)
    for residue in sorted(set(_CODON_TO_RESIDUE.values()))
}


class LegacyFeatureError(ValueError):
    pass


def _canonical_gene_key(value: str) -> str:
    key = value.strip()
    if _HGNC_ID.fullmatch(key) is None:
        raise LegacyFeatureError(f"gene_key is not a canonical HGNC ID: {value!r}")
    return key


def _stable_ensembl_gene_id(value: str) -> str:
    cleaned = value.strip().upper()
    match = _ENSEMBL_GENE_ID.fullmatch(cleaned)
    if match is None:
        raise LegacyFeatureError(f"invalid Ensembl gene ID: {value!r}")
    return match.group("stable") + (match.group("par") or "")


def _sha256_path(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _content_hash(records: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(records),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DeepHESequence89Row:

    gene_key: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _canonical_gene_key(self.gene_key)
        if len(self.values) != len(DEEPHE_SEQUENCE_89_COLUMNS):
            raise LegacyFeatureError(f"DeepHE row has {len(self.values)} values; expected 89")
        if any(not math.isfinite(value) for value in self.values):
            raise LegacyFeatureError(f"DeepHE row for {self.gene_key} contains non-finite values")

    def as_record(self) -> dict[str, str | float]:
        return {
            "gene_key": self.gene_key,
            **dict(zip(DEEPHE_SEQUENCE_89_COLUMNS, self.values, strict=True)),
        }


@dataclass(frozen=True)
class DeepHESequence89Audit:
    input_assets: int
    output_genes: int
    genes_with_cds_n: int
    ambiguous_codons_excluded: int
    genes_with_protein_x: int
    protein_x_residues_excluded: int
    genes_with_selenocysteine: int
    protein_u_residues_excluded: int
    terminal_protein_stops_removed: int
    feature_content_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "view": "deephe_sequence_89",
            "input_assets": self.input_assets,
            "output_genes": self.output_genes,
            "feature_count": 89,
            "feature_columns": list(DEEPHE_SEQUENCE_89_COLUMNS),
            "codon_order": list(CODON_ORDER),
            "amino_acid_order": list(AMINO_ACID_ORDER),
            "genes_with_cds_n": self.genes_with_cds_n,
            "ambiguous_codons_excluded": self.ambiguous_codons_excluded,
            "genes_with_protein_x": self.genes_with_protein_x,
            "protein_x_residues_excluded": self.protein_x_residues_excluded,
            "genes_with_selenocysteine": self.genes_with_selenocysteine,
            "protein_u_residues_excluded": self.protein_u_residues_excluded,
            "terminal_protein_stops_removed": self.terminal_protein_stops_removed,
            "scaled": False,
            "label_input_used": False,
            "cds_n_rule": (
                "N-containing codons excluded from codon/RSCU/CAI; N excluded from GC "
                "denominator and included in CDS length"
            ),
            "protein_x_u_rule": (
                "X and U excluded from standard-amino-acid frequency denominator and "
                "included in protein length"
            ),
            "cai_rule": (
                "within-gene geometric mean of RSCU_i/max_synonymous_RSCU, excluding "
                "Met/Trp/stop/ambiguous codons; empty product equals 1"
            ),
            "rscu_max_rule": "maximum gene-wide RSCU under NCBI genetic code table 1",
        }


@dataclass(frozen=True)
class DeepHESequence89Build:
    rows: tuple[DeepHESequence89Row, ...]
    audit: DeepHESequence89Audit


@dataclass(frozen=True)
class _DeepHECounters:
    has_n: bool
    ambiguous_codons: int
    x_count: int
    u_count: int
    terminal_stop_removed: bool


def _normalize_deephe_sequences(cds_sequence: str, protein_sequence: str) -> tuple[str, str, bool]:
    cds = "".join(cds_sequence.split()).upper()
    protein_sequence = "".join(protein_sequence.split()).upper()
    terminal_stop_removed = protein_sequence.endswith("*")
    protein = protein_sequence[:-1] if terminal_stop_removed else protein_sequence

    if not cds or len(cds) % 3:
        raise LegacyFeatureError("DeepHE CDS must be non-empty and divisible by three")
    illegal_cds = sorted(set(cds) - _LEGAL_CDS)
    if illegal_cds:
        raise LegacyFeatureError(f"DeepHE CDS contains unsupported characters: {illegal_cds!r}")
    if not protein:
        raise LegacyFeatureError("DeepHE protein must be non-empty after terminal-stop removal")
    if "*" in protein:
        raise LegacyFeatureError("DeepHE protein contains an internal stop")
    illegal_protein = sorted(set(protein) - _LEGAL_PROTEIN)
    if illegal_protein:
        raise LegacyFeatureError(
            f"DeepHE protein contains unsupported characters: {illegal_protein!r}"
        )
    return cds, protein, terminal_stop_removed


def _rscu(codon_counts: Counter[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for synonymous in _SYNONYMOUS_CODONS.values():
        family_total = sum(codon_counts[codon] for codon in synonymous)
        if family_total == 0:
            continue
        expected = family_total / len(synonymous)
        values.update({codon: codon_counts[codon] / expected for codon in synonymous})
    return values


def _within_gene_cai(valid_codons: Sequence[str], rscu: Mapping[str, float]) -> float:
    log_weights: list[float] = []
    for codon in valid_codons:
        residue = _CODON_TO_RESIDUE[codon]
        if residue in {"M", "W", "*"}:
            continue
        family_max = max(rscu[item] for item in _SYNONYMOUS_CODONS[residue])
        weight = rscu[codon] / family_max
        if weight <= 0.0:
            raise LegacyFeatureError("encountered codon has non-positive CAI weight")
        log_weights.append(math.log(weight))
    return math.exp(math.fsum(log_weights) / len(log_weights)) if log_weights else 1.0


def _compute_deephe_sequence_89(
    gene_key: str,
    cds_sequence: str,
    protein_sequence: str,
) -> tuple[DeepHESequence89Row, _DeepHECounters]:
    key = _canonical_gene_key(gene_key)
    cds, protein, terminal_stop_removed = _normalize_deephe_sequences(
        cds_sequence, protein_sequence
    )
    codons = tuple(cds[offset : offset + 3] for offset in range(0, len(cds), 3))
    valid_codons = tuple(codon for codon in codons if "N" not in codon)
    if not valid_codons:
        raise LegacyFeatureError(f"{key} has no unambiguous codon")
    codon_counts = Counter(valid_codons)
    codon_total = len(valid_codons)
    codon_frequencies = tuple(codon_counts[codon] / codon_total for codon in CODON_ORDER)

    determinate_bases = sum(base != "N" for base in cds)
    if determinate_bases == 0:
        raise LegacyFeatureError(f"{key} has no determinate CDS base")
    gc_content = sum(base in "GC" for base in cds) / determinate_bases

    rscu = _rscu(codon_counts)
    cai = _within_gene_cai(valid_codons, rscu)
    rscu_max = max(rscu.values())

    protein_counts = Counter(protein)
    standard_residue_total = sum(protein_counts[residue] for residue in AMINO_ACID_ORDER)
    if standard_residue_total == 0:
        raise LegacyFeatureError(f"{key} has no standard amino-acid residue")
    amino_acid_frequencies = tuple(
        protein_counts[residue] / standard_residue_total for residue in AMINO_ACID_ORDER
    )
    values = (
        *codon_frequencies,
        gc_content,
        float(len(cds)),
        cai,
        rscu_max,
        *amino_acid_frequencies,
        float(len(protein)),
    )
    row = DeepHESequence89Row(key, tuple(float(value) for value in values))
    counters = _DeepHECounters(
        has_n="N" in cds,
        ambiguous_codons=len(codons) - len(valid_codons),
        x_count=protein_counts["X"],
        u_count=protein_counts["U"],
        terminal_stop_removed=terminal_stop_removed,
    )
    return row, counters


def compute_deephe_sequence_89(
    gene_key: str,
    cds_sequence: str,
    protein_sequence: str,
) -> DeepHESequence89Row:

    row, _ = _compute_deephe_sequence_89(gene_key, cds_sequence, protein_sequence)
    return row


def assert_deephe_sequence_89_contract(rows: Sequence[DeepHESequence89Row]) -> None:

    if not rows:
        raise LegacyFeatureError("deephe_sequence_89 has no rows")
    keys = [row.gene_key for row in rows]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise LegacyFeatureError("DeepHE rows must have sorted unique gene_key values")
    for row in rows:
        if len(row.values) != 89 or any(not math.isfinite(value) for value in row.values):
            raise LegacyFeatureError(f"invalid DeepHE feature vector for {row.gene_key}")


def build_deephe_sequence_89(assets: Iterable[SequenceAsset]) -> DeepHESequence89Build:

    rows: list[DeepHESequence89Row] = []
    counters: list[_DeepHECounters] = []
    seen: set[str] = set()
    input_assets = 0
    for asset in assets:
        input_assets += 1
        key = _canonical_gene_key(asset.gene_key)
        if key in seen:
            raise LegacyFeatureError(f"duplicate DeepHE gene_key: {key}")
        seen.add(key)
        if not asset.sequence_qc_pass:
            raise LegacyFeatureError(f"{key} did not pass the frozen sequence-asset QC")
        if asset.cds_sequence is None:
            raise LegacyFeatureError(f"{key} has no reconstructed CDS")
        row, row_counters = _compute_deephe_sequence_89(
            key, asset.cds_sequence, asset.protein_sequence
        )
        rows.append(row)
        counters.append(row_counters)

    ordered = tuple(sorted(rows, key=lambda row: row.gene_key))
    assert_deephe_sequence_89_contract(ordered)
    records = tuple(row.as_record() for row in ordered)
    audit = DeepHESequence89Audit(
        input_assets=input_assets,
        output_genes=len(ordered),
        genes_with_cds_n=sum(item.has_n for item in counters),
        ambiguous_codons_excluded=sum(item.ambiguous_codons for item in counters),
        genes_with_protein_x=sum(item.x_count > 0 for item in counters),
        protein_x_residues_excluded=sum(item.x_count for item in counters),
        genes_with_selenocysteine=sum(item.u_count > 0 for item in counters),
        protein_u_residues_excluded=sum(item.u_count for item in counters),
        terminal_protein_stops_removed=sum(item.terminal_stop_removed for item in counters),
        feature_content_sha256=_content_hash(records),
    )
    return DeepHESequence89Build(rows=ordered, audit=audit)


@dataclass(frozen=True)
class XGEP150Row:

    gene_key: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _canonical_gene_key(self.gene_key)
        if len(self.values) != 150:
            raise LegacyFeatureError(f"XGEP row has {len(self.values)} values; expected 150")
        if any(not math.isfinite(value) for value in self.values):
            raise LegacyFeatureError(f"XGEP row for {self.gene_key} contains non-finite values")

    def as_record(self) -> dict[str, str | float]:
        return {
            "gene_key": self.gene_key,
            **dict(zip(XGEP_150_COLUMNS, self.values, strict=True)),
        }


@dataclass(frozen=True)
class XGEP150Audit:
    positive_rows: int
    negative_rows: int
    prediction_rows: int
    mapped_rows: int
    unmapped_rows: int
    unmapped_stable_id_sha256: str
    unmapped_examples: tuple[str, ...]
    feature_content_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "view": "xgep_150",
            "source_row_counts": {
                "positive": self.positive_rows,
                "negative": self.negative_rows,
                "prediction": self.prediction_rows,
            },
            "source_rows_total": (self.positive_rows + self.negative_rows + self.prediction_rows),
            "mapped_rows": self.mapped_rows,
            "unmapped_rows": self.unmapped_rows,
            "unmapped_examples": list(self.unmapped_examples),
            "feature_count": 150,
            "feature_columns": list(XGEP_150_COLUMNS),
            "historical_label_columns_read": ["positive", "negative"],
            "historical_label_columns_discarded": True,
            "output_columns": ["gene_key", *XGEP_150_COLUMNS],
            "label_input_used": False,
            "scaled": False,
            "mapping_key": "Ensembl stable gene ID",
        }


@dataclass(frozen=True)
class XGEP150Build:
    rows: tuple[XGEP150Row, ...]
    audit: XGEP150Audit


@dataclass(frozen=True)
class _XGEPSourceRow:
    source: Literal["positive", "negative", "prediction"]
    stable_ensembl_gene_id: str
    values: tuple[float, ...]


def _parse_embedding_values(
    values: Sequence[str], *, path: Path, line_number: int
) -> tuple[float, ...]:
    if len(values) != 150:
        raise LegacyFeatureError(
            f"{path.name}:{line_number} has {len(values)} embedding values; expected 150"
        )
    try:
        parsed = tuple(float(value) for value in values)
    except ValueError as error:
        raise LegacyFeatureError(
            f"{path.name}:{line_number} contains a non-numeric embedding value"
        ) from error
    if any(not math.isfinite(value) for value in parsed):
        raise LegacyFeatureError(f"{path.name}:{line_number} contains a non-finite embedding value")
    return parsed


def _read_labeled_xgep(
    path: Path,
    *,
    source: Literal["positive", "negative"],
) -> tuple[_XGEPSourceRow, ...]:
    expected_label = "1" if source == "positive" else "0"
    rows: list[_XGEPSourceRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, fields in enumerate(csv.reader(handle), start=1):
            if len(fields) != 152:
                raise LegacyFeatureError(
                    f"{path.name}:{line_number} has {len(fields)} columns; expected 152"
                )
            if fields[0].strip() != expected_label:
                raise LegacyFeatureError(
                    f"{path.name}:{line_number} has historical label {fields[0]!r}; "
                    f"expected {expected_label}"
                )
            rows.append(
                _XGEPSourceRow(
                    source=source,
                    stable_ensembl_gene_id=_stable_ensembl_gene_id(fields[1]),
                    values=_parse_embedding_values(fields[2:], path=path, line_number=line_number),
                )
            )
    if not rows:
        raise LegacyFeatureError(f"{path.name} has no data rows")
    return tuple(rows)


def _read_prediction_xgep(path: Path) -> tuple[_XGEPSourceRow, ...]:
    rows: list[_XGEPSourceRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise LegacyFeatureError(f"{path.name} is empty") from error
        if len(header) != 151 or tuple(header[1:]) != tuple(str(index) for index in range(150)):
            raise LegacyFeatureError(
                f"{path.name} does not have the frozen identifier plus 0..149 header"
            )
        for line_number, fields in enumerate(reader, start=2):
            if len(fields) != 151:
                raise LegacyFeatureError(
                    f"{path.name}:{line_number} has {len(fields)} columns; expected 151"
                )
            rows.append(
                _XGEPSourceRow(
                    source="prediction",
                    stable_ensembl_gene_id=_stable_ensembl_gene_id(fields[0]),
                    values=_parse_embedding_values(fields[1:], path=path, line_number=line_number),
                )
            )
    if not rows:
        raise LegacyFeatureError(f"{path.name} has no data rows")
    return tuple(rows)


def _canonical_ensembl_mapping(mapping: Mapping[str, str]) -> dict[str, str]:
    canonical: dict[str, str] = {}
    for source_id, gene_key in mapping.items():
        stable_id = _stable_ensembl_gene_id(source_id)
        target = _canonical_gene_key(gene_key)
        previous = canonical.get(stable_id)
        if previous is not None and previous != target:
            raise LegacyFeatureError(
                f"Ensembl stable ID {stable_id} maps to conflicting HGNC IDs: {previous}, {target}"
            )
        canonical[stable_id] = target
    return canonical


def assert_xgep_150_contract(rows: Sequence[XGEP150Row]) -> None:

    if not rows:
        raise LegacyFeatureError("xgep_150 has no mapped rows")
    keys = [row.gene_key for row in rows]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise LegacyFeatureError("XGEP rows must have sorted unique gene_key values")
    for row in rows:
        if len(row.values) != 150 or any(not math.isfinite(value) for value in row.values):
            raise LegacyFeatureError(f"invalid XGEP feature vector for {row.gene_key}")


def build_xgep_150(
    positive_path: Path,
    negative_path: Path,
    prediction_path: Path,
    ensembl_to_gene_key: Mapping[str, str],
) -> XGEP150Build:

    source_groups = (
        _read_labeled_xgep(positive_path, source="positive"),
        _read_labeled_xgep(negative_path, source="negative"),
        _read_prediction_xgep(prediction_path),
    )
    source_rows = tuple(row for group in source_groups for row in group)
    counts = Counter(row.source for row in source_rows)
    mapping = _canonical_ensembl_mapping(ensembl_to_gene_key)

    seen_stable: dict[str, str] = {}
    seen_gene_key: dict[str, str] = {}
    rows: list[XGEP150Row] = []
    unmapped: list[str] = []
    for source_row in source_rows:
        stable_id = source_row.stable_ensembl_gene_id
        previous_source = seen_stable.get(stable_id)
        if previous_source is not None:
            raise LegacyFeatureError(
                f"duplicate XGEP Ensembl stable ID {stable_id} in "
                f"{previous_source} and {source_row.source}"
            )
        seen_stable[stable_id] = source_row.source
        gene_key = mapping.get(stable_id)
        if gene_key is None:
            unmapped.append(stable_id)
            continue
        previous_stable = seen_gene_key.get(gene_key)
        if previous_stable is not None:
            raise LegacyFeatureError(
                f"multiple XGEP Ensembl IDs map to {gene_key}: {previous_stable}, {stable_id}"
            )
        seen_gene_key[gene_key] = stable_id
        rows.append(XGEP150Row(gene_key=gene_key, values=source_row.values))

    ordered = tuple(sorted(rows, key=lambda row: row.gene_key))
    assert_xgep_150_contract(ordered)
    unmapped_sorted = tuple(sorted(unmapped))
    unmapped_hash = hashlib.sha256(
        ("\n".join(unmapped_sorted) + ("\n" if unmapped_sorted else "")).encode("utf-8")
    ).hexdigest()
    records = tuple(row.as_record() for row in ordered)
    audit = XGEP150Audit(
        positive_rows=counts["positive"],
        negative_rows=counts["negative"],
        prediction_rows=counts["prediction"],
        mapped_rows=len(ordered),
        unmapped_rows=len(unmapped_sorted),
        unmapped_stable_id_sha256=unmapped_hash,
        unmapped_examples=unmapped_sorted[:20],
        feature_content_sha256=_content_hash(records),
    )
    return XGEP150Build(rows=ordered, audit=audit)


def build_xgep_pancancer_150(
    data_directory: Path,
    ensembl_to_gene_key: Mapping[str, str],
) -> XGEP150Build:

    return build_xgep_150(
        data_directory / "Essential_genes_pancancer_emb_len150",
        data_directory / "Non_essential_genes_pancancer_emb_len150",
        data_directory / "Genes_for_prediction_pancancer_emb_len150",
        ensembl_to_gene_key,
    )


@dataclass(frozen=True)
class LegacyFeatureWriteResult:
    feature_sha256: str
    audit_sha256: str


def _write_feature_build(
    records: Sequence[Mapping[str, Any]],
    audit: Mapping[str, object],
    feature_path: Path,
    audit_path: Path,
    *,
    columns: Sequence[str],
) -> LegacyFeatureWriteResult:
    if feature_path.resolve() == audit_path.resolve():
        raise LegacyFeatureError("feature and audit paths must differ")
    write_deterministic_parquet(
        records,
        feature_path,
        sort_by=("gene_key",),
        column_order=("gene_key", *columns),
    )
    feature_sha256 = _sha256_path(feature_path)
    audit_sha256 = write_json_atomic(audit, audit_path)
    return LegacyFeatureWriteResult(feature_sha256, audit_sha256)


def write_deephe_sequence_89(
    build: DeepHESequence89Build,
    feature_path: Path,
    audit_path: Path,
) -> LegacyFeatureWriteResult:

    assert_deephe_sequence_89_contract(build.rows)
    return _write_feature_build(
        tuple(row.as_record() for row in build.rows),
        build.audit.as_dict(),
        feature_path,
        audit_path,
        columns=DEEPHE_SEQUENCE_89_COLUMNS,
    )


def write_xgep_150(
    build: XGEP150Build,
    feature_path: Path,
    audit_path: Path,
) -> LegacyFeatureWriteResult:

    assert_xgep_150_contract(build.rows)
    return _write_feature_build(
        tuple(row.as_record() for row in build.rows),
        build.audit.as_dict(),
        feature_path,
        audit_path,
        columns=XGEP_150_COLUMNS,
    )
