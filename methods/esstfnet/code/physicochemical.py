"""Physicochemical DNA descriptors of the EssTFNet feature set.
Imported by the EssTFNet feature builders.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np


BASE_ALPHABET = "AGCT"
LZ_ALPHABET = "ATGC"
PSEKNC_ALPHABET = "ACGT"
BASE_FEATURE_NAMES = (
    "GC Content",
    "zCurveX",
    "zCurveY",
    "zCurveZ",
    "GC Skew",
    "AT Skew",
    "ATGC Ratio",
)
RAW_CANDIDATE_DIMENSION = 86151
RAW_CANDIDATE_FAMILY_COUNTS = {
    "base": 7,
    "kmer": 84,
    "PseKNC": 65,
    "lambda-Z": 85995,
}


@dataclass(frozen=True)
class RawCandidateColumn:

    feature_index: int
    feature_name: str
    feature_family: str
    w: int | None
    k: int | None


def _require_dna(sequence: str) -> str:
    sequence = str(sequence).upper()
    if not sequence or set(sequence) - set("ACGT"):
        raise ValueError("EssTFNet DNA features require a non-empty ACGT sequence")
    return sequence


def _counts(sequence: str) -> Counter[str]:
    return Counter(sequence)


def _base_values_from_counts(
    counts: Counter[str], length: int, required: Iterable[str]
) -> dict[str, float]:
    a, c, g, t = (counts[base] for base in "ACGT")
    required = set(required)
    values: dict[str, float] = {}
    if "GC Content" in required:
        values["GC Content"] = (g + c) / length
    if "zCurveX" in required:
        values["zCurveX"] = (a + g - c - t) / length
    if "zCurveY" in required:
        values["zCurveY"] = (a + c - g - t) / length
    if "zCurveZ" in required:
        values["zCurveZ"] = (a + t - c - g) / length
    if "GC Skew" in required:
        denominator = g + c
        if denominator == 0:
            raise ValueError("official GC Skew formula is undefined because G+C is zero")
        values["GC Skew"] = (g - c) / denominator
    if "AT Skew" in required:
        denominator = a + t
        if denominator == 0:
            raise ValueError("official AT Skew formula is undefined because A+T is zero")
        values["AT Skew"] = (a - t) / denominator
    if "ATGC Ratio" in required:
        denominator = g + c
        if denominator == 0:
            raise ValueError("official ATGC Ratio formula is undefined because G+C is zero")
        values["ATGC Ratio"] = (a + t) / denominator
    return values


def _kmer_names(k: int) -> list[str]:
    if k == 0:
        return [""]
    if k == 1:
        return list(BASE_ALPHABET)
    previous = _kmer_names(k - 1)
    return [prefix + base for prefix in previous for base in BASE_ALPHABET]


def _pseknc_kmer_names(k: int) -> list[str]:
    if k == 0:
        return [""]
    if k == 1:
        return list(PSEKNC_ALPHABET)
    previous = _pseknc_kmer_names(k - 1)
    return [prefix + base for prefix in previous for base in PSEKNC_ALPHABET]


def _frequency(sequence: str, motif: str) -> int:

    count = 0
    start = 0
    while True:
        index = sequence.find(motif, start)
        if index < 0:
            return count
        count += 1
        start = index + 1


def _sample_normalized(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    standard_deviation = math.sqrt(
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )
    return [round((value - mean) / standard_deviation, 2) for value in values]


_PSEKNC_EXTRA_VALUES = _sample_normalized(
    [
        1.019,
        -0.918,
        0.488,
        0.567,
        0.567,
        -0.070,
        -0.579,
        0.488,
        -0.654,
        -2.455,
        -0.070,
        -0.918,
        1.603,
        -0.654,
        0.567,
        1.019,
    ]
)
_PSEKNC_PHYCHE = {
    "AA": [0.06, 0.5, 0.09, 1.59, 0.11, -0.11],
    "AC": [1.5, 0.5, 1.19, 0.13, 1.29, 1.04],
    "AG": [0.78, 0.36, -0.28, 0.68, -0.24, -0.62],
    "AT": [1.07, 0.22, 0.83, -1.02, 2.51, 1.17],
    "CA": [-1.38, -1.36, -1.01, -0.86, -0.62, -1.25],
    "CC": [0.06, 1.08, -0.28, 0.56, -0.82, 0.24],
    "CG": [-1.66, -1.22, -1.38, -0.82, -0.29, -1.39],
    "CT": [0.78, 0.36, -0.28, 0.68, -0.24, -0.62],
    "GA": [-0.08, 0.5, 0.09, 0.13, -0.39, 0.71],
    "GC": [-0.08, 0.22, 2.3, -0.35, 0.65, 1.59],
    "GG": [0.06, 1.08, -0.28, 0.56, -0.82, 0.24],
    "GT": [1.5, 0.5, 1.19, 0.13, 1.29, 1.04],
    "TA": [-1.23, -2.37, -1.38, -2.24, -1.51, -1.39],
    "TC": [-0.08, 0.5, 0.09, 0.13, -0.39, 0.71],
    "TG": [-1.38, -1.36, -1.01, -0.86, -0.62, -1.25],
    "TT": [0.06, 0.5, 0.09, 1.59, 0.11, -0.11],
}
for _index, _key in enumerate((a + b for a in "ACGT" for b in "ACGT")):
    _PSEKNC_PHYCHE[_key] = [*_PSEKNC_PHYCHE[_key], _PSEKNC_EXTRA_VALUES[_index]]


def pseknc_feature_values(sequence: str) -> dict[str, float]:

    sequence = _require_dna(sequence)
    if len(sequence) < 4:
        raise ValueError("official PseKNC formula requires sequence length at least 4")
    frequencies = [_frequency(sequence, motif) for motif in _pseknc_kmer_names(3)]
    frequency_sum = float(sum(frequencies))
    normalized = [value / frequency_sum for value in frequencies]

    theta_sum = 0.0
    for index in range(0, len(sequence) - 2):
        first_values = _PSEKNC_PHYCHE[sequence[index : index + 2]]
        second_values = _PSEKNC_PHYCHE[sequence[index + 1 : index + 3]]
        theta_sum += sum(
            (left - right) ** 2 for left, right in zip(first_values, second_values)
        ) / len(first_values)
    theta = theta_sum / (len(sequence) - 2)
    denominator = 1.0 + 0.5 * theta
    values = [frequency / denominator for frequency in normalized]
    values.append(0.5 * theta / denominator)
    return {f"pseknc:{index}": value for index, value in enumerate(values, start=1)}


def _kmer_names_for_lz(k: int) -> list[str]:
    if k == 0:
        return [""]
    return [prefix + base for prefix in _kmer_names_for_lz(k - 1) for base in LZ_ALPHABET]


def _raw_lambda_names(w: int, k: int) -> list[str]:
    names: list[str] = []
    motifs = _kmer_names_for_lz(w - 1)
    for phase in ("first", "second", "third"):
        for axis in "XYZ":
            names.extend(
                f"Forward w:{w}_k:{k}_{axis}_{motif}_{phase}" for motif in motifs
            )
    return names


def generate_raw_candidate_columns() -> tuple[RawCandidateColumn, ...]:

    columns: list[RawCandidateColumn] = []

    def add(name: str, family: str, w: int | None, k: int | None) -> None:
        columns.append(RawCandidateColumn(len(columns), name, family, w, k))

    for name in BASE_FEATURE_NAMES:
        add(name, "base", None, None)
    for k in range(1, 4):
        for motif in _kmer_names(k):
            add(f"kmer:{motif}", "kmer", None, k)
    for index in range(1, 66):
        add(f"pseknc:{index}", "PseKNC", None, 3)
    for w in range(1, 7):
        for k in range(0, 7):
            for name in _raw_lambda_names(w, k):
                add(name, "lambda-Z", w, k)

    if len(columns) != RAW_CANDIDATE_DIMENSION:
        raise AssertionError(f"raw candidate dimension is {len(columns)}, expected {RAW_CANDIDATE_DIMENSION}")
    family_counts = Counter(column.feature_family for column in columns)
    if family_counts != Counter(RAW_CANDIDATE_FAMILY_COUNTS):
        raise AssertionError(
            f"raw candidate family counts are {dict(family_counts)}, "
            f"expected {RAW_CANDIDATE_FAMILY_COUNTS}"
        )
    names = [column.feature_name for column in columns]
    if len(set(names)) != RAW_CANDIDATE_DIMENSION:
        raise AssertionError("raw candidate feature names are not unique")
    if [column.feature_index for column in columns] != list(range(RAW_CANDIDATE_DIMENSION)):
        raise AssertionError("raw candidate feature indices are not contiguous")
    return tuple(columns)


def raw_candidate_columns() -> tuple[RawCandidateColumn, ...]:

    return generate_raw_candidate_columns()


def _subsequence_counters(sequence: str, w: int, k: int) -> tuple[Counter[str], Counter[str], Counter[str]]:

    if w == 1:
        return tuple(Counter(sequence[phase::3]) for phase in range(3))

    counters: list[Counter[str]] = []
    for phase in range(3):
        counter: Counter[str] = Counter()
        for index in range(phase, len(sequence), 3):
            target = w - 2 + index + k
            if target < len(sequence):
                counter[sequence[index : index + w - 1] + sequence[target]] += 1
        counters.append(counter)
    return tuple(counters)


def _lambda_value(counter: Counter[str], length: int, axis: int, motif: str) -> float:
    if axis == 0:
        positive_bases, negative_bases = "AG", "CT"
    elif axis == 1:
        positive_bases, negative_bases = "AC", "GT"
    else:
        positive_bases, negative_bases = "AT", "GC"
    positive = sum(counter[motif + base] for base in positive_bases)
    negative = sum(counter[motif + base] for base in negative_bases)
    return (positive - negative) / length


def feature_vector(sequence: str) -> np.ndarray:

    sequence = _require_dna(sequence)
    values = np.empty(RAW_CANDIDATE_DIMENSION, dtype=np.float32)
    length = len(sequence)
    counts = _counts(sequence)
    base_values = _base_values_from_counts(counts, length, BASE_FEATURE_NAMES)
    for index, name in enumerate(BASE_FEATURE_NAMES):
        values[index] = base_values[name]

    offset = len(BASE_FEATURE_NAMES)
    for k in range(1, 4):
        total = length - k + 1
        if total <= 0:
            raise ValueError(f"official kmer formula is undefined for k={k} and sequence length {length}")
        kmer_counts = Counter(sequence[index : index + k] for index in range(total))
        for motif in _kmer_names(k):
            values[offset] = kmer_counts[motif] / total
            offset += 1

    if length < 4:
        raise ValueError("official PseKNC formula requires sequence length at least 4")
    pseknc_values = pseknc_feature_values(sequence)
    for index in range(1, 66):
        values[offset] = pseknc_values[f"pseknc:{index}"]
        offset += 1

    for w in range(1, 7):
        motifs = _kmer_names_for_lz(w - 1)
        for k in range(0, 7):
            phase_counters = _subsequence_counters(sequence, w, k)
            phase_lengths = tuple(sum(counter.values()) for counter in phase_counters)
            for phase, (counter, phase_length) in enumerate(zip(phase_counters, phase_lengths)):
                if phase_length == 0:
                    raise ValueError(
                        f"official lambda-Z-curve formula is undefined for "
                        f"(w={w}, k={k}) phase {phase + 1}"
                    )
                for axis in range(3):
                    for motif in motifs:
                        values[offset] = _lambda_value(counter, phase_length, axis, motif)
                        offset += 1

    if offset != RAW_CANDIDATE_DIMENSION:
        raise AssertionError(f"raw candidate vector filled {offset} values")
    return values


