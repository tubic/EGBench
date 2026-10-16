"""Pairwise Jaccard index between the essential gene sets of the three 2015 genome-wide screens in DEG.

Run from the release root: python analysis/label_sensitivity/screen_agreement.py
Writes analysis/label_sensitivity/results/screen_agreement.csv.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "DEG" / "data"
COMMON = ROOT / "datasets" / "DEG" / "folds" / "common_binary.parquet"
RESULTS = Path(__file__).resolve().parent / "results"

SCREEN_PUBMED_IDS = ("26472760", "26472758", "26627737")
SURNAME_PATTERN = re.compile(r"[A-Za-zÀ-ɏ]+")
YEAR_PATTERN = re.compile(r"\((\d{4})\)")


def short_reference(citation: str) -> str:
    surname = SURNAME_PATTERN.search(citation.lstrip('" '))
    year = YEAR_PATTERN.search(citation)
    if surname is None or year is None:
        raise ValueError(f"cannot derive a short reference from {citation!r}")
    return f"{surname.group(0)} {year.group(1)}"


def publication_groups(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    frame["pubmed_id"] = frame["pubmed_id"].astype(str).str.strip()
    frame["reference"] = frame["reference"].map(short_reference)
    grouped = (
        frame.groupby("pubmed_id")
        .agg(
            reference=("reference", "first"),
            method=("method", lambda values: "/".join(sorted(set(values)))),
            source_set_count=("dataset_id", "size"),
            source_sets=("dataset_id", lambda values: ",".join(sorted(values))),
        )
        .reset_index()
    )
    return grouped.sort_values(["source_set_count", "pubmed_id"], ascending=[False, True]).reset_index(drop=True)


def read_groups() -> pd.DataFrame:
    return publication_groups(pd.read_csv(DATA / "deg16_source_datasets.tsv", sep="\t"))


def read_membership() -> pd.DataFrame:
    return pd.read_parquet(DATA / "deg16_membership_hgnc.parquet").set_index("gene_key")


def indicator_matrix(membership: pd.DataFrame, groups: pd.DataFrame, gene_keys: Sequence[str]) -> pd.DataFrame:
    columns = sorted(name for value in groups["source_sets"] for name in str(value).split(","))
    return membership.reindex(gene_keys)[columns].astype(int)


def essential_sets(
    membership: pd.DataFrame, groups: pd.DataFrame, gene_keys: Sequence[str], criterion: str
) -> dict[str, set[str]]:
    indicator = indicator_matrix(membership, groups, gene_keys)
    result: dict[str, set[str]] = {}
    for record in groups.to_dict("records"):
        sets = str(record["source_sets"]).split(",")
        threshold = 1.0 if criterion == "source_set_union" else len(sets) / 2
        support = indicator[sets].sum(axis=1)
        result[str(record["pubmed_id"])] = set(indicator.index[support >= threshold])
    return result


def agreement_rows(
    membership: pd.DataFrame, groups: pd.DataFrame, gene_sets: Mapping[str, Sequence[str]]
) -> pd.DataFrame:
    reference = dict(zip(groups["pubmed_id"].astype(str), groups["reference"]))
    rows: list[dict[str, object]] = []
    for criterion in ("own_paper_majority", "source_set_union"):
        for gene_set_name, gene_keys in gene_sets.items():
            called = essential_sets(membership, groups, gene_keys, criterion)
            for index, first in enumerate(SCREEN_PUBMED_IDS):
                for second in SCREEN_PUBMED_IDS[index + 1 :]:
                    left, right = called[first], called[second]
                    union = left | right
                    rows.append(
                        {
                            "criterion": criterion,
                            "gene_set": gene_set_name,
                            "gene_count": int(len(gene_keys)),
                            "study_a": reference[first],
                            "study_b": reference[second],
                            "essential_a": int(len(left)),
                            "essential_b": int(len(right)),
                            "intersection": int(len(left & right)),
                            "union": int(len(union)),
                            "jaccard": round(len(left & right) / len(union), 4),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    membership = read_membership()
    common = pd.read_parquet(COMMON, columns=["gene_key"])
    gene_sets = {
        "benchmark_common": list(common["gene_key"]),
        "candidate_universe": list(membership.index),
    }
    table = agreement_rows(membership, read_groups(), gene_sets)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "screen_agreement.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
