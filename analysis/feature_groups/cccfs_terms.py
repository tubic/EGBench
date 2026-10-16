"""Name each HELP CCcfs column by its GO cellular component term and assign it to the complex, compartment or other tier.

Run from the release root: python analysis/feature_groups/cccfs_terms.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import re

import pandas as pd

from common.feature_blocks import ANNOTATIONS, CCCFS_TERMS, GO_OBO, GO_OBO_SHA256

CCCFS_CSV = ANNOTATIONS / "Human_CCcfs.csv"
CCCFS_CSV_SHA256 = "aa385ed4380b4d43da601cd73f81fb75d509f51dfc14623e20e56335827356f9"
COMPARTMENT_TABLE = ANNOTATIONS / "human_compartment_integrated_full.tsv"
COMPLEX_ROOT = "GO:0032991"
ANATOMICAL_ROOT = "GO:0110165"
GO_IDENTIFIER = re.compile(r"GO:\d{7}")
COMPLEX_TIER = "complex"
COMPARTMENT_TIER = "compartment"
OTHER_TIER = "other"


def verify_downloads() -> None:
    for path, expected in ((CCCFS_CSV, CCCFS_CSV_SHA256), (GO_OBO, GO_OBO_SHA256)):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"{path} has sha256 {digest.hexdigest()}, expected {expected}")


def cccfs_column_names() -> pd.DataFrame:
    with CCCFS_CSV.open(encoding="utf-8", errors="replace") as handle:
        header = next(csv.reader(io.StringIO(handle.readline())))
    frame = pd.read_csv(CCCFS_CSV, index_col=0, engine="c", dtype={name: "float32" for name in header[1:]})
    if list(frame.columns) != header[1:]:
        raise ValueError("the header read by hand differs from the one pandas read")
    names = list(frame.columns)
    imputed = frame.fillna(frame.mean(numeric_only=True))
    kept = [name for name, unique in zip(names, imputed.nunique().to_numpy()) if unique > 1]
    return pd.DataFrame({"column": [f"cc__{i}" for i in range(len(kept))], "term_name": kept})


def _obo_records() -> list[dict[str, list[str]]]:
    records: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] | None = None
    for line in GO_OBO.read_text(encoding="utf-8").splitlines():
        if line == "[Term]":
            current = {}
            records.append(current)
        elif line.startswith("["):
            current = None
        elif current is not None and ": " in line:
            tag, _, value = line.partition(": ")
            current.setdefault(tag, []).append(value.strip())
    return records


def term_identifiers() -> list[dict[str, set[str]]]:
    levels: list[dict[str, set[str]]] = [{}, {}, {}, {}]

    def offer(level: int, name: str, identifier: str) -> None:
        key = name.strip().lower()
        if key:
            levels[level].setdefault(key, set()).add(identifier)

    for record in _obo_records():
        identifiers = record.get("id", [])
        if not identifiers or not identifiers[0].startswith("GO:"):
            continue
        if record.get("namespace", [""])[0] != "cellular_component":
            continue
        identifier = identifiers[0]
        retired = record.get("is_obsolete", ["false"])[0] == "true"
        for name in record.get("name", []):
            if retired and name.lower().startswith("obsolete "):
                name = name[len("obsolete "):]
            offer(2 if retired else 0, name, identifier)
        for synonym in record.get("synonym", []):
            if "EXACT" not in synonym or not synonym.startswith('"'):
                continue
            offer(2 if retired else 1, synonym[1:].split('"', 1)[0], identifier)
    table = pd.read_csv(
        COMPARTMENT_TABLE, sep="\t", header=None,
        names=["identifier", "symbol", "go_id", "go_name", "score"],
        usecols=["go_id", "go_name"], dtype="string",
    ).drop_duplicates()
    for go_id, go_name in zip(table["go_id"], table["go_name"]):
        offer(3, str(go_name), str(go_id))
    return levels


def resolve(name: str, levels: list[dict[str, set[str]]]) -> str:
    name = name.strip()
    if GO_IDENTIFIER.fullmatch(name):
        return name
    key = name.lower()
    for level in levels:
        if key in level:
            found = level[key]
            return next(iter(found)) if len(found) == 1 else ""
    return ""


def canonical_identifiers() -> dict[str, str]:
    direct: dict[str, str] = {}
    for record in _obo_records():
        identifiers = record.get("id", [])
        if not identifiers or not identifiers[0].startswith("GO:"):
            continue
        identifier = identifiers[0]
        for alternative in record.get("alt_id", []):
            if alternative.startswith("GO:"):
                direct[alternative] = identifier
        if record.get("is_obsolete", ["false"])[0] == "true":
            for replacement in record.get("replaced_by", []):
                if replacement.startswith("GO:"):
                    direct[identifier] = replacement
    resolved: dict[str, str] = {}
    for start in direct:
        seen = {start}
        current = start
        while current in direct and direct[current] not in seen:
            current = direct[current]
            seen.add(current)
        resolved[start] = current
    return resolved


def descendants(root: str) -> set[str]:
    children: dict[str, set[str]] = {}
    current: str | None = None
    for line in GO_OBO.read_text(encoding="utf-8").splitlines():
        if line == "[Term]":
            current = None
            continue
        tag, _, rest = line.partition(": ")
        if tag == "id" and rest.startswith("GO:"):
            current = rest.strip()
        elif current is None:
            continue
        elif tag == "is_a" or (tag == "relationship" and rest.startswith("part_of ")):
            tokens = [t for t in rest.replace("!", " ! ").split() if t.startswith("GO:")]
            if tokens:
                children.setdefault(tokens[0], set()).add(current)
    seen = {root}
    stack = [root]
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def tier_table() -> pd.DataFrame:
    frame = cccfs_column_names()
    levels = term_identifiers()
    complexes = descendants(COMPLEX_ROOT)
    anatomical = descendants(ANATOMICAL_ROOT)
    canonical = canonical_identifiers()
    frame["go_id"] = [resolve(name, levels) for name in frame["term_name"]]
    frame["current_go_id"] = [canonical.get(go_id, go_id) for go_id in frame["go_id"]]
    tiers = []
    for go_id in frame["current_go_id"]:
        if go_id in complexes:
            tiers.append(COMPLEX_TIER)
        elif go_id in anatomical:
            tiers.append(COMPARTMENT_TIER)
        else:
            tiers.append(OTHER_TIER)
    frame["tier"] = tiers
    return frame


def main() -> None:
    verify_downloads()
    table = tier_table()
    CCCFS_TERMS.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CCCFS_TERMS, index=False)
    print(f"wrote {CCCFS_TERMS}: {len(table)} columns, tiers {table['tier'].value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
