"""AUPRC and rank of the 15 methods on DEG under the main label and five alternative label definitions.

Run from the release root with PYTHONPATH=src:analysis:
python analysis/label_sensitivity/label_definitions.py [workers]
Writes label_sensitivity.csv and label_sensitivity_rank_agreement.csv to analysis/label_sensitivity/results/.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from egbench.contracts import METHOD_IDS, TargetKind
from egbench.data import read_cohort_labels
from egbench.fusion import pooled
from egbench.metrics.discrimination import auprc
from egbench.stats import confidence_interval
from label_sensitivity.screen_agreement import read_groups, read_membership

RESULTS = Path(__file__).resolve().parent / "results"
COHORT = "deg"
SPLIT = "family_5fold"
GENE_KEY = "gene_key"
SCORE = "y_score"
FOLD = "fold"
SOURCES = tuple(
    f"DEG{n}" for n in (2006, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)
)
DEFINITIONS = ("base", "threshold_6", "drop_hart", "drop_wang", "drop_blomen", "depmap")
POSITIVE_VOTES = 5


def definitions(genes: pd.Index) -> dict[str, pd.Series]:
    membership = read_membership()
    votes = membership[list(SOURCES)].astype("int64")
    if not (votes.sum(axis=1) == membership["deg_vote_count"].astype("int64")).all():
        raise ValueError("deg16 membership: the source columns do not sum to deg_vote_count")
    votes = votes.reindex(genes)
    if votes.isna().any().any():
        raise ValueError("a DEG gene is missing from the membership table")
    groups = read_groups()
    papers = {row.reference.split()[0].lower(): row.source_sets.split(",") for row in groups.itertuples()}

    def by_threshold(count: pd.Series, threshold: int) -> pd.Series:
        label = pd.Series(np.nan, index=genes)
        label[count >= threshold] = 1.0
        label[count == 0] = 0.0
        return label

    total = votes.sum(axis=1)
    out = {"base": by_threshold(total, POSITIVE_VOTES), "threshold_6": by_threshold(total, 6)}
    for paper in ("hart", "wang", "blomen"):
        kept = [source for source in SOURCES if source not in papers[paper]]
        out["drop_" + paper] = by_threshold(votes[kept].sum(axis=1), POSITIVE_VOTES)
    depmap = read_cohort_labels("depmap").set_index(GENE_KEY)["label"].astype("float64")
    out["depmap"] = depmap.reindex(genes)
    return {name: out[name] for name in DEFINITIONS}


def run(predictor: str) -> pd.DataFrame:
    frame = pooled(predictor, COHORT, split_strategy=SPLIT).set_index(GENE_KEY)
    rows = []
    for name, label in definitions(frame.index).items():
        kept = label.notna().to_numpy()
        y = label.to_numpy("float64")[kept]
        interval = confidence_interval(
            y,
            frame[SCORE].to_numpy("float64")[kept],
            auprc,
            fold=frame[FOLD].to_numpy()[kept],
            target_kind=TargetKind.BINARY,
        )
        rows.append(
            {
                "split_strategy": SPLIT,
                "predictor": predictor,
                "definition": name,
                "genes": int(kept.sum()),
                "positives": int(y.sum()),
                "positive_rate": float(y.mean()),
                "auprc": float(interval.point),
                "auprc_half_width": float(interval.half_width),
            }
        )
    return pd.DataFrame(rows)


def rank_agreement(table: pd.DataFrame) -> pd.DataFrame:
    base = table[table.definition == "base"].set_index("predictor")["rank"]
    rows = []
    for name in DEFINITIONS:
        other = table[table.definition == name].set_index("predictor")["rank"].reindex(base.index)
        rho, p = stats.spearmanr(base.to_numpy(), other.to_numpy())
        rows.append({"split_strategy": SPLIT, "definition": name, "spearman": rho, "p": p})
    return pd.DataFrame(rows)


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    started = time.monotonic()
    tables = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for table in pool.map(run, METHOD_IDS):
            tables.append(table)
            print(f"{len(tables)}/{len(METHOD_IDS)} methods done, {time.monotonic() - started:.0f} s", flush=True)
    table = pd.concat(tables, ignore_index=True)
    table["rank"] = table.groupby(["split_strategy", "definition"])["auprc"].rank(ascending=False, method="min")
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "label_sensitivity.csv", index=False)
    rank_agreement(table).to_csv(RESULTS / "label_sensitivity_rank_agreement.csv", index=False)


if __name__ == "__main__":
    main()
