"""L2 logistic regression on seven frozen pretrained embeddings with C chosen on validation AUPRC, compared with the methods built on them.

Run from the release root after esm2_pool.py, esm2_extract.py and geneformer_embeddings.py:
python analysis/foundation_model_probes/embedding_probes.py [--workers N]
"""

from __future__ import annotations

import argparse
import time
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from egbench.contracts import SEED, TargetKind
from egbench.data import TRAIN_ROLE, VALIDATION_ROLE, prediction_cohort, read_training_labels
from egbench.metrics.discrimination import auprc, auroc

from common.stats import (
    ALL_COHORTS, ANALYSIS_DIRECTORY, FOLD, FOLDS, GENE_KEY, METHODS_DIRECTORY, SCORE, SPLIT, holm, interval, kind_of,
    paired, pooled, primary_metric, process_pool, target_of, test_frame, write_table,
)

PROBES = ANALYSIS_DIRECTORY / "foundation_model_probes"
WORK = PROBES / "work"
RESULTS = PROBES / "results"
GENE_LEVEL_GENES = PROBES / "data" / "gene_level_genes.csv"
SEQUENCE: tuple[str, ...] = ("esm2_150m", "esm2_650m", "esm2_3b", "esmc_600m", "proteomelm_l", "evo2_7b")
EMBEDDINGS: tuple[str, ...] = SEQUENCE + ("geneformer",)
COMPARATORS: tuple[str, ...] = ("bingo", "proteomelm", "esstfnet")
SCALE_PAIRS = (("esm2_650m", "esm2_150m"), ("esm2_3b", "esm2_650m"), ("esm2_3b", "esm2_150m"))
GRID: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 1e-1, 1.0, 10.0)


def load_embedding(name: str, cohort: str) -> pd.DataFrame:
    source = prediction_cohort(cohort)
    if name.startswith("esm2_"):
        frame = pd.read_parquet(WORK / f"{name}_mean.parquet")
        return frame.drop(columns=[c for c in ("cached_length", "truncated") if c in frame.columns]
                          ).set_index(GENE_KEY).astype("float32")
    if name == "esmc_600m":
        return (pd.read_parquet(METHODS_DIRECTORY / "proteomelm" / "feature" / "source" / "esmc_base_embeddings.parquet")
                .drop(columns=["sequence_sha256"]).set_index(GENE_KEY).astype("float32"))
    if name == "proteomelm_l":
        return (pd.read_parquet(METHODS_DIRECTORY / "proteomelm" / "feature" / "source" / f"{source}_features.parquet")
                .set_index(GENE_KEY).astype("float32"))
    if name == "evo2_7b":
        frame = pd.read_parquet(METHODS_DIRECTORY / "esstfnet" / "feature" / "source" / source
                                / "evo2_7b_base_embeddings.parquet")
        return pd.DataFrame(np.stack(frame["embedding"].to_numpy()).astype("float32"),
                            index=pd.Index(frame[GENE_KEY].astype("object")))
    if name == "geneformer":
        return pd.read_parquet(WORK / "geneformer_gene_key.parquet").set_index(GENE_KEY).astype("float32")
    raise ValueError(name)


def fit_probe(job: tuple[str, str]) -> dict:
    cohort, name = job
    started = time.time()
    source = prediction_cohort(cohort)
    target = target_of(cohort)
    matrix = load_embedding(name, cohort)
    scope = None
    if name == "geneformer":
        scope = pd.Index(pd.read_csv(GENE_LEVEL_GENES)[GENE_KEY].astype("object"))
    frames = []
    for fold in FOLDS:
        train = read_training_labels(source, fold, TRAIN_ROLE, split_strategy=SPLIT)
        train_keys = pd.Index(train[GENE_KEY].astype("object"))
        kept_train = np.ones(train_keys.size, dtype=bool) if scope is None else np.asarray(train_keys.isin(scope))
        validation = read_training_labels(source, fold, VALIDATION_ROLE, split_strategy=SPLIT)
        validation_keys = pd.Index(validation[GENE_KEY].astype("object"))
        kept_validation = (np.ones(validation_keys.size, dtype=bool) if scope is None
                           else np.asarray(validation_keys.isin(scope)))
        test = test_frame("mahi", cohort, fold)[[GENE_KEY, target]]
        test_keys = pd.Index(test[GENE_KEY].astype("object"))
        kept_test = np.ones(test_keys.size, dtype=bool) if scope is None else np.asarray(test_keys.isin(scope))
        x = matrix.reindex(train_keys[kept_train]).to_numpy("float64")
        if not np.isfinite(x).all():
            raise ValueError(f"{name} {cohort} fold {fold}: a train gene has no embedding")
        center, spread = x.mean(axis=0), x.std(axis=0)
        spread = np.where(spread > 0, spread, 1.0)
        x = (x - center) / spread
        labels = train["label"].to_numpy("float64")[kept_train]
        z_validation = (matrix.reindex(validation_keys[kept_validation]).to_numpy("float64") - center) / spread
        y_validation = validation["label"].to_numpy("float64")[kept_validation]
        scored = test.loc[kept_test].reset_index(drop=True)
        z_test = (matrix.reindex(pd.Index(scored[GENE_KEY].astype("object"))).to_numpy("float64") - center) / spread
        block = {}
        for value in GRID:
            model = LogisticRegression(C=value, max_iter=2000, random_state=SEED)
            model.fit(x, labels)
            block[value] = (float(auprc(y_validation, model.predict_proba(z_validation)[:, 1])),
                            model.predict_proba(z_test)[:, 1])
        best = max(GRID, key=lambda value: block[value][0])
        scored[SCORE] = block[best][1]
        scored[FOLD] = fold
        frames.append(scored)
    scores = pd.concat(frames, ignore_index=True)
    scores.to_csv(WORK / f"probe_scores_{cohort}_{name}.csv", index=False)
    print(f"[probe {cohort} {name}] {len(scores):,} genes, {time.time() - started:.0f} s", flush=True)
    return {"cohort_id": cohort, "embedding": name}


def evaluate(cohort: str) -> dict:
    metric_name, metric = primary_metric(cohort)
    kind = kind_of(cohort)
    target = target_of(cohort)
    tables = {name: pd.read_csv(WORK / f"probe_scores_{cohort}_{name}.csv") for name in EMBEDDINGS}
    rows = []
    for name in EMBEDDINGS:
        frame = tables[name]
        y = frame[target].to_numpy("float64")
        fold = frame[FOLD].to_numpy()
        keys = pd.Index(frame[GENE_KEY].astype("object"))
        s = frame[SCORE].to_numpy("float64")
        row = {"cohort_id": cohort, "embedding": name, "metric": metric_name, "genes": int(len(frame))}
        row["point"], row["half_width"] = interval(y, s, metric, fold, kind)
        if kind is TargetKind.BINARY:
            row["auroc"], row["auroc_half_width"] = interval(y, s, auroc, fold, kind)
        if name in SEQUENCE:
            for comparator in COMPARATORS:
                other = pooled(comparator, cohort).set_index(GENE_KEY)[SCORE].reindex(keys).to_numpy("float64")
                if np.isnan(other).any():
                    raise ValueError(f"{comparator} does not score every gene of {cohort}")
                row[f"{comparator}_point"], _ = interval(y, other, metric, fold, kind)
                (row[f"minus_{comparator}"], row[f"minus_{comparator}_half_width"],
                 row[f"minus_{comparator}_p"]) = paired(y, s, other, metric, fold, kind)
        rows.append(row)
    scale = []
    base = tables["esm2_150m"]
    y = base[target].to_numpy("float64")
    fold = base[FOLD].to_numpy()
    keys = pd.Index(base[GENE_KEY].astype("object"))
    for larger, smaller in SCALE_PAIRS:
        a = tables[larger].set_index(GENE_KEY)[SCORE].reindex(keys).to_numpy("float64")
        b = tables[smaller].set_index(GENE_KEY)[SCORE].reindex(keys).to_numpy("float64")
        d, half, p = paired(y, a, b, metric, fold, kind)
        scale.append({"cohort_id": cohort, "metric": metric_name, "larger": larger, "smaller": smaller,
                      "difference": d, "half_width": half, "p": p})
    print(f"[evaluate {cohort}] done", flush=True)
    return {"table": pd.DataFrame(rows), "scale": pd.DataFrame(scale)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    jobs = list(product(ALL_COHORTS, EMBEDDINGS))
    with process_pool(arguments.workers) as pool:
        list(pool.map(fit_probe, jobs))
    with process_pool(min(arguments.workers, len(ALL_COHORTS))) as pool:
        results = list(pool.map(evaluate, ALL_COHORTS))
    table = pd.concat([r["table"] for r in results], ignore_index=True)
    sequence = table[table.embedding.isin(SEQUENCE)]
    for comparator in COMPARATORS:
        table[f"holm_{comparator}"] = np.nan
        table.loc[sequence.index, f"holm_{comparator}"] = holm(sequence, ["cohort_id"], f"minus_{comparator}_p")
    scale = pd.concat([r["scale"] for r in results], ignore_index=True)
    scale["holm_p"] = holm(scale, ["cohort_id"])
    write_table(table, RESULTS / "probe_selected_c.csv")
    write_table(scale, RESULTS / "probe_selected_c_differences.csv")
    print("wrote probe_selected_c.csv and probe_selected_c_differences.csv", flush=True)


if __name__ == "__main__":
    main()
