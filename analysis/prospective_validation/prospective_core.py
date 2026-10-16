"""Prospective validation on DepMap 23Q4: the core feature set model, the 15 methods and the hallmark count on the
genes 26Q1 newly calls common essential, overall and in twelve gene groups, and the hallmarks of those genes.
Run: python analysis/prospective_validation/prospective_core.py [--workers 12]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from egbench.baselines import literature_counts
from egbench.data import (
    TRAIN_ROLE, feature_directory, read_cohort_labels, read_evaluation_truth, read_label_confidence,
    read_training_labels,
)
from egbench.metrics.discrimination import auprc, average_ranks, hits_at_k
from egbench.modules import COMPLEX, PROCESS, TERM_ID, cohort_modules
from egbench.prospective import (
    BACKGROUND, CLASS, FOREGROUND, PROSPECTIVE_COHORT, prospective_genes, scored_frames, source_predictions,
)
from egbench.stats import RESAMPLE_COUNT
from egbench.strata import QUARTILE_BANDS, quartile_bands

from common.feature_blocks import induced_degree, make_learner, read_indexed, run_pool
from common.hallmark_counts import RICH, YEAST, bootstrap, cells_of, hallmark_frame
from common.hallmarks import expression_median
from common.stats import FOLDS, METHODS, METHODS_DIRECTORY, SPLIT, Resampler, holm, summarise

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
WORK = HERE / "work"
CORE = "core_feature_set"
COUNT = "hallmark_count"
PREDICTORS: tuple[str, ...] = METHODS + (CORE, COUNT)
ALL = "all"
BANDINGS: dict[str, tuple[str, ...]] = {
    ALL: (ALL,),
    "literature_quartile": QUARTILE_BANDS,
    "complex_partner": ("complex_partner_yes", "complex_partner_no"),
    "process_partner": ("process_partner_yes", "process_partner_no"),
    "any_partner": ("any_partner_yes", "any_partner_no"),
    "annotated": ("annotated_yes", "annotated_no"),
}
ENTRIES: tuple[tuple[str, str], ...] = tuple((banding, band) for banding, bands in BANDINGS.items() for band in bands)
METRICS: tuple[str, ...] = ("auroc", "auprc", "hits_at_100")
SHORTLIST = 20
CHUNK = 100


def assembled_paths() -> tuple[Path, Path]:
    return WORK / f"assembled_{SPLIT}.csv", WORK / f"assembled_{SPLIT}.npy"


def core_design(fold: int) -> pd.DataFrame:
    cohort = PROSPECTIVE_COHORT
    table = read_indexed(feature_directory(METHODS_DIRECTORY / "help", cohort, split_strategy=SPLIT, outer_fold=fold)
                         / "features.parquet")
    confidence = [c for c in table.columns if c.startswith("cc__")]
    annotation = [f"bio__{i}" for i in range(18)]
    components = read_indexed(feature_directory(METHODS_DIRECTORY / "esssubgraph", cohort, split_strategy=SPLIT,
                                                outer_fold=fold) / "features.parquet")
    static = read_indexed(METHODS_DIRECTORY / "epgat" / "feature" / "source" / "generated"
                          / f"{cohort}_static_features.parquet")
    orthology = [c for c in static.columns if c.startswith("ortholog__")] + ["orthology_missing"]
    if len(confidence) != 3301 or components.shape[1] != 50 or len(orthology) != 195:
        raise ValueError(f"fold {fold}: unexpected feature group widths")
    genes = pd.Index(sorted(table.index), dtype="object")
    return pd.concat([table.loc[genes, confidence], components.loc[genes], table.loc[genes, annotation],
                      static.loc[genes, orthology]], axis=1).astype("float64")


def fit_job(fold: int) -> pd.DataFrame:
    design = core_design(fold)
    train = read_training_labels(PROSPECTIVE_COHORT, fold, TRAIN_ROLE, split_strategy=SPLIT)
    test = read_evaluation_truth(PROSPECTIVE_COHORT, fold, split_strategy=SPLIT)
    model = make_learner()
    model.fit(design.loc[train["gene_key"].astype(str)].to_numpy(), train["label"].to_numpy("float64"))
    keys = test["gene_key"].astype(str).to_numpy("object")
    scores = model.predict_proba(design.loc[keys].to_numpy())[:, 1]
    return pd.DataFrame({"gene_key": keys, "fold": fold, "y_score": scores})


def cohort_hallmarks() -> pd.DataFrame:
    cohort = PROSPECTIVE_COHORT
    corum = read_indexed(feature_directory(METHODS_DIRECTORY / "deepsnap", cohort, split_strategy=SPLIT,
                                           outer_fold=0) / "features.parquet", columns=["corum_membership_count"])
    genes = pd.Index(sorted(corum.index), dtype="object", name="gene_key")
    static = read_indexed(METHODS_DIRECTORY / "epgat" / "feature" / "source" / "generated"
                          / f"{cohort}_static_features.parquet", columns=[YEAST])
    frame, _ = hallmark_frame(
        genes, corum.loc[genes, "corum_membership_count"].to_numpy("float64"), static.loc[genes, YEAST].to_numpy("float64"),
        induced_degree(genes), expression_median(genes))
    labels = read_cohort_labels(cohort).set_index("gene_key")["label"]
    labels.index = labels.index.astype(str)
    frame.insert(1, "label_23q4", labels.reindex(genes).to_numpy("float64"))
    if frame["label_23q4"].isna().any():
        raise ValueError("a gene of the 23Q4 cohort has no label")
    later = read_label_confidence("depmap")
    later["gene_key"] = later["gene_key"].astype(str)
    later = later.set_index("gene_key")
    for column in ("official_common_essential", "strongly_selective"):
        frame[f"later_{column}"] = later[column].astype("float64").reindex(genes).to_numpy()
    classes = prospective_genes()
    classes["gene_key"] = classes["gene_key"].astype(str)
    frame["prospective_class"] = classes.set_index("gene_key")[CLASS].reindex(genes).fillna("").to_numpy()
    return frame


def hallmark_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pool = frame[frame.prospective_class.isin((FOREGROUND, BACKGROUND))].reset_index(drop=True)
    for name in (FOREGROUND, BACKGROUND):
        part = pool[pool.prospective_class == name]
        row = {"group": name, "genes": int(len(part))}
        for k in range(5):
            row[f"count_{k}"] = int((part["count"] == k).sum())
        for column in ("complex", "old", "connected", "expressed"):
            row[f"share_{column}"] = float(part[column].mean())
        rows.append(row)
    negatives = frame[frame.label_23q4 == 0].reset_index(drop=True)
    for name, part in (("negatives_rich", negatives[negatives["count"] >= RICH]),
                       ("negatives_poor", negatives[negatives["count"] < RICH])):
        rows.append({"group": name, "genes": int(len(part)),
                     "became_common_essential": int((part.later_official_common_essential == 1).sum()),
                     "became_strongly_selective": int(((part.later_official_common_essential != 1)
                                                       & (part.later_strongly_selective == 1)).sum())})
    described = pd.DataFrame(rows)

    foreground = (pool.prospective_class == FOREGROUND).to_numpy()
    rich = (pool["count"] >= RICH).to_numpy("float64")

    def c1(take):
        f, r = (foreground, rich) if take is None else (foreground[take], rich[take])
        out = {"c1_foreground_rich_share": float(r[f].mean()), "c1_background_rich_share": float(r[~f].mean())}
        out["c1_difference"] = out["c1_foreground_rich_share"] - out["c1_background_rich_share"]
        return out

    negative_rich = (negatives["count"] >= RICH).to_numpy()
    became = (negatives.later_official_common_essential == 1).to_numpy("float64")
    selective = ((negatives.later_official_common_essential != 1) & (negatives.later_strongly_selective == 1)).to_numpy("float64")

    def c2(take):
        r, b, s = (negative_rich, became, selective) if take is None else (negative_rich[take], became[take], selective[take])
        out = {"c2_rich_became_common_essential": float(b[r].mean()), "c2_poor_became_common_essential": float(b[~r].mean()),
               "c2_rich_became_strongly_selective": float(s[r].mean()), "c2_poor_became_strongly_selective": float(s[~r].mean())}
        out["c2_difference"] = out["c2_rich_became_common_essential"] - out["c2_poor_became_common_essential"]
        return out

    first, _ = bootstrap(foreground.astype("int64"), c1, "C1")
    second, _ = bootstrap(negative_rich.astype("int64"), c2, "C2")
    return described, pd.concat([first, second], ignore_index=True)


def adjacency(granularity: str, genes: pd.Index) -> sp.csr_matrix:
    position = pd.Series(np.arange(genes.size), index=genes)
    pairs = cohort_modules(PROSPECTIVE_COHORT, granularity)
    pairs = pairs[pairs["gene_key"].isin(genes)]
    terms = pd.factorize(pairs[TERM_ID])[0]
    rows = position.reindex(pairs["gene_key"]).to_numpy()
    width = terms.max() + 1 if terms.size else 1
    incidence = sp.coo_matrix((np.ones(rows.size), (rows, terms)), shape=(genes.size, width)).tocsr()
    matrix = ((incidence @ incidence.T) > 0).astype("float64")
    matrix = matrix.tolil()
    matrix.setdiag(0.0)
    return matrix.tocsr()


def module_bands(genes: pd.DataFrame) -> pd.DataFrame:
    labeled = pd.Index(read_cohort_labels(PROSPECTIVE_COHORT)["gene_key"].astype("object"))
    position = pd.Series(np.arange(labeled.size), index=labeled)
    matrices = {g: adjacency(g, labeled) for g in (COMPLEX, PROCESS)}
    annotated = {g: np.asarray(matrices[g].sum(axis=1)).ravel() > 0 for g in (COMPLEX, PROCESS)}
    frames = []
    for fold in FOLDS:
        train = read_training_labels(PROSPECTIVE_COHORT, fold, TRAIN_ROLE, split_strategy=SPLIT)
        essential = np.zeros(labeled.size)
        places = position.reindex(pd.Index(train["gene_key"].astype("object"))).to_numpy("int64")
        essential[places] = train["label"].to_numpy("float64")
        rows = genes.loc[genes["fold"] == fold]
        index = position.reindex(pd.Index(rows["gene_key"].astype("object"))).to_numpy("int64")
        counts = {g: (matrices[g] @ essential)[index] for g in (COMPLEX, PROCESS)}
        frame = pd.DataFrame({"gene_key": rows["gene_key"].to_numpy("object")})
        frame["complex_partner"] = np.where(counts[COMPLEX] > 0, "complex_partner_yes", "complex_partner_no")
        frame["process_partner"] = np.where(counts[PROCESS] > 0, "process_partner_yes", "process_partner_no")
        frame["any_partner"] = np.where((counts[COMPLEX] + counts[PROCESS]) > 0, "any_partner_yes", "any_partner_no")
        any_annotation = annotated[COMPLEX][index] | annotated[PROCESS][index]
        frame["annotated"] = np.where(any_annotation, "annotated_yes", "annotated_no")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def predictor_frames(predictor: str, core: pd.DataFrame, counts: pd.Series) -> list[pd.DataFrame]:
    if predictor == CORE:
        return [core.loc[core.fold == fold, ["gene_key", "y_score"]].reset_index(drop=True) for fold in FOLDS]
    if predictor == COUNT:
        frames = []
        for fold in FOLDS:
            keys = core.loc[core.fold == fold, "gene_key"].to_numpy("object")
            frames.append(pd.DataFrame({"gene_key": keys, "y_score": counts.reindex(keys).to_numpy("float64")}))
        return frames
    frames = source_predictions(predictor, split_strategy=SPLIT)
    return [frame.assign(gene_key=frame["gene_key"].astype(str)) for frame in frames]


def assemble(core: pd.DataFrame, counts: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    genes = prospective_genes()
    genes["gene_key"] = genes["gene_key"].astype(str)
    base, columns = None, {}
    for predictor in PREDICTORS:
        frames = predictor_frames(predictor, core, counts)
        scored = pd.concat(scored_frames(frames, genes), ignore_index=True)
        scored = scored.sort_values(["fold", "gene_key"], kind="stable").reset_index(drop=True)
        if base is None:
            base = scored[["gene_key", "label", "fold"]].copy()
        elif not np.array_equal(base["gene_key"].to_numpy("object"), scored["gene_key"].to_numpy("object")):
            raise ValueError(f"{predictor}: a different gene order")
        columns[predictor] = scored["y_score"].to_numpy("float64")
    literature = literature_counts().reindex(pd.Index(base["gene_key"].astype("object"))).fillna(0.0)
    base["literature_quartile"] = quartile_bands(literature).to_numpy("object")
    bands = module_bands(base).set_index("gene_key")
    for column in ("complex_partner", "process_partner", "any_partner", "annotated"):
        base[column] = bands[column].reindex(base["gene_key"].to_numpy("object")).to_numpy("object")
    base[ALL] = ALL
    return base, np.column_stack([columns[p] for p in PREDICTORS])


_DATA: dict[str, dict] = {}


def load() -> dict:
    if SPLIT not in _DATA:
        table_path, block_path = assembled_paths()
        base = pd.read_csv(table_path)
        block = np.load(block_path)
        y = base["label"].to_numpy("float64")
        fold = base["fold"].to_numpy()
        cells = cells_of(fold, y)
        order = np.argsort(cells, kind="stable")
        _DATA[SPLIT] = {"y": y, "block": block, "cells": cells, "order": order,
                        "slots": [fold[order] == f for f in FOLDS],
                        "bands": {name: base[name].to_numpy("object") for name in BANDINGS}}
    return _DATA[SPLIT]


def band_statistics(data: dict, index: np.ndarray) -> np.ndarray:
    y = data["y"][index]
    values = np.ascontiguousarray(data["block"][index].T)
    drawn = {name: values_[index] for name, values_ in data["bands"].items()}
    out = np.full((len(PREDICTORS), len(ENTRIES), len(METRICS)), np.nan)
    for position, (banding, band) in enumerate(ENTRIES):
        inside = drawn[banding] == band
        pieces = []
        for slot in data["slots"]:
            mask = slot & inside
            labels = y[mask]
            if labels.size == 0:
                continue
            pieces.append((mask, labels, float(labels.sum())))
        if not pieces:
            continue
        for column in range(len(PREDICTORS)):
            aurocs, auprcs, shortlist = [], [], 0.0
            for mask, labels, n_pos in pieces:
                scores = values[column][mask]
                ranks = average_ranks(scores)
                n_neg = labels.size - n_pos
                if n_pos > 0 and n_neg > 0:
                    aurocs.append((float(ranks[labels == 1].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
                    auprcs.append(float(auprc(labels, scores)))
                if banding == ALL:
                    shortlist += float(hits_at_k(labels, scores, k=SHORTLIST))
            out[column, position, 0] = float(np.mean(aurocs)) if aurocs else np.nan
            out[column, position, 1] = float(np.mean(auprcs)) if auprcs else np.nan
            if banding == ALL:
                out[column, position, 2] = shortlist
    return out


def chunk_job(job: tuple[int, int]) -> np.ndarray:
    start, stop = job
    data = load()
    resampler = Resampler(data["cells"])
    for _ in range(start):
        resampler.draw()
    return np.stack([band_statistics(data, resampler.draw()) for _ in range(stop - start)])


def band_table(point: np.ndarray, replicates: np.ndarray) -> pd.DataFrame:
    data = load()
    core = PREDICTORS.index(CORE)
    rows = []
    for position, (banding, band) in enumerate(ENTRIES):
        mask = data["bands"][banding] == band
        for column, predictor in enumerate(PREDICTORS):
            row = {"split_strategy": SPLIT, "banding": banding, "band": band, "predictor": predictor,
                   "genes": int(mask.sum()), "foreground": int(data["y"][mask].sum())}
            for offset, metric in enumerate(METRICS):
                value = point[column, position, offset]
                half_width, _ = summarise(value, replicates[:, column, position, offset])
                row[metric], row[f"{metric}_half_width"] = value, half_width
                if metric in ("auroc", "auprc"):
                    difference = value - point[core, position, offset]
                    half_width, p = summarise(difference, replicates[:, column, position, offset]
                                              - replicates[:, core, position, offset])
                    row[f"minus_core_{metric}"] = difference
                    row[f"minus_core_{metric}_half_width"] = half_width
                    row[f"minus_core_{metric}_p"] = p
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    fits = run_pool(fit_job, list(FOLDS), arguments.workers, "core feature set fits")
    core = pd.concat([fits[fold] for fold in FOLDS], ignore_index=True)

    hallmarks = cohort_hallmarks()
    hallmarks.to_csv(RESULTS / "hallmarks_depmap_23q4.csv", index=False)
    described, statistics = hallmark_tables(hallmarks)
    described.to_csv(RESULTS / "prospective_hallmark_counts.csv", index=False)
    statistics.to_csv(RESULTS / "prospective_hallmarks.csv", index=False)
    counts = hallmarks.set_index("gene_key")["count"].astype("float64")

    base, block = assemble(core, counts)
    table_path, block_path = assembled_paths()
    base.to_csv(table_path, index=False)
    np.save(block_path, block)
    print(f"assembled {len(base):,} genes, {int(base.label.sum())} newly essential", flush=True)

    chunks = [(start, min(start + CHUNK, RESAMPLE_COUNT)) for start in range(0, RESAMPLE_COUNT, CHUNK)]
    drawn = run_pool(chunk_job, chunks, arguments.workers, "resampling")
    data = load()
    point = band_statistics(data, data["order"])
    replicates = np.concatenate([drawn[job] for job in chunks])
    table = band_table(point, replicates)
    rows = table[table.predictor.isin(METHODS)]
    for metric in ("auroc", "auprc"):
        table[f"holm_{metric}"] = np.nan
        table.loc[rows.index, f"holm_{metric}"] = holm(rows, ["split_strategy", "banding", "band"], f"minus_core_{metric}_p")
    table.to_csv(RESULTS / "prospective_core_bands.csv", index=False)


if __name__ == "__main__":
    main()
