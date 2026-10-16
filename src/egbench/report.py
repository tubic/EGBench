"""Builds the benchmark tables reported in the manuscript: leaderboard, fusion gain, rank comparisons, matched subsets and false positives."""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from egbench.baselines import BASELINE_IDS, COVARIATE_COLUMNS, gene_covariates, score_baseline
from egbench.contracts import LEADERBOARD_COHORT_IDS, METHOD_IDS, TargetKind
from egbench.data import PROJECT_ROOT, cohort_target, read_cohort_labels
from egbench.evaluate import PRIMARY_METRICS, Scored, score_folds
from egbench.fusion import (
    FUSION_TABLE_COLUMNS,
    FUSION_TOP,
    FusionComparison,
    compare_fusion,
    fusion_table,
    pooled,
    score_fusion,
)
from egbench.metrics import discrimination, regression
from egbench.metrics.discrimination import average_ranks
from egbench.stats import (
    CONFIDENCE_LEVEL,
    Comparison,
    benjamini_hochberg_adjusted,
    holm_adjusted,
    paired_difference,
    paired_view_differences,
)
from egbench.strata import MATCHED, SliceScore, matched_mask, score_slice, strata_table
from egbench.top_errors import FalsePositiveProfile, false_positive_profile, false_positive_table

SPLIT = "family_5fold"
GENE_KEY = "gene_key"
SCORE = "y_score"
FOLD = "fold"

METHOD_NAMES: dict[str, str] = {
    "bingo": "BiNGO",
    "deepcelless": "DeepCellEss",
    "deephe": "DeepHE",
    "deeplyessential": "DeeplyEssential",
    "deepsnap": "DeEPsnap",
    "ednn_eg": "eDNN-EG",
    "epgat": "EPGAT",
    "esssubgraph": "EssSubgraph",
    "esstfnet": "EssTFNet",
    "gcnn_sfm": "GCNN-SFM",
    "help": "HELP",
    "mahi": "Mahi",
    "pheg": "Pheg",
    "proteomelm": "ProteomeLM-Ess",
    "xgep": "XGEP-XGB",
}
OTHER_NAMES: dict[str, str] = {
    FUSION_TOP: "Top-5 fusion",
    "ppi_degree": "Baseline PPI degree",
    "literature": "Baseline literature",
}

MATCHED_COHORTS: tuple[str, ...] = ("deg", "depmap")
FALSE_POSITIVE_COHORTS: tuple[str, ...] = ("depmap",)
DROP_METRICS: dict[str, Callable[..., float]] = {
    "auprc": discrimination.auprc,
    "auroc": discrimination.auroc,
}

LEADERBOARD_RANK = "leaderboard_rank"
FUSION_GAIN = "fusion_gain"
FAMILIES: tuple[str, ...] = (LEADERBOARD_RANK, FUSION_GAIN)
DIRECTED: dict[str, bool] = {LEADERBOARD_RANK: True, FUSION_GAIN: True}
ALPHA = 1.0 - CONFIDENCE_LEVEL

RANK_COLUMNS: tuple[str, ...] = (
    "predictor",
    "cohort_id",
    "scope",
    "split_strategy",
    "metric",
    "point",
    "half_width",
    "rank",
)
MEAN_RANK_COLUMNS: tuple[str, ...] = ("predictor", "split_strategy", "mean_rank", "task_count")
SIGNIFICANCE_COLUMNS: tuple[str, ...] = (
    "family",
    "cohort_id",
    "split_strategy",
    "metric",
    "subject_a",
    "subject_b",
    "gene_count",
    "difference",
    "half_width",
    "p_value",
    "family_size",
    "holm_p",
    "bh_p",
    "directed",
    "holds",
)
SMD_COLUMNS: tuple[str, ...] = ("covariate", "smd_before", "smd_after")


class Predictor(NamedTuple):
    predictor_id: str
    display_name: str
    ranked: bool


class Claim(NamedTuple):
    family: str
    cohort_id: str
    split_strategy: str
    metric: str
    subject_a: str
    subject_b: str
    gene_count: int
    comparison: Comparison


class MatchedResult(NamedTuple):
    predictor_id: str
    cohort_id: str
    score: SliceScore
    drops: dict[str, Comparison]


def predictors() -> dict[str, Predictor]:
    listed = {name: Predictor(name, METHOD_NAMES[name], True) for name in METHOD_IDS}
    for name in (FUSION_TOP, *BASELINE_IDS):
        listed[name] = Predictor(name, OTHER_NAMES[name], False)
    return listed


def primary_metric(cohort_id: str) -> str:
    return PRIMARY_METRICS[cohort_target(cohort_id).kind]


def metric_function(cohort_id: str) -> Callable[..., float]:
    if cohort_target(cohort_id).kind is TargetKind.BINARY:
        return discrimination.auprc
    return regression.spearman


def run_jobs(
    queue: Sequence[object],
    function: Callable[..., object],
    *arguments: object,
    workers: int,
    label: str,
    progress: Callable[[str], None],
) -> dict[object, object]:
    results: dict[object, object] = {}
    started = time.monotonic()
    total = len(queue)

    def note() -> None:
        progress(f"{time.strftime('%H:%M:%S')} {label}: {len(results)}/{total} done, {time.monotonic() - started:.0f} s")

    if workers == 1 or total <= 1:
        for job in queue:
            results[job] = function(job, *arguments)
            note()
        return results
    interval = 30.0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        submitted: dict[Future, object] = {pool.submit(function, job, *arguments): job for job in queue}
        pending = set(submitted)
        next_beat = started + interval
        while pending:
            done, pending = wait(pending, timeout=max(0.0, next_beat - time.monotonic()), return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    results[submitted[future]] = future.result()
                except BaseException:
                    for other in pending:
                        other.cancel()
                    raise
            if time.monotonic() >= next_beat or not pending:
                note()
                next_beat = time.monotonic() + interval
    return results


def score_job(job: tuple[str, str], methods_directory: Path) -> Scored:
    predictor_id, cohort_id = job
    if predictor_id in BASELINE_IDS:
        return score_baseline(predictor_id, cohort_id, split_strategy=SPLIT)
    if predictor_id == FUSION_TOP:
        return score_fusion(
            predictor_id,
            cohort_id,
            split_strategy=SPLIT,
            member_ids=METHOD_IDS,
            methods_directory=methods_directory,
        )
    return score_folds(methods_directory / predictor_id, cohort_id, split_strategy=SPLIT)


def long_table(scored: Iterable[Scored]) -> pd.DataFrame:
    rows = []
    for item in scored:
        for metric, value in item.values.items():
            rows.append(
                (
                    item.method_id,
                    item.cohort_id,
                    item.scope,
                    item.split_strategy,
                    metric,
                    float(value),
                    float(item.intervals[metric].half_width),
                )
            )
    return pd.DataFrame(rows, columns=list(RANK_COLUMNS[:-1]))


def primary_ranks(table: pd.DataFrame, listed: Mapping[str, Predictor]) -> pd.DataFrame:
    order = {name: position for position, name in enumerate(listed)}
    frames = []
    for (cohort_id, scope, split_strategy), group in table.groupby(
        ["cohort_id", "scope", "split_strategy"], sort=True
    ):
        metric = primary_metric(str(cohort_id))
        rows = group.loc[group["metric"] == metric, list(RANK_COLUMNS[:-1])].copy()
        ranked = rows["predictor"].map(lambda name: listed[name].ranked).to_numpy(bool)
        points = rows.loc[ranked, "point"].to_numpy("float64")
        if np.isnan(points).any():
            raise ValueError(f"primary_ranks: {metric!r} is undefined for a method on {cohort_id}")
        rank = np.full(len(rows), float("nan"))
        if ranked.any():
            rank[ranked] = average_ranks(-points)
        rows["rank"] = rank
        rows["_unranked"] = ~ranked
        rows["_order"] = rows["predictor"].map(order)
        frames.append(
            rows.sort_values(["_unranked", "rank", "_order"], kind="stable").drop(
                columns=["_unranked", "_order"]
            )
        )
    return pd.concat(frames, ignore_index=True)[list(RANK_COLUMNS)]


def mean_ranks(ranks: pd.DataFrame) -> pd.DataFrame:
    ranked = ranks.dropna(subset=["rank"])
    rows = []
    for split_strategy, group in ranked.groupby("split_strategy", sort=True):
        cohorts = group.groupby("predictor")["cohort_id"].apply(frozenset)
        if cohorts.nunique() != 1:
            raise ValueError(f"mean_ranks: the methods of {split_strategy} were ranked on different cohorts")
        means = group.groupby("predictor")["rank"].mean()
        for predictor, mean in means.items():
            rows.append((predictor, split_strategy, float(mean), len(cohorts.iat[0])))
    result = pd.DataFrame(rows, columns=list(MEAN_RANK_COLUMNS))
    return result.sort_values(["split_strategy", "mean_rank", "predictor"], kind="stable").reset_index(drop=True)


def ranked_methods(ranks: pd.DataFrame, cohort_id: str) -> list[str]:
    rows = ranks.loc[(ranks["cohort_id"] == cohort_id) & ranks["rank"].notna()]
    return rows.sort_values("rank", kind="stable")["predictor"].tolist()


def predictor_order(ranks: pd.DataFrame, cohort_id: str, listed: Mapping[str, Predictor]) -> list[str]:
    order = ranks.loc[ranks["cohort_id"] == cohort_id, "predictor"].tolist()
    return order + [name for name in listed if name not in order]


def fusion_job(cohort_id: str, methods_directory: Path) -> FusionComparison:
    return compare_fusion(
        FUSION_TOP,
        cohort_id,
        split_strategy=SPLIT,
        member_ids=METHOD_IDS,
        methods_directory=methods_directory,
    )


def rank_claims_job(job: tuple[str, tuple[str, ...]], methods_directory: Path) -> list[Claim]:
    cohort_id, top = job
    target = cohort_target(cohort_id)
    frames = {
        name: pooled(name, cohort_id, split_strategy=SPLIT, methods_directory=methods_directory)
        .sort_values([FOLD, GENE_KEY], kind="stable")
        .reset_index(drop=True)
        for name in top
    }
    first = frames[top[0]]
    keys = first[GENE_KEY].to_numpy("object")
    y_true = first[target.column].to_numpy("float64")
    fold = first[FOLD].to_numpy("int64")
    for name, frame in frames.items():
        if not np.array_equal(keys, frame[GENE_KEY].to_numpy("object")):
            raise ValueError(f"rank claims: {name!r} is scored on other genes of {cohort_id}")
    claims = []
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            claims.append(
                Claim(
                    LEADERBOARD_RANK,
                    cohort_id,
                    SPLIT,
                    primary_metric(cohort_id),
                    top[a],
                    top[b],
                    int(y_true.size),
                    paired_difference(
                        y_true,
                        frames[top[a]][SCORE].to_numpy("float64"),
                        frames[top[b]][SCORE].to_numpy("float64"),
                        metric_function(cohort_id),
                        fold=fold,
                        target_kind=target.kind,
                    ),
                )
            )
    return claims


def fusion_claims(comparisons: Sequence[FusionComparison]) -> list[Claim]:
    claims = []
    for item in comparisons:
        metric = primary_metric(item.cohort_id)
        claims.append(
            Claim(
                FUSION_GAIN,
                item.cohort_id,
                item.split_strategy,
                metric,
                item.fusion_id,
                "+".join(sorted(set(item.best_members))),
                item.gene_count,
                item.comparisons[metric],
            )
        )
    return claims


def significance_table(claims: Sequence[Claim]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "family": claim.family,
                "cohort_id": claim.cohort_id,
                "split_strategy": claim.split_strategy,
                "metric": claim.metric,
                "subject_a": claim.subject_a,
                "subject_b": claim.subject_b,
                "gene_count": claim.gene_count,
                "difference": claim.comparison.difference,
                "half_width": claim.comparison.half_width,
                "p_value": claim.comparison.p_value,
                "directed": DIRECTED[claim.family],
            }
            for claim in claims
        ]
    )
    frame["_family_order"] = frame["family"].map({name: index for index, name in enumerate(FAMILIES)})
    frame = frame.sort_values(["_family_order"], kind="stable").drop(columns=["_family_order"])
    keys = ["family", "cohort_id", "split_strategy"]
    frame["family_size"] = frame.groupby(keys, sort=False)["p_value"].transform(
        lambda values: int(values.notna().sum())
    )
    frame["holm_p"] = frame.groupby(keys, sort=False)["p_value"].transform(
        lambda values: holm_adjusted(values.to_numpy("float64"))
    )
    frame["bh_p"] = frame.groupby(keys, sort=False)["p_value"].transform(
        lambda values: benjamini_hochberg_adjusted(values.to_numpy("float64"))
    )
    signed = (~frame["directed"]) | (frame["difference"] > 0.0)
    frame["holds"] = frame["holm_p"].notna() & (frame["holm_p"] <= ALPHA) & signed
    return frame.reset_index(drop=True)[list(SIGNIFICANCE_COLUMNS)]


def matched_job(job: tuple[str, str], masks: Mapping[str, pd.Series], methods_directory: Path) -> MatchedResult:
    predictor_id, cohort_id = job
    frame = pooled(predictor_id, cohort_id, split_strategy=SPLIT, methods_directory=methods_directory)
    target = cohort_target(cohort_id)
    keys = pd.Index(frame[GENE_KEY].to_numpy("object"))
    y_true = frame[target.column].to_numpy("float64")
    y_score = frame[SCORE].to_numpy("float64")
    fold = frame[FOLD].to_numpy("int64")
    keep = masks[cohort_id].reindex(keys, fill_value=False).astype(bool).to_numpy()
    score = score_slice(y_true, y_score, keep, fold=fold, target_kind=target.kind)
    drops = paired_view_differences(
        y_true, y_score, y_score, dict(DROP_METRICS), fold_a=fold, fold_b=fold, keep_a=keep
    )
    return MatchedResult(predictor_id, cohort_id, score, drops)


def _smd(values: pd.Series, labels: pd.Series) -> float:
    present = values.notna()
    kept = values[present]
    classes = labels[present].astype("float64")
    spread = float(kept.std(ddof=1))
    if kept.size < 2 or spread == 0.0 or not (classes == 1.0).any() or not (classes == 0.0).any():
        return float("nan")
    return float((kept[classes == 1.0].mean() - kept[classes == 0.0].mean()) / spread)


def standardised_mean_differences(cohort_id: str, mask: pd.Series) -> pd.DataFrame:
    values = gene_covariates(cohort_id, split_strategy=SPLIT).set_index(GENE_KEY)
    labels = read_cohort_labels(cohort_id).set_index(GENE_KEY)[cohort_target(cohort_id).column]
    keys = labels.index
    logged = np.log1p(values.reindex(keys)[list(COVARIATE_COLUMNS)].astype("float64"))
    kept = mask.reindex(keys).fillna(False).astype(bool)
    rows = [
        {
            "covariate": column,
            "smd_before": _smd(logged[column], labels),
            "smd_after": _smd(logged[column][kept], labels[kept]),
        }
        for column in COVARIATE_COLUMNS
    ]
    return pd.DataFrame(rows, columns=list(SMD_COLUMNS))


def matched_frame(
    results: Sequence[MatchedResult],
    table: pd.DataFrame,
    *,
    cohort_id: str,
    order: Sequence[str],
    listed: Mapping[str, Predictor],
) -> pd.DataFrame:
    rows = table[(table["cohort_id"] == cohort_id) & table["metric"].isin(list(DROP_METRICS))]
    full = {
        (str(row.predictor), str(row.metric)): (float(row.point), float(row.half_width))
        for row in rows.itertuples(index=False)
    }
    by_predictor = {item.predictor_id: item for item in results if item.cohort_id == cohort_id}
    out = []
    for predictor_id in order:
        item = by_predictor.get(predictor_id)
        if item is None:
            continue
        row: dict[str, object] = {
            "matching": MATCHED,
            "predictor": predictor_id,
            "display_name": listed[predictor_id].display_name,
            "gene_count": item.score.gene_count,
            "positive_count": item.score.positive_count,
        }
        for metric in DROP_METRICS:
            whole = full.get((predictor_id, metric), (float("nan"), float("nan")))
            interval = item.score.intervals[metric]
            drop = item.drops[metric]
            row[f"full_{metric}"] = whole[0]
            row[f"full_{metric}_half_width"] = whole[1]
            row[f"matched_{metric}"] = interval.point
            row[f"matched_{metric}_half_width"] = interval.half_width
            row[f"{metric}_drop"] = drop.difference
            row[f"{metric}_drop_half_width"] = drop.half_width
            row[f"{metric}_drop_p_value"] = drop.p_value
        out.append(row)
    return pd.DataFrame(out)


def false_positive_job(job: tuple[str, str], methods_directory: Path) -> FalsePositiveProfile:
    predictor_id, cohort_id = job
    return false_positive_profile(
        predictor_id,
        cohort_id,
        split_strategy=SPLIT,
        member_ids=METHOD_IDS,
        methods_directory=methods_directory,
    )


def _named(frame: pd.DataFrame, listed: Mapping[str, Predictor]) -> pd.DataFrame:
    named = frame.copy()
    named.insert(1, "display_name", named["predictor"].map(lambda name: listed[name].display_name))
    return named


def build(
    output: str | Path | None = None,
    *,
    workers: int = 1,
    progress: Callable[[str], None] | None = None,
    methods_directory: str | Path | None = None,
) -> list[Path]:
    report = progress if progress is not None else (lambda line: None)
    target = PROJECT_ROOT / "analysis" / "benchmark_report" / "results" if output is None else Path(output)
    target.mkdir(parents=True, exist_ok=True)
    directory = PROJECT_ROOT / "methods" if methods_directory is None else Path(methods_directory)
    listed = predictors()
    written: list[Path] = []

    def write(frame: pd.DataFrame, name: str) -> None:
        path = target / name
        frame.to_csv(path, index=False)
        written.append(path)

    jobs = [
        (name, cohort_id)
        for name in (*METHOD_IDS, FUSION_TOP)
        for cohort_id in LEADERBOARD_COHORT_IDS
    ] + [(name, cohort_id) for name in BASELINE_IDS for cohort_id in MATCHED_COHORTS]
    scored = run_jobs(jobs, score_job, directory, workers=workers, label="scoring", progress=report)
    table = long_table(scored[job] for job in jobs)
    leaderboard = table[~table["predictor"].isin(BASELINE_IDS)]
    ranks = primary_ranks(leaderboard, listed)
    write(_named(ranks, listed), "primary_metric.csv")
    write(_named(mean_ranks(ranks), listed), "mean_rank.csv")

    fused = run_jobs(
        list(LEADERBOARD_COHORT_IDS), fusion_job, directory, workers=workers, label="fusion", progress=report
    )
    comparisons = [fused[cohort_id] for cohort_id in LEADERBOARD_COHORT_IDS]
    fusions = fusion_table(comparisons)
    for cohort_id in LEADERBOARD_COHORT_IDS:
        rows = fusions[fusions["cohort_id"] == cohort_id]
        frame = rows[list(FUSION_TABLE_COLUMNS)].copy()
        write(_named(frame, listed), f"{cohort_id}_{SPLIT}_fusion.csv")

    top = [(cohort_id, tuple(ranked_methods(ranks, cohort_id)[:3])) for cohort_id in LEADERBOARD_COHORT_IDS]
    claimed = run_jobs(top, rank_claims_job, directory, workers=workers, label="rank comparisons", progress=report)
    claims = [claim for job in top for claim in claimed[job]] + fusion_claims(comparisons)
    write(significance_table(claims), "significance.csv")

    masks = {cohort_id: matched_mask(strata_table(cohort_id)) for cohort_id in MATCHED_COHORTS}
    matched_jobs = [(name, cohort_id) for cohort_id in MATCHED_COHORTS for name in listed]
    matched = run_jobs(
        matched_jobs, matched_job, masks, directory, workers=workers, label="matched subsets", progress=report
    )
    for cohort_id in MATCHED_COHORTS:
        order = predictor_order(ranks, cohort_id, listed)
        frame = matched_frame(
            [matched[job] for job in matched_jobs], table, cohort_id=cohort_id, order=order, listed=listed
        )
        write(frame, f"{cohort_id}_{SPLIT}_matched.csv")
        balance = standardised_mean_differences(cohort_id, masks[cohort_id])
        balance.insert(0, "matching", MATCHED)
        write(balance, f"{cohort_id}_{SPLIT}_matched_balance.csv")

    profile_jobs = [
        (name, cohort_id) for cohort_id in FALSE_POSITIVE_COHORTS for name in (FUSION_TOP, *METHOD_IDS)
    ]
    profiles = run_jobs(
        profile_jobs, false_positive_job, directory, workers=workers, label="false positives", progress=report
    )
    table_fp = false_positive_table([profiles[job] for job in profile_jobs])
    for cohort_id in FALSE_POSITIVE_COHORTS:
        rows = table_fp[table_fp["cohort_id"] == cohort_id]
        order = predictor_order(ranks, cohort_id, listed)
        names = [name for name in order if name in set(rows["predictor"])]
        names += sorted(set(rows["predictor"]) - set(names))
        frame = pd.concat([rows[rows["predictor"] == name] for name in names], ignore_index=True)
        write(_named(frame, listed), f"{cohort_id}_{SPLIT}_false_positives.csv")
    return written
