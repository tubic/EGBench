"""Builds the assets of the agent server: the test-fold predictions of the 15 methods per cohort, an example predictor on DEG with a flawed copy, and an example table of a user's own labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from egbench.contracts import (
    COHORT_IDS,
    LEADERBOARD_COHORT_IDS,
    METHOD_IDS,
    OUTER_FOLDS,
    SEED,
    ContractError,
    check_prediction_identity,
    read_fold_products,
)
from egbench.data import (
    EXPRESSION_COLUMN,
    FAMILY_SIZE_COLUMN,
    PPI_DEGREE_COLUMN,
    PROJECT_ROOT,
    TEST_ROLE,
    TRAIN_ROLE,
    VALIDATION_ROLE,
    cohort_target,
    fold_gene_keys,
    model_directory,
    prediction_cohort,
    primary_scope,
    read_evaluation_truth,
    read_split_assignments,
    read_strata_inputs,
    read_training_labels,
)
from egbench.evaluate import PRIMARY_METRICS, join_targets, metric_table, scope_frame, score_frames

SPLIT = "family_5fold"
GENE_KEY = "gene_key"
SCORE = "y_score"
FOLD = "outer_fold"

AGENT_DIRECTORY = Path(__file__).resolve().parent
PREDICTION_DIRECTORY = AGENT_DIRECTORY / "predictions"
EXAMPLE_DIRECTORY = AGENT_DIRECTORY / "examples"
LEADERBOARD_FILE = PROJECT_ROOT / "analysis" / "benchmark_report" / "results" / "primary_metric.csv"

PREDICTION_COHORTS: tuple[str, ...] = tuple(
    cohort_id for cohort_id in COHORT_IDS if prediction_cohort(cohort_id) == cohort_id
)

EXAMPLE_COHORT = "deg"
EXAMPLE_ID = "deg_logistic"
LABELS_ID = "mouse_lethality_labels"
MOUSE_UNIVERSE = PROJECT_ROOT / "datasets" / "Mouse" / "data" / "gene_universe.parquet"
EXAMPLE_COVARIATES: tuple[str, ...] = (FAMILY_SIZE_COLUMN, PPI_DEGREE_COLUMN, EXPRESSION_COLUMN)
FIT_ROLES: tuple[str, ...] = (TRAIN_ROLE, VALIDATION_ROLE)

# outer fold and number of genes of each flaw in the flawed copy of the example
DROPPED = (2, 3)
DUPLICATED = (0, 1)
BLANKED = (4, 1)


def _report(line: str) -> None:
    print(line, flush=True)


def fold_scores(
    method_id: str, cohort_id: str, outer_fold: int, assignments: pd.DataFrame
) -> pd.Series:
    directory = model_directory(
        PROJECT_ROOT / "methods" / method_id, cohort_id, split_strategy=SPLIT, outer_fold=outer_fold
    )
    products = read_fold_products(directory)
    expected = {
        "method_id": method_id,
        "cohort_id": cohort_id,
        "split_strategy": SPLIT,
        "outer_fold": outer_fold,
        "split_build_hash": str(assignments["split_build_hash"].iat[0]),
    }
    differing = [
        f"{key}: {products.manifest[key]!r}, the fold table gives {value!r}"
        for key, value in expected.items()
        if products.manifest[key] != value
    ]
    if differing:
        raise ContractError(f"{directory}: the run manifest describes another fold, {differing}")
    check_prediction_identity(
        products.blind,
        assignments,
        cohort_id=cohort_id,
        split_strategy=SPLIT,
        outer_fold=outer_fold,
        role=TEST_ROLE,
    )
    return products.blind.set_index(GENE_KEY)[SCORE]


def cohort_table(cohort_id: str) -> pd.DataFrame:
    assignments = read_split_assignments(cohort_id, split_strategy=SPLIT)
    folds = []
    for outer_fold in OUTER_FOLDS:
        tested = assignments[FOLD].eq(outer_fold) & assignments["role"].eq(TEST_ROLE)
        keys = np.sort(assignments.loc[tested, GENE_KEY].to_numpy(dtype=object))
        frame = pd.DataFrame({GENE_KEY: keys, FOLD: np.full(keys.size, outer_fold, dtype="int8")})
        for method_id in METHOD_IDS:
            scores = fold_scores(method_id, cohort_id, outer_fold, assignments)
            frame[method_id] = scores.reindex(keys).to_numpy(dtype="float64")
        folds.append(frame)
    return pd.concat(folds, ignore_index=True).sort_values([FOLD, GENE_KEY], ignore_index=True)


def leaderboard_points(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cohort_id in LEADERBOARD_COHORT_IDS:
        table = tables[prediction_cohort(cohort_id)]
        target = cohort_target(cohort_id)
        scope = primary_scope(cohort_id)
        metric = PRIMARY_METRICS[target.kind]
        function = metric_table(cohort_id)[metric]
        values: dict[str, list[float]] = {method_id: [] for method_id in METHOD_IDS}
        for outer_fold in OUTER_FOLDS:
            truth = read_evaluation_truth(cohort_id, outer_fold, split_strategy=SPLIT)
            fold = table.loc[table[FOLD].eq(outer_fold)]
            for method_id in METHOD_IDS:
                frame = join_targets(
                    fold[[GENE_KEY, method_id]].rename(columns={method_id: SCORE}),
                    truth[[GENE_KEY, target.column]],
                    cohort_id=cohort_id,
                    split_strategy=SPLIT,
                    source=f"{prediction_cohort(cohort_id)}.parquet, {method_id}, fold {outer_fold}",
                )
                frame = scope_frame(frame, cohort_id=cohort_id, scope=scope)
                values[method_id].append(
                    function(frame[target.column].to_numpy(), frame[SCORE].to_numpy())
                )
        for method_id, per_fold in values.items():
            rows.append((method_id, cohort_id, scope, metric, float(np.nanmean(per_fold))))
    return pd.DataFrame(rows, columns=["predictor", "cohort_id", "scope", "metric", "recomputed"])


def check_leaderboard(tables: dict[str, pd.DataFrame]) -> None:
    published = pd.read_csv(LEADERBOARD_FILE)
    published = published.loc[published["predictor"].isin(METHOD_IDS)]
    merged = leaderboard_points(tables).merge(
        published, on=["predictor", "cohort_id", "scope", "metric"], how="left", validate="one_to_one"
    )
    if merged["point"].isna().any():
        missing = merged.loc[merged["point"].isna(), ["predictor", "cohort_id"]].values.tolist()
        raise ContractError(f"{LEADERBOARD_FILE}: no published point for {missing}")
    merged["difference"] = (merged["recomputed"] - merged["point"]).abs()
    for cohort_id, group in merged.groupby("cohort_id", sort=False):
        _report(
            f"{cohort_id} {group['metric'].iat[0]} on {group['scope'].iat[0]}: largest absolute "
            f"difference from primary_metric.csv over {len(group)} methods {group['difference'].max():.3g}"
        )


def build_predictions() -> None:
    PREDICTION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    tables = {}
    for cohort_id in PREDICTION_COHORTS:
        table = cohort_table(cohort_id)
        path = PREDICTION_DIRECTORY / f"{cohort_id}.parquet"
        table.to_parquet(path, index=False, compression="zstd")
        tables[cohort_id] = table
        _report(
            f"{cohort_id}: {len(METHOD_IDS)} methods x {len(OUTER_FOLDS)} folds checked, "
            f"{len(table)} genes written to {path}"
        )
    check_leaderboard(tables)


def example_predictions() -> pd.DataFrame:
    target = cohort_target(EXAMPLE_COHORT)
    covariates = (
        read_strata_inputs(EXAMPLE_COHORT).set_index(GENE_KEY)[list(EXAMPLE_COVARIATES)].astype("float64")
    )
    folds = []
    for outer_fold in OUTER_FOLDS:
        fitted = pd.concat(
            [
                read_training_labels(EXAMPLE_COHORT, outer_fold, role, split_strategy=SPLIT)
                for role in FIT_ROLES
            ],
            ignore_index=True,
        )
        tested = np.sort(
            fold_gene_keys(EXAMPLE_COHORT, outer_fold, TEST_ROLE, split_strategy=SPLIT).to_numpy(dtype=object)
        )
        # log1p of the three covariates, missing values filled with the median of the fitted genes
        model = make_pipeline(
            FunctionTransformer(np.log1p),
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(random_state=SEED),
        )
        model.fit(covariates.reindex(fitted[GENE_KEY]).to_numpy(), fitted[target.column].to_numpy())
        scores = model.predict_proba(covariates.reindex(tested).to_numpy())[:, 1]
        folds.append(pd.DataFrame({GENE_KEY: tested, SCORE: scores, FOLD: outer_fold}))
    return pd.concat(folds, ignore_index=True).sort_values([FOLD, GENE_KEY], ignore_index=True)


def flawed_copy(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    generator = np.random.default_rng(SEED)

    def pick(flaw: tuple[int, int]) -> list[str]:
        outer_fold, count = flaw
        keys = predictions.loc[predictions[FOLD].eq(outer_fold), GENE_KEY].to_numpy(dtype=object)
        return sorted(generator.choice(keys, size=count, replace=False).tolist())

    changed = {"dropped": pick(DROPPED), "duplicated": pick(DUPLICATED), "blanked": pick(BLANKED)}
    flawed = predictions.loc[~predictions[GENE_KEY].isin(changed["dropped"])]
    flawed = pd.concat(
        [flawed, flawed.loc[flawed[GENE_KEY].isin(changed["duplicated"])]], ignore_index=True
    )
    flawed.loc[flawed[GENE_KEY].isin(changed["blanked"]), SCORE] = np.nan
    return flawed.sort_values([FOLD, GENE_KEY], kind="stable", ignore_index=True), changed


def score_example(path: Path) -> None:
    predictions = pd.read_csv(path, dtype={GENE_KEY: object})
    target = cohort_target(EXAMPLE_COHORT)
    folds = []
    for outer_fold in OUTER_FOLDS:
        truth = read_evaluation_truth(EXAMPLE_COHORT, outer_fold, split_strategy=SPLIT)
        folds.append(
            join_targets(
                predictions.loc[predictions[FOLD].eq(outer_fold), [GENE_KEY, SCORE]],
                truth[[GENE_KEY, target.column]],
                cohort_id=EXAMPLE_COHORT,
                split_strategy=SPLIT,
                source=f"{path}, fold {outer_fold}",
            )
        )
    scored = score_frames(folds, method_id=EXAMPLE_ID, cohort_id=EXAMPLE_COHORT, split_strategy=SPLIT)
    for metric, value in scored.values.items():
        _report(f"{EXAMPLE_ID} {metric}: {value:.4f} +/- {scored.intervals[metric].half_width:.4f}")
    metric = PRIMARY_METRICS[target.kind]
    published = pd.read_csv(LEADERBOARD_FILE)
    listed = published.loc[
        published["predictor"].isin(METHOD_IDS)
        & published["cohort_id"].eq(EXAMPLE_COHORT)
        & published["metric"].eq(metric)
    ]
    above = int((listed["point"] > scored.values[metric]).sum())
    _report(f"{EXAMPLE_ID} would rank {above + 1} by {metric} among itself and the {len(listed)} methods")


def build_example() -> None:
    EXAMPLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    predictions = example_predictions()
    correct = EXAMPLE_DIRECTORY / f"{EXAMPLE_ID}.csv"
    predictions.to_csv(correct, index=False)
    flawed, changed = flawed_copy(predictions)
    flawed_path = EXAMPLE_DIRECTORY / f"{EXAMPLE_ID}_flawed.csv"
    flawed.to_csv(flawed_path, index=False)
    _report(f"wrote {len(predictions)} rows to {correct} and {len(flawed)} rows to {flawed_path}")
    for flaw, keys in changed.items():
        _report(f"flawed copy, {flaw}: {', '.join(keys)}")
    score_example(correct)


def build_labels() -> None:
    """Write the knockout calls of the Mouse task, 1 lethal and 0 viable, by gene symbol, as a user would give labels."""
    universe = pd.read_parquet(MOUSE_UNIVERSE, columns=["gene_symbol", "label"]).dropna(subset=["label"])
    table = universe.rename(columns={"gene_symbol": "gene"}).astype({"label": "int64"}).sort_values("gene")
    path = EXAMPLE_DIRECTORY / f"{LABELS_ID}.csv"
    table.to_csv(path, index=False)
    _report(f"wrote {len(table)} genes, {int(table['label'].sum())} of them lethal, to {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the prediction tables and the example predictor of the agent server.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "predictions",
        help="write the test-fold scores of the 15 methods, one table per cohort, and recompute the leaderboard from them",
    )
    commands.add_parser(
        "example",
        help="fit the example logistic predictor on DEG, write it and a flawed copy, and score the correct one",
    )
    commands.add_parser("labels", help="write the knockout calls of the Mouse task as an example of a user's own labels")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "predictions":
        build_predictions()
    elif arguments.command == "example":
        build_example()
    else:
        build_labels()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
