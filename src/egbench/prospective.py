"""Genes that became common essential between DepMap 23Q4 and 26Q1, and the method predictions scored on them."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from egbench.baselines import BASELINE_IDS, baseline_predictions
from egbench.contracts import PREDICTION_FILES
from egbench.data import (
    PROJECT_ROOT,
    TEST_ROLE,
    model_directory,
    read_cohort_labels,
    read_label_confidence,
    read_predictions,
)
from egbench.evaluate import FOLD_COUNT

PROSPECTIVE_COHORT = "depmap_23q4"
LATER_COHORT = "depmap"
LATER_COMMON_ESSENTIAL = "official_common_essential"
LATER_STRONGLY_SELECTIVE = "strongly_selective"

GENE_KEY = "gene_key"
SCORE = "y_score"
CLASS = "prospective_class"
FOREGROUND = "foreground"
BACKGROUND = "background"
FOLD = "fold"


def prospective_genes(root: str | Path | None = None) -> pd.DataFrame:
    labels = read_cohort_labels(PROSPECTIVE_COHORT, root=root)
    later = read_label_confidence(LATER_COHORT, root=root)
    joined = labels.merge(later, on=GENE_KEY, how="left", validate="one_to_one")
    if joined[LATER_COMMON_ESSENTIAL].isna().any():
        missing = joined.loc[joined[LATER_COMMON_ESSENTIAL].isna(), GENE_KEY].tolist()
        raise ValueError(
            f"prospective_genes: {len(missing)} genes of {PROSPECTIVE_COHORT} have no "
            f"{LATER_COHORT} verdict, for example {missing[:5]}"
        )
    negative = joined["label"].astype("int64").eq(0)
    later_common = joined[LATER_COMMON_ESSENTIAL].astype("int64").eq(1)
    later_selective = joined[LATER_STRONGLY_SELECTIVE].astype("int64").eq(1)
    frame = pd.DataFrame({GENE_KEY: joined[GENE_KEY].astype("object")})
    frame[CLASS] = ""
    frame.loc[negative & later_common, CLASS] = FOREGROUND
    frame.loc[negative & ~later_common & ~later_selective, CLASS] = BACKGROUND
    kept = frame[frame[CLASS].ne("")].reset_index(drop=True)
    if not kept[CLASS].eq(FOREGROUND).any():
        raise ValueError("prospective_genes: no gene became common essential")
    return kept


def source_predictions(
    predictor_id: str,
    *,
    split_strategy: str,
    methods_directory: str | Path | None = None,
    root: str | Path | None = None,
) -> list[pd.DataFrame]:
    frames = []
    for outer_fold in range(FOLD_COUNT):
        if predictor_id in BASELINE_IDS:
            frame = baseline_predictions(
                predictor_id,
                PROSPECTIVE_COHORT,
                split_strategy=split_strategy,
                outer_fold=outer_fold,
                root=root,
            )
        else:
            directory = model_directory(
                (PROJECT_ROOT / "methods" if methods_directory is None else Path(methods_directory))
                / predictor_id,
                PROSPECTIVE_COHORT,
                split_strategy=split_strategy,
                outer_fold=outer_fold,
            )
            frame = read_predictions(
                directory / PREDICTION_FILES["blind"],
                cohort_id=PROSPECTIVE_COHORT,
                split_strategy=split_strategy,
                outer_fold=outer_fold,
                role=TEST_ROLE,
                root=root,
            )
        frames.append(frame[[GENE_KEY, SCORE]].reset_index(drop=True))
    return frames


def scored_frames(frames: Sequence[pd.DataFrame], genes: pd.DataFrame) -> list[pd.DataFrame]:
    class_of = genes.set_index(GENE_KEY)[CLASS]
    kept = []
    seen = 0
    for index, frame in enumerate(frames):
        classes = class_of.reindex(pd.Index(frame[GENE_KEY]))
        mask = classes.notna().to_numpy()
        seen += int(mask.sum())
        kept.append(
            pd.DataFrame(
                {
                    GENE_KEY: frame[GENE_KEY].to_numpy("object")[mask],
                    "label": (classes.to_numpy("object")[mask] == FOREGROUND).astype("int64"),
                    SCORE: frame[SCORE].to_numpy("float64")[mask],
                    FOLD: np.full(int(mask.sum()), index, dtype="int64"),
                }
            )
        )
    if seen != len(genes):
        raise ValueError(
            f"scored_frames: the folds score {seen} of the {len(genes)} genes the "
            f"validation names"
        )
    return kept


