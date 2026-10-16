"""eDNN-EG implementation package.
Imported by run.py; not run directly.
"""
from .adapter import EDNNEGAdapter, ScoreBatch
from .calibration import PlattScoreCalibrator
from .contracts import (
    BASE_LEARNER_ORDER,
    FORMAL_CALIBRATION_SPEC,
    FORMAL_CNN_TRAINING_SPEC,
    FORMAL_FASTTEXT_SPEC,
    FORMAL_MLP_TRAINING_SPEC,
)
from .errors import ContractViolation, MissingRequiredDependency
from .frozen_data import FrozenPartition, FrozenSplit, load_frozen_split

__all__ = [
    "BASE_LEARNER_ORDER",
    "ContractViolation",
    "EDNNEGAdapter",
    "FORMAL_CALIBRATION_SPEC",
    "FORMAL_CNN_TRAINING_SPEC",
    "FORMAL_FASTTEXT_SPEC",
    "FORMAL_MLP_TRAINING_SPEC",
    "FrozenPartition",
    "FrozenSplit",
    "MissingRequiredDependency",
    "PlattScoreCalibrator",
    "ScoreBatch",
    "load_frozen_split",
]
