"""Platt calibration of the eDNN-EG ensemble score."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from .contracts import FORMAL_CALIBRATION_SPEC
from .errors import ContractViolation


class PlattScoreCalibrator:
    def __init__(self, *, seed: int) -> None:
        spec = FORMAL_CALIBRATION_SPEC
        self.seed = int(seed)
        self.estimator = LogisticRegression(
            C=spec.c,
            solver=spec.solver,
            penalty=spec.penalty,
            tol=spec.tolerance,
            max_iter=spec.max_iterations,
            fit_intercept=spec.fit_intercept,
            random_state=self.seed,
        )
        self.metadata: dict[str, object] = {}

    @staticmethod
    def _raw_matrix(raw_scores: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
        if not len(values) or not np.isfinite(values).all():
            raise ContractViolation("raw linear-regression scores must be finite and non-empty")
        return values.reshape(-1, 1)

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> "PlattScoreCalibrator":
        matrix = self._raw_matrix(raw_scores)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(y) != len(matrix) or set(np.unique(y).tolist()) != {0, 1}:
            raise ContractViolation(
                "Platt calibration requires aligned validation scores with both binary classes"
            )
        self.estimator.fit(matrix, y)
        if int(np.max(self.estimator.n_iter_)) >= FORMAL_CALIBRATION_SPEC.max_iterations:
            raise ContractViolation("Platt calibration did not converge")
        self.metadata = {
            "specification": asdict(FORMAL_CALIBRATION_SPEC),
            "seed": self.seed,
            "fit_rows": len(y),
            "fit_class_counts": {
                "negative": int(np.sum(y == 0)),
                "positive": int(np.sum(y == 1)),
            },
            "coefficient": float(self.estimator.coef_[0, 0]),
            "intercept": float(self.estimator.intercept_[0]),
            "iterations": int(np.max(self.estimator.n_iter_)),
            "test_labels_used": False,
        }
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        if not hasattr(self.estimator, "classes_"):
            raise ContractViolation("Platt calibration transform called before fit")
        matrix = self._raw_matrix(raw_scores)
        classes = list(self.estimator.classes_)
        probabilities = np.asarray(
            self.estimator.predict_proba(matrix)[:, classes.index(1)], dtype=np.float64
        )
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
        ):
            raise ContractViolation("Platt calibration returned invalid probabilities")
        return probabilities
