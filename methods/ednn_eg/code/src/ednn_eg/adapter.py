"""eDNN-EG stacked ensemble: fastText representation, five base learners and the linear meta-learner."""
from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedGroupKFold

from .contracts import BASE_LEARNER_ORDER, architecture_manifest, normalize_cds
from .errors import ContractViolation
from .fasttext_backend import EmbeddingBackend, OfficialFastTextBackend
from .models import (
    ProbabilityLearner,
    SklearnProbabilityLearner,
    TorchBinaryLearner,
    make_official_base_learners,
)

FastTextFactory = Callable[[int, Path], EmbeddingBackend]
LearnerFactory = Callable[[int, str], MutableMapping[str, ProbabilityLearner]]

FASTTEXT_FILE = "fasttext.bin"
BASE_LEARNERS_FILE = "base_learners.joblib"
META_LEARNER_FILE = "meta_learner.joblib"
TORCH_LEARNER_FILES = {"mlp": "mlp.pt", "cnn": "cnn.pt"}
SKLEARN_LEARNERS = tuple(
    name for name in BASE_LEARNER_ORDER if name not in TORCH_LEARNER_FILES
)


@dataclass(frozen=True)
class ScoreBatch:
    ids: tuple[str, ...]
    base_scores: np.ndarray
    meta_scores: np.ndarray


class EDNNEGAdapter:

    def __init__(
        self,
        *,
        work_dir: str | Path,
        seed: int,
        device: str = "cuda",
        fasttext_thread: int = 1,
        fasttext_factory: FastTextFactory | None = None,
        learner_factory: LearnerFactory | None = None,
        allow_test_components: bool = False,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.seed = int(seed)
        self.device = device
        self.fasttext_thread = int(fasttext_thread)
        if self.fasttext_thread != 1:
            raise ContractViolation(
                "formal fastText training requires fasttext_thread=1"
            )

        custom_components = fasttext_factory is not None or learner_factory is not None
        if custom_components and not allow_test_components:
            raise ContractViolation(
                "custom components are restricted to explicit tests and cannot produce formal results"
            )
        if custom_components and (fasttext_factory is None or learner_factory is None):
            raise ContractViolation("tests must supply both component factories")
        self.formal_components = not custom_components
        self._fasttext_factory = fasttext_factory or self._make_fasttext
        self._learner_factory = learner_factory or self._make_learners

        self.final_fasttext_: EmbeddingBackend | None = None
        self.final_learners_: MutableMapping[str, ProbabilityLearner] | None = None
        self.meta_learner_: LinearRegression | None = None
        self.train_ids_: tuple[str, ...] = ()
        self.oof_base_scores_: np.ndarray | None = None
        self.meta_fitted_scores_: np.ndarray | None = None
        self.audit_record_: dict[str, object] = {}

    def _make_fasttext(self, seed: int, work_dir: Path) -> EmbeddingBackend:
        return OfficialFastTextBackend(
            seed=seed, work_dir=work_dir, thread=self.fasttext_thread
        )

    def _make_learners(
        self, seed: int, device: str
    ) -> MutableMapping[str, ProbabilityLearner]:
        return make_official_base_learners(seed=seed, device=device)

    def fit(
        self,
        sequences: Mapping[str, str],
        labels: Mapping[str, int],
        *,
        groups: Mapping[str, str] | None = None,
        split_strategy: str = "family_5fold",
    ) -> "EDNNEGAdapter":
        ids, normalized_sequences, y = self._validate_training_input(sequences, labels)
        group_values = self._validate_inner_groups(ids, groups, split_strategy)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        oof = np.full((len(ids), len(BASE_LEARNER_ORDER)), np.nan, dtype=np.float64)
        folds: list[dict[str, object]] = []
        splitter = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=self.seed
        )
        split_iterator = splitter.split(np.zeros(len(y)), y, group_values)
        splitter_kind = "StratifiedGroupKFold"

        for fold_index, (train_index, validation_index) in enumerate(split_iterator):
            train_ids = [ids[index] for index in train_index]
            validation_ids = [ids[index] for index in validation_index]
            overlap = sorted(set(train_ids) & set(validation_ids))
            if overlap:
                raise ContractViolation(f"inner fold {fold_index} ID overlap: {overlap[:5]}")
            for role, fold_labels in (
                ("train", y[train_index]),
                ("validation", y[validation_index]),
            ):
                if set(np.unique(fold_labels).tolist()) != {0, 1}:
                    raise ContractViolation(
                        f"inner fold {fold_index} {role} partition lacks a binary class"
                    )

            train_groups: list[str] = []
            validation_groups: list[str] = []
            group_overlap: list[str] = []
            if group_values is not None:
                train_groups = sorted(set(group_values[train_index].tolist()))
                validation_groups = sorted(set(group_values[validation_index].tolist()))
                group_overlap = sorted(set(train_groups) & set(validation_groups))
                if split_strategy == "family_5fold" and group_overlap:
                    raise ContractViolation(
                        f"inner family fold {fold_index} cluster overlap: {group_overlap[:5]}"
                    )

            backend = self._fasttext_factory(
                self.seed, self.work_dir / f"inner_fold_{fold_index}"
            )
            backend.fit(
                train_ids,
                [normalized_sequences[index] for index in train_index],
                y[train_index],
            )
            train_features = backend.transform(
                train_ids, [normalized_sequences[index] for index in train_index]
            )
            validation_features = backend.transform(
                validation_ids, [normalized_sequences[index] for index in validation_index]
            )
            self._validate_embedding_matrix(train_features, len(train_index))
            self._validate_embedding_matrix(validation_features, len(validation_index))

            learners = self._learner_factory(self.seed, self.device)
            self._validate_learner_suite(learners)
            for column, name in enumerate(BASE_LEARNER_ORDER):
                learner = learners[name]
                learner.fit(train_features, y[train_index])
                scores = np.asarray(
                    learner.predict_positive(validation_features), dtype=np.float64
                ).reshape(-1)
                self._validate_base_scores(scores, len(validation_index), name)
                oof[validation_index, column] = scores

            folds.append(
                {
                    "fold": fold_index,
                    "train_ids": train_ids,
                    "validation_ids": validation_ids,
                    "overlap_count": 0,
                    "train_groups": train_groups,
                    "validation_groups": validation_groups,
                    "group_overlap_count": len(group_overlap),
                    "fasttext": dict(getattr(backend, "metadata", {})),
                    "base_learners": list(BASE_LEARNER_ORDER),
                }
            )

        if not np.isfinite(oof).all():
            missing = np.argwhere(~np.isfinite(oof))[:10].tolist()
            raise ContractViolation(f"OOF matrix contains unfilled or non-finite cells: {missing}")

        meta_learner = LinearRegression()
        meta_learner.fit(oof, y)
        meta_fitted = np.asarray(meta_learner.predict(oof), dtype=np.float64).reshape(-1)
        if not np.isfinite(meta_fitted).all():
            raise ContractViolation(
                "linear regression meta learner returned non-finite training fitted scores"
            )

        final_backend = self._fasttext_factory(self.seed, self.work_dir / "outer_train_full")
        final_backend.fit(ids, normalized_sequences, y)
        final_features = final_backend.transform(ids, normalized_sequences)
        self._validate_embedding_matrix(final_features, len(ids))
        final_learners = self._learner_factory(self.seed, self.device)
        self._validate_learner_suite(final_learners)
        final_summaries: dict[str, object] = {}
        for name in BASE_LEARNER_ORDER:
            final_learners[name].fit(final_features, y)
            final_summaries[name] = dict(
                getattr(final_learners[name], "training_summary", {})
            )

        self.final_fasttext_ = final_backend
        self.final_learners_ = final_learners
        self.meta_learner_ = meta_learner
        self.train_ids_ = tuple(ids)
        self.oof_base_scores_ = oof
        self.meta_fitted_scores_ = meta_fitted
        self.audit_record_ = {
            "method_id": "ednn_eg",
            "implementation_class": "adapted_variant_clean_room",
            "formal_components": self.formal_components,
            "seed": self.seed,
            "device": self.device,
            "published_variant": "original_no_smote",
            "split_strategy": split_strategy,
            "inner_splitter": splitter_kind,
            "train_rows": len(ids),
            "train_ids": ids,
            "inner_fold_count": 5,
            "fasttext_fit_count": 6,
            "folds": folds,
            "final_fasttext": dict(getattr(final_backend, "metadata", {})),
            "final_base_training": final_summaries,
            "meta_learner": {
                "type": "LinearRegression",
                "coef": np.asarray(meta_learner.coef_, dtype=float).tolist(),
                "intercept": float(meta_learner.intercept_),
                "input_columns": list(BASE_LEARNER_ORDER),
                "training_score_semantics": "fitted_on_complete_base_oof_matrix",
            },
            "architecture": architecture_manifest(),
        }
        return self

    @staticmethod
    def _validate_inner_groups(
        ids: Sequence[str],
        groups: Mapping[str, str] | None,
        split_strategy: str,
    ) -> np.ndarray | None:
        if split_strategy != "family_5fold":
            raise ContractViolation("split_strategy must be family_5fold")
        if groups is None:
            raise ContractViolation("family_5fold requires a cluster ID for every training gene")
        if set(groups) != set(ids):
            raise ContractViolation(
                "family inner-fold group IDs differ from outer training IDs"
            )
        values = np.asarray([str(groups[gene_id]).strip() for gene_id in ids], dtype=object)
        if any(not value for value in values):
            raise ContractViolation("family inner-fold cluster IDs must be non-empty")
        if len(set(values.tolist())) < 5:
            raise ContractViolation("family inner-fold stacking requires at least five clusters")
        return values

    def predict_base_scores(self, sequences: Mapping[str, str]) -> tuple[tuple[str, ...], np.ndarray]:
        if self.final_fasttext_ is None or self.final_learners_ is None:
            raise ContractViolation("predict called before fit")
        ids = tuple(sequences)
        if not ids:
            raise ContractViolation("prediction input is empty")
        if len(set(ids)) != len(ids):
            raise ContractViolation("prediction identifiers are duplicated")
        normalized = [normalize_cds(sequences[gene_id]) for gene_id in ids]
        features = self.final_fasttext_.transform(ids, normalized)
        self._validate_embedding_matrix(features, len(ids))
        columns = []
        for name in BASE_LEARNER_ORDER:
            scores = np.asarray(
                self.final_learners_[name].predict_positive(features), dtype=np.float64
            ).reshape(-1)
            self._validate_base_scores(scores, len(ids), name)
            columns.append(scores)
        return ids, np.column_stack(columns)

    def predict_scores(self, sequences: Mapping[str, str]) -> ScoreBatch:
        if self.meta_learner_ is None:
            raise ContractViolation("predict called before fit")
        ids, base_scores = self.predict_base_scores(sequences)
        meta_scores = np.asarray(
            self.meta_learner_.predict(base_scores), dtype=np.float64
        ).reshape(-1)
        if not np.isfinite(meta_scores).all():
            raise ContractViolation("linear regression meta learner returned non-finite scores")
        return ScoreBatch(ids=ids, base_scores=base_scores, meta_scores=meta_scores)

    def save(self, directory: str | Path) -> None:

        if (
            self.final_fasttext_ is None
            or self.final_learners_ is None
            or self.meta_learner_ is None
        ):
            raise ContractViolation("save called before fit")
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        self.final_fasttext_.save_model(destination / FASTTEXT_FILE)
        joblib.dump(
            {
                name: self.final_learners_[name].estimator
                for name in SKLEARN_LEARNERS
            },
            destination / BASE_LEARNERS_FILE,
        )
        for name, filename in TORCH_LEARNER_FILES.items():
            learner = self.final_learners_[name]
            learner.torch.save(learner.state(), destination / filename)
        joblib.dump(self.meta_learner_, destination / META_LEARNER_FILE)

    @classmethod
    def load(
        cls, directory: str | Path, *, seed: int, device: str
    ) -> "EDNNEGAdapter":

        source = Path(directory)
        adapter = cls(work_dir=source, seed=seed, device=device)
        backend = OfficialFastTextBackend(
            seed=seed, work_dir=source, thread=adapter.fasttext_thread
        )
        backend.load_model(source / FASTTEXT_FILE)
        estimators = joblib.load(source / BASE_LEARNERS_FILE)
        if tuple(estimators) != SKLEARN_LEARNERS:
            raise ContractViolation(
                f"stored base learners are {tuple(estimators)}; expected "
                f"{SKLEARN_LEARNERS}"
            )
        learners: MutableMapping[str, ProbabilityLearner] = {}
        for name in BASE_LEARNER_ORDER:
            if name in TORCH_LEARNER_FILES:
                learner = TorchBinaryLearner(
                    architecture=name, seed=seed, device=device
                )
                learner.load_state(
                    learner.torch.load(
                        source / TORCH_LEARNER_FILES[name],
                        map_location=learner.device,
                        weights_only=True,
                    )
                )
                learners[name] = learner
            else:
                learners[name] = SklearnProbabilityLearner(estimators[name])
        adapter.final_fasttext_ = backend
        adapter.final_learners_ = learners
        adapter.meta_learner_ = joblib.load(source / META_LEARNER_FILE)
        return adapter

    @staticmethod
    def _validate_embedding_matrix(features: np.ndarray, expected_rows: int) -> None:
        matrix = np.asarray(features)
        if matrix.shape != (expected_rows, 100):
            raise ContractViolation(
                f"fastText feature shape is {matrix.shape}; expected ({expected_rows}, 100)"
            )
        if not np.isfinite(matrix).all():
            raise ContractViolation("fastText feature matrix contains non-finite values")

    @staticmethod
    def _validate_base_scores(scores: np.ndarray, expected_rows: int, name: str) -> None:
        if scores.shape != (expected_rows,):
            raise ContractViolation(
                f"{name} returned shape {scores.shape}; expected ({expected_rows},)"
            )
        if not np.isfinite(scores).all():
            raise ContractViolation(f"{name} returned non-finite probabilities")
        if np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ContractViolation(f"{name} returned values outside [0, 1]")

    @staticmethod
    def _validate_learner_suite(
        learners: MutableMapping[str, ProbabilityLearner]
    ) -> None:
        if tuple(learners) != BASE_LEARNER_ORDER:
            raise ContractViolation(
                f"base learner suite must be {BASE_LEARNER_ORDER}; received {tuple(learners)}"
            )

    @staticmethod
    def _validate_training_input(
        sequences: Mapping[str, str], labels: Mapping[str, int]
    ) -> tuple[list[str], list[str], np.ndarray]:
        sequence_ids = set(sequences)
        label_ids = set(labels)
        if sequence_ids != label_ids:
            raise ContractViolation(
                "outer training sequence and label IDs differ: "
                f"sequence_only={sorted(sequence_ids - label_ids)[:5]}, "
                f"label_only={sorted(label_ids - sequence_ids)[:5]}"
            )
        ids = sorted(sequence_ids)
        if len(ids) < 20:
            raise ContractViolation("strict five-fold stack requires at least 20 outer training rows")
        normalized_sequences = [normalize_cds(sequences[gene_id]) for gene_id in ids]
        y = np.asarray([int(labels[gene_id]) for gene_id in ids], dtype=np.int64)
        if not set(np.unique(y).tolist()).issubset({0, 1}):
            raise ContractViolation("outer training labels must be binary 0/1")
        class_counts = np.bincount(y, minlength=2)
        if np.any(class_counts < 5):
            raise ContractViolation(
                f"strict stratified five-fold stack requires at least five rows per class: {class_counts}"
            )
        return ids, normalized_sequences, y
