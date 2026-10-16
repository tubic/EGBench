"""Fixed eDNN-EG hyperparameters and sequence normalisation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .errors import ContractViolation


@dataclass(frozen=True)
class FastTextSpec:
    supervised: bool = True
    dim: int = 100
    ws: int = 5
    word_ngrams: int = 6
    epoch: int = 200
    learning_rate: float = 0.1
    loss: str = "softmax"
    min_count: int = 1
    min_count_label: int = 1
    min_char_ngram: int = 0
    max_char_ngram: int = 0
    negatives: int = 5
    bucket: int = 2_000_000
    learning_rate_update_rate: int = 100
    sampling_threshold: float = 1e-4
    label_prefix: str = "__label__"
    pretrained_vectors: str = ""

    def assert_formal(self) -> None:
        expected = FastTextSpec()
        if self != expected:
            raise ContractViolation(
                f"fastText parameters differ from the frozen formal contract: "
                f"expected={asdict(expected)}, received={asdict(self)}"
            )

    def as_fasttext_kwargs(self, *, seed: int, thread: int) -> dict[str, Any]:
        self.assert_formal()
        return {
            "dim": self.dim,
            "ws": self.ws,
            "wordNgrams": self.word_ngrams,
            "epoch": self.epoch,
            "lr": self.learning_rate,
            "loss": self.loss,
            "minCount": self.min_count,
            "minCountLabel": self.min_count_label,
            "minn": self.min_char_ngram,
            "maxn": self.max_char_ngram,
            "neg": self.negatives,
            "bucket": self.bucket,
            "lrUpdateRate": self.learning_rate_update_rate,
            "t": self.sampling_threshold,
            "label": self.label_prefix,
            "pretrainedVectors": self.pretrained_vectors,
            "seed": seed,
            "thread": thread,
            "verbose": 0,
        }


@dataclass(frozen=True)
class MLPTrainingSpec:
    epochs: int = 100
    batch_size: int = 20
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False


@dataclass(frozen=True)
class CNNTrainingSpec:
    epochs: int = 50
    batch_size: int = 20
    learning_rate: float = 1.0
    rho: float = 0.9
    epsilon: float = 1e-6
    weight_decay: float = 0.0


@dataclass(frozen=True)
class CalibrationSpec:
    method: str = "platt_logistic_regression"
    fit_partition: str = "outer_validation_predictions_from_outer_train_model"
    solver: str = "lbfgs"
    penalty: str = "l2"
    c: float = 1.0
    tolerance: float = 1e-4
    max_iterations: int = 1000
    fit_intercept: bool = True
    threshold: float = 0.5


FORMAL_FASTTEXT_SPEC = FastTextSpec()
FORMAL_MLP_TRAINING_SPEC = MLPTrainingSpec()
FORMAL_CNN_TRAINING_SPEC = CNNTrainingSpec()
FORMAL_CALIBRATION_SPEC = CalibrationSpec()
BASE_LEARNER_ORDER = ("knn", "random_forest", "rbf_svm", "mlp", "cnn")


def architecture_manifest() -> dict[str, Any]:
    return {
        "fasttext": {
            "supervised": True,
            "dim": 100,
            "ws": 5,
            "wordNgrams": 6,
            "epoch": 200,
            "lr": 0.1,
            "loss": "softmax",
            "minCount": 1,
            "minCountLabel": 1,
            "minn": 0,
            "maxn": 0,
            "neg": 5,
            "bucket": 2_000_000,
            "lrUpdateRate": 100,
            "t": 1e-4,
            "label": "__label__",
            "pretrainedVectors": "",
            "thread": 1,
        },
        "inner_folds": 5,
        "knn": {"n_neighbors": 10, "metric": "euclidean"},
        "random_forest": {
            "n_estimators": 500,
            "criterion": "gini",
            "max_features": "sqrt",
        },
        "rbf_svm": {"C": 23768.0, "gamma": 0.001953125, "probability": True},
        "mlp": {
            "hidden": [100, 50],
            "activation": "relu",
            "dropout_each_hidden": 0.5,
            "output": "sigmoid",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "epochs": 100,
            "batch_size": 20,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "adam_amsgrad": False,
            "device": "cuda",
        },
        "cnn": {
            "layers": ["conv32_k3", "maxpool2", "conv64_k3", "maxpool2", "dense64"],
            "activation": "relu",
            "dropout": 0.1,
            "output": "softmax_2",
            "optimizer": "adadelta",
            "learning_rate": 1.0,
            "rho": 0.9,
            "epsilon": 1e-6,
            "weight_decay": 0.0,
            "epochs": 50,
            "batch_size": 20,
            "device": "cuda",
        },
        "meta_learner": "linear_regression",
        "published_variant": "original_no_smote",
        "score_calibration": asdict(FORMAL_CALIBRATION_SPEC),
        "svm_source_conflict": {
            "selected_C": 23768,
            "paper_table_and_author_code_C": 23768,
            "paper_prose_C": 32768,
        },
    }


def validate_cuda_device_name(device: str) -> None:
    if not device.startswith("cuda"):
        raise ContractViolation(
            f"MLP and CNN require a CUDA device for formal eDNN-EG runs; received {device!r}"
        )


def normalize_cds(sequence: str) -> str:
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ContractViolation("empty CDS sequence")
    invalid = sorted(set(normalized) - {"A", "C", "G", "T"})
    if invalid:
        raise ContractViolation(f"CDS contains unsupported symbols: {invalid}")
    return normalized
