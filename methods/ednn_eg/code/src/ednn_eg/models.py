"""eDNN-EG base learners (k-nearest neighbours, random forest, RBF SVM, MLP and CNN)."""
from __future__ import annotations

import os
from collections.abc import MutableMapping
from typing import Protocol

import numpy as np

from .contracts import (
    BASE_LEARNER_ORDER,
    FORMAL_CNN_TRAINING_SPEC,
    FORMAL_MLP_TRAINING_SPEC,
    validate_cuda_device_name,
)
from .errors import ContractViolation, MissingRequiredDependency


class ProbabilityLearner(Protocol):
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None: ...

    def predict_positive(self, features: np.ndarray) -> np.ndarray: ...


def configure_cuda_determinism_environment() -> None:
    expected = ":4096:8"
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed not in {None, expected}:
        raise ContractViolation(
            "CUBLAS_WORKSPACE_CONFIG differs from the frozen :4096:8 contract"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected


class SklearnProbabilityLearner:
    def __init__(self, estimator) -> None:
        self.estimator = estimator

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.estimator.fit(features, labels)

    def predict_positive(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(self.estimator.predict_proba(features), dtype=np.float64)
        classes = list(self.estimator.classes_)
        if 1 not in classes:
            raise ContractViolation("base learner did not fit the positive class")
        result = probabilities[:, classes.index(1)]
        _validate_probability_vector(result, len(features))
        return result


def _require_torch():
    try:
        import torch
        from torch import nn
    except Exception as exc:
        raise MissingRequiredDependency(
            "torch with CUDA support is required for MLP and CNN; no CPU fallback is permitted"
        ) from exc
    return torch, nn


def require_cuda(device: str):
    validate_cuda_device_name(device)
    torch, _ = _require_torch()
    if not torch.cuda.is_available():
        raise MissingRequiredDependency(
            "torch.cuda.is_available() is false; no CPU fallback is permitted"
        )
    try:
        selected = torch.device(device)
        torch.empty(1, device=selected)
    except Exception as exc:
        raise MissingRequiredDependency(f"CUDA device {device!r} cannot allocate a tensor") from exc
    return torch, selected


def build_mlp_module():
    torch, nn = _require_torch()

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(50, 1),
            )

        def forward(self, inputs):
            return self.network(inputs).squeeze(1)

    return MLP()


def build_cnn_module():
    torch, nn = _require_torch()

    class CNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
            self.pool1 = nn.MaxPool1d(kernel_size=2)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
            self.pool2 = nn.MaxPool1d(kernel_size=2)
            self.relu = nn.ReLU()
            self.dense = nn.Linear(64 * 23, 64)
            self.dropout = nn.Dropout(0.1)
            self.output = nn.Linear(64, 2)

        def forward(self, inputs):
            values = inputs.unsqueeze(1)
            values = self.pool1(self.relu(self.conv1(values)))
            values = self.pool2(self.relu(self.conv2(values)))
            values = values.flatten(start_dim=1)
            values = self.dropout(self.relu(self.dense(values)))
            return self.output(values)

    return CNN()


class TorchBinaryLearner:
    def __init__(self, *, architecture: str, seed: int, device: str) -> None:
        if architecture not in {"mlp", "cnn"}:
            raise ContractViolation(f"unknown neural architecture: {architecture}")
        configure_cuda_determinism_environment()
        self.architecture = architecture
        self.seed = int(seed)
        self.torch, self.device = require_cuda(device)
        self.torch.manual_seed(self.seed)
        self.torch.cuda.manual_seed_all(self.seed)
        self.torch.backends.cudnn.benchmark = False
        self.torch.backends.cudnn.deterministic = True
        self.torch.use_deterministic_algorithms(True)
        self.model = (build_mlp_module() if architecture == "mlp" else build_cnn_module()).to(
            self.device
        )
        self.training_summary: dict[str, object] = {}

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        torch = self.torch
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if x.ndim != 2 or x.shape[1] != 100 or len(x) != len(y):
            raise ContractViolation("deep base learners require an n by 100 feature matrix")
        if set(np.unique(y).tolist()) != {0, 1}:
            raise ContractViolation("deep base learner training requires both classes")

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        x_tensor = torch.from_numpy(x)
        if self.architecture == "mlp":
            spec = FORMAL_MLP_TRAINING_SPEC
            y_tensor = torch.from_numpy(y.astype(np.float32))
            loss_fn = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=spec.learning_rate,
                betas=(spec.beta1, spec.beta2),
                eps=spec.epsilon,
                weight_decay=spec.weight_decay,
                amsgrad=spec.amsgrad,
            )
            epochs = spec.epochs
            batch_size = spec.batch_size
        else:
            spec = FORMAL_CNN_TRAINING_SPEC
            y_tensor = torch.from_numpy(y)
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adadelta(
                self.model.parameters(),
                lr=spec.learning_rate,
                rho=spec.rho,
                eps=spec.epsilon,
                weight_decay=spec.weight_decay,
            )
            epochs = spec.epochs
            batch_size = spec.batch_size

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_tensor, y_tensor),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        final_loss = None
        self.model.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch_x)
                loss = loss_fn(logits, batch_y)
                if not torch.isfinite(loss):
                    raise ContractViolation(f"{self.architecture} produced a non-finite loss")
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach().cpu())
        self.training_summary = {
            "architecture": self.architecture,
            "device": str(self.device),
            "epochs": epochs,
            "batch_size": batch_size,
            "final_batch_loss": final_loss,
            "optimizer": (
                {
                    "name": "Adam",
                    "learning_rate": FORMAL_MLP_TRAINING_SPEC.learning_rate,
                    "betas": [
                        FORMAL_MLP_TRAINING_SPEC.beta1,
                        FORMAL_MLP_TRAINING_SPEC.beta2,
                    ],
                    "epsilon": FORMAL_MLP_TRAINING_SPEC.epsilon,
                    "weight_decay": FORMAL_MLP_TRAINING_SPEC.weight_decay,
                    "amsgrad": FORMAL_MLP_TRAINING_SPEC.amsgrad,
                }
                if self.architecture == "mlp"
                else {
                    "name": "Adadelta",
                    "learning_rate": FORMAL_CNN_TRAINING_SPEC.learning_rate,
                    "rho": FORMAL_CNN_TRAINING_SPEC.rho,
                    "epsilon": FORMAL_CNN_TRAINING_SPEC.epsilon,
                    "weight_decay": FORMAL_CNN_TRAINING_SPEC.weight_decay,
                }
            ),
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        }

    def state(self) -> dict:

        return {
            name: tensor.detach().cpu()
            for name, tensor in self.model.state_dict().items()
        }

    def load_state(self, state) -> None:

        self.model.load_state_dict(state)
        self.model.to(self.device)

    def predict_positive(self, features: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 100:
            raise ContractViolation("deep base learners require an n by 100 feature matrix")
        loader = torch.utils.data.DataLoader(
            torch.from_numpy(x), batch_size=256, shuffle=False, num_workers=0, pin_memory=True
        )
        outputs: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for batch_x in loader:
                logits = self.model(batch_x.to(self.device, non_blocking=True))
                if self.architecture == "mlp":
                    probabilities = torch.sigmoid(logits)
                else:
                    probabilities = torch.softmax(logits, dim=1)[:, 1]
                outputs.append(probabilities.detach().cpu().numpy())
        result = np.concatenate(outputs).astype(np.float64, copy=False)
        _validate_probability_vector(result, len(x))
        return result


def make_official_base_learners(
    *, seed: int, device: str
) -> MutableMapping[str, ProbabilityLearner]:
    configure_cuda_determinism_environment()
    require_cuda(device)
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
    except Exception as exc:
        raise MissingRequiredDependency(
            "scikit-learn==1.6.1 is required; no base-learner fallback is permitted"
        ) from exc

    learners: MutableMapping[str, ProbabilityLearner] = {
        "knn": SklearnProbabilityLearner(
            KNeighborsClassifier(n_neighbors=10, metric="euclidean")
        ),
        "random_forest": SklearnProbabilityLearner(
            RandomForestClassifier(
                n_estimators=500,
                criterion="gini",
                max_features="sqrt",
                bootstrap=True,
                random_state=seed,
                n_jobs=-1,
            )
        ),
        "rbf_svm": SklearnProbabilityLearner(
            SVC(
                C=23768.0,
                gamma=0.001953125,
                kernel="rbf",
                probability=True,
                random_state=seed,
            )
        ),
        "mlp": TorchBinaryLearner(architecture="mlp", seed=seed, device=device),
        "cnn": TorchBinaryLearner(architecture="cnn", seed=seed, device=device),
    }
    if tuple(learners) != BASE_LEARNER_ORDER:
        raise ContractViolation("base learner order differs from the frozen contract")
    return learners


def _validate_probability_vector(values: np.ndarray, expected_rows: int) -> None:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(vector) != expected_rows:
        raise ContractViolation(
            f"base learner returned {len(vector)} probabilities for {expected_rows} rows"
        )
    if not np.isfinite(vector).all():
        raise ContractViolation("base learner returned non-finite probabilities")
    if np.any(vector < 0.0) or np.any(vector > 1.0):
        raise ContractViolation("base learner probabilities fall outside [0, 1]")
