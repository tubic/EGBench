"""BiNGO graph attention network, adversarial training step and threshold selection.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import copy
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, matthews_corrcoef
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from egbench.contracts import SEED


BINGO_MODEL_NAME = "esm2_t33_650M_UR50D"
ESM_LAYER = 33
ESM_EMBEDDING_DIM = 1280
MAX_PROTEIN_LENGTH = 1000
CONTACT_RETAIN_FRACTION = 0.20
SOURCE_COMMIT = "fe8d579b6c1a0ea68a1c4f6a996b0037796a0375"
CACHE_SCHEMA = "bingo_esm2_residue_graph_v1"


def high_contact_edges(
    contacts: torch.Tensor | np.ndarray,
    retain_fraction: float = CONTACT_RETAIN_FRACTION,
) -> tuple[torch.Tensor, float]:

    matrix = torch.as_tensor(contacts, dtype=torch.float32, device="cpu")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("contact map must be square")
    if not torch.isfinite(matrix).all():
        raise ValueError("contact map contains NaN or infinite values")
    if not 0.0 < retain_fraction <= 1.0:
        raise ValueError("retain_fraction must be in (0, 1]")
    n = int(matrix.shape[0])
    if n == 0:
        raise ValueError("contact map is empty")
    if n == 1:
        return torch.empty((2, 0), dtype=torch.long), float("nan")

    row, col = torch.triu_indices(n, n, offset=1)
    values = matrix[row, col]
    keep = max(1, math.ceil(values.numel() * retain_fraction))
    order = torch.argsort(values, descending=True, stable=True)[:keep]
    selected_row = row[order]
    selected_col = col[order]
    edge_index = torch.stack(
        [
            torch.cat([selected_row, selected_col]),
            torch.cat([selected_col, selected_row]),
        ],
        dim=0,
    ).to(dtype=torch.long)
    threshold = float(values[order[-1]].item())
    return edge_index, threshold


def architecture_contract() -> dict[str, Any]:
    return {
        "esm_model": BINGO_MODEL_NAME,
        "esm_layer": ESM_LAYER,
        "residue_embedding_dim": ESM_EMBEDDING_DIM,
        "maximum_protein_length": MAX_PROTEIN_LENGTH,
        "contact_retain_fraction": CONTACT_RETAIN_FRACTION,
        "contact_direction": "largest_probabilities",
        "edge_directionality": "bidirectional",
        "gat_1": {"in": 1280, "out": 1280, "heads": 2},
        "gat_2": {"in": 2560, "out": 1280, "heads": 1},
        "pool": "global_max",
        "classifier": [1280, 16, 1],
        "dropout": 0.5,
    }


class BingoGAT(nn.Module):

    def __init__(
        self,
        *,
        embedding_dim: int = ESM_EMBEDDING_DIM,
        hidden_dim: int | None = None,
        heads: int = 2,
        dropout: float = 0.5,
        classifier_hidden: int = 16,
    ) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import GATConv, global_max_pool
        except ImportError as exc:
            raise RuntimeError("Bingo requires torch-geometric") from exc
        hidden = embedding_dim if hidden_dim is None else hidden_dim
        self.dropout_probability = dropout
        self._global_max_pool = global_max_pool
        self.gat1 = GATConv(
            embedding_dim,
            hidden,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.gat2 = GATConv(
            hidden * heads,
            hidden,
            heads=1,
            concat=True,
            dropout=dropout,
        )
        self.fc1 = nn.Linear(hidden, classifier_hidden)
        self.fc2 = nn.Linear(classifier_hidden, 1)

    def forward(self, data: Any) -> torch.Tensor:
        x = nn.functional.dropout(
            data.x, p=self.dropout_probability, training=self.training
        )
        x = nn.functional.elu(self.gat1(x, data.edge_index))
        x = nn.functional.dropout(x, p=self.dropout_probability, training=self.training)
        x = torch.relu(self.gat2(x, data.edge_index))
        x = self._global_max_pool(x, data.batch)
        x = torch.relu(self.fc1(x))
        return self.fc2(x).view(-1)


class FastGradientPerturbation:

    def __init__(
        self,
        model: nn.Module,
        *,
        prefixes: tuple[str, ...] = ("gat1.", "gat2."),
    ) -> None:
        self.model = model
        self.prefixes = prefixes
        self._backup: dict[str, torch.Tensor] = {}
        self.last_delta_norm = 0.0
        self.last_attacked_parameters: tuple[str, ...] = ()

    def attack(self, epsilon: float = 1.0) -> tuple[str, ...]:
        if epsilon <= 0:
            raise ValueError("FGM epsilon must be positive")
        if self._backup:
            raise RuntimeError("FGM attack called before restore")
        attacked: list[str] = []
        total_squared = 0.0
        for name, parameter in self.model.named_parameters():
            if not name.startswith(self.prefixes) or parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            norm = torch.linalg.vector_norm(gradient)
            if not torch.isfinite(norm) or float(norm.item()) == 0.0:
                continue
            self._backup[name] = parameter.detach().clone()
            delta = epsilon * gradient / norm
            with torch.no_grad():
                parameter.add_(delta)
            attacked.append(name)
            total_squared += float(torch.sum(delta.float() ** 2).item())
        self.last_attacked_parameters = tuple(attacked)
        self.last_delta_norm = math.sqrt(total_squared)
        return self.last_attacked_parameters

    def restore(self) -> None:
        if not self._backup:
            raise RuntimeError("FGM restore called without an active attack")
        for name, parameter in self.model.named_parameters():
            if name not in self._backup:
                continue
            with torch.no_grad():
                parameter.copy_(self._backup[name])
            if not torch.equal(parameter.detach(), self._backup[name]):
                raise RuntimeError(f"FGM failed to restore {name}")
        self._backup = {}


@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 40
    batch_size: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 5e-4
    dropout: float = 0.5
    fgm_epsilon: float = 1.0
    early_stopping_patience: int = 8
    num_workers: int = 2

    def validate(self) -> None:
        if self.max_epochs < 1 or self.batch_size < 1:
            raise ValueError("max_epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer hyperparameters")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.fgm_epsilon <= 0 or self.early_stopping_patience < 1:
            raise ValueError("invalid FGM or early-stopping configuration")
        if self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")


@dataclass(frozen=True)
class FitResult:
    model: BingoGAT
    history: tuple[dict[str, float | int], ...]
    best_epoch: int
    best_validation_aupr: float
    validation_threshold: float
    fgm_attacked_parameters: tuple[str, ...]
    fgm_delta_norm: float
    config: dict[str, Any]


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _epoch_train(
    model: BingoGAT,
    loader: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    fgm: FastGradientPerturbation,
    device: torch.device,
    epsilon: float,
) -> tuple[float, float]:
    model.train()
    total_base = 0.0
    total_adv = 0.0
    total_rows = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        labels = batch.y.view(-1).float()
        optimizer.zero_grad(set_to_none=True)
        base_loss = loss_fn(model(batch), labels)
        base_loss.backward()
        attacked = fgm.attack(epsilon)
        if attacked:
            adversarial_loss = loss_fn(model(batch), labels)
            adversarial_loss.backward()
            fgm.restore()
            adversarial_value = float(adversarial_loss.detach().item())
        else:
            adversarial_value = float(base_loss.detach().item())
        optimizer.step()
        rows = int(labels.numel())
        total_base += float(base_loss.detach().item()) * rows
        total_adv += adversarial_value * rows
        total_rows += rows
    if total_rows == 0:
        raise RuntimeError("empty training loader")
    return total_base / total_rows, total_adv / total_rows


def predict_probabilities(
    model: BingoGAT,
    loader: Iterable[Any],
    device: torch.device,
    *,
    require_labels: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    _, probabilities, labels = predict_scores(
        model, loader, device, require_labels=require_labels
    )
    return probabilities, labels


def predict_scores(
    model: BingoGAT,
    loader: Iterable[Any],
    device: torch.device,
    *,
    require_labels: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:

    model.eval()
    raw_scores: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            logits = model(batch)
            raw_scores.append(logits.cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
            if require_labels:
                if getattr(batch, "y", None) is None:
                    raise ValueError("labelled prediction loader lacks labels")
                labels.append(batch.y.view(-1).cpu().numpy())
    if not scores:
        raise RuntimeError("empty prediction loader")
    y_raw = np.concatenate(raw_scores).astype(np.float64, copy=False)
    y_score = np.concatenate(scores).astype(np.float64, copy=False)
    y_true = (
        np.concatenate(labels).astype(np.int64, copy=False) if require_labels else None
    )
    return y_raw, y_score, y_true


def mcc_threshold(labels: np.ndarray, scores: np.ndarray) -> float:

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    if labels.size == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be a nonempty binary array")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("scores must be finite probabilities in [0, 1]")
    unique_scores = np.unique(scores)
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    candidates = np.unique(np.concatenate(([0.0], midpoints, [1.0])))
    best_mcc = float("-inf")
    best_threshold = 0.0
    for threshold in candidates:
        mcc = float(matthews_corrcoef(labels, scores >= threshold))
        if mcc > best_mcc + 1e-12 or (
            math.isclose(mcc, best_mcc, rel_tol=0.0, abs_tol=1e-12)
            and float(threshold) > best_threshold
        ):
            best_mcc = mcc
            best_threshold = float(threshold)
    return best_threshold


def fit(
    train_loader: Iterable[Any],
    validation_loader: Iterable[Any],
    device: torch.device,
    config: TrainConfig | None = None,
    *,
    model: BingoGAT | None = None,
) -> FitResult:
    config = config or TrainConfig()
    config.validate()
    seed_everything(SEED)
    model = model or BingoGAT(dropout=config.dropout)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    fgm = FastGradientPerturbation(model)

    best_aupr = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_threshold = 0.5
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.max_epochs + 1):
        base_loss, adversarial_loss = _epoch_train(
            model,
            train_loader,
            optimizer,
            loss_fn,
            fgm,
            device,
            config.fgm_epsilon,
        )
        validation_scores, validation_labels = predict_probabilities(
            model, validation_loader, device, require_labels=True
        )
        if validation_labels is None:
            raise RuntimeError("validation labels are unavailable")
        validation_aupr = float(
            average_precision_score(validation_labels, validation_scores)
        )
        threshold = mcc_threshold(validation_labels, validation_scores)
        history.append(
            {
                "epoch": epoch,
                "train_base_loss": base_loss,
                "train_adversarial_loss": adversarial_loss,
                "validation_aupr": validation_aupr,
                "validation_threshold": threshold,
            }
        )
        if validation_aupr > best_aupr:
            best_aupr = validation_aupr
            best_epoch = epoch
            best_threshold = threshold
            best_state = copy.deepcopy(
                {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                }
            )
            stale = 0
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return FitResult(
        model=model,
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_aupr=best_aupr,
        validation_threshold=best_threshold,
        fgm_attacked_parameters=fgm.last_attacked_parameters,
        fgm_delta_norm=fgm.last_delta_norm,
        config=asdict(config),
    )
