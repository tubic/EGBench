"""Training loop and prediction for DeeplyEssential.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import DeeplyEssentialDNN, TrainConfig, set_deterministic

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from egbench.contracts import SEED


@dataclass(frozen=True)
class TrainingResult:
    model: DeeplyEssentialDNN
    history: tuple[dict[str, float | int], ...]
    best_epoch: int
    best_validation_loss: float
    positive_weight: float
    config: dict[str, Any]


def _loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(features, dtype=np.float32)),
        torch.from_numpy(np.asarray(labels, dtype=np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        generator=generator,
        drop_last=False,
    )


def _epoch_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_rows = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            rows = labels.shape[0]
            total_loss += float(loss.detach().item()) * rows
            total_rows += rows
    if total_rows == 0:
        raise RuntimeError("empty data loader")
    return total_loss / total_rows


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    device: torch.device,
    config: TrainConfig | None = None,
) -> TrainingResult:
    config = config or TrainConfig()
    config.validate()
    set_deterministic(SEED)

    x_train = np.asarray(x_train, dtype=np.float32)
    x_validation = np.asarray(x_validation, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_validation = np.asarray(y_validation, dtype=np.int64)
    for role, features, labels in (
        ("train", x_train, y_train),
        ("validation", x_validation, y_validation),
    ):
        if features.ndim != 2:
            raise ValueError(f"{role} features must be a two-dimensional matrix")
        if labels.ndim != 1 or len(labels) != len(features):
            raise ValueError(f"{role} labels do not align with features")
        if not np.isfinite(features).all():
            raise ValueError(f"{role} features contain NaN or infinite values")
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"{role} labels must contain both binary classes")
    negatives = int(np.count_nonzero(y_train == 0))
    positives = int(np.count_nonzero(y_train == 1))
    if positives == 0 or negatives == 0:
        raise ValueError("training labels must contain both classes")
    positive_weight = negatives / positives

    pin_memory = device.type == "cuda"
    train_loader = _loader(
        x_train,
        y_train,
        config.batch_size,
        True,
        SEED,
        pin_memory,
    )
    validation_loader = _loader(
        x_validation,
        y_validation,
        config.batch_size,
        False,
        SEED,
        pin_memory,
    )
    model = DeeplyEssentialDNN(dropout=config.dropout).to(device)
    optimizer = torch.optim.Adadelta(
        model.parameters(),
        lr=config.learning_rate,
        rho=config.rho,
        eps=config.epsilon,
    )
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.max_epochs + 1):
        train_loss = _epoch_loss(model, train_loader, loss_fn, device, optimizer)
        validation_loss = _epoch_loss(
            model, validation_loader, loss_fn, device, optimizer=None
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return TrainingResult(
        model=model,
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        positive_weight=positive_weight,
        config=asdict(config),
    )


def predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    dummy_labels = np.zeros(len(features), dtype=np.float32)
    loader = _loader(
        features,
        dummy_labels,
        batch_size,
        False,
        seed=SEED,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for batch, _ in loader:
            logits = model(batch.to(device, non_blocking=True))
            output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output).astype(np.float64, copy=False)
