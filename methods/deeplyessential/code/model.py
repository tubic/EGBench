"""DeeplyEssential feed-forward network.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


INPUT_FEATURES = 89
HIDDEN_WIDTHS = (128, 256, 512, 1024, 1024, 1024)


class DeeplyEssentialDNN(nn.Module):

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        layers: list[nn.Module] = []
        previous = INPUT_FEATURES
        for width in HIDDEN_WIDTHS:
            linear = nn.Linear(previous, width)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.ReLU(), nn.Dropout(dropout)))
            previous = width
        output = nn.Linear(previous, 1)
        nn.init.xavier_uniform_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    max_epochs: int = 100
    dropout: float = 0.3
    learning_rate: float = 1.0
    rho: float = 0.95
    epsilon: float = 1e-7
    early_stopping_patience: int = 15

    def validate(self) -> None:
        if self.batch_size < 1 or self.max_epochs < 1:
            raise ValueError("batch_size and max_epochs must be positive")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if self.learning_rate <= 0 or not 0 < self.rho < 1 or self.epsilon <= 0:
            raise ValueError("invalid Adadelta parameters")


def set_deterministic(seed: int) -> None:

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str, allow_cpu: bool = False) -> torch.device:

    requested = requested.strip().lower()
    if requested == "cpu":
        if not allow_cpu:
            raise RuntimeError("CPU execution is restricted to explicit synthetic smoke tests")
        return torch.device("cpu")
    if not requested.startswith("cuda"):
        raise ValueError("device must be 'cuda', 'cuda:N', or smoke-only 'cpu'")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(requested)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    try:
        torch.empty(1, device=device)
    except Exception as exc:
        raise RuntimeError(f"requested CUDA device is unavailable: {requested}") from exc
    return device
