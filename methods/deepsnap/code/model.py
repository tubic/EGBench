"""DeEPsnap multilayer perceptron and its snapshot learning-rate schedule.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import math
from typing import Any

try:
    import torch.nn as nn
except ImportError as exc:
    nn = None
    _TORCH_IMPORT_ERROR: Exception | None = exc
else:
    _TORCH_IMPORT_ERROR = None


def architecture_contract() -> dict[str, Any]:
    return {
        "input_dimensions": 258,
        "hidden_layers": [1024, 512, 256],
        "hidden_activation": "relu",
        "dropout_after_each_hidden": 0.3,
        "output": "single_logit_with_sigmoid_at_inference",
        "optimizer": "Adam",
        "batch_size": 32,
        "epochs": 50,
        "snapshot_cycle_epochs": 5,
        "snapshot_count": 10,
        "initial_learning_rate": 0.001,
        "ensemble": "arithmetic_mean_of_ten_snapshot_probabilities",
    }


def snapshot_learning_rate(epoch: int, *, initial_lr: float = 0.001, cycle: int = 5) -> float:
    if epoch < 0 or cycle <= 0 or initial_lr <= 0:
        raise ValueError("epoch, cycle, and initial learning rate are invalid")
    scaler = math.sin(math.pi * (1 + (epoch % cycle) / (2 * cycle))) + 1
    return scaler * initial_lr


if nn is not None:

    class DeEPsnapMLP(nn.Module):
        def __init__(self, input_dimensions: int = 258) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dimensions, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
            )
            self.reset_parameters()

        def reset_parameters(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

        def forward(self, features: Any) -> Any:
            if features.ndim != 2:
                raise ValueError("DeEPsnap expects a two-dimensional feature matrix")
            return self.network(features).squeeze(-1)

else:

    class DeEPsnapMLP:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("DeEPsnap requires PyTorch") from _TORCH_IMPORT_ERROR
