"""DeepCellEss network (convolution, BiLSTM and attention over one-hot protein sequences) and its training loop.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from egbench.contracts import SEED

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None
    _TORCH_IMPORT_ERROR: Exception | None = exc
else:
    _TORCH_IMPORT_ERROR = None


AMINO_ACID_ORDER = "RSPFICETYKGANUMLWDVHQ"
MAX_PROTEIN_LENGTH = 600
EMBEDDING_DIM = 21
EXPECTED_TRUNCATED_GENES = 1_659
FORMAL_BATCH_SIZE = 256
PAD_TOKEN = "*"

_AA_TO_INDEX = {residue: index for index, residue in enumerate(AMINO_ACID_ORDER)}


@dataclass(frozen=True)
class EncodingResult:
    values: np.ndarray
    original_length: int
    kept_length: int
    truncated: bool


@dataclass(frozen=True)
class EncodingSummary:
    sample_count: int
    truncated_count: int
    maximum_original_length: int


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    max_epochs: int = 500
    patience: int = 30
    device: str = "cuda"


@dataclass(frozen=True)
class FitResult:
    model: Any
    threshold: float
    best_epoch: int
    best_validation_auc: float
    positive_weight: float
    history: tuple[dict[str, float], ...]


def architecture_contract() -> dict[str, Any]:

    return {
        "protein_length": MAX_PROTEIN_LENGTH,
        "amino_acid_order": AMINO_ACID_ORDER,
        "embedding_dim": EMBEDDING_DIM,
        "padding": "right_zero_one_hot",
        "truncation": "n_terminal_prefix",
        "convolution": {"channels": 21, "kernel_size": 3, "padding": "same"},
        "attention": {"embedding_dim": 21, "heads": 3},
        "bilstm": {"input_dim": 21, "hidden_dim": 64, "layers": 1},
        "pooling": "adaptive_max",
        "output": "single_sigmoid",
    }


def _normalize_sequence(sequence: str) -> str:
    if not isinstance(sequence, str):
        raise TypeError("protein sequence must be a string")
    normalized = "".join(sequence.split()).upper()
    invalid = sorted(set(normalized).difference(_AA_TO_INDEX))
    if invalid:
        raise ValueError(
            "unsupported protein residue(s): "
            + ", ".join(invalid)
            + f"; accepted alphabet is {AMINO_ACID_ORDER}"
        )
    return normalized


def encode_protein(sequence: str) -> EncodingResult:

    normalized = _normalize_sequence(sequence)
    original_length = len(normalized)
    kept = normalized[:MAX_PROTEIN_LENGTH]
    values = np.zeros((MAX_PROTEIN_LENGTH, EMBEDDING_DIM), dtype=np.float32)
    for position, residue in enumerate(kept):
        values[position, _AA_TO_INDEX[residue]] = 1.0
    return EncodingResult(
        values=values,
        original_length=original_length,
        kept_length=len(kept),
        truncated=original_length > MAX_PROTEIN_LENGTH,
    )


def encode_proteins(sequences: Iterable[str]) -> tuple[np.ndarray, EncodingSummary]:
    encoded = [encode_protein(sequence) for sequence in sequences]
    if not encoded:
        raise ValueError("at least one protein sequence is required")
    values = np.stack([item.values for item in encoded], axis=0)
    summary = EncodingSummary(
        sample_count=len(encoded),
        truncated_count=sum(item.truncated for item in encoded),
        maximum_original_length=max(item.original_length for item in encoded),
    )
    return values, summary


if nn is not None:

    class DeepCellEss(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.text_cnn = nn.Conv1d(
                in_channels=EMBEDDING_DIM,
                out_channels=EMBEDDING_DIM,
                kernel_size=3,
                padding="same",
            )
            self.attention = nn.MultiheadAttention(
                embed_dim=EMBEDDING_DIM,
                num_heads=3,
                dropout=0.0,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(EMBEDDING_DIM)
            self.bilstm = nn.LSTM(
                input_size=EMBEDDING_DIM,
                hidden_size=64,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
            self.output = nn.Linear(128, 1)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.xavier_normal_(self.text_cnn.weight)
            nn.init.zeros_(self.text_cnn.bias)
            nn.init.xavier_normal_(self.attention.out_proj.weight)
            nn.init.zeros_(self.attention.out_proj.bias)
            for name, parameter in self.bilstm.named_parameters():
                if "weight_ih" in name or "weight_hh" in name:
                    nn.init.orthogonal_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
            nn.init.xavier_normal_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def forward(self, features: Any, return_attention: bool = False) -> Any:
            if features.ndim != 3 or tuple(features.shape[1:]) != (
                MAX_PROTEIN_LENGTH,
                EMBEDDING_DIM,
            ):
                raise ValueError(
                    "DeepCellEss expects [batch, 600, 21] one-hot features"
                )
            residual = features
            convolved = F.relu(self.text_cnn(features.transpose(1, 2))).transpose(1, 2)
            hidden = residual + convolved
            attended, attention_weights = self.attention(hidden, hidden, hidden)
            hidden = hidden + self.attention_norm(attended)
            hidden, _ = self.bilstm(hidden)
            pooled = torch.max(hidden, dim=1).values
            probability = torch.sigmoid(self.output(pooled))
            if return_attention:
                return probability, attention_weights
            return probability

else:

    class DeepCellEss:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "DeepCellEss requires PyTorch; install the environment in environment.yaml"
            ) from _TORCH_IMPORT_ERROR


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "training requires PyTorch; install the environment in environment.yaml"
        ) from _TORCH_IMPORT_ERROR


def deterministic_kernels() -> None:

    _require_torch()
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def set_random_seed(seed: int) -> None:

    _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic_kernels()


def _validate_features(features: np.ndarray, labels: Sequence[int], role: str) -> None:
    if features.ndim != 3 or features.shape[1:] != (
        MAX_PROTEIN_LENGTH,
        EMBEDDING_DIM,
    ):
        raise ValueError(f"{role} features must have shape [n, 600, 21]")
    if len(features) != len(labels):
        raise ValueError(f"{role} features and labels have different row counts")
    if not np.isfinite(features).all():
        raise ValueError(f"{role} features contain non-finite values")
    raw_labels = np.asarray(labels)
    if not np.isfinite(raw_labels).all() or not np.isin(raw_labels, [0, 1]).all():
        raise ValueError(f"{role} labels must be binary")


def binary_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:

    y = np.asarray(labels, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float64)
    if y.shape != score.shape or y.ndim != 1:
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both classes")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[y == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def select_threshold(
    validation_labels: Sequence[int],
    validation_scores: Sequence[float],
) -> tuple[float, float]:

    labels = np.asarray(validation_labels, dtype=np.int64)
    scores = np.asarray(validation_scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1 or len(labels) == 0:
        raise ValueError("validation labels and scores must be aligned and non-empty")
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in np.arange(0.01, 1.0, 0.01):
        accuracy = float(np.mean((scores >= threshold).astype(np.int64) == labels))
        if accuracy > best_accuracy:
            best_threshold = float(round(threshold, 2))
            best_accuracy = accuracy
    return best_threshold, best_accuracy


def predict_scores(
    model: Any,
    features: np.ndarray,
    *,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    _require_torch()
    dummy_labels = np.zeros(len(features), dtype=np.float32)
    _validate_features(features, dummy_labels, "prediction")
    loader = DataLoader(
        TensorDataset(torch.as_tensor(features, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            scores = model(batch.to(device)).reshape(-1)
            output.append(scores.detach().cpu().numpy())
    return np.concatenate(output).astype(np.float64, copy=False)


def fit(
    model: Any,
    train_features: np.ndarray,
    train_labels: Sequence[int],
    validation_features: np.ndarray,
    validation_labels: Sequence[int],
    *,
    config: TrainConfig = TrainConfig(),
) -> FitResult:

    _require_torch()
    train_y = np.asarray(train_labels, dtype=np.float32)
    validation_y = np.asarray(validation_labels, dtype=np.float32)
    _validate_features(train_features, train_y, "train")
    _validate_features(validation_features, validation_y, "validation")
    positive_count = int(train_y.sum())
    negative_count = len(train_y) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("training requires both classes")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    set_random_seed(SEED)

    device = torch.device(config.device)
    model.reset_parameters()
    model = model.to(device)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(train_features, dtype=torch.float32),
            torch.as_tensor(train_y, dtype=torch.float32),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    positive_weight = negative_count / positive_count
    best_auc = -np.inf
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(config.max_epochs):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            probabilities = model(batch_features).reshape(-1)
            base_loss = F.binary_cross_entropy(
                probabilities,
                batch_labels,
                reduction="none",
            )
            weights = torch.where(
                batch_labels > 0.5,
                torch.as_tensor(positive_weight, device=device),
                torch.as_tensor(1.0, device=device),
            )
            loss = (base_loss * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_labels)
            sample_count += len(batch_labels)

        validation_scores = predict_scores(
            model,
            validation_features,
            device=str(device),
            batch_size=config.batch_size,
        )
        validation_auc = binary_roc_auc(validation_y.astype(int), validation_scores)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": loss_sum / sample_count,
                "validation_auc": validation_auc,
            }
        )
        if validation_auc > best_auc:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    final_validation_scores = predict_scores(
        model,
        validation_features,
        device=str(device),
        batch_size=config.batch_size,
    )
    threshold, _ = select_threshold(validation_y.astype(int), final_validation_scores)
    return FitResult(
        model=model,
        threshold=threshold,
        best_epoch=best_epoch,
        best_validation_auc=float(best_auc),
        positive_weight=float(positive_weight),
        history=tuple(history),
    )


