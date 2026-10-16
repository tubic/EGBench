"""GCNN-SFM sequence encoding, network and training loop.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from egbench.contracts import SEED

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
except ImportError as exc:
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None
    WeightedRandomSampler = None
    _TORCH_IMPORT_ERROR: Exception | None = exc
else:
    _TORCH_IMPORT_ERROR = None


KMER_SIZE = 3
DISTANCE_GROUP_COUNT = 3
FEATURE_NODE_COUNT = 4**KMER_SIZE
PAIR_NODE_COUNT = 4 ** (2 * KMER_SIZE)
EDGE_COUNT = 2 * PAIR_NODE_COUNT
VECTOR_LENGTH = DISTANCE_GROUP_COUNT * PAIR_NODE_COUNT
NODE_HIDDEN_DIM = 3
GRAPH_HIDDEN_DIM = 128
CNN_HIDDEN_DIM = 64
GRAPH_LAYER_COUNT = 4
CNN_LAYER_COUNT = 3
CNN_KERNEL_SIZE = 8
FULLY_CONNECTED_DIM = 100
FORMAL_BATCH_SIZE = 64

_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "official_c_golden.json"
SOURCE_COMMIT = "77a81ad7427d415caaf5bc4911ceec4fbcd1f137"
SOURCE_C_SHA256 = "86076d988e039adacdde9301405d3e1adf926c91810ca2c547867f862922781e"
_OFFICIAL_C_CASES = {
    "ACGTACGTACGT": (
        300.0,
        "008b748ca30b76ef8a85ea9f039ed110b9b453d765ee6cf159d754d3d9795664",
    ),
    "AAACCCGGGTTTAAACCCGGGTTT": (
        300.0,
        "05d158f04078d1a6a86ea6fe4712d645b0c5237b3888bdcf4085a266be55709d",
    ),
    "ACGTNNNACGTACGT": (
        199.16666666666666,
        "f39b49262985e9ee2a172cc66fb92b6b262a741daedf5a21922649b7908f8571",
    ),
    "TTTTTTTTTTTTTTT": (
        300.0,
        "8b29df4b6577e43e978020a0f7150c1e61d3464b8bc44ac9968a693a572c08d7",
    ),
}
_BASE_TO_DIGIT = str.maketrans({"A": "0", "C": "1", "G": "2", "T": "3"})
_DISTANCE_GROUPS = (
    tuple(range(0, 1)),
    tuple(range(1, 2)),
    tuple(range(2, 3)),
    tuple(range(3, 5)),
    tuple(range(5, 9)),
    tuple(range(9, 17)),
    tuple(range(17, 33)),
    tuple(range(33, 65)),
)


@dataclass(frozen=True)
class EncodedGraph:
    pair_node_features: np.ndarray
    feature_node_features: np.ndarray
    edge_index: np.ndarray
    vector: np.ndarray


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 10
    device: str = "cuda"


@dataclass(frozen=True)
class FitResult:
    model: Any
    threshold: float
    best_epoch: int
    best_validation_loss: float
    history: tuple[dict[str, float], ...]


def architecture_contract() -> dict[str, Any]:
    return {
        "k": KMER_SIZE,
        "d": DISTANCE_GROUP_COUNT,
        "encoded_vector_length": VECTOR_LENGTH,
        "feature_nodes": FEATURE_NODE_COUNT,
        "pair_nodes": PAIR_NODE_COUNT,
        "directed_edges": EDGE_COUNT,
        "pair_node_feature_dimension": DISTANCE_GROUP_COUNT,
        "feature_node_feature_dimension": 1,
        "bidirectional_graphsage_layers": GRAPH_LAYER_COUNT,
        "graph_hidden_dimension": GRAPH_HIDDEN_DIM,
        "conv1d_layers": CNN_LAYER_COUNT,
        "conv1d_channels": CNN_HIDDEN_DIM,
        "conv1d_kernel": CNN_KERNEL_SIZE,
        "fully_connected_dimension": FULLY_CONNECTED_DIM,
        "output": "two_logits",
    }


def _numeric_locations(sequence: str, k: int) -> tuple[list[int], int]:
    if not isinstance(sequence, str):
        raise TypeError("DNA sequence must be a string")
    normalized = sequence.upper()
    original_length = len(normalized)
    translated = normalized.translate(_BASE_TO_DIGIT)
    numeric = "".join(character for character in translated if character.isdigit())
    locations = [
        int(numeric[index : index + k], 4) for index in range(len(numeric) - k + 1)
    ]
    return locations, original_length


def _location_matrix(
    locations: Sequence[int],
    distances: Sequence[int],
    *,
    k: int,
    original_length: int,
) -> np.ndarray:
    matrix = np.zeros((4**k, 4**k), dtype=np.float64)
    denominator = 0
    for distance in distances:
        for index in range(0, len(locations) - k - distance):
            matrix[locations[index], locations[index + k + distance]] += 1
        denominator = int(denominator + (original_length - 2 * k - distance + 1.0))
    if denominator <= 0:
        raise ValueError("sequence is too short for the fixed k=3,d=3 encoder")
    return (matrix / denominator).reshape(-1)


def matrix_encoding(sequence: str, *, k: int = 3, d: int = 3) -> np.ndarray:

    if k != KMER_SIZE or d != DISTANCE_GROUP_COUNT:
        raise ValueError("formal GCNN-SFM encoding is fixed to k=3,d=3")
    locations, original_length = _numeric_locations(sequence, k)
    components = [
        _location_matrix(
            locations,
            _DISTANCE_GROUPS[group_index],
            k=k,
            original_length=original_length,
        )
        for group_index in range(d)
    ]
    encoded = np.hstack(components) * 100
    if encoded.shape != (VECTOR_LENGTH,):
        raise AssertionError("GCNN-SFM encoder produced an invalid vector length")
    return encoded


def build_edge_index() -> np.ndarray:

    edge_index = np.empty((2, EDGE_COUNT), dtype=np.int64)
    for pair_index in range(PAIR_NODE_COUNT):
        left_feature = pair_index // FEATURE_NODE_COUNT
        right_feature = pair_index % FEATURE_NODE_COUNT
        edge_index[:, 2 * pair_index] = (left_feature, pair_index)
        edge_index[:, 2 * pair_index + 1] = (right_feature, pair_index)
    return edge_index


def encode_graph(sequence: str) -> EncodedGraph:
    vector = matrix_encoding(sequence)
    distance_matrices = vector.reshape(
        DISTANCE_GROUP_COUNT,
        FEATURE_NODE_COUNT,
        FEATURE_NODE_COUNT,
    )
    pair_node_features = vector.reshape(DISTANCE_GROUP_COUNT, PAIR_NODE_COUNT).T
    feature_node_features = (
        distance_matrices[0]
        .sum(axis=1)
        .reshape(
            FEATURE_NODE_COUNT,
            1,
        )
    )
    return EncodedGraph(
        pair_node_features=pair_node_features,
        feature_node_features=feature_node_features,
        edge_index=build_edge_index(),
        vector=vector,
    )


def verify_official_c_fixture(path: str | Path = _FIXTURE_PATH) -> dict[str, Any]:
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata_checks = {
        "source_commit": fixture.get("source_commit") == SOURCE_COMMIT,
        "source_c_sha256": fixture.get("source_c_sha256") == SOURCE_C_SHA256,
        "case_count": len(fixture.get("cases", [])) == len(_OFFICIAL_C_CASES),
    }
    checks: list[dict[str, Any]] = []
    for case in fixture["cases"]:
        sequence = case["sequence"]
        expected = _OFFICIAL_C_CASES.get(sequence)
        encoded = matrix_encoding(case["sequence"])
        digest = hashlib.sha256(encoded.tobytes()).hexdigest()
        bound_to_code = bool(
            expected is not None
            and case.get("vector_length") == VECTOR_LENGTH
            and case.get("vector_sum") == expected[0]
            and case.get("vector_sha256") == expected[1]
        )
        checks.append(
            {
                "sequence": sequence,
                "sha256": digest,
                "expected_sha256": case["vector_sha256"],
                "sum": float(np.sum(encoded)),
                "expected_sum": case["vector_sum"],
                "bound_to_code_constants": bound_to_code,
                "passed": bool(
                    bound_to_code
                    and encoded.shape == (VECTOR_LENGTH,)
                    and digest == case["vector_sha256"]
                    and np.isclose(float(np.sum(encoded)), case["vector_sum"])
                ),
            }
        )
    return {
        "source_commit": fixture["source_commit"],
        "source_c_sha256": fixture["source_c_sha256"],
        "metadata_checks": metadata_checks,
        "all_passed": all(metadata_checks.values())
        and set(case["sequence"] for case in fixture["cases"]) == set(_OFFICIAL_C_CASES)
        and all(item["passed"] for item in checks),
        "checks": checks,
    }


if nn is not None:

    class BipartiteMeanSAGE(nn.Module):

        def __init__(self, source_dim: int, target_dim: int, output_dim: int) -> None:
            super().__init__()
            self.neighbor = nn.Linear(source_dim, output_dim, bias=True)
            self.root = nn.Linear(target_dim, output_dim, bias=False)

        def forward(
            self,
            source: Any,
            target: Any,
            edge_index: Any,
        ) -> Any:
            source_index, target_index = edge_index
            messages = source.index_select(1, source_index)
            aggregated = source.new_zeros(
                (source.shape[0], target.shape[1], source.shape[2])
            )
            aggregated.index_add_(1, target_index, messages)
            counts = torch.bincount(target_index, minlength=target.shape[1]).clamp_min(
                1
            )
            aggregated = aggregated / counts.to(aggregated.dtype).view(1, -1, 1)
            return self.neighbor(aggregated) + self.root(target)

    class GCNNSFM(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.pair_projection = nn.LazyLinear(PAIR_NODE_COUNT * NODE_HIDDEN_DIM)
            self.feature_projection = nn.Linear(
                FEATURE_NODE_COUNT,
                FEATURE_NODE_COUNT * NODE_HIDDEN_DIM,
            )
            self.forward_sage = nn.ModuleList()
            self.backward_sage = nn.ModuleList()
            for layer_index in range(GRAPH_LAYER_COUNT):
                if layer_index == 0:
                    self.forward_sage.append(
                        BipartiteMeanSAGE(
                            NODE_HIDDEN_DIM, NODE_HIDDEN_DIM, GRAPH_HIDDEN_DIM
                        )
                    )
                    self.backward_sage.append(
                        BipartiteMeanSAGE(
                            GRAPH_HIDDEN_DIM, NODE_HIDDEN_DIM, GRAPH_HIDDEN_DIM
                        )
                    )
                else:
                    self.forward_sage.append(
                        BipartiteMeanSAGE(
                            GRAPH_HIDDEN_DIM,
                            GRAPH_HIDDEN_DIM,
                            GRAPH_HIDDEN_DIM,
                        )
                    )
                    self.backward_sage.append(
                        BipartiteMeanSAGE(
                            GRAPH_HIDDEN_DIM,
                            GRAPH_HIDDEN_DIM,
                            GRAPH_HIDDEN_DIM,
                        )
                    )
            self.graph_norms = nn.ModuleList(
                nn.LayerNorm(GRAPH_HIDDEN_DIM) for _ in range(GRAPH_LAYER_COUNT - 1)
            )
            self.convolutions = nn.ModuleList(
                [
                    nn.Conv1d(GRAPH_HIDDEN_DIM, CNN_HIDDEN_DIM, CNN_KERNEL_SIZE),
                    nn.Conv1d(CNN_HIDDEN_DIM, CNN_HIDDEN_DIM, CNN_KERNEL_SIZE),
                    nn.Conv1d(CNN_HIDDEN_DIM, CNN_HIDDEN_DIM, CNN_KERNEL_SIZE),
                ]
            )
            self.classifier_hidden = nn.LazyLinear(FULLY_CONNECTED_DIM)
            self.classifier_output = nn.Linear(FULLY_CONNECTED_DIM, 2)
            edge_index = torch.as_tensor(build_edge_index(), dtype=torch.long)
            self.register_buffer("forward_edges", edge_index[:, 0::2], persistent=False)
            self.register_buffer(
                "backward_edges",
                edge_index[[1, 0], 1::2],
                persistent=False,
            )

        def forward(self, pair_features: Any, feature_features: Any) -> Any:
            if pair_features.ndim != 3 or tuple(pair_features.shape[1:]) != (
                PAIR_NODE_COUNT,
                DISTANCE_GROUP_COUNT,
            ):
                raise ValueError("pair features must have shape [batch, 4096, 3]")
            if feature_features.ndim != 3 or tuple(feature_features.shape[1:]) != (
                FEATURE_NODE_COUNT,
                1,
            ):
                raise ValueError("feature-node features must have shape [batch, 64, 1]")
            batch_size = pair_features.shape[0]
            pair_hidden = self.pair_projection(
                pair_features.reshape(batch_size, -1)
            ).reshape(
                batch_size,
                PAIR_NODE_COUNT,
                NODE_HIDDEN_DIM,
            )
            feature_hidden = self.feature_projection(
                feature_features.reshape(batch_size, FEATURE_NODE_COUNT)
            ).reshape(batch_size, FEATURE_NODE_COUNT, NODE_HIDDEN_DIM)

            for layer_index in range(GRAPH_LAYER_COUNT):
                pair_hidden = F.relu(
                    self.forward_sage[layer_index](
                        feature_hidden,
                        pair_hidden,
                        self.forward_edges,
                    )
                )
                pair_hidden = F.dropout(pair_hidden, p=0.2, training=self.training)
                feature_hidden = F.relu(
                    self.backward_sage[layer_index](
                        pair_hidden,
                        feature_hidden,
                        self.backward_edges,
                    )
                )
                feature_hidden = F.dropout(
                    feature_hidden, p=0.2, training=self.training
                )
                if layer_index < GRAPH_LAYER_COUNT - 1:
                    pair_hidden = self.graph_norms[layer_index](pair_hidden)
                    feature_hidden = self.graph_norms[layer_index](feature_hidden)

            hidden = pair_hidden.transpose(1, 2)
            for layer_index, convolution in enumerate(self.convolutions):
                hidden = F.relu(convolution(hidden))
                if layer_index > 0:
                    hidden = F.dropout(hidden, p=0.2, training=self.training)
            hidden = F.relu(self.classifier_hidden(hidden.flatten(start_dim=1)))
            return self.classifier_output(hidden)

else:

    class BipartiteMeanSAGE:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("PyTorch is required") from _TORCH_IMPORT_ERROR

    class GCNNSFM:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "GCNN-SFM requires PyTorch; install environment.yaml"
            ) from _TORCH_IMPORT_ERROR


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "training requires PyTorch; install environment.yaml"
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


def training_sample_weights(train_labels: Sequence[int]) -> np.ndarray:

    raw_labels = np.asarray(train_labels)
    if (
        raw_labels.ndim != 1
        or len(raw_labels) == 0
        or not np.isfinite(raw_labels).all()
        or not np.isin(raw_labels, [0, 1]).all()
    ):
        raise ValueError("training labels must be a non-empty binary vector")
    labels = raw_labels.astype(np.int64)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("weighted sampling requires both classes")
    return (1.0 / counts[labels]).astype(np.float64)


def select_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(scores, dtype=np.float64)
    if y.shape != probability.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("validation labels and probabilities must align")
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in np.arange(0.01, 1.0, 0.01):
        accuracy = float(np.mean((probability >= threshold) == y))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(round(threshold, 2))
    return best_threshold


def _validate_arrays(
    pair_features: np.ndarray,
    feature_features: np.ndarray,
    labels: Sequence[int],
    role: str,
) -> np.ndarray:
    raw_labels = np.asarray(labels)
    labels_array = raw_labels.astype(np.int64)
    if pair_features.shape != (
        len(labels_array),
        PAIR_NODE_COUNT,
        DISTANCE_GROUP_COUNT,
    ):
        raise ValueError(f"{role} pair features must have shape [n,4096,3]")
    if feature_features.shape != (len(labels_array), FEATURE_NODE_COUNT, 1):
        raise ValueError(f"{role} feature-node features must have shape [n,64,1]")
    if not np.isfinite(pair_features).all() or not np.isfinite(feature_features).all():
        raise ValueError(f"{role} features contain non-finite values")
    if not np.isfinite(raw_labels).all() or not np.isin(raw_labels, [0, 1]).all():
        raise ValueError(f"{role} labels must be binary")
    return labels_array


def predict_scores(
    model: Any,
    pair_features: np.ndarray,
    feature_features: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    _require_torch()
    dummy = np.zeros(len(pair_features), dtype=np.int64)
    _validate_arrays(pair_features, feature_features, dummy, "prediction")
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(pair_features, dtype=torch.float32),
            torch.as_tensor(feature_features, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for pair_batch, feature_batch in loader:
            logits = model(pair_batch.to(device), feature_batch.to(device))
            chunks.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(chunks).astype(np.float64, copy=False)


def fit(
    model: Any,
    train_pair_features: np.ndarray,
    train_feature_features: np.ndarray,
    train_labels: Sequence[int],
    validation_pair_features: np.ndarray,
    validation_feature_features: np.ndarray,
    validation_labels: Sequence[int],
    *,
    config: TrainConfig = TrainConfig(),
) -> FitResult:

    _require_torch()
    source_gate = verify_official_c_fixture()
    if not source_gate["all_passed"]:
        raise RuntimeError(f"official C encoder fixture gate failed: {source_gate}")
    train_y = _validate_arrays(
        train_pair_features,
        train_feature_features,
        train_labels,
        "train",
    )
    validation_y = _validate_arrays(
        validation_pair_features,
        validation_feature_features,
        validation_labels,
        "validation",
    )
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_random_seed(SEED)
    for module in model.modules():
        if module is not model and hasattr(module, "reset_parameters"):
            module.reset_parameters()

    generator = torch.Generator().manual_seed(SEED)
    weights = torch.as_tensor(training_sample_weights(train_y), dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(train_y),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(train_pair_features, dtype=torch.float32),
            torch.as_tensor(train_feature_features, dtype=torch.float32),
            torch.as_tensor(train_y, dtype=torch.long),
        ),
        batch_size=config.batch_size,
        sampler=sampler,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(validation_pair_features, dtype=torch.float32),
            torch.as_tensor(validation_feature_features, dtype=torch.float32),
            torch.as_tensor(validation_y, dtype=torch.long),
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        model(
            torch.as_tensor(
                train_pair_features[:1], dtype=torch.float32, device=device
            ),
            torch.as_tensor(
                train_feature_features[:1],
                dtype=torch.float32,
                device=device,
            ),
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_loss = np.inf
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(config.max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for pair_batch, feature_batch, label_batch in train_loader:
            logits = model(pair_batch.to(device), feature_batch.to(device))
            loss = F.cross_entropy(logits, label_batch.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(label_batch)
            train_count += len(label_batch)

        model.eval()
        validation_loss_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for pair_batch, feature_batch, label_batch in validation_loader:
                logits = model(pair_batch.to(device), feature_batch.to(device))
                batch_loss = F.cross_entropy(
                    logits,
                    label_batch.to(device),
                    reduction="sum",
                )
                validation_loss_sum += float(batch_loss)
                validation_count += len(label_batch)
        validation_loss = validation_loss_sum / validation_count
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss_sum / train_count,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
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
    validation_scores = predict_scores(
        model,
        validation_pair_features,
        validation_feature_features,
        device=str(device),
        batch_size=config.batch_size,
    )
    threshold = select_threshold(validation_y, validation_scores)
    return FitResult(
        model=model,
        threshold=threshold,
        best_epoch=best_epoch,
        best_validation_loss=float(best_loss),
        history=tuple(history),
    )


