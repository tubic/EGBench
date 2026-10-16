"""Two-layer graph attention network of EPGAT.
Imported by run.py; not run directly.
"""
from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR: Exception | None = exc
else:
    _TORCH_IMPORT_ERROR = None


def architecture_contract() -> dict[str, Any]:
    return {
        "layers": 2,
        "hidden_units_per_head": 8,
        "hidden_attention_heads": 8,
        "output_units": 1,
        "output_attention_heads": 1,
        "hidden_head_merge": "concatenate",
        "output_head_merge": "mean",
        "activation": "relu_after_hidden_gat",
        "dropout": 0.4,
        "negative_slope": 0.2,
        "attention_parameterization": "one concatenated target-source vector per head",
        "self_loops": True,
        "learning_rate": 0.005,
        "weight_decay": 5e-4,
        "max_epochs": 1_000,
        "early_stopping": (
            "validation class-balanced loss, patience 100 epochs, "
            "parameters of the best epoch restored"
        ),
        "loss": "sum_of_positive_and_negative_class_mean_BCE",
    }


if nn is not None:

    class GATConv(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            heads: int = 1,
            concat: bool = True,
            attention_dropout: float = 0.0,
            negative_slope: float = 0.2,
        ) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.heads = heads
            self.concat = concat
            self.attention_dropout = attention_dropout
            self.negative_slope = negative_slope
            self.weight = nn.Parameter(torch.empty(in_channels, heads * out_channels))
            self.attention = nn.Parameter(torch.empty(1, heads, 2 * out_channels))
            bias_size = heads * out_channels if concat else out_channels
            self.bias = nn.Parameter(torch.zeros(bias_size))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.xavier_uniform_(self.weight)
            bound = (6.0 / (self.attention.size(-2) + self.attention.size(-1))) ** 0.5
            nn.init.uniform_(self.attention, -bound, bound)
            nn.init.zeros_(self.bias)

        def forward(self, features: Any, edge_index: Any) -> Any:
            if features.ndim != 2 or features.shape[1] != self.in_channels:
                raise ValueError("GATConv received an invalid node-feature matrix")
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("edge_index must have shape [2, edges]")
            node_count = features.shape[0]
            source, target = edge_index
            transformed = (features @ self.weight).view(node_count, self.heads, self.out_channels)
            target_values = transformed.index_select(0, target)
            source_values = transformed.index_select(0, source)
            logits = (
                torch.cat([target_values, source_values], dim=-1) * self.attention
            ).sum(dim=-1)
            logits = F.leaky_relu(logits, negative_slope=self.negative_slope)
            expanded_target = target.view(-1, 1).expand(-1, self.heads)
            maxima = logits.new_full((node_count, self.heads), -torch.inf)
            maxima.scatter_reduce_(0, expanded_target, logits, reduce="amax", include_self=True)
            exponentiated = torch.exp(logits - maxima.index_select(0, target))
            denominators = logits.new_zeros((node_count, self.heads))
            denominators.index_add_(0, target, exponentiated)
            attention = exponentiated / denominators.index_select(0, target).clamp_min(1e-12)
            attention = F.dropout(
                attention,
                p=self.attention_dropout,
                training=self.training,
            )
            messages = transformed.index_select(0, source) * attention.unsqueeze(-1)
            aggregated = transformed.new_zeros((node_count, self.heads, self.out_channels))
            aggregated.index_add_(0, target, messages)
            if self.concat:
                return aggregated.reshape(node_count, self.heads * self.out_channels) + self.bias
            return aggregated.mean(dim=1) + self.bias


    class EPGAT(nn.Module):
        def __init__(self, input_dimensions: int, *, dropout: float = 0.4) -> None:
            super().__init__()
            self.dropout = dropout
            self.hidden = GATConv(
                input_dimensions,
                8,
                heads=8,
                concat=True,
                attention_dropout=dropout,
                negative_slope=0.2,
            )
            self.output = GATConv(
                64,
                1,
                heads=1,
                concat=False,
                attention_dropout=dropout,
                negative_slope=0.2,
            )

        def forward(self, features: Any, edge_index: Any) -> Any:
            hidden = F.relu(self.hidden(features, edge_index))
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            return self.output(hidden, edge_index).squeeze(-1)

else:

    class GATConv:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("EPGAT requires PyTorch") from _TORCH_IMPORT_ERROR


    class EPGAT:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("EPGAT requires PyTorch") from _TORCH_IMPORT_ERROR
