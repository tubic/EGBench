"""EssTFNet model built on the upstream time-frequency modules in code/upstream.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn


UPSTREAM_ROOT = Path(__file__).resolve().parent / "upstream"
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from module.ATFNet import Model as ATFNet
from module.initial import initial_model
from module.lossfunc import FocalLoss
from utils.params import ATFNetConfigs


DNA_ALPHABET = "ATCG"
PROTEIN_ALPHABET = "RSPFICETYKGANUMLWDVHQ"
LANGUAGE_MODEL_PROJECTION = 256


@dataclass(frozen=True)
class Hyperparameters:

    max_len: int = 1800
    protein_max_len: int = 600
    kernel_size: int = 5
    head_num: int = 1
    hidden_dim: int = 32
    layer_num: int = 2
    attn_drop: float = 0.0
    lstm_drop: float = 0.0
    linear_drop: float = 0.2
    d_model: int = 1
    factor: int = 2
    n_heads: int = 1
    e_layers: int = 1
    d_ff: int = 4
    atf_drop: float = 0.1
    fnet_d_ff: int = 8
    fnet_d_model: int = 8
    complex_dropout: float = 0.0
    fnet_layers: int = 1
    is_emb: bool = True
    epoch_num: int = 500
    batch_size: int = 32
    patience: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4


def one_hot_table(alphabet: str) -> torch.Tensor:

    return torch.cat(
        [torch.zeros(1, len(alphabet)), torch.eye(len(alphabet))]
    ).to(torch.float32)


class SequenceProcessor(nn.Module):

    def __init__(
        self,
        input_dim: int,
        parameters: Hyperparameters,
        *,
        config: ATFNetConfigs | None,
    ) -> None:
        super().__init__()
        self.cnn = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=parameters.kernel_size,
            padding="same",
        )
        self.attention_is_time_frequency = config is not None
        if config is not None:
            self.attention = ATFNet(config)
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=parameters.head_num,
                dropout=parameters.attn_drop,
                batch_first=True,
            )
            self.layer_norm = nn.LayerNorm(input_dim)
        self.bilstm = nn.LSTM(
            input_dim,
            parameters.hidden_dim,
            bidirectional=True,
            batch_first=True,
            num_layers=parameters.layer_num,
            dropout=parameters.lstm_drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = functional.relu(self.cnn(x.permute(0, 2, 1)))
        x = residual + x.permute(0, 2, 1)
        if self.attention_is_time_frequency:
            x, _ = self.attention(x)
        else:
            attended, _ = self.attention(x, x, x)
            x = x + self.layer_norm(attended)
        x, _ = self.bilstm(x)
        return torch.max(x, dim=-1).values


class EssTFNet(nn.Module):

    def __init__(
        self,
        parameters: Hyperparameters,
        *,
        feature_dimensions: int,
        language_model_dimensions: int,
    ) -> None:
        super().__init__()
        self.register_buffer("dna_one_hot", one_hot_table(DNA_ALPHABET))
        self.register_buffer("protein_one_hot", one_hot_table(PROTEIN_ALPHABET))
        dna_config = ATFNetConfigs(
            parameters.max_len,
            len(DNA_ALPHABET),
            parameters.max_len,
            parameters.d_model,
            parameters.factor,
            parameters.n_heads,
            parameters.e_layers,
            parameters.d_ff,
            parameters.atf_drop,
            parameters.fnet_d_ff,
            parameters.fnet_d_model,
            parameters.complex_dropout,
            parameters.fnet_layers,
            parameters.is_emb,
        )
        self.dna_processor = SequenceProcessor(
            len(DNA_ALPHABET), parameters, config=dna_config
        )
        self.protein_processor = SequenceProcessor(
            len(PROTEIN_ALPHABET), parameters, config=None
        )
        self.language_model_linear = nn.Linear(
            language_model_dimensions, LANGUAGE_MODEL_PROJECTION
        )
        concatenated = (
            parameters.max_len
            + parameters.protein_max_len
            + feature_dimensions
            + LANGUAGE_MODEL_PROJECTION
        )
        self.concatenated_dimensions = concatenated
        self.shared_layers = nn.Sequential(
            nn.Linear(concatenated, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(parameters.linear_drop),
        )
        self.output_head = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid())

    def forward(
        self,
        dna_tokens: torch.Tensor,
        protein_tokens: torch.Tensor,
        features: torch.Tensor,
        language_model_embedding: torch.Tensor,
    ) -> torch.Tensor:
        dna = self.dna_processor(self.dna_one_hot[dna_tokens.long()])
        protein = self.protein_processor(self.protein_one_hot[protein_tokens.long()])
        projected = self.language_model_linear(language_model_embedding)
        merged = torch.cat([dna, protein, features, projected], dim=1)
        return self.output_head(self.shared_layers(merged)).reshape(-1)


__all__ = [
    "DNA_ALPHABET",
    "EssTFNet",
    "FocalLoss",
    "Hyperparameters",
    "PROTEIN_ALPHABET",
    "initial_model",
]
