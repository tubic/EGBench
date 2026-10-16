"""Removes the genes outside a fold's training set from the training graph of the network methods."""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from egbench.contracts import ContractError

EDGE_COLUMNS: tuple[str, str] = ("gene_key_a", "gene_key_b")


def perturb_edges(
    edges: pd.DataFrame,
    *,
    columns: Sequence[str] = EDGE_COLUMNS,
    training_nodes: Sequence[str] | None = None,
) -> pd.DataFrame:
    if training_nodes is None:
        return edges
    left, right = columns
    missing = [name for name in (left, right) if name not in edges.columns]
    if missing:
        raise ContractError(
            f"perturb_edges: the edge table has no {missing} column, it holds "
            f"{list(edges.columns)}"
        )
    allowed = set(map(str, training_nodes))
    keep = edges[left].astype(str).isin(allowed) & edges[right].astype(str).isin(allowed)
    return edges.loc[keep].reset_index(drop=True)
