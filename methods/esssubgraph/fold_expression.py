"""Fit the TCGA expression principal components on the fold's train role and project all genes.
Imported by run.py; not run directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def build_expression_fold(
    *,
    expression_path: Path,
    assignments_path: Path | None = None,
    fold_genes_path: Path | None = None,
    cohort_id: str,
    split_strategy: str,
    outer_fold: int,
    output_directory: Path,
    n_components: int = 50,
) -> dict[str, object]:
    if (assignments_path is None) == (fold_genes_path is None):
        raise ValueError("provide exactly one of assignments_path or fold_genes_path")
    if fold_genes_path is not None:
        context = pd.read_parquet(fold_genes_path, columns=["gene_key", "role"]).copy()
    else:
        assert assignments_path is not None
        assignments = pd.read_parquet(assignments_path)
        context = assignments.loc[
            assignments["cohort_id"].astype(str).eq(cohort_id)
            & assignments["split_strategy"].astype(str).eq(split_strategy)
            & pd.to_numeric(assignments["outer_fold"], errors="coerce").eq(outer_fold)
        ].copy()
    if set(context["role"].astype(str)) != {"train", "validation", "test"}:
        raise ValueError("expression fold requires train, validation and test roles")
    expression = pd.read_parquet(expression_path)
    if "gene_key" not in expression or expression["gene_key"].duplicated().any():
        raise ValueError("expression matrix requires unique gene_key values")
    expression = expression.set_index(expression["gene_key"].astype(str)).drop(
        columns="gene_key"
    )
    role_keys = {
        role: sorted(
            context.loc[context["role"].eq(role), "gene_key"].astype(str).tolist()
        )
        for role in ("train", "validation", "test")
    }
    missing = sorted(set().union(*map(set, role_keys.values())).difference(expression.index))
    if missing:
        raise ValueError(f"{len(missing)} split genes lack expression rows")
    train_raw = expression.loc[role_keys["train"]].to_numpy(dtype=np.float64, copy=True)
    if np.isinf(train_raw).any() or (train_raw[~np.isnan(train_raw)] < 0).any():
        raise ValueError("expression values must be non-negative finite values or NaN")
    train_log = np.log2(train_raw + 1.0)
    if np.isnan(train_log).all(axis=0).any():
        raise ValueError("one or more TCGA samples are all-missing in the training genes")
    medians = np.nanmedian(train_log, axis=0)
    train_imputed = np.where(np.isnan(train_log), medians, train_log)
    means = train_imputed.mean(axis=0)
    scales = train_imputed.std(axis=0, ddof=0)
    scales[scales == 0.0] = 1.0
    standardized_train = (train_imputed - means) / scales
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(standardized_train)
    output_directory.mkdir(parents=True, exist_ok=True)
    columns = [f"expression_pc_{index:03d}" for index in range(1, n_components + 1)]
    output_paths: dict[str, Path] = {}
    for role, keys in role_keys.items():
        raw = expression.loc[keys].to_numpy(dtype=np.float64, copy=True)
        logged = np.log2(raw + 1.0)
        imputed = np.where(np.isnan(logged), medians, logged)
        transformed = pca.transform((imputed - means) / scales)
        if not np.isfinite(transformed).all():
            raise ValueError(f"non-finite PCA values in {role}")
        frame = pd.DataFrame(transformed, columns=columns)
        frame.insert(0, "gene_key", keys)
        path = output_directory / f"expression_pca_50_{role}.parquet"
        frame.to_parquet(path, index=False)
        output_paths[role] = path
    state_path = output_directory / "expression_pca_50_state.npz"
    np.savez_compressed(
        state_path,
        medians=medians,
        means=means,
        scales=scales,
        pca_mean=pca.mean_,
        components=pca.components_,
        explained_variance=pca.explained_variance_,
    )
    manifest = {
        "schema_version": "1.0",
        "cohort_id": cohort_id,
        "split_strategy": split_strategy,
        "outer_fold": outer_fold,
        "n_components": n_components,
        "solver": "full",
        "fit_role": "train",
        "fit_gene_count": len(role_keys["train"]),
        "role_counts": {role: len(keys) for role, keys in role_keys.items()},
        "log_transform": "log2(x+1)",
        "imputation": "per-TCGA-sample median fitted on train genes",
        "standardization": "per-TCGA-sample train mean and population standard deviation",
    }
    manifest_path = output_directory / "expression_pca_50_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
