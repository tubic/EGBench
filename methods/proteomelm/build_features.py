#!/usr/bin/env python3
"""Write the ProteomeLM-Ess fold views, and rebuild the ProteomeLM layer-8 source matrices.
Rebuild from methods/proteomelm: python build_features.py --dataset deg --dataset depmap
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]

_spec = importlib.util.spec_from_file_location(
    "family_feature_layout", REPO_ROOT / "methods" / "family_feature_layout.py"
)
family_feature_layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(family_feature_layout)
materialize_static_feature_folds = family_feature_layout.materialize_static_feature_folds

sys.path.insert(0, str(REPO_ROOT / "src"))
from egbench.contracts import COHORT_IDS


METHOD_ID = "proteomelm"
FEATURE_DIMENSIONS = 1152
SOURCE_ROOT = METHOD_ROOT / "feature" / "source"
MODEL_ROOT = SOURCE_ROOT / "model_assets" / "proteomelm_l"
ESMC_CACHE = SOURCE_ROOT / "esmc_base_embeddings.parquet"
ESMC_RUNTIME_ROOT = SOURCE_ROOT / "esmc_runtime"

DEVICE = "cuda"
ESM_BLOCK = 512
TOKEN_BUDGET = 2048
ESM_CACHE_CHUNK_GENES = 512


def _matrix_path(cohort_id: str) -> Path:
    source = SOURCE_ROOT / f"{cohort_id}_features.parquet"
    if not source.is_file():
        raise FileNotFoundError(
            f"missing ProteomeLM source matrix: {source}. Rebuild it with the "
            f"command in method.yaml's rebuild_command: "
            f"'python build_features.py --dataset {cohort_id}'"
        )
    return source


def rebuild_source(datasets: tuple[str, ...]) -> None:
    model_files = (MODEL_ROOT / "config.json", MODEL_ROOT / "model.safetensors")
    if missing := [str(path) for path in model_files if not path.is_file()]:
        raise FileNotFoundError(
            "ProteomeLM-L model assets are absent: " + ", ".join(missing)
        )
    esmc_weights = ESMC_RUNTIME_ROOT / "data/weights/esmc_600m_2024_12_v0.pth"
    if not esmc_weights.is_file():
        raise FileNotFoundError(
            f"ESM-C checkpoint is absent: {esmc_weights}. Run acquire_assets.py "
            "--include-esmc."
        )
    environment = os.environ.copy()
    upstream = METHOD_ROOT / "code" / "upstream"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(upstream)
        if not existing_pythonpath
        else os.pathsep.join((str(upstream), existing_pythonpath))
    )
    environment["HF_HOME"] = str(SOURCE_ROOT / "model_assets/huggingface")
    environment["HF_HUB_CACHE"] = str(
        SOURCE_ROOT / "model_assets/huggingface/hub"
    )
    environment["TRANSFORMERS_CACHE"] = str(
        SOURCE_ROOT / "model_assets/huggingface/transformers"
    )
    for cohort_id in datasets:
        slug = cohort_id
        proteins = SOURCE_ROOT / f"{slug}_protein_sequences.fa.gz"
        if not proteins.is_file():
            raise FileNotFoundError(
                f"protein FASTA is absent: {proteins}. Install the source asset "
                "before rebuilding ProteomeLM features."
            )
        gene_list = SOURCE_ROOT / f"{slug}_common_extractable_genes.parquet"
        if not gene_list.is_file():
            raise FileNotFoundError(
                f"ProteomeLM source gene list is absent: {gene_list}"
            )
        subprocess.run(
            [
                sys.executable,
                str(METHOD_ROOT / "code/embed_proteome.py"),
                "--proteins",
                str(proteins),
                "--gene-list",
                str(gene_list),
                "--proteomelm-dir",
                str(MODEL_ROOT),
                "--layer",
                "8",
                "--out",
                str(SOURCE_ROOT / f"{slug}_features.parquet"),
                "--esm-cache",
                str(ESMC_CACHE),
                "--block-cache-dir",
                str(SOURCE_ROOT / f"{slug}_proteomelm_layer8_blocks"),
                "--device",
                DEVICE,
                "--esm-block",
                str(ESM_BLOCK),
                "--token-budget",
                str(TOKEN_BUDGET),
                "--esm-cache-chunk-genes",
                str(ESM_CACHE_CHUNK_GENES),
            ],
            check=True,
            cwd=METHOD_ROOT / "code",
            env=environment,
        )


def materialize(datasets: tuple[str, ...]) -> None:
    for cohort_id in datasets:
        _, observed = materialize_static_feature_folds(
            METHOD_ROOT,
            cohort_id,
            repo_root=REPO_ROOT,
            matrix_path=_matrix_path(cohort_id),
        )
        if observed != FEATURE_DIMENSIONS:
            raise ValueError(
                f"ProteomeLM {cohort_id} has {observed} dimensions; "
                f"expected {FEATURE_DIMENSIONS}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=COHORT_IDS, action="append", required=True)
    args = parser.parse_args()
    datasets = tuple(args.dataset)
    rebuild_source(datasets)
    materialize(datasets)


if __name__ == "__main__":
    main()
