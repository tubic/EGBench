#!/usr/bin/env python3
"""Download the ProteomeLM-L and ESM-C 600M checkpoints from Hugging Face.
Run from methods/proteomelm: python acquire_assets.py --include-esmc
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


METHOD_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = METHOD_ROOT / "feature/source"
DEFAULT_OUTPUT = SOURCE_ROOT / "model_assets/proteomelm_l"
ESMC_RUNTIME_ROOT = SOURCE_ROOT / "esmc_runtime"
DEFAULT_ESMC_WEIGHTS = (
    ESMC_RUNTIME_ROOT / "data/weights/esmc_600m_2024_12_v0.pth"
)
HF_HOME = SOURCE_ROOT / "model_assets/huggingface"
HF_HUB_CACHE = HF_HOME / "hub"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-esmc", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(METHOD_ROOT):
        raise ValueError(f"output must be located inside {METHOD_ROOT}")
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="Bitbol-Lab/ProteomeLM-L",
        local_dir=output,
        cache_dir=HF_HUB_CACHE,
        allow_patterns=["config.json", "model.safetensors"],
    )
    print(output.relative_to(METHOD_ROOT))
    if args.include_esmc:
        snapshot_download(
            repo_id="EvolutionaryScale/esmc-600m-2024-12",
            local_dir=ESMC_RUNTIME_ROOT,
            cache_dir=HF_HUB_CACHE,
            allow_patterns=["data/weights/esmc_600m_2024_12_v0.pth"],
        )
        if not DEFAULT_ESMC_WEIGHTS.is_file():
            raise FileNotFoundError(
                f"ESM-C checkpoint was not materialized at {DEFAULT_ESMC_WEIGHTS}"
            )
        print(DEFAULT_ESMC_WEIGHTS.relative_to(METHOD_ROOT))


if __name__ == "__main__":
    main()
