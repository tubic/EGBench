"""Wheel build hook: puts the files the MCP server reads under ``egbench_mcp/``, laid out as in the repository.

Each file is added by name, because a directory added whole would also take the caches that runs
leave in a clone (``work/``, ``feature/``, ``model/``). The library is installed only there, under
``egbench_mcp/src/``, which ``egbench_mcp.pth`` puts on the import path: ``egbench.data`` finds the
repository two levels above its own file, so the library and the server then read the datasets
beside it. An editable install keeps the library in the clone.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

MIRROR = "egbench_mcp"
FILES = (
    "README.md",
    "src/egbench/**/*.py",
    # The server and the modules of its tools; the layer-1 tables are built by annotation, core_model and sources.
    *(f"agent/{module}.py" for module in ("launch", "server", "benchmark", "external", "labels", "annotation", "core_model", "sources")),
    "agent/external.json",
    "agent/README.md",
    "agent/predictions/*.parquet",
    "agent/examples/*.csv",
    "analysis/common/*.py",
    "analysis/*/results/*",
    "datasets/**/*.parquet",
    "datasets/**/*.json",
    "datasets/**/*.csv",
    "datasets/**/*.tsv",
    "methods/*/method.yaml",
)


class MirrorHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel" or version == "editable":
            return
        root = Path(self.root)
        for pattern in FILES:
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    build_data["force_include"][str(path)] = f"{MIRROR}/{path.relative_to(root).as_posix()}"
        path_file = Path(tempfile.mkdtemp()) / f"{MIRROR}.pth"
        path_file.write_text(f"{MIRROR}/src\n")
        build_data["force_include"][str(path_file)] = path_file.name
