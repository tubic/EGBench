"""Entry point of the installed package: the command ``egbench-mcp`` starts the MCP server.

The wheel holds the files the server reads under ``egbench_mcp/``, laid out as in the repository
(``hatch_build.py``), so the modules find them as they do in a clone. The installed package is no
place for downloads and the tables built from them, so the data root is ``EGBENCH_DATA_ROOT`` or,
when it is unset, ``~/egbench-data``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

DATA_ROOT = Path.home() / "egbench-data"


def main() -> None:
    os.environ.setdefault("EGBENCH_DATA_ROOT", str(DATA_ROOT))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import server

    server.server.run()
