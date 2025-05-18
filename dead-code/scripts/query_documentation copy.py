"""Compatibility wrapper for PathRAG documentation queries.

Maintains the public import surface used by older tests::

    from scripts.query_documentation import query_documentation

The actual implementation now lives in ``src.cli.query``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from src.cli.query import (  # noqa: E402  pylint: disable=wrong-import-position
    query_documentation,  # re-export
    main as _cli_main,
)

__all__ = [
    "query_documentation",
]


if __name__ == "__main__":
    sys.exit(_cli_main())
