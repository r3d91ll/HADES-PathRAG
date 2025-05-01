"""Compatibility wrapper for documentation ingestion.

This script re-exports the public helpers in ``src.cli.ingest`` so that
legacy imports like::

    from scripts.ingest_documentation import create_documentation_config, ingest_documentation

continue to work after the codebase re-organisation.

The real implementation lives in ``src.cli.ingest``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on ``sys.path`` when the module is executed directly.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from src.cli.ingest import (  # noqa: E402  pylint: disable=wrong-import-position
    create_documentation_config,  # re-export
    ingest_documentation,  # re-export
    main as _cli_main,
)

__all__ = [
    "create_documentation_config",
    "ingest_documentation",
]


if __name__ == "__main__":
    # Delegate to the new CLI entry-point.
    sys.exit(_cli_main())
