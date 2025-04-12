"""
HADES-PathRAG: Path-based Retrieval Augmented Generation with ISNE embeddings.

This package implements PathRAG with ISNE embeddings for efficient graph-based retrieval.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("hades-pathrag")
except PackageNotFoundError:
    # Package is not installed
    try:
        from ._version import version as __version__  # type: ignore
    except ImportError:
        __version__ = "unknown"
