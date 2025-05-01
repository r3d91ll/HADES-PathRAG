"""Embedding utilities for HADES-PathRAG.

This sub-package will host text and graph embedding back-ends.  For now we
expose an ISNE façade so the orchestrator can trigger graph embeddings
without importing top-level ``src.isne`` objects directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from src.ingest.isne_connector import ISNEIngestorConnector
from src.storage.arango.connection import ArangoConnection

__all__: list[str] = [
    "embed_graph_with_isne",
]


def embed_graph_with_isne(
    repo_path: Union[str, Path],
    *,
    connection: Optional[ArangoConnection] = None,
    repo_name: Optional[str] = None,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
) -> None:
    """Run ISNE over an already-ingested repository directory.

    This helper simply wraps :class:`src.ingest.isne_connector.ISNEIngestorConnector`
    so that the main orchestrator can remain agnostic of ISNE internals.
    """
    connector = ISNEIngestorConnector(
        arango_connection=connection,
        embedding_model=embedding_model,
    )
    connector.process_repository(Path(repo_path), repo_name=repo_name, store_in_arango=True)
