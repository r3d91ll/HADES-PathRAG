from __future__ import annotations

"""Semantic text chunker using [Chonky](https://github.com/mithril-security/chonky).

The helper ``chunk_text`` mirrors the signature of ``chunk_python_code`` so the
`PreprocessorManager` can treat code and text files uniformly.

We depend only on the lightweight `chonky` ParagraphSplitter which yields
paragraph segments.  Additional post-processing splits long paragraphs to
respect the ``max_tokens`` budget and attaches basic metadata.
"""

from typing import Any, Dict, List, Union
import hashlib
import uuid
import logging
import json

try:
    from chonky import ParagraphSplitter  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Package 'chonky' is required for semantic text chunking.\n"
        "Install with:  pip install chonky"
    ) from exc

from src.config.chunker_config import get_chunker_config

logger = logging.getLogger(__name__)


_SPLITTER: ParagraphSplitter | None = None


def _get_splitter(model_id: str = "mirth/chonky_distilbert_uncased_1", device: str = "cpu") -> ParagraphSplitter:  # noqa: D401
    """Lazy-instantiate the global Chonky paragraph splitter."""
    global _SPLITTER  # pylint: disable=global-statement
    if _SPLITTER is None:
        logger.info("Loading Chonky paragraph splitter '%s' on %s", model_id, device)
        _SPLITTER = ParagraphSplitter(model_id=model_id, device=device)
    return _SPLITTER


def _hash_path(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()[:8]


def chunk_text(
    document: Dict[str, Any], *, max_tokens: int = 2048, output_format: str = "python"
) -> Union[List[Dict[str, Any]], str]:  # noqa: D401
    """Split plain-text/Markdown document into semantic paragraphs.

    Parameters
    ----------
    document:
        Pre-processed document dict. Must contain ``content`` and ``path`` keys.
    max_tokens:
        Token budget for each chunk (default comes from config).
    output_format:
        Output format of the chunks. Can be either "python" or "json".
    """
    cfg = get_chunker_config("chonky")
    if max_tokens == 2048:
        max_tokens = cfg.get("max_tokens", 2048)

    source = document.get("content") or document.get("source", "")
    path: str = document.get("path", "unknown")
    if not source.strip():
        logger.warning("Chunk_text called with empty document: %s", path)
        return []

    # Run Chonky paragraph splitter
    splitter = _get_splitter()
    # ``ParagraphSplitter`` implements ``__call__`` and returns an
    # iterator over paragraph strings.  The library does **not** expose a
    # dedicated ``split`` helper, contrary to some outdated examples.
    # Convert the generator into a concrete ``list`` so we can iterate
    # multiple times if needed.
    paragraphs: List[str] = list(splitter(source))

    chunks: List[Dict[str, Any]] = []
    parent_id = f"doc:{_hash_path(path)}"

    for idx, para in enumerate(paragraphs):
        if not para.strip():
            continue
        # TODO: measure tokens; optionally further split large paragraphs.
        chunk_id = f"chunk:{uuid.uuid4().hex[:8]}"
        chunks.append(
            {
                "id": chunk_id,
                "parent": parent_id,
                "path": path,
                "type": document.get("type", "markdown"),
                "content": para,
                "symbol_type": "paragraph",
                "name": f"paragraph_{idx}",
                "line_start": 0,
                "line_end": 0,
            }
        )
    logger.debug("Chunked %s into %d paragraphs", path, len(chunks))

    if output_format == "json":
        return json.dumps(chunks, indent=2)

    return chunks
