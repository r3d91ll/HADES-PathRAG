'''
MarkdownPreProcessor: Compatibility wrapper for DoclingPreProcessor
------------------------------------------------------------------
This thin wrapper exists to maintain backwards-compatibility with older
code and tests that still import `MarkdownPreProcessor` from
`src.ingest.pre_processor.markdown_pre_processor`.

Internally it simply re-exports the newer `DoclingPreProcessor`, so all
functionality remains exactly the same.  Going forward, new code should
prefer importing `DoclingPreProcessor` directly, and this shim can be
removed once all references have been migrated.
'''
from __future__ import annotations

from .docling_pre_processor import DoclingPreProcessor


class MarkdownPreProcessor(DoclingPreProcessor):
    """Alias subclass for backwards-compatibility."""

    # No additional implementation is necessary – all behaviour is
    # inherited from ``DoclingPreProcessor``.  We keep the subclass in
    # place (rather than a simple assignment) so that ``isinstance`` and
    # ``issubclass`` checks continue to work as expected.

    pass

__all__ = ["MarkdownPreProcessor"]
