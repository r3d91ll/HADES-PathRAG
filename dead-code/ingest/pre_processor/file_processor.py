"""Directory scanner and batch planner for HADES-PathRAG pre-processing.

Moved from ``src.ingest.processing.file_processor`` to ``src.ingest.pre_processor``
so that *all* steps executed before language-specific processors live in the
``pre_processor`` package.

The implementation is unchanged; only the import path moved.  Any existing
imports that still reference the old location will keep working because a thin
re-export stub remains in the previous module.
"""

from __future__ import annotations

# NOTE: We copy the original implementation verbatim to preserve git history
# across a move.  Future edits should occur **only** in this file.

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Union

from src.types.common import PreProcessorConfig

logger = logging.getLogger(__name__)


class FileProcessor:
    """Discover files, filter via exclude patterns, and create balanced batches."""

    DEFAULT_MAX_WORKERS = 4
    DEFAULT_BATCH_SIZE = 10

    def __init__(self, config: Optional[PreProcessorConfig] = None):
        self.config = config or {}
        self.exclude_patterns = self.config.get("exclude_patterns", [])
        self.max_workers = self.config.get("max_workers", self.DEFAULT_MAX_WORKERS)
        self.recursive = self.config.get("recursive", True)
        self.file_type_map = self.config.get("file_type_map", {})

        self.exclude_regex: List[Pattern] = []
        for pattern in self.exclude_patterns:
            try:
                self.exclude_regex.append(re.compile(pattern))
            except re.error as exc:
                logger.warning("Invalid exclude pattern '%s': %s", pattern, exc)

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------

    def should_exclude(self, file_path: Union[str, Path]) -> bool:
        path_str = str(file_path)
        return any(pattern.search(path_str) for pattern in self.exclude_regex)

    def get_file_type(self, file_path: Union[str, Path]) -> str:
        file_path = Path(file_path)
        ext = file_path.suffix.lstrip(".").lower()
        for file_type, extensions in self.file_type_map.items():
            if ext in [e.lower().lstrip(".") for e in extensions]:
                return file_type
        return ext or "unknown"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_files(self, root_dir: Union[str, Path]) -> Dict[str, List[Path]]:
        root_dir = Path(root_dir)
        if not root_dir.exists() or not root_dir.is_dir():
            logger.error("Directory does not exist or is not a directory: %s", root_dir)
            return {}

        files_by_type: Dict[str, List[Path]] = {}
        for current_dir, _dirs, files in os.walk(root_dir):
            current_path = Path(current_dir)
            if self.should_exclude(current_path):
                continue
            for file_name in files:
                file_path = current_path / file_name
                if self.should_exclude(file_path):
                    continue
                file_type = self.get_file_type(file_path)
                files_by_type.setdefault(file_type, []).append(file_path)
            if not self.recursive:
                break
        return files_by_type

    def create_batches(
        self,
        files_by_type: Dict[str, List[Path]],
        *,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, List[Path]]]:
        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE
        batches: List[Dict[str, List[Path]]] = []
        for file_type, files in files_by_type.items():
            file_batches = [files[i : i + batch_size] for i in range(0, len(files), batch_size)]
            for i, file_batch in enumerate(file_batches):
                while i >= len(batches):
                    batches.append({})
                batches[i][file_type] = file_batch
        return batches

    def process_directory(
        self, directory: Union[str, Path], *, batch_size: Optional[int] = None
    ) -> List[Dict[str, List[Path]]]:
        files_by_type = self.collect_files(directory)
        return self.create_batches(files_by_type, batch_size=batch_size)
