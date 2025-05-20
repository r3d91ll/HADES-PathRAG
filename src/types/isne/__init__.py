"""ISNE type definitions.

This module provides type definitions for the ISNE (Inductive Shallow Node Embedding) system.
"""

from src.types.isne.models import *

__all__ = [
    "ISNEConfig",
    "ISNETrainingConfig",
    "ISNEModelConfig",
    "DocumentType",
    "RelationType",
    "IngestDocument",
    "DocumentRelation"
]
