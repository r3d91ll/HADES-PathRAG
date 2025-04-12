"""
Type stubs for libraries without official mypy type stubs.

This module provides type-ignoring stubs for external libraries that don't have
official type stubs, allowing mypy to continue checking the rest of the codebase.
"""
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

# Create stubs for torch
import torch
import torch.nn
import torch.nn.functional
import torch.utils.data

# Create stubs for sentence_transformers
import sentence_transformers

# Create stubs for scikit-learn
import sklearn
import sklearn.preprocessing

# Create stubs for tiktoken
import tiktoken

# Create stubs for pydantic
import pydantic

# Create stubs for arango
import arango
import arango.database
import arango.collection
import arango.graph
import arango.exceptions

# Define common type aliases 
Embedding = Any  # Typically a numpy.ndarray
NodeID = Any  # Typically a string or numeric ID

# This file is imported by .mypy.ini or setup.cfg to tell mypy to ignore these imports
