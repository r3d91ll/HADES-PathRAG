"""
Pre-processor configuration loader for HADES-PathRAG (YAML-based).

Loads, merges, and validates pre-processor configuration from YAML.
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from src.types.common import PreProcessorConfig

# Defaults (should match those in preprocessor_config.yaml)
DEFAULT_FILE_TYPE_MAP = {
    'python': ['.py'],
    'javascript': ['.js', '.jsx', '.ts', '.tsx'],
    'java': ['.java'],
    'cpp': ['.cpp', '.hpp', '.cc', '.h'],
    'markdown': ['.md', '.markdown'],
    'pdf': ['.pdf'],
    'json': ['.json'],
    'csv': ['.csv'],
    'text': ['.txt'],
    'html': ['.html', '.htm'],
    'xml': ['.xml'],
}

DEFAULT_PREPROCESSOR_CONFIG = {
    'python': {
        'create_symbol_table': True,
        'extract_docstrings': True,
        'analyze_imports': True,
    },
    'markdown': {
        'extract_mermaid': True,
        'extract_code_blocks': True,
        'extract_links': True,
    }
}

DEFAULTS = {
    'version': 1,
    'exclude_patterns': ['__pycache__', '.git'],
    'recursive': True,
    'max_workers': 4,
    'file_type_map': DEFAULT_FILE_TYPE_MAP,
    'preprocessor_config': DEFAULT_PREPROCESSOR_CONFIG,
    'options': {},
}

CONFIG_PATH = Path(__file__).parent / 'preprocessor_config.yaml'

def load_config(config_path: Optional[Union[str, Path]] = None) -> PreProcessorConfig:
    """
    Load configuration from a YAML file, merging with defaults.
    """
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        user_config = yaml.safe_load(f) or {}

    # Merge user config with defaults
    # Merge top-level keys
    config = dict(DEFAULTS)
    config.update(user_config)
    # Merge nested dicts
    config['file_type_map'] = {**DEFAULTS['file_type_map'], **user_config.get('file_type_map', {})}
    config['preprocessor_config'] = {**DEFAULTS['preprocessor_config'], **user_config.get('preprocessor_config', {})}

    # Build PreProcessorConfig TypedDict
    return PreProcessorConfig(
        input_dir=Path(user_config.get('input_dir', '.')),
        output_dir=Path(user_config.get('output_dir', user_config.get('input_dir', '.') + '/.symbol_table')),
        exclude_patterns=config['exclude_patterns'],
        recursive=config['recursive'],
        max_workers=config['max_workers'],
        file_type_map=config['file_type_map'],
        preprocessor_config=config['preprocessor_config'],
        options=config['options'],
    )
