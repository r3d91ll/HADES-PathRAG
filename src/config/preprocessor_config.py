"""
Pre-processor configuration loader for HADES-PathRAG (YAML-based).

Loads, merges, and validates pre-processor configuration from YAML.
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, cast
from src.types.common import PreProcessorConfig

# Defaults (should match those in preprocessor_config.yaml)
DEFAULT_FILE_TYPE_MAP: Dict[str, List[str]] = {
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

DEFAULT_PREPROCESSOR_CONFIG: Dict[str, Dict[str, Any]] = {
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

DEFAULTS: Dict[str, Any] = {
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

    # Create fresh dictionaries with proper types
    exclude_patterns: List[str] = list(DEFAULTS['exclude_patterns'])
    recursive: bool = bool(DEFAULTS['recursive'])
    max_workers: int = int(DEFAULTS['max_workers'])
    options: Dict[str, Any] = dict(DEFAULTS['options'])
    
    # Update from user config
    if 'exclude_patterns' in user_config and isinstance(user_config['exclude_patterns'], list):
        exclude_patterns = list(user_config['exclude_patterns'])
    if 'recursive' in user_config:
        recursive = bool(user_config['recursive'])
    if 'max_workers' in user_config:
        max_workers = int(user_config['max_workers'])
    if 'options' in user_config and isinstance(user_config['options'], dict):
        options = dict(user_config['options'])
    
    # Handle nested dictionaries
    file_type_map: Dict[str, List[str]] = dict(DEFAULT_FILE_TYPE_MAP)
    if 'file_type_map' in user_config and isinstance(user_config['file_type_map'], dict):
        # Merge file type map
        for key, value in user_config['file_type_map'].items():
            if isinstance(value, list):
                file_type_map[key] = list(value)
    
    preprocessor_config: Dict[str, Dict[str, Any]] = dict(DEFAULT_PREPROCESSOR_CONFIG)
    if 'preprocessor_config' in user_config and isinstance(user_config['preprocessor_config'], dict):
        # Merge preprocessor config
        for key, value in user_config['preprocessor_config'].items():
            if isinstance(value, dict):
                if key in preprocessor_config:
                    # Update existing dictionary
                    preprocessor_config[key].update(value)
                else:
                    # Add new dictionary
                    preprocessor_config[key] = dict(value)

    # Build PreProcessorConfig TypedDict
    result = PreProcessorConfig(
        input_dir=Path(user_config.get('input_dir', '.')),
        output_dir=Path(user_config.get('output_dir', user_config.get('input_dir', '.') + '/.symbol_table')),
        exclude_patterns=exclude_patterns,
        recursive=recursive,
        max_workers=max_workers,
        file_type_map=file_type_map,
        preprocessor_config=preprocessor_config,
        options=options,
    )
    
    return result
