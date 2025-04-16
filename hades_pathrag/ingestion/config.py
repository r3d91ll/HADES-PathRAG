import yaml  # type: ignore
from typing import Any


def load_pipeline_config(config_path: str) -> Any:
    """
    Load the pipeline configuration from a YAML file.
    Args:
        config_path: Path to the YAML config file.
    Returns:
        YAML config as a dict (Any type for mypy compatibility).
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
