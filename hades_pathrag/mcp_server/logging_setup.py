"""
Logging configuration for the MCP server.

This module sets up logging for the server components.
"""
import logging
import os
import sys
from typing import Optional
from logging.handlers import RotatingFileHandler
from pathlib import Path
import datetime


def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging for the MCP server.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Convert level string to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers to prevent duplicate logging
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Add console handler for stdout (for IDE integration)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    
    # Also add a stderr handler as backup (some IDEs capture this better)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(logging.WARNING)  # Only warnings and above go to stderr
    root_logger.addHandler(stderr_handler)
    
    # Use an absolute path for logs directory
    logs_dir = Path("/home/todd/ML-Lab/HADES-PathRAG/hades_pathrag/logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate log filename with timestamp to make it unique per session
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = logs_dir / f"mcp_server_{timestamp}.log"
    
    # Add file handler with rotation (10MB max size, keep 5 backup files)
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)  # Same level as console for complete logs
    root_logger.addHandler(file_handler)
    
    # Add a symbolic link to the latest log for convenience
    latest_log_link = logs_dir / "mcp_server_latest.log"
    try:
        if os.path.exists(latest_log_link):
            os.unlink(latest_log_link)
        os.symlink(log_file, latest_log_link)
    except (OSError, IOError) as e:
        # Don't fail logging setup just because of symlink issues
        print(f"Warning: Could not create symlink to latest log: {e}")
    
    # Log the log file location
    root_logger.info(f"Logging to: {log_file}")
    
    # Only set lower log levels for noisy libraries if we're not in debug mode
    if numeric_level > logging.DEBUG:
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("fastapi").setLevel(logging.WARNING)
    else:
        # In debug mode, allow all debug logs to flow
        logging.getLogger("uvicorn").setLevel(numeric_level)
        logging.getLogger("uvicorn.access").setLevel(numeric_level)
        logging.getLogger("fastapi").setLevel(numeric_level)
        
    # Log the effective log level
    root_logger.debug(f"Logging initialized at level: {logging.getLevelName(numeric_level)}")
