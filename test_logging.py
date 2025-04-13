#!/usr/bin/env python
"""Test script to verify logging functionality."""
import logging
import os
from pathlib import Path
import datetime
from logging.handlers import RotatingFileHandler

# Create logs directory (using absolute path for certainty)
logs_dir = Path("/home/todd/ML-Lab/HADES-PathRAG/hades_pathrag/logs")
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
log_file = logs_dir / f"test_logging_{current_date}.log"

# Set up a logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create and add file handler
handler = RotatingFileHandler(
    filename=log_file,
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,
    encoding="utf-8"
)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Log some test messages
logger.info(f"Test logging to file: {log_file}")
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")

print(f"Log file created at: {log_file}")
print("Check if the file exists and contains the log messages.")
