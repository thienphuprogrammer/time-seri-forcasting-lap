"""
Logging utilities for time series forecasting package.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from ..config import CONFIG


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup a logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file path (optional)
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level or CONFIG["logging"]["level"]))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(CONFIG["logging"]["format"])
    
    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with standard configuration.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return setup_logger(name, log_file=CONFIG["logging"]["file"])


# Package logger
package_logger = get_logger(__name__)

__all__ = ["setup_logger", "get_logger", "package_logger"] 