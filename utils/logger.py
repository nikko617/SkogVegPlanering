# -*- coding: utf-8 -*-
"""
logger.py – Proper logging setup for SkogVegPlanering
Ensures StreamHandler is configured to prevent NoneType errors
"""

import logging
import sys


def setup_logger(name):
    """
    Setup a logger with proper StreamHandler.

    Args:
        name: Logger name (typically __name__)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Set logging level
    logger.setLevel(logging.DEBUG)

    # Create StreamHandler (stdout)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger
