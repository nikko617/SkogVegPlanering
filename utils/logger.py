# -*- coding: utf-8 -*-
"""
logger.py – Proper logging setup for SkogVegPlanering
Ensures StreamHandler is configured to prevent NoneType errors
"""

import logging
import sys

_MISSING = object()


def setup_logger(name):
    """
    Setup a logger with proper StreamHandler.

    Args:
        name: Logger name (typically __name__)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.propagate = False

    # Remove stale handlers (can happen across QGIS/plugin reload cycles)
    valid_handlers = []
    for existing_handler in logger.handlers:
        stream = getattr(existing_handler, "stream", _MISSING)
        if isinstance(existing_handler, logging.StreamHandler) and stream is None:
            try:
                existing_handler.close()
            except Exception:
                pass
            continue
        valid_handlers.append(existing_handler)
    logger.handlers = valid_handlers

    # Avoid duplicate handlers when at least one valid handler remains
    if logger.handlers:
        return logger

    # Set logging level
    logger.setLevel(logging.DEBUG)

    # Create StreamHandler (stdout preferred, then stderr fallback)
    stream = sys.stdout if sys.stdout is not None else sys.stderr
    if stream is not None:
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
    else:
        handler = logging.NullHandler()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger
