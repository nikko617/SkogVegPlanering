# -*- coding: utf-8 -*-
"""
logger.py – Proper logging setup for SkogVegPlanering
Ensures StreamHandler is configured to prevent NoneType errors
"""

import logging
import sys

# Sentinel to distinguish missing "stream" attributes from stream=None.
_MISSING = object()


def _pick_stream():
    """Select the best available output stream for logging."""
    stream = sys.stdout if sys.stdout is not None else sys.stderr
    if stream is None or getattr(stream, "closed", False):
        return None
    return stream


class _SafeStreamHandler(logging.StreamHandler):
    """Stream handler that tolerates stream shutdown during QGIS/plugin reloads."""

    def emit(self, record):
        stream = self.stream
        if stream is None or getattr(stream, "closed", False):
            fallback = _pick_stream()
            if fallback is None:
                return
            self.setStream(fallback)
        super().emit(record)


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
        stream_closed = getattr(stream, "closed", False) if stream is not _MISSING else False
        if isinstance(existing_handler, logging.StreamHandler) and (
            stream is None or stream_closed
        ):
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
    stream = _pick_stream()
    if stream is not None:
        handler = _SafeStreamHandler(stream)
        handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
    else:
        handler = logging.NullHandler()

    # Add handler to logger
    logger.addHandler(handler)

    return logger
