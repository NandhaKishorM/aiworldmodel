"""
Logging Utilities — Structured Logging with Rich
===================================================
Provides consistent, configurable logging across all modules.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_configured = False


def setup_logging(
    level: str = "INFO",
    fmt: str = "rich",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure the root logger.

    Parameters
    ----------
    level    : DEBUG, INFO, WARNING, ERROR
    fmt      : "rich" for colorized output, "json" for structured
    log_file : optional file path for persistent logging
    """
    global _configured
    if _configured:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove default handlers
    root.handlers.clear()

    if fmt == "rich":
        try:
            from rich.logging import RichHandler
            handler = RichHandler(
                level=log_level,
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
        except ImportError:
            handler = _make_stream_handler(log_level)
    else:
        handler = _make_stream_handler(log_level)

    root.addHandler(handler)

    # File handler
    if log_file:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(fh)

    _configured = True


def _make_stream_handler(level: int) -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    return handler


def get_logger(name: str) -> logging.Logger:
    """Get a named logger (use __name__ in each module)."""
    return logging.getLogger(name)
