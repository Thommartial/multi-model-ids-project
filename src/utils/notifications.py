"""Project logging and milestone notifications.

Built on ``loguru``. :func:`configure_logging` sets up a clean console
log plus a rotating file log under ``experiments/logs/``.
:func:`notify` records a milestone (the start or finish of a long
experiment) so progress is visible both live and in the log file.

Email/Slack notifications are intentionally not wired up here -- they
need credentials and add fragility. If they are wanted later, send them
from :func:`notify` using values read from environment variables (see
``.env.example``).
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def configure_logging(log_dir: str = "experiments/logs", level: str = "INFO"):
    """Set up console + rotating file logging. Safe to call repeatedly."""
    global _CONFIGURED
    if _CONFIGURED:
        return logger

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()  # drop loguru's default handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    logger.add(
        Path(log_dir) / "project_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )
    _CONFIGURED = True
    return logger


def notify(message: str, level: str = "SUCCESS") -> None:
    """Record a project milestone to the console and the log file."""
    configure_logging()
    logger.log(level, message)
