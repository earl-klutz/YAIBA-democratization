"""Logging helpers for visualization pipelines."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from . import naming


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_logger(run_id: str) -> logging.Logger:
    """Configure a logger that emits to stdout and to a file."""

    logger_name = f"yaiba.movie.{run_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    log_path = naming.META_ROOT / "logs" / f"run_{run_id}.log"
    _ensure_parent(log_path)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def log_summary(logger: logging.Logger, stats: Mapping[str, object]) -> None:
    """Emit a formatted summary to *logger*."""

    logger.info("=== 処理サマリー ===")
    for key, value in stats.items():
        logger.info("%s: %s", key, value)
