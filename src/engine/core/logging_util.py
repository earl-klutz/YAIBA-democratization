"""Logging helpers shared by the visualization pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import naming


def get_logger(run_id: str, *, log_path: Path | None = None) -> logging.Logger:
    """Return a configured logger for the current execution."""

    logger_name = f"yaiba.movie.{run_id}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    if log_path is None:
        log_path = naming.meta_paths(run_id, run_id)["log_path"]

    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def log_summary(logger: logging.Logger, stats: dict[str, Any]) -> None:
    """Output a summary of the execution statistics."""

    lines = ["=== 実行サマリー ==="]
    for key, value in stats.items():
        lines.append(f"{key}: {value}")
    logger.info("\n".join(lines))
