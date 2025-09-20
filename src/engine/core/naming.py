"""Naming helpers for YAIBA visualization outputs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

_KIND_DIR_MAP: Final[dict[str, str]] = {
    "image": "images",
    "table": "tables",
    "movie": "movies",
}

_BASE_OUTPUT_DIR: Final[Path] = Path("YAIBA_data") / "output"


def _sanitize(text: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "-", text.strip())
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    return sanitized or "untitled"


def build_basename(
    event_day: str,
    filename: str,
    dt: str,
    ver: str,
    *,
    duration: int | None = None,
) -> str:
    """Build a canonical basename used across the project."""

    event_day_part = _sanitize(str(event_day))
    filename_part = _sanitize(filename)
    dt_part = _sanitize(dt)
    ver_part = _sanitize(ver)

    parts: list[str] = [event_day_part, filename_part, dt_part]
    if duration is not None:
        parts.append(f"{int(duration)}s")

    joined = "-".join(parts)
    return f"{joined}_{ver_part}"


def result_path(kind: str, basename: str) -> Path:
    """Return the path to an artefact stored under /YAIBA_data/output/results."""

    if kind not in _KIND_DIR_MAP:
        raise ValueError(f"未知の成果物種別です: {kind}")

    directory = _BASE_OUTPUT_DIR / "results" / _KIND_DIR_MAP[kind]
    directory.mkdir(parents=True, exist_ok=True)
    return directory / basename


def meta_paths(dt: str, ver: str) -> dict[str, Path]:
    """Return output paths for configuration and log artefacts."""

    base_meta = _BASE_OUTPUT_DIR / "meta"
    config_dir = base_meta / "configs"
    logs_dir = base_meta / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / f"run_{_sanitize(ver)}_params.yaml"
    log_path = logs_dir / f"run_{_sanitize(dt)}.log"
    return {"config_path": config_path, "log_path": log_path}
