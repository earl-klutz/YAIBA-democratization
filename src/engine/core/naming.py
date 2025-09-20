"""Naming helpers for YAIBA visualization outputs."""
from __future__ import annotations

from pathlib import Path

RESULT_ROOT = Path("YAIBA_data") / "output" / "results"
META_ROOT = Path("YAIBA_data") / "output" / "meta"

_KIND_DIR = {
    "image": "images",
    "table": "tables",
    "movie": "movies",
}


def _sanitize(component: str) -> str:
    return component.replace(" ", "_")


def build_basename(event_day: str, filename: str, dt: str, ver: str, *, duration: int | None = None) -> str:
    """Build a canonical basename according to the design document."""

    components: list[str] = [
        _sanitize(event_day),
        _sanitize(filename),
        _sanitize(dt),
    ]
    if duration is not None:
        components.append(f"{int(duration)}s")
    base = "-".join(filter(None, components))
    return f"{base}_{_sanitize(ver)}"


def result_path(kind: str, basename: str) -> Path:
    """Return the storage path for the given *kind* and *basename*."""

    try:
        subdir = _KIND_DIR[kind]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"未知の成果物種別です: {kind}") from exc
    return RESULT_ROOT / subdir / basename


def meta_paths(dt: str, ver: str) -> dict[str, Path]:
    """Return the metadata file paths for the given run."""

    configs = META_ROOT / "configs"
    logs = META_ROOT / "logs"
    return {
        "config_path": configs / f"run_{_sanitize(ver)}_params.yaml",
        "log_path": logs / f"run_{_sanitize(dt)}.log",
    }
