# src/yaiba_bi/core/__init__.py

# Public re-exports (Core APIs)

from .movie import MovieGenerator, run_movie_xz
from .histogram import HistogramGenerator
from .validation import (
    require_columns,
    drop_invalid_types,
    clip_by_boundary,
    enforce_min_seconds,
)
from .naming import build_basename, result_path, meta_paths
from .logging_util import get_logger, log_summary

__all__ = [
    # Generators
    "MovieGenerator",
    "HistogramGenerator",
    "run_movie_xz",

    # Validation utilities
    "require_columns",
    "drop_invalid_types",
    "clip_by_boundary",
    "enforce_min_seconds",

    # Naming & paths
    "build_basename",
    "result_path",
    "meta_paths",

    # Logging
    "get_logger",
    "log_summary",
]
