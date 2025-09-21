"""Core engine public API exports.

This module exposes visualization/statistics entrypoints for B-stage processing.
"""

from .viz_people import (
    RenderConfig,
    TrajectoryConfig,
    validate_df_cc,
    validate_df_pos,
    compute_cc_stats,
    render_concurrency_png,
    render_trajectory2d_png,
    save_stats_txt,
    run,
)

__all__ = [
    "RenderConfig",
    "TrajectoryConfig",
    "validate_df_cc",
    "validate_df_pos",
    "compute_cc_stats",
    "render_concurrency_png",
    "render_trajectory2d_png",
    "save_stats_txt",
    "run",
]


