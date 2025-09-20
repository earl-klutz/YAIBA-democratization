"""Utilities for validating intermediate YAIBA dataframes."""
from __future__ import annotations

from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = [
    "second",
    "user_id",
    "location_x",
    "location_z",
    "event_day",
]


def require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """Ensure all *cols* exist in *df*."""

    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(
            "必須列が不足しています: " + ", ".join(sorted(missing))
        )


def _coerce_seconds(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _coerce_event_day(series: pd.Series) -> pd.Series:
    days = pd.to_datetime(series, errors="coerce")
    if hasattr(days.dt, "tz") and days.dt.tz is not None:
        days = days.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return days.dt.normalize()


_NUMERIC_COLUMNS = [
    "location_x",
    "location_y",
    "location_z",
    "rotation_1",
    "rotation_2",
    "rotation_3",
]


def drop_invalid_types(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows that cannot be coerced to the expected dtypes."""

    if df.empty:
        return df.copy(), 0

    work = df.copy()
    valid_mask = pd.Series(True, index=work.index)
    converted: dict[str, pd.Series] = {}

    if "second" in work.columns:
        seconds = _coerce_seconds(work["second"])
        converted["second"] = seconds
        valid_mask &= seconds.notna()

    if "event_day" in work.columns:
        days = _coerce_event_day(work["event_day"])
        converted["event_day"] = days
        valid_mask &= days.notna()

    for col in _NUMERIC_COLUMNS:
        if col in work.columns:
            values = pd.to_numeric(work[col], errors="coerce")
            converted[col] = values
            valid_mask &= values.notna()

    cleaned = work[valid_mask].copy()
    for col, series in converted.items():
        cleaned[col] = series.loc[cleaned.index]

    dropped = int((~valid_mask).sum())
    return cleaned, dropped


def clip_by_boundary(df: pd.DataFrame, boundary: dict) -> pd.DataFrame:
    """Clip rows outside of the configured spatial boundary."""

    if not boundary:
        return df.copy()

    work = df.copy()
    for axis in ("location_x", "location_y", "location_z"):
        if axis not in work.columns:
            continue
        min_key = f"{axis}_min"
        max_key = f"{axis}_max"
        lower = boundary.get(min_key)
        upper = boundary.get(max_key)
        if lower is not None:
            work = work[work[axis] >= float(lower)]
        if upper is not None:
            work = work[work[axis] <= float(upper)]
    return work


def enforce_min_seconds(df: pd.DataFrame, min_unique_seconds: int) -> None:
    """Ensure the dataframe has at least *min_unique_seconds* unique timestamps."""

    unique_seconds = df["second"].dt.floor("s").nunique() if not df.empty else 0
    if unique_seconds < min_unique_seconds:
        raise ValueError(
            f"ユニークなsecondが不足しています (必要: {min_unique_seconds}, 実際: {unique_seconds})"
        )
