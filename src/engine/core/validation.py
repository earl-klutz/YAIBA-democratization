"""Utility helpers for validating intermediate tables used by the C工程."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import pandas as pd

_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "second",
    "user_id",
    "location_x",
    "location_z",
    "event_day",
)

_OPTIONAL_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "location_y",
    "rotation_1",
    "rotation_2",
    "rotation_3",
    "location_dx",
    "location_dy",
    "location_dz",
)


class ValidationError(ValueError):
    """Raised when the incoming DataFrame does not satisfy hard requirements."""


def require_columns(df: pd.DataFrame, cols: Iterable[str] | None = None) -> None:
    """Ensure that all columns required by downstream processing exist.

    Parameters
    ----------
    df:
        The DataFrame to check.
    cols:
        Additional columns that must be present.  When *cols* is ``None`` the
        default set defined by the specification is used.
    """

    required = set(_REQUIRED_COLUMNS if cols is None else cols)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValidationError(f"中間形式テーブルに必須列が不足しています: {', '.join(missing)}")


def drop_invalid_types(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows that contain non-convertible values.

    Returns
    -------
    (DataFrame, int)
        A tuple consisting of the cleaned DataFrame and the number of rows that
        were removed during the process.
    """

    working = df.copy()
    removed_mask = pd.Series(False, index=working.index)

    # second – convert to timezone-aware UTC timestamps.
    seconds = pd.to_datetime(working["second"], errors="coerce", utc=True)
    removed_mask |= seconds.isna()
    working["second"] = seconds

    # event_day – keep as date object.
    if "event_day" in working.columns:
        event_day = pd.to_datetime(working["event_day"], errors="coerce")
        removed_mask |= event_day.isna()
        working["event_day"] = event_day.dt.date

    # User ID is treated as a string identifier.
    working["user_id"] = working["user_id"].astype(str)

    numeric_required = ("location_x", "location_z")
    numeric_optional = [col for col in _OPTIONAL_NUMERIC_COLUMNS if col in working.columns]

    for column in numeric_required:
        values = pd.to_numeric(working[column], errors="coerce")
        removed_mask |= values.isna()
        working[column] = values

    for column in numeric_optional:
        values = pd.to_numeric(working[column], errors="coerce")
        working[column] = values

    cleaned = working.loc[~removed_mask].copy()
    removed = int(removed_mask.sum())
    cleaned.reset_index(drop=True, inplace=True)
    return cleaned, removed


def clip_by_boundary(df: pd.DataFrame, boundary: dict | None) -> pd.DataFrame:
    """Remove rows that fall outside the world boundary definition."""

    if not boundary:
        return df.copy()

    working = df.copy()
    mask = pd.Series(True, index=working.index)

    for axis in ("x", "y", "z"):
        min_key = f"location_{axis}_min"
        max_key = f"location_{axis}_max"
        min_value = boundary.get(min_key)
        max_value = boundary.get(max_key)

        column = f"location_{axis}"
        if min_value is not None:
            mask &= working[column] >= float(min_value)
        if max_value is not None:
            mask &= working[column] <= float(max_value)

    clipped = working.loc[mask].copy()
    clipped.reset_index(drop=True, inplace=True)
    return clipped


def enforce_min_seconds(df: pd.DataFrame, min_unique_seconds: int) -> None:
    """Ensure that the DataFrame spans at least *min_unique_seconds* seconds."""

    if min_unique_seconds <= 0:
        return

    if "second" not in df.columns:
        raise ValidationError("`second` 列が見つからないため、秒数の検証ができません。")

    unique_seconds = df["second"].dt.floor("S").nunique()
    if unique_seconds < min_unique_seconds:
        raise ValidationError(
            "ユニークな秒数が不足しています"
            f" (必要: {min_unique_seconds}, 実際: {unique_seconds})."
        )
