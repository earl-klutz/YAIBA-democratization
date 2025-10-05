from __future__ import annotations

from typing import Iterable

import pandas as pd

from .spec_errors import SpecError


REQUIRED_CC_COLUMNS = ("t", "cc")
REQUIRED_POS_COLUMNS = ("t", "user_id", "x", "z")


def _check_columns(df: pd.DataFrame, required: Iterable[str], *, code: int) -> None:
    """要求列が揃っているかを確認する。

    Args:
        df: チェック対象DataFrame。
        required: 必須列の反復可能オブジェクト。
        code: 不足時に送出するエラーコード。

    Raises:
        SpecError: DataFrameでない、または列不足の場合。
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise SpecError(-2101, "input is not a DataFrame")
    missing = set(required) - set(df.columns)
    if missing:
        raise SpecError(code, f"missing required columns: {sorted(missing)}")


def validate_df_cc(df_cc: pd.DataFrame) -> None:
    """同時接続数テーブルの必須列・非負・時系列昇順を検証する。

    Args:
        df_cc: 同時接続数DataFrame。

    Raises:
        SpecError: 列不足、値不正、時系列逆順などが検出された場合。
    """
    _check_columns(df_cc, REQUIRED_CC_COLUMNS, code=-2102)
    if df_cc["cc"].isna().all():
        raise SpecError(-2204, "concurrency series is empty")
    if (pd.to_numeric(df_cc["cc"], errors="coerce") < 0).any():
        raise SpecError(-2204, "cc must be non-negative")
    if not pd.to_datetime(df_cc["t"], errors="coerce").is_monotonic_increasing:
        df_cc.sort_values("t", inplace=True, kind="mergesort")


def validate_df_pos(df_pos: pd.DataFrame) -> None:
    """位置ログの必須列と有効な座標値を検証し、時系列順に整列する。

    Args:
        df_pos: 位置ログDataFrame。

    Raises:
        SpecError: 列不足、座標が全てNaNなどの異常がある場合。
    """
    _check_columns(df_pos, REQUIRED_POS_COLUMNS, code=-2102)
    coords = df_pos[["x", "z"]]
    if coords.isna().all(axis=None):
        raise SpecError(-2301, "trajectory positions are entirely null")
    df_pos.sort_values(["user_id", "t"], inplace=True, kind="mergesort")
