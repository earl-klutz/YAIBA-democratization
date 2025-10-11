from __future__ import annotations
import numpy as np
import pandas as pd

# エラーコード（15章）に準拠
EC_INPUT_FORMAT = -2102
EC_INPUT_TYPE_MISMATCH = -2104
EC_STATS_SAMPLE_SHORT = -2403

REQUIRED_COLS = ["second", "user_id", "location_x", "location_y", "location_z", "event_day"]

class PipelineError(Exception):
    def __init__(self, code: int, msg: str) -> None:
        super().__init__(f"EC={code} {msg}")
        self.code = code
        self.msg = msg

def require_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise PipelineError(EC_INPUT_FORMAT, f"必須列欠如: {missing}")

def _to_jst_series(dt_like: pd.Series) -> pd.Series:
    s = pd.to_datetime(dt_like, errors="coerce")
    try:
        if getattr(s.dtype, "tz", None) is not None:
            return s.dt.tz_convert("Asia/Tokyo")
        return s.dt.tz_localize("Asia/Tokyo")
    except Exception:
        return pd.to_datetime(s, errors="coerce")

def drop_invalid_types(df: pd.DataFrame, logger) -> pd.DataFrame:
    df = df.copy()
    df["second"] = _to_jst_series(df["second"])      # 入力はJST基準で運用
    for c in ["location_x", "location_y", "location_z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["second", "user_id", "location_x", "location_z"])
    removed = before - len(df)
    if removed > 0:
        logger.warning(f"EC={EC_INPUT_TYPE_MISMATCH} drop_invalid_rows count={removed}")
    return df

def clip_by_boundary(df: pd.DataFrame, boundary: dict | None) -> pd.DataFrame:
    if not boundary:
        return df
    xmn, xmx = boundary.get("location_x_min", -np.inf), boundary.get("location_x_max", np.inf)
    ymn, ymx = boundary.get("location_y_min", -np.inf), boundary.get("location_y_max", np.inf)
    zmn, zmx = boundary.get("location_z_min", -np.inf), boundary.get("location_z_max", np.inf)
    m = (df["location_x"].between(xmn, xmx) &
         df["location_y"].between(ymn, ymx) &
         df["location_z"].between(zmn, zmx))
    return df.loc[m].copy()

def enforce_min_seconds(df: pd.DataFrame, min_unique_seconds: int) -> None:
    uniq = df["second"].dt.floor("s").nunique()
    if uniq < min_unique_seconds:
        raise PipelineError(EC_STATS_SAMPLE_SHORT, f"ユニークsecond不足: {uniq} < {min_unique_seconds}")
