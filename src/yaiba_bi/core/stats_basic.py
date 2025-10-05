"""同時接続数統計量の算出とテキスト出力を扱うモジュール。"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .spec_errors import SpecError


def compute_cc_stats(df_cc: pd.DataFrame) -> Dict[str, float]:
    """同時接続数DataFrameから基本統計量を計算して返す。

    Args:
        df_cc: 列`t`と`cc`を含むDataFrame。

    Returns:
        max/mean/median/p95 を格納した辞書。

    Raises:
        SpecError: 有効なサンプルが存在しない場合。
    """
    series = pd.to_numeric(df_cc["cc"], errors="coerce").dropna()
    if series.empty:
        raise SpecError(-2403, "no samples for stats")
    stats = {
        "max": int(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p95": float(np.percentile(series.to_numpy(), 95)),
    }
    return stats


def save_stats_txt(stats: Dict[str, float], path: str) -> str:
    """統計情報を仕様に基づくフォーマットでテキスト出力する。

    Args:
        stats: `compute_cc_stats` が返した統計辞書。
        path: 保存先ファイルパス。

    Returns:
        保存後のファイルパス。

    Raises:
        SpecError: ファイルI/Oに失敗した場合。
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"max: {int(round(stats['max']))}\n",
        f"mean: {stats['mean']:.1f}\n",
        f"median: {stats['median']:.0f}\n",
        f"p95: {stats['p95']:.0f}\n",
    ]
    try:
        out_path.write_text("".join(lines), encoding="utf-8")
        return str(out_path)
    except OSError as exc:  # noqa: BLE001
        raise SpecError(-2704, f"failed to save stats: {exc}")
