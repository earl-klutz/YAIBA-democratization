"""同時接続数系列の描画およびPNG出力を提供するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .spec_errors import SpecError
from .validation import validate_df_cc
from .naming import NamingHelper


@dataclass(frozen=True)
class ConcurrencyPlotter:
    """同時接続数DataFrameから折れ線グラフを生成する描画クラス。"""

    render: Dict[str, object]
    io: Dict[str, object]

    def __post_init__(self) -> None:
        """描画設定を初期化する。

        Raises:
            SpecError: `agg` が許容外の場合。
        """
        defaults = {
            "dpi": 144,
            "width_px": 1280,
            "height_px": 720,
            "resample_sec": 1,
            "gap_factor": 5,
            "agg": "mean",
        }
        render = dict(defaults | self.render)
        object.__setattr__(self, "render", render)

    def resample_cc(self, df_cc: pd.DataFrame, sec: Optional[int] = None) -> pd.DataFrame:
        """所定粒度で再サンプリングした同時接続数を返す。

        Args:
            df_cc: 列`t`と`cc`を持つ同時接続数テーブル。
            sec: 再サンプリング粒度（秒）。`None`なら設定値を利用。

        Returns:
            再サンプリング後のDataFrame。

        Raises:
            SpecError: 粒度や集計方法が不正な場合。
        """
        if sec is None:
            sec = int(self.render.get("resample_sec", 1))
        if sec <= 0:
            raise SpecError(-2201, "resample_sec must be positive")
        series = df_cc.copy()
        series["t"] = pd.to_datetime(series["t"], errors="coerce", utc=True)
        series.set_index("t", inplace=True)
        agg = self.render.get("agg", "mean")
        if agg not in {"mean", "sum"}:
            raise SpecError(-2201, "unsupported aggregation")
        rule = f"{sec}S"
        if agg == "mean":
            resampled = series.resample(rule).mean(numeric_only=True)
        else:
            resampled = series.resample(rule).sum(numeric_only=True)
        resampled = resampled.dropna()
        resampled.reset_index(inplace=True)
        resampled.rename(columns={"t": "t", "cc": "cc"}, inplace=True)
        return resampled

    def plot(self, df_cc: pd.DataFrame) -> plt.Figure:
        """同時接続数の折れ線グラフを描画する。

        Args:
            df_cc: 同時接続数データ。

        Returns:
            描画済みのMatplotlib Figure。

        Raises:
            SpecError: 入力が不正、または描画処理に失敗した場合。
        """
        validate_df_cc(df_cc)
        data = self.resample_cc(df_cc)
        if data.empty:
            raise SpecError(-2202, "no data after resampling")
        fig, ax = plt.subplots(
            figsize=(self.render["width_px"] / self.render["dpi"],
                     self.render["height_px"] / self.render["dpi"]),
            dpi=self.render["dpi"],
        )
        try:
            t = pd.to_datetime(data["t"], utc=True)
            y = pd.to_numeric(data["cc"], errors="coerce")
            dt = t.diff().dt.total_seconds()
            median_dt = float(np.nanmedian(dt.to_numpy())) if dt.notna().any() else 1.0
            gap_threshold = float(self.render.get("gap_factor", 5)) * max(median_dt, 1.0)
            y_plot = y.copy()
            y_plot[dt > gap_threshold] = np.nan
            ax.plot(t, y_plot, linewidth=1.5, color="#1f77b4")
            ax.set_title("Concurrent Users")
            ax.set_xlabel("Time (JST)")
            ax.set_ylabel("Users [count]")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            return fig
        except Exception as exc:  # noqa: BLE001
            plt.close(fig)
            raise SpecError(-2202, f"failed to plot concurrency: {exc}")

    def save_png(self, fig: plt.Figure, path: str) -> str:
        """FigureをPNG保存する。

        Args:
            fig: 保存対象のFigure。
            path: 保存先パス。

        Returns:
            保存後のファイルパス。

        Raises:
            SpecError: ファイル保存に失敗した場合。
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(out_path, dpi=self.render["dpi"], bbox_inches="tight")
            return str(out_path)
        except OSError as exc:
            raise SpecError(-2702, f"failed to save concurrency png: {exc}")
        finally:
            plt.close(fig)

    def run(self, df_cc: pd.DataFrame, naming: NamingHelper) -> Dict[str, str]:
        """PNG生成から保存までを一括実行する。

        Args:
            df_cc: 同時接続数データ。
            naming: 保存パス生成に使用するNamingHelper。

        Returns:
            `{"cc_png": 保存パス}` の辞書。
        """
        validate_df_cc(df_cc)
        fig = self.plot(df_cc)
        path = naming.build_output_path("cc_line", "png")
        saved = self.save_png(fig, path)
        return {"cc_png": saved}
