"""イベントログ可視化および統計出力エンジン。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "SpecError",
    "RenderConfig",
    "TrajectoryConfig",
    "EventLogVisualizer",
]


@dataclass(frozen=True)
class ErrorInfo:
    """エラーコードとメッセージを保持する軽量データ構造。"""
    code: int
    message: str


class SpecError(Exception):
    """仕様逸脱エラーを表す例外。

    Args:
        code: 仕様書で定義されたエラーコード。
        message: エラーの詳細説明。

    Attributes:
        info: エラーコードとメッセージをまとめた `ErrorInfo`。
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.info = ErrorInfo(code=code, message=message)

    @property
    def code(self) -> int:  # pragma: no cover
        return self.info.code

    @property
    def message(self) -> str:  # pragma: no cover
        return self.info.message


@dataclass(frozen=True)
class RenderConfig:
    """レンダリング出力の共通設定。

    Args:
        event_day: イベント日 (YYYY-MM-DD)。
        filename: 出力ファイルの識別子。
        dpi: 画像DPI。
        width_px: 画像幅(px)。
        height_px: 画像高さ(px)。
    """

    event_day: str
    filename: str
    dpi: int = 144
    width_px: int = 1280
    height_px: int = 720

    @property
    def size_inches(self) -> Tuple[float, float]:
        """インチ単位の描画サイズ。

        Returns:
            (幅, 高さ) をインチで表したタプル。
        """
        return (self.width_px / self.dpi, self.height_px / self.dpi)


@dataclass(frozen=True)
class TrajectoryConfig:
    """軌跡描画に関する設定。

    Args:
        color_scheme: 軌跡の着色方式。
        break_gap_factor: 線分分割に用いる時間ギャップ倍率。
        start_marker_size_px: 始点マーカー直径(px)。
        end_marker_size_px: 終点マーカー一辺(px)。
        filter_user_ids: 描画対象とするユーザーIDのリスト。
        bounds: 描画範囲を上書きする境界辞書。
        fit_mode: フィット方式 (未使用)。
        margin_px: 余白(px)。
        clip_oob: 範囲外データの取り扱い。
    """

    color_scheme: Literal["by_user", "by_speed", "by_time"] = "by_user"
    break_gap_factor: float = 3.0
    start_marker_size_px: int = 6
    end_marker_size_px: int = 10
    filter_user_ids: Optional[List[str]] = None
    bounds: Optional[Dict[str, float]] = None
    fit_mode: Literal["fit", "fill", "stretch"] = "fit"
    margin_px: int = 0
    clip_oob: bool = True


class _Naming:
    """成果物ファイルのパス組み立てユーティリティ。"""

    RESULT_ROOT: Path = Path("content/YAIBA_data/output")

    def ensure_dirs(self) -> None:
        """必要なディレクトリを作成する。"""

        self.RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    def cc_png_path(self, event_day: str, filename: str) -> Path:
        """同時接続数PNGの出力パス。

        Args:
            event_day: 出力対象イベント日。
            filename: ファイル識別子。

        Returns:
            生成されるPNGの `Path`。
        """

        self.ensure_dirs()
        return self.RESULT_ROOT / f"cc_line_{event_day}_{filename}.png"

    def trajectory_png_path(self, event_day: str, filename: str) -> Path:
        """軌跡PNGの出力パス。

        Args:
            event_day: 出力対象イベント日。
            filename: ファイル識別子。

        Returns:
            生成されるPNGの `Path`。
        """

        self.ensure_dirs()
        return self.RESULT_ROOT / f"traj_{event_day}_{filename}.png"

    def stats_txt_path(self, event_day: str, filename: str) -> Path:
        """統計TXTの出力パス。

        Args:
            event_day: 出力対象イベント日。
            filename: ファイル識別子。

        Returns:
            生成されるテキストファイルの `Path`。
        """

        self.ensure_dirs()
        return self.RESULT_ROOT / f"stats_{event_day}_{filename}.txt"


class EventLogVisualizer:
    """イベントログ可視化処理を提供するクラス。"""

    def __init__(self, render_config: RenderConfig, trajectory_config: TrajectoryConfig) -> None:
        """設定クラスを受け取りインスタンス化する。

        Args:
            render_config: レンダリング設定。
            trajectory_config: 軌跡描画設定。
        """

        self.render_config = render_config
        self.trajectory_config = trajectory_config
        self._naming = _Naming()

    @staticmethod
    def _percentile(values: np.ndarray, q: float) -> float:
        """NumPyバージョン差異を吸収した分位点計算。

        Args:
            values: 分位点を計算する数値配列。
            q: 取得したい百分位。

        Returns:
            分位点の値。
        """

        try:
            return float(np.percentile(values, q, method="linear"))
        except TypeError:  # numpy<1.22
            return float(np.percentile(values, q, interpolation="linear"))

    @staticmethod
    def _normalize_area(area: Any) -> Dict[str, float]:
        """描画領域情報を辞書形式へ揃える。

        Args:
            area: `{"min_x":..., "max_x":..., ...}` 形式の辞書、または
                `x_min/x_max/z_min/z_max` 属性を持つオブジェクト。

        Returns:
            境界情報を浮動小数で格納した辞書。
        """
        if isinstance(area, dict):
            return area
        attrs = {"min_x": "x_min", "max_x": "x_max", "min_z": "z_min", "max_z": "z_max"}
        converted: Dict[str, float] = {}
        for key, attr in attrs.items():
            if hasattr(area, attr):
                converted[key] = float(getattr(area, attr))
        return converted

    @staticmethod
    def _build_concurrency(df_att: pd.DataFrame) -> pd.DataFrame:
        """参加・離脱ログから同時接続数のタイムラインを生成する。

        Args:
            df_att: 列 `second`, `action` を含む参加離脱ログ。

        Returns:
            列 `second`, `cc` を持つ DataFrame。1秒刻みの累積接続数。

        Raises:
            SpecError: 必須列不足、または算出結果が空の場合。
        """
        required = {"second", "action"}
        if not required.issubset(df_att.columns):
            raise SpecError(-2101, "attendance に必須列 second/action が存在しません")
        df_att = df_att.copy()
        df_att["second"] = pd.to_datetime(df_att["second"])
        df_att = df_att.sort_values("second")
        action_map = {"join": 1, "left": -1}
        df_att["delta"] = df_att["action"].map(action_map)
        df_att = df_att.dropna(subset=["delta"])
        events = df_att.groupby("second")["delta"].sum().sort_index()
        if events.empty:
            raise SpecError(-2101, "attendance から同時接続数を構成できません")
        timeline = events.reindex(pd.date_range(events.index.min(), events.index.max(), freq="s"), fill_value=0.0)
        cc_series = timeline.cumsum().astype(float)
        df_cc = pd.DataFrame({"second": timeline.index, "cc": cc_series})
        return df_cc

    def compute_cc_stats(self, df_cc: pd.DataFrame) -> dict:
        """同時接続数系列の統計値を算出する。

        Args:
            df_cc: 同時接続数を含むDataFrame。

        Returns:
            最大値・平均値・中央値・95パーセンタイルを格納した辞書。

        Raises:
            SpecError: 検証エラー時。
        """

        series = df_cc["cc"].dropna()
        values = series.to_numpy(dtype=float)
        return {
            "max": int(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": self._percentile(values, 95.0),
        }

    def render_concurrency_png(self, df_cc: pd.DataFrame) -> str:
        """同時接続数折れ線グラフを描画する。

        Args:
            df_cc: 列 `cc` と時間列 (`t` または `second`) を含む DataFrame。

        Returns:
            保存したPNGのパス文字列。

        Raises:
            SpecError: 検証エラー時。
        """

        rcfg = self.render_config
        out_path = self._naming.cc_png_path(rcfg.event_day, rcfg.filename)

        fig, ax = plt.subplots(figsize=rcfg.size_inches, dpi=rcfg.dpi)
        time_col = "t" if "t" in df_cc.columns else "second"
        if time_col not in df_cc.columns:
            raise SpecError(-2101, "cc データに時間列が存在しません")
        ax.plot(df_cc[time_col], df_cc["cc"], linewidth=1.5, color="#1f77b4")
        ax.set_xlabel("時間")
        ax.set_ylabel("同時接続数")
        ax.grid(True, linestyle="--", alpha=0.3)

        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_path, dpi=rcfg.dpi)
        plt.close(fig)
        return str(out_path)

    def _fit_bounds(self, bounds: Dict[str, float]) -> Tuple[float, float, float, float]:
        """描画範囲を設定値に従って調整する。

        Args:
            bounds: 基準となるエリア境界。

        Returns:
            (min_x, max_x, min_z, max_z) のタプル。
        """

        cfg_bounds = self.trajectory_config.bounds or {}
        return (
            cfg_bounds.get("min_x", bounds["min_x"]),
            cfg_bounds.get("max_x", bounds["max_x"]),
            cfg_bounds.get("min_z", bounds["min_z"]),
            cfg_bounds.get("max_z", bounds["max_z"]),
        )

    def _filter_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """ユーザーIDフィルタを適用する。

        Args:
            df: フィルタ対象のDataFrame。

        Returns:
            フィルタ後のDataFrame。
        """

        user_ids = self.trajectory_config.filter_user_ids
        if user_ids:
            return df[df["user_id"].isin(user_ids)].copy()
        return df.copy()

    @staticmethod
    def _apply_breaks(df: pd.DataFrame, break_gap_factor: float) -> Iterable[pd.DataFrame]:
        """時間ギャップに基づき線分を分割する。

        Args:
            df: 分割対象のDataFrame。
            break_gap_factor: ギャップ閾値を決定する倍率。

        Returns:
            分割後のDataFrameリスト。
        """

        if df.empty:
            return []
        time_col = "t" if "t" in df.columns else "second"
        if time_col not in df.columns:
            raise SpecError(-2101, "軌跡データに時間列が存在しません")
        df_sorted = df.sort_values(time_col)
        times = df_sorted[time_col].astype("int64") / 1e9
        diffs = np.diff(times)
        if len(diffs) == 0:
            return [df_sorted]

        median_gap = np.median(diffs)
        if median_gap <= 0:
            median_gap = 1.0
        threshold = median_gap * break_gap_factor

        segments = np.split(df_sorted, np.where(diffs > threshold)[0] + 1)
        return [segment for segment in segments if not segment.empty]

    @staticmethod
    def _color_for(index: int) -> str:
        """ユーザーごとの色を決定する補助関数。

        Args:
            index: グループインデックス。

        Returns:
            描画に使用する色コード。
        """

        palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
        return palette[index % len(palette)]

    def render_trajectory2d_png(self, df_pos: pd.DataFrame, area: Dict[str, float]) -> str:
        """ユーザー位置軌跡を2Dに描画し保存する。

        Args:
            df_pos: 列 `second`, `user_id`, `location_x`, `location_z` を含む位置情報。
            area: 描画エリア境界辞書。

        Returns:
            保存したPNGのパス文字列。

        Raises:
            SpecError: 検証エラーや領域不正時。
        """

        required = {"second", "user_id", "location_x", "location_z"}
        if not required.issubset(df_pos.columns):
            missing = required - set(df_pos.columns)
            raise SpecError(-2101, f"df_pos に必須列 {sorted(missing)} が存在しません")
        df_pos = df_pos.copy()
        df_pos["second"] = pd.to_datetime(df_pos["second"])
        df_pos = self._filter_users(df_pos)
        if df_pos.empty:
            raise SpecError(-2101, "df_pos が空です")

        rcfg = self.render_config
        tcfg = self.trajectory_config
        out_path = self._naming.trajectory_png_path(rcfg.event_day, rcfg.filename)

        fig, ax = plt.subplots(figsize=rcfg.size_inches, dpi=rcfg.dpi)
        min_x, max_x, min_z, max_z = self._fit_bounds(area)

        width = max_x - min_x
        height = max_z - min_z
        if width <= 0 or height <= 0:
            raise SpecError(-2301, "描画領域が不正です")

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_z, max_z)

        handles = []
        labels = []

        for idx, (user_id, group) in enumerate(df_pos.groupby("user_id")):
            color = self._color_for(idx)
            group = group.sort_values("second")
            segments = self._apply_breaks(group, tcfg.break_gap_factor)
            line_handle = None
            for seg in segments:
                [line_handle] = ax.plot(seg["location_x"], seg["location_z"], linewidth=1.5, color=color)
            start = group.iloc[0]
            end = group.iloc[-1]
            ax.scatter(start["location_x"], start["location_z"], s=tcfg.start_marker_size_px**2, marker="o", color=color, zorder=3)
            ax.scatter(end["location_x"], end["location_z"], s=tcfg.end_marker_size_px**2, marker="^", color=color, zorder=3)
            if line_handle is not None:
                handles.append(line_handle)
                labels.append(str(user_id))

        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.3)

        if handles:
            ax.legend(handles, labels, loc="upper right")

        fig.tight_layout()
        fig.savefig(out_path, dpi=rcfg.dpi)
        plt.close(fig)
        return str(out_path)

    def save_stats_txt(self, stats: Dict[str, float]) -> str:
        """統計値をテキストファイルとして保存する。

        Args:
            stats: `max` `mean` `median` `p95` を含む辞書。

        Returns:
            保存したテキストファイルのパス文字列。
        """

        rcfg = self.render_config
        out_path = self._naming.stats_txt_path(rcfg.event_day, rcfg.filename)
        lines = [
            f"max: {int(round(stats['max']))}",
            f"mean: {stats['mean']:.1f}",
            f"median: {stats['median']:.1f}",
            f"p95: {stats['p95']:.1f}",
        ]
        Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(out_path)

    def run(self, attendance: pd.DataFrame, position: pd.DataFrame, area: Any) -> Dict[str, str]:
        """一連の可視化・統計出力処理を実行する。

        Args:
            attendance: 列 `second` と `action` を含む参加／離脱ログ。
            position: 位置情報 DataFrame。
            area: 描画エリアを示す辞書または `TestArea` 等のオブジェクト。

        Returns:
            生成した成果物パス (`cc_png`, `traj_png`, `stats_txt`) をまとめた辞書。
        """

        df_cc = self._build_concurrency(attendance)
        stats = self.compute_cc_stats(df_cc)
        cc_png = self.render_concurrency_png(df_cc)

        try:
            traj_png = self.render_trajectory2d_png(position, self._normalize_area(area))
        except SpecError:
            traj_png = ""

        stats_txt = self.save_stats_txt(stats)

        return {
            "cc_png": cc_png,
            "traj_png": traj_png,
            "stats_txt": stats_txt,
        }
