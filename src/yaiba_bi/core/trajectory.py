"""位置軌跡を2D平面に描画しPNG出力するためのユーティリティ。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .spec_errors import SpecError
from .validation import validate_df_pos


@dataclass(frozen=True)
class TrajectoryPlotter:
    """位置ログから軌跡PNGを生成するプロットユーティリティ。"""

    area: Dict[str, float]
    style: Dict[str, object]
    io: Dict[str, object]

    def __post_init__(self) -> None:
        """デフォルトスタイルパラメータを適用して初期化する。

        Raises:
            SpecError: スタイル設定に矛盾がある場合。
        """
        defaults = {
            "break_gap_factor": 3.0,
            "start_marker_size_px": 6,
            "end_marker_size_px": 10,
            "line_width": 1.5,
            "clip_oob": True,
            "color_scheme": "by_user",
        }
        style = dict(defaults | self.style)
        object.__setattr__(self, "style", style)

    def clip_by_area(self, df_pos: pd.DataFrame, area: Dict[str, float]) -> pd.DataFrame:
        """領域外の点を除外し、描画対象に適合するデータを返す。

        Args:
            df_pos: 位置ログDataFrame。
            area: `min_x`/`max_x`/`min_z`/`max_z` を含む境界辞書。

        Returns:
            領域内のデータだけに絞ったDataFrame。

        Raises:
            SpecError: 定義領域内に点が存在しない場合。
        """
        clipped = df_pos[
            (df_pos["x"].between(area["min_x"], area["max_x"], inclusive="both"))
            & (df_pos["z"].between(area["min_z"], area["max_z"], inclusive="both"))
        ].copy()
        if clipped.empty:
            raise SpecError(-2302, "no positions within area")
        return clipped

    def segment_by_gap(self, df_pos: pd.DataFrame, gap_factor: float) -> List[pd.DataFrame]:
        """ユーザー毎に時間ギャップで分割した軌跡セグメント群を返す。

        Args:
            df_pos: 位置ログDataFrame。
            gap_factor: 時間差を閾値化する倍率。

        Returns:
            連続区間ごとに分割したDataFrameリスト。

        Raises:
            SpecError: 有効な区間が生成されない場合。
        """
        df = df_pos.copy()
        df["t"] = pd.to_datetime(df["t"], errors="coerce", utc=True)
        df.sort_values(["user_id", "t"], inplace=True, kind="mergesort")
        segments: List[pd.DataFrame] = []
        for user_id, group in df.groupby("user_id", sort=False):
            dt = group["t"].diff().dt.total_seconds()
            median_dt = float(np.nanmedian(dt.to_numpy())) if dt.notna().any() else 1.0
            threshold = gap_factor * max(median_dt, 1.0)
            breaks = dt.isna() | (dt > threshold)
            indices = group.index[breaks].tolist()
            if not indices or indices[0] != group.index.min():
                indices = [group.index.min()] + indices
            indices.append(group.index.max() + 1)
            for start, end in zip(indices[:-1], indices[1:]):
                segment = group.loc[(group.index >= start) & (group.index < end)]
                if len(segment) >= 2:
                    segments.append(segment.copy())
        if not segments:
            raise SpecError(-2303, "insufficient trajectory segments")
        return segments

    def plot(self, df_pos: pd.DataFrame) -> plt.Figure:
        """位置データを描画し、Matplotlib Figureを返す。

        Args:
            df_pos: 位置ログDataFrame。

        Returns:
            描画済みFigure。

        Raises:
            SpecError: 入力不正・描画失敗時。
        """
        validate_df_pos(df_pos)
        clipped = self.clip_by_area(df_pos, self.area)
        segments = self.segment_by_gap(clipped, float(self.style.get("break_gap_factor", 3.0)))
        fig, ax = plt.subplots(
            figsize=(self.style.get("width_px", 1280) / self.style.get("dpi", 144),
                     self.style.get("height_px", 720) / self.style.get("dpi", 144)),
            dpi=self.style.get("dpi", 144),
        )
        try:
            ax.set_facecolor("white")
            x_min = float(self.area["min_x"])
            x_max = float(self.area["max_x"])
            z_min = float(self.area["min_z"])
            z_max = float(self.area["max_z"])
            if x_min == x_max:
                pad = max(1.0, abs(x_min) * 0.05 or 1.0)
                x_min -= pad
                x_max += pad
            if z_min == z_max:
                pad = max(1.0, abs(z_min) * 0.05 or 1.0)
                z_min -= pad
                z_max += pad
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(z_min, z_max)
            ax.set_aspect("equal")
            colors = self._color_cycle({seg["user_id"].iloc[0] for seg in segments})
            for segment in segments:
                user_id = segment["user_id"].iloc[0]
                color = colors.get(user_id, "#1f77b4")
                ax.plot(
                    segment["x"],
                    segment["z"],
                    linewidth=float(self.style["line_width"]),
                    color=color,
                    alpha=0.85,
                    solid_capstyle="round",
                )
                ax.scatter(
                    segment["x"].iloc[0],
                    segment["z"].iloc[0],
                    s=self.style["start_marker_size_px"] ** 2,
                    c="#2ca02c",
                    marker="o",
                    zorder=3,
                    clip_on=self.style["clip_oob"],
                )
                ax.scatter(
                    segment["x"].iloc[-1],
                    segment["z"].iloc[-1],
                    s=self.style["end_marker_size_px"] ** 2,
                    c="#d62728",
                    marker="^",
                    zorder=3,
                    clip_on=self.style["clip_oob"],
                )
            ax.set_title("Trajectories (x-z)")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("z [m]")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            return fig
        except Exception as exc:  # noqa: BLE001
            plt.close(fig)
            raise SpecError(-2303, f"failed to render trajectories: {exc}")

    def save_png(self, fig: plt.Figure, path: str) -> str:
        """描画結果をPNGとして保存する。

        Args:
            fig: 保存対象のFigure。
            path: 保存先パス。

        Returns:
            保存後パス。

        Raises:
            SpecError: ファイル保存に失敗した場合。
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(out_path, dpi=self.style.get("dpi", 144), bbox_inches="tight")
            return str(out_path)
        except OSError as exc:  # noqa: BLE001
            raise SpecError(-2702, f"failed to save trajectory png: {exc}")
        finally:
            plt.close(fig)

    def run(self, df_pos: pd.DataFrame, naming) -> Dict[str, str]:
        """軌跡図生成から保存までを一括実行する。

        Args:
            df_pos: 位置ログDataFrame。
            naming: 命名ヘルパー。

        Returns:
            `{"traj_png": 保存パス}` を含む辞書。
        """
        fig = self.plot(df_pos)
        path = naming.build_output_path("traj2D", "png")
        saved = self.save_png(fig, path)
        return {"traj_png": saved}

    @staticmethod
    def _color_cycle(user_ids: Iterable[str]) -> Dict[str, str]:
        """ユーザーID集合に対して安定した色割り当てを提供する。

        Args:
            user_ids: 色割り当て対象のユーザー集合。

        Returns:
            ユーザーID→色のマッピング。
        """
        cmap = plt.get_cmap("tab20")
        return {uid: cmap(i % cmap.N) for i, uid in enumerate(user_ids)}
