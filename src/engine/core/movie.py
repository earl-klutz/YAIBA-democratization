"""動画生成(C-2工程)のコア実装。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from zoneinfo import ZoneInfo

from . import logging_util, naming, validation


@dataclass(slots=True)
class MovieStats:
    """保持しておきたい統計情報。"""

    records: int
    excluded_rows: int
    unique_users: int
    unique_seconds: int
    frames: int


class MovieGenerator:
    """C工程でx–z平面のアニメーションを生成するクラス。"""

    def __init__(
        self,
        boundary: dict | None,
        outlier: dict | None,
        theme: dict | None,
        movie: dict | None,
        point: dict | None,
        trail: dict | None,
        io: dict | None,
    ) -> None:
        self.boundary = boundary or {}
        self.outlier = outlier or {}
        self.theme = {
            "palette": "tab10",
            "bg_color": "#eeeeee",
            "font": "Meiryo",
            "font_size": 16,
        }
        if theme:
            self.theme.update(theme)

        self.movie_config = {
            "duration_real": 10_800,
            "format": "mp4",
            "fps": 30,
            "duration_sec": 60,
            "bitrate": "2M",
            "version": "c1.0",
            "min_unique_seconds": 180,
        }
        if movie:
            self.movie_config.update(movie)

        self.point_config = {"radius_px": 6, "alpha": 1.0}
        if point:
            self.point_config.update(point)

        self.trail_config = {
            "length_real_seconds": 30,
            "alpha_start": 1.0,
            "alpha_end": 0.1,
        }
        if trail:
            self.trail_config.update(trail)

        self.io_config = {"output_filename": "movie", "overwrite": False}
        if io:
            self.io_config.update(io)

        self._frame_seconds: pd.DatetimeIndex | None = None
        self._frame_groups: dict[pd.Timestamp, pd.DataFrame] = {}
        self._user_tracks: dict[str, pd.DataFrame] = {}
        self._color_map: dict[str, tuple[float, float, float, float]] = {}
        self._axis_limits: dict[str, float] | None = None
        self._stats: MovieStats | None = None
        self._event_day: str | None = None
        self._excluded_rows = 0
        self._type_drop_count = 0

    # ------------------------------------------------------------------
    # 前処理
    # ------------------------------------------------------------------
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """整形・間引き・外れ値処理後の描画用データを返す。"""

        validation.require_columns(df)
        working = df.copy()
        working.sort_values("second", inplace=True)

        working, type_drop_count = validation.drop_invalid_types(working)
        self._type_drop_count = type_drop_count
        working = validation.clip_by_boundary(working, self.boundary)
        validation.enforce_min_seconds(
            working, int(self.movie_config.get("min_unique_seconds", 0))
        )

        duration_real = int(self.movie_config.get("duration_real", 10_800))
        if duration_real <= 0:
            raise validation.ValidationError("duration_real は正の数である必要があります。")

        start_second = working["second"].min()
        limit_second = start_second + pd.Timedelta(seconds=duration_real)
        working = working[working["second"] <= limit_second].copy()

        # 再検証：フィルタ後でも秒数が足りなければエラーにする。
        validation.enforce_min_seconds(
            working, int(self.movie_config.get("min_unique_seconds", 0))
        )

        self._excluded_rows = len(df) - len(working)

        working["second_floor"] = working["second"].dt.floor("S")
        unique_seconds = pd.DatetimeIndex(sorted(working["second_floor"].unique()))
        if unique_seconds.empty:
            raise validation.ValidationError("有効な時刻情報が存在しません。")

        fps = int(self.movie_config.get("fps", 30))
        duration_sec = int(self.movie_config.get("duration_sec", 60))
        self._frame_seconds = self._build_frame_sequence(unique_seconds, fps, duration_sec)

        working["user_id"] = working["user_id"].astype(str)

        palette = plt.get_cmap(self.theme.get("palette", "tab10"))
        users = working["user_id"].unique()
        colors = {
            user: palette(idx % palette.N) if hasattr(palette, "N") else palette(idx)
            for idx, user in enumerate(users)
        }
        self._color_map = {
            user: mcolors.to_rgba(color)
            for user, color in colors.items()
        }
        working["color"] = working["user_id"].map(self._color_map)

        self._frame_groups = {
            ts: group.copy()
            for ts, group in working.groupby("second_floor")
        }
        self._user_tracks = {
            user: group.sort_values("second_floor")[
                ["second_floor", "location_x", "location_z", "color"]
            ].copy()
            for user, group in working.groupby("user_id")
        }

        self._axis_limits = self._resolve_axis_limits(working)
        self._event_day = self._extract_event_day(working)

        frame_count = len(self._frame_seconds) if self._frame_seconds is not None else 0
        self._stats = MovieStats(
            records=len(working),
            excluded_rows=self._excluded_rows,
            unique_users=len(users),
            unique_seconds=len(unique_seconds),
            frames=frame_count,
        )

        return working

    # ------------------------------------------------------------------
    def _extract_event_day(self, df: pd.DataFrame) -> str:
        value = df["event_day"].dropna().astype(str)
        if value.empty:
            return "unknown"
        return value.iloc[0]

    def _resolve_axis_limits(self, df: pd.DataFrame) -> dict[str, float]:
        if not self.boundary:
            padding = 1.0
            x_min = math.floor(df["location_x"].min() - padding)
            x_max = math.ceil(df["location_x"].max() + padding)
            z_min = math.floor(df["location_z"].min() - padding)
            z_max = math.ceil(df["location_z"].max() + padding)
            return {"x_min": x_min, "x_max": x_max, "z_min": z_min, "z_max": z_max}

        return {
            "x_min": float(self.boundary.get("location_x_min", df["location_x"].min())),
            "x_max": float(self.boundary.get("location_x_max", df["location_x"].max())),
            "z_min": float(self.boundary.get("location_z_min", df["location_z"].min())),
            "z_max": float(self.boundary.get("location_z_max", df["location_z"].max())),
        }

    def _build_frame_sequence(
        self,
        unique_seconds: pd.DatetimeIndex,
        fps: int,
        duration_sec: int,
    ) -> pd.DatetimeIndex:
        if fps <= 0 or duration_sec <= 0:
            raise validation.ValidationError("fps と duration_sec は正の整数である必要があります。")

        total_frames = fps * duration_sec
        start = unique_seconds[0]
        data_span = unique_seconds[-1] - start
        span_seconds = max(data_span.total_seconds(), 1.0)
        duration_real = min(float(self.movie_config.get("duration_real", span_seconds)), span_seconds)
        compression = duration_real / duration_sec
        time_step = compression / fps

        offsets = np.arange(total_frames) * time_step
        target_times = start + pd.to_timedelta(offsets, unit="s")

        seconds_int = unique_seconds.asi8
        frame_seconds: list[pd.Timestamp] = []
        for ts in target_times:
            idx = np.searchsorted(seconds_int, ts.value, side="left")
            if idx >= len(unique_seconds):
                idx = len(unique_seconds) - 1
            frame_seconds.append(unique_seconds[idx])

        return pd.DatetimeIndex(frame_seconds)

    # ------------------------------------------------------------------
    # 描画処理
    # ------------------------------------------------------------------
    def render(self, df: pd.DataFrame) -> animation.FuncAnimation:
        """アニメーションを生成して返す。"""

        if self._frame_seconds is None:
            raise RuntimeError("prepare() を実行してから render() を呼び出してください。")

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(self.theme.get("bg_color", "#ffffff"))
        ax.set_facecolor(self.theme.get("bg_color", "#ffffff"))
        ax.set_xlabel("X [m]", fontname=self.theme.get("font"), fontsize=self.theme.get("font_size"))
        ax.set_ylabel("Z [m]", fontname=self.theme.get("font"), fontsize=self.theme.get("font_size"))

        limits = self._axis_limits or {}
        ax.set_xlim(limits.get("x_min", df["location_x"].min()), limits.get("x_max", df["location_x"].max()))
        ax.set_ylim(limits.get("z_min", df["location_z"].min()), limits.get("z_max", df["location_z"].max()))
        ax.set_title("YAIBA Visitor Flow (x–z plane)", fontname=self.theme.get("font"), fontsize=self.theme.get("font_size", 16))
        ax.grid(True, linestyle="--", alpha=0.3)

        scatter = ax.scatter([], [], s=self._point_size(), c=[], alpha=self.point_config.get("alpha", 1.0))
        trail_collection = LineCollection([], linewidths=2)
        ax.add_collection(trail_collection)

        time_text = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=self.theme.get("font_size", 16),
            fontname=self.theme.get("font"),
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        trail_window = pd.Timedelta(seconds=self.trail_config.get("length_real_seconds", 30))

        def _frame_dataset(timestamp: pd.Timestamp) -> pd.DataFrame:
            base = self._frame_groups.get(timestamp)
            if base is None:
                return pd.DataFrame(columns=df.columns)
            return base

        def init() -> Iterable[Any]:
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_array(np.array([]))
            trail_collection.set_segments([])
            time_text.set_text("")
            return scatter, trail_collection, time_text

        def update(frame_index: int) -> Iterable[Any]:
            timestamp = self._frame_seconds[frame_index]
            frame_df = _frame_dataset(timestamp)

            if not frame_df.empty:
                coords = frame_df[["location_x", "location_z"]].to_numpy()
                scatter.set_offsets(coords)
                colors = np.array([self._color_map[user] for user in frame_df["user_id"]])
                scatter.set_color(colors)
                scatter.set_sizes(np.full(len(frame_df), self._point_size()))
                scatter.set_alpha(self.point_config.get("alpha", 1.0))
            else:
                scatter.set_offsets(np.empty((0, 2)))
                scatter.set_color([])
                scatter.set_sizes([])

            segments: list[np.ndarray] = []
            segment_colors: list[tuple[float, float, float, float]] = []

            for user, track in self._user_tracks.items():
                user_color = self._color_map.get(user, (0.2, 0.2, 0.2, 1.0))
                past = track[(track["second_floor"] <= timestamp) & (track["second_floor"] >= timestamp - trail_window)]
                if len(past) < 2:
                    continue
                coords = past[["location_x", "location_z"]].to_numpy()
                segs = np.stack([coords[:-1], coords[1:]], axis=1)
                alpha_values = np.linspace(
                    self.trail_config.get("alpha_start", 1.0),
                    self.trail_config.get("alpha_end", 0.1),
                    len(segs),
                )
                for seg, alpha in zip(segs, alpha_values, strict=False):
                    segments.append(seg)
                    segment_colors.append((user_color[0], user_color[1], user_color[2], float(alpha)))

            trail_collection.set_segments(segments)
            if segment_colors:
                trail_collection.set_colors(segment_colors)
            else:
                trail_collection.set_colors([])

            if timestamp.tzinfo is None:
                timestamp_local = timestamp.tz_localize("UTC").tz_convert("Asia/Tokyo")
            else:
                timestamp_local = timestamp.tz_convert("Asia/Tokyo")
            time_text.set_text(timestamp_local.strftime("%Y-%m-%d %H:%M:%S %Z"))
            return scatter, trail_collection, time_text

        frame_count = len(self._frame_seconds)
        anim = animation.FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=frame_count,
            interval=1000 / self.movie_config.get("fps", 30),
            blit=False,
        )
        return anim

    # ------------------------------------------------------------------
    def _point_size(self) -> float:
        radius = float(self.point_config.get("radius_px", 6))
        return radius ** 2

    # ------------------------------------------------------------------
    def save_mp4(self, anim: animation.FuncAnimation, path: str | Path) -> str:
        """MP4を保存する。"""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        bitrate = self._parse_bitrate(self.movie_config.get("bitrate"))
        writer = animation.FFMpegWriter(
            fps=int(self.movie_config.get("fps", 30)),
            bitrate=bitrate,
        )
        anim.save(path, writer=writer)
        plt.close(anim._fig)
        return str(path)

    def _parse_bitrate(self, value: Any) -> int | None:
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip().upper()
        if text.endswith("M"):
            return int(float(text[:-1]) * 1000)
        if text.endswith("K"):
            return int(float(text[:-1]))
        return int(float(text))

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame, output_basename: str | None = None) -> dict[str, str | int]:
        """一括実行（命名・保存・ログを含む）。"""

        prepared = self.prepare(df)
        anim = self.render(prepared)

        now = datetime.now(ZoneInfo("Asia/Tokyo"))
        dt_label = now.strftime("%Y%m%d_%H%M%S")
        version = self.movie_config.get("version", "c1.0")

        basename_core = naming.build_basename(
            self._event_day or "unknown",
            output_basename or self.io_config.get("output_filename", "movie"),
            dt_label,
            version,
            duration=self.movie_config.get("duration_sec"),
        )
        full_basename = f"movie_xz-{basename_core}"
        movie_path = naming.result_path("movie", full_basename).with_suffix(
            f".{self.movie_config.get('format', 'mp4')}"
        )

        if movie_path.exists() and not self.io_config.get("overwrite", False):
            raise FileExistsError(f"出力ファイルが既に存在します: {movie_path}")

        saved_path = self.save_mp4(anim, movie_path)
        meta = naming.meta_paths(dt_label, version)

        logger = logging_util.get_logger(dt_label, log_path=meta["log_path"])
        if self._stats is None:
            stats_dict = {}
        else:
            stats_dict = {
                "レコード数": self._stats.records,
                "除外件数": self._stats.excluded_rows,
                "型不整合による除外": self._type_drop_count,
                "ユニークユーザー数": self._stats.unique_users,
                "ユニーク秒数": self._stats.unique_seconds,
                "総フレーム数": self._stats.frames,
                "保存先": saved_path,
            }
        logging_util.log_summary(logger, stats_dict)

        frame_count = len(self._frame_seconds) if self._frame_seconds is not None else 0
        return {"mp4_path": saved_path, "log_path": str(meta["log_path"]), "frames": frame_count}
