"""Movie generation for YAIBA trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import animation
from zoneinfo import ZoneInfo

from . import logging_util, naming, validation

JST = ZoneInfo("Asia/Tokyo")


@dataclass
class MovieParams:
    duration_real: int = 10800
    format: str = "mp4"
    fps: int = 30
    duration_sec: int = 60
    bitrate: str = "2M"
    version: str = "c1.0"
    min_unique_seconds: int = 180


@dataclass
class PointParams:
    radius_px: int = 6
    alpha: float = 1.0


@dataclass
class TrailParams:
    length_real_seconds: int = 30
    alpha_start: float = 1.0
    alpha_end: float = 0.1


@dataclass
class ThemeParams:
    palette: str = "tab10"
    bg_color: str = "#eeeeee"
    font: str = "Meiryo"
    font_size: int = 16


@dataclass
class IOParams:
    output_filename: str = "movie"
    overwrite: bool = False


class MovieGenerator:
    """Generate x–z plane trajectory animations from YAIBA data."""

    def __init__(
        self,
        boundary: dict | None = None,
        outlier: dict | None = None,
        theme: dict | None = None,
        movie: dict | None = None,
        point: dict | None = None,
        trail: dict | None = None,
        io: dict | None = None,
    ) -> None:
        self.boundary = boundary or {}
        self.outlier = outlier or {}
        self.theme = self._merge_params(ThemeParams(), theme)
        self.movie = self._merge_params(MovieParams(), movie)
        self.point = self._merge_params(PointParams(), point)
        self.trail = self._merge_params(TrailParams(), trail)
        self.io = self._merge_params(IOParams(), io)
        self._prepare_stats: dict[str, Any] = {}

    @staticmethod
    def _merge_params(defaults: Any, updates: dict | None) -> dict:
        data = defaults.__dict__ if hasattr(defaults, "__dict__") else dict(defaults)
        merged = dict(data)
        if updates:
            merged.update(updates)
        return merged

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the raw intermediate dataframe."""

        validation.require_columns(df, validation.REQUIRED_COLUMNS)

        work = df.copy()
        work = work.sort_values(["second", "user_id"])
        work = work.drop_duplicates(subset=["second", "user_id"], keep="last")

        cleaned, dropped = validation.drop_invalid_types(work)
        cleaned = cleaned.sort_values(["second", "user_id"])

        event_days = cleaned["event_day"].dropna().unique()
        if len(event_days) > 1:
            raise ValueError("event_dayは単一である必要があります")

        if not cleaned.empty:
            start_time = cleaned["second"].min()
            duration_real = int(self.movie["duration_real"])
            cutoff = start_time + pd.Timedelta(seconds=duration_real)
            cleaned = cleaned[cleaned["second"] <= cutoff]

        cleaned = validation.clip_by_boundary(cleaned, self.boundary)

        validation.enforce_min_seconds(
            cleaned,
            int(self.movie.get("min_unique_seconds", MovieParams.min_unique_seconds)),
        )

        cleaned = cleaned.assign(
            second_jst=cleaned["second"].dt.tz_convert(JST)
        )

        self._prepare_stats = {
            "input_rows": len(df),
            "dropped_invalid_rows": dropped,
            "output_rows": len(cleaned),
            "unique_users": int(cleaned["user_id"].nunique()),
            "unique_seconds": int(cleaned["second"].dt.floor("s").nunique()),
        }
        return cleaned

    def _build_user_colors(self, df: pd.DataFrame) -> dict[Any, tuple[float, float, float, float]]:
        cmap = cm.get_cmap(self.theme["palette"])
        users = sorted(df["user_id"].unique())
        if not users:
            return {}
        colors: dict[Any, tuple[float, float, float, float]] = {}
        denom = max(len(users) - 1, 1)
        for idx, user in enumerate(users):
            rgba = cmap(idx / denom)
            colors[user] = (rgba[0], rgba[1], rgba[2], self.point["alpha"])
        return colors

    def _calc_limits(self, df: pd.DataFrame) -> tuple[float, float, float, float]:
        x_min = self.boundary.get("location_x_min", float(df["location_x"].min()))
        x_max = self.boundary.get("location_x_max", float(df["location_x"].max()))
        z_min = self.boundary.get("location_z_min", float(df["location_z"].min()))
        z_max = self.boundary.get("location_z_max", float(df["location_z"].max()))
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0
        if z_min == z_max:
            z_min -= 1.0
            z_max += 1.0
        return x_min, x_max, z_min, z_max

    def render(self, df: pd.DataFrame) -> animation.FuncAnimation:
        """Create the matplotlib animation object."""

        fps = int(self.movie["fps"])
        duration_sec = int(self.movie["duration_sec"])
        num_frames = fps * duration_sec
        if num_frames <= 0:
            raise ValueError("duration_sec と fps は正の値である必要があります")

        user_colors = self._build_user_colors(df)
        x_min, x_max, z_min, z_max = self._calc_limits(df)

        plt.rcParams["font.family"] = self.theme["font"]
        fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=100)
        ax.set_facecolor(self.theme["bg_color"])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Z [m]")
        title = ax.set_title("YAIBA: ユーザー位置 2Dプロット", fontsize=self.theme["font_size"])
        time_text = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=self.theme["font_size"],
        )

        base_size = (self.point["radius_px"] * (72 / fig.dpi)) ** 2

        df_sorted = df.sort_values("second")
        seconds_index = pd.DatetimeIndex(df_sorted["second"].drop_duplicates())
        start_time = seconds_index[0]
        if len(seconds_index) > 1:
            real_span = (seconds_index[-1] - seconds_index[0]).total_seconds()
        else:
            real_span = 1.0
        time_step = real_span / max(num_frames - 1, 1)

        trail_length = float(self.trail["length_real_seconds"])
        alpha_start = float(self.trail["alpha_start"])
        alpha_end = float(self.trail["alpha_end"])

        scatter = ax.scatter([], [], s=[], c=[])

        def frame_time(i: int) -> pd.Timestamp:
            return start_time + pd.to_timedelta(i * time_step, unit="s")

        def update(frame_index: int):
            current_time = frame_time(frame_index)
            trail_start = current_time - pd.Timedelta(seconds=trail_length)
            mask = (df_sorted["second"] >= trail_start) & (df_sorted["second"] <= current_time)
            frame_df = df_sorted.loc[mask]

            if frame_df.empty:
                scatter.set_offsets(np.empty((0, 2)))
                scatter.set_sizes([])
                scatter.set_facecolors([])
            else:
                seconds_ago = (current_time - frame_df["second"]).dt.total_seconds().to_numpy()
                alphas = np.interp(
                    seconds_ago,
                    (0.0, trail_length if trail_length > 0 else 1.0),
                    (alpha_start, alpha_end),
                    left=alpha_start,
                    right=alpha_end,
                )
                offsets = frame_df[["location_x", "location_z"]].to_numpy()
                colors = []
                for uid, alpha in zip(frame_df["user_id"], alphas):
                    base = user_colors.get(uid, (0.2, 0.2, 0.2, self.point["alpha"]))
                    colors.append((base[0], base[1], base[2], float(alpha)))
                scatter.set_offsets(offsets)
                scatter.set_sizes(np.full(len(frame_df), base_size))
                scatter.set_facecolors(colors)
                scatter.set_edgecolors(colors)

            current_jst = current_time.tz_convert(JST)
            time_text.set_text(current_jst.strftime("%Y-%m-%d %H:%M:%S JST"))
            return scatter, time_text

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=num_frames,
            interval=1000 / fps,
            blit=False,
            repeat=False,
        )
        return anim

    def save_mp4(self, anim: animation.FuncAnimation, path: str | Path) -> str:
        """Save the animation as an MP4 file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = animation.FFMpegWriter(fps=self.movie["fps"], bitrate=self.movie["bitrate"])
        anim.save(output_path, writer=writer)
        plt.close(anim._fig)
        return str(output_path)

    def run(self, df: pd.DataFrame, output_basename: str | None = None) -> dict[str, str | int]:
        """Execute the full pipeline."""

        prepared = self.prepare(df)
        event_day = prepared["event_day"].iloc[0]
        if isinstance(event_day, pd.Timestamp):
            event_day_str = event_day.strftime("%Y-%m-%d")
        else:
            event_day_str = str(event_day)

        now = datetime.now(tz=JST)
        dt_str = now.strftime("%Y%m%d_%H%M%S")
        filename = output_basename or self.io["output_filename"]
        version = self.movie["version"]
        basename_core = naming.build_basename(
            event_day_str,
            filename,
            dt_str,
            version,
            duration=self.movie["duration_sec"],
        )
        movie_filename = f"movie_xz-{basename_core}.{self.movie['format']}"
        movie_path = naming.result_path("movie", movie_filename)
        if movie_path.exists() and not self.io.get("overwrite", False):
            raise FileExistsError(
                f"既に同名のファイルが存在します: {movie_path}"
            )

        logger = logging_util.get_logger(dt_str)
        logger.info("C-2 動画生成を開始します")
        logger.info("入力行数: %s", self._prepare_stats.get("input_rows"))
        logger.info("除外行数: %s", self._prepare_stats.get("dropped_invalid_rows"))

        anim = self.render(prepared)
        frames = int(self.movie["fps"] * self.movie["duration_sec"])
        logger.info("生成フレーム数: %s", frames)

        output_path = self.save_mp4(anim, movie_path)
        logger.info("動画を保存しました: %s", output_path)

        logging_util.log_summary(
            logger,
            {
                "mp4_path": output_path,
                "unique_users": self._prepare_stats.get("unique_users"),
                "unique_seconds": self._prepare_stats.get("unique_seconds"),
            },
        )

        meta = naming.meta_paths(dt_str, version)
        log_path = meta["log_path"]
        log_path.parent.mkdir(parents=True, exist_ok=True)

        return {
            "mp4_path": str(output_path),
            "log_path": str(log_path),
            "frames": frames,
        }
