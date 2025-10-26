from __future__ import annotations
import os, gc, math, logging, tracemalloc
from dataclasses import dataclass
from typing import Optional, Dict, Deque, Tuple
from collections import deque
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .fontconfig import setup_fonts
setup_fonts()
import matplotlib.animation as animation


from tqdm import tqdm

from .logging_util import get_logger, log_summary
from .naming import build_basename, result_path, meta_paths
from .validation import (
    PipelineError,
    require_columns, drop_invalid_types, clip_by_boundary, enforce_min_seconds
)

# エラーコード
EC_STORAGE_DST_INVALID = -2701
EC_STORAGE_PERM = -2702
EC_STORAGE_IO = -2704
EC_STATS_UNKNOWN = -2400
EC_CU_DATA_EMPTY = -2204

TZ_JST = ZoneInfo("Asia/Tokyo")

@dataclass
class Theme:
    palette: str = "tab10"
    bg_color: str = "#eeeeee"
    font: str = "Meiryo"
    font_size: int = 16

@dataclass
class MovieParams:

    duration_real: int = 10800   # [sec] 解析対象上限
    format: str = "mp4"
    fps: int = 30                 # ← ここは 30 固定で使う
    duration_sec: int | None = None  # ← None/<=0 なら自動決定
    auto_min_sec: int = 10           # ← 自動時の最小尺
    auto_max_sec: int = 120           # ← 自動時の最大尺
    # bitrate は後方互換で解決（int/str/bitrate_kbps を許容）
    bitrate: int | str = 2000

@dataclass
class PointParams:
    radius_px: int = 6
    alpha: float = 1.0

@dataclass
class TrailParams:
    length_real_seconds: int = 0
    alpha_start: float = 1.0
    alpha_end: float = 0.1

@dataclass
class IOParams:
    output_filename: str = "movie"
    out_dir: Optional[str] = None
    overwrite: bool = False

MovieIOParams = IOParams


def _resolve_bitrate_kbps(movie: MovieParams) -> int:
    val = getattr(movie, "bitrate_kbps", None)
    if isinstance(val, (int, float)) and val > 0:
        return int(val)
    legacy = movie.bitrate
    if isinstance(legacy, (int, float)) and legacy > 0:
        return int(legacy)
    if isinstance(legacy, str):
        s = legacy.strip().lower()
        try:
            if s.endswith("k"): return int(float(s[:-1]))
            if s.endswith("m"): return int(float(s[:-1]) * 1000)
            return int(float(s))
        except ValueError:
            pass
    return 2000

def _ensure_dir(p: str, logger: logging.Logger) -> None:
    try:
        os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    except Exception as e:
        (logger or logging.getLogger(__name__)).error(f"EC={EC_STORAGE_DST_INVALID} make_dir_failed path={p} err={e}")
        raise PipelineError(EC_STORAGE_DST_INVALID, f"出力先フォルダ作成失敗: {p}") from e

class MovieGenerator:
    def __init__(
        self,
        boundary: Optional[Dict[str, float]] = None,
        theme: Theme = Theme(),
        movie: MovieParams = MovieParams(),
        point: PointParams = PointParams(),
        trail: TrailParams = TrailParams(),
        io: IOParams = IOParams(),
        ver: str = "c2.0",
    ) -> None:
        self.boundary = boundary or {}
        self.theme = theme
        self.movie = movie
        self.point = point
        self.trail = trail
        self.io = io
        self.ver = ver

    # --- 準備 ---
    def prepare(self, df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
        require_columns(df)
        df = drop_invalid_types(df, logger)

        # event_day を命名用 YYYY-MM-DD に（JST最頻・フォールバック現在日）
        try:
            ed = pd.to_datetime(df["event_day"], errors="coerce")
            ed = ed.dt.tz_localize("Asia/Tokyo") if getattr(ed.dtype, "tz", None) is None else ed.dt.tz_convert("Asia/Tokyo")
            s = pd.Series(pd.to_datetime(ed, errors="coerce")).dt.date.astype("string")
            mode = s.mode()
            event_day = str(mode.iat[0]) if not mode.empty else None
        except Exception:
            event_day = None
        if not event_day:
            event_day = datetime.now(TZ_JST).date().isoformat()
        self._event_day_str = event_day

        # ソート & 後勝ち重複解決（秒は小文字 "s"）
        df = df.sort_values(["second", "user_id"], ascending=[True, True])
        df["sec_floor"] = df["second"].dt.floor("s")
        df = df.drop_duplicates(subset=["sec_floor", "user_id"], keep="last")

        # 境界クリップ
        df = clip_by_boundary(df, self.boundary)
        if df.empty:
            raise PipelineError(EC_CU_DATA_EMPTY, "描画用データが空です")

        # 解析窓の制限（JST）
        t0 = df["sec_floor"].min()
        t1 = t0 + timedelta(seconds=self.movie.duration_real)
        df = df[(df["sec_floor"] >= t0) & (df["sec_floor"] <= t1)].copy()

        # 180秒未満は停止（設計仕様）
        enforce_min_seconds(df, 180)
        return df

    # --- 描画 ---
    def render(self, df: pd.DataFrame, logger: logging.Logger) -> Tuple[animation.FuncAnimation, Dict]:
        xmn = self.boundary.get("location_x_min", float(df["location_x"].min()))
        xmx = self.boundary.get("location_x_max", float(df["location_x"].max()))
        zmn = self.boundary.get("location_z_min", float(df["location_z"].min()))
        zmx = self.boundary.get("location_z_max", float(df["location_z"].max()))

        users = pd.Index(df["user_id"].unique())
        cmap = plt.get_cmap(self.theme.palette, max(10, len(users)))
        uid_to_rgba = {uid: cmap(i % cmap.N) for i, uid in enumerate(users)}

        t0 = df["sec_floor"].min()
        tmax = df["sec_floor"].max()
        real_span = min((tmax - t0).total_seconds(), self.movie.duration_real)
        dur = getattr(self, "_duration_sec_eff", None) or self.movie.duration_sec or self.movie.auto_min_sec
        total_frames = max(1, int(round(dur * self.movie.fps)))
        sec_per_frame = (real_span / (total_frames - 1)) if total_frames > 1 else 0.0

        by_sec: Dict[pd.Timestamp, pd.DataFrame] = dict(tuple(df.groupby("sec_floor")))
        # ── trail を使うか判定（長さ>0 かつ 可視alphaのときのみ有効）
        use_trail = (
            (self.trail.length_real_seconds > 0) and
            (self.trail.alpha_start > 0 or self.trail.alpha_end > 0)
        )
        if use_trail:
            trail_frames: int = max(1, int(math.ceil(self.trail.length_real_seconds / max(1e-9, sec_per_frame))))
            trail_buf: Deque[pd.DataFrame] = deque(maxlen=trail_frames)
        else:
            trail_buf = None  # type: ignore[assignment]

        dpi = 120
        fig, ax = plt.subplots(figsize=(960/dpi, 720/dpi), dpi=dpi)
        fig.patch.set_facecolor(self.theme.bg_color)
        ax.set_facecolor("white")
        ax.set_xlim(xmn, xmx); ax.set_ylim(zmn, zmx)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Z [m]")
        title = ax.text(0.5, 1.02, "YAIBA: ユーザー位置 2Dプロット", transform=ax.transAxes, ha="center", va="bottom")
        tlabel = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top")

        curr = ax.scatter([], [], s=self.point.radius_px**2, alpha=self.point.alpha)
        if use_trail:
            trail_sc = ax.scatter([], [], s=(self.point.radius_px * 0.35) ** 2)
        else:
            trail_sc = ax.scatter([], [], s=1, alpha=0)  

        times = []
        times: list[pd.Timestamp] = []
        for i in range(total_frames):
            ti = t0 if total_frames == 1 else t0 + pd.to_timedelta(i * sec_per_frame, unit="s")
            times.append(min(ti, tmax))

        def _frame(i: int):
            ts = times[i].floor("s")
        def _frame(ts):
            # Matplotlib から渡される frame を「時刻」として扱う
            # 念のため秒丸め（小文字 "s"）
            ts = pd.Timestamp(ts).floor("s")

            # 秒ごとのデータを取得（なければ空DF）
            df_now = by_sec.get(ts)
            if df_now is None:
                df_now = pd.DataFrame(columns=["user_id", "location_x", "location_z"])

            if use_trail:
                if df_now is None:
                    df_now = pd.DataFrame(columns=["user_id", "location_x", "location_z"])
                trail_buf.append(df_now)

            if not df_now.empty:
                offs = np.c_[df_now["location_x"].to_numpy(), df_now["location_z"].to_numpy()]
                cols = [uid_to_rgba[uid] for uid in df_now["user_id"]]
                curr.set_offsets(offs); curr.set_facecolors(cols)
            else:
                curr.set_offsets(np.empty((0, 2))); curr.set_facecolors([])

            if use_trail and len(trail_buf) > 0:
                xs, zs, cols = [], [], []
                K = len(trail_buf)
                for k, dfk in enumerate(trail_buf):
                    if dfk is None or dfk.empty: continue
                    alpha = self.trail.alpha_start + (self.trail.alpha_end - self.trail.alpha_start) * (k / max(1, K-1))
                    for uid, x, z in zip(dfk["user_id"], dfk["location_x"], dfk["location_z"]):
                        r,g,b,_ = uid_to_rgba[uid]; xs.append(x); zs.append(z); cols.append((r,g,b,alpha))
                if xs:
                    trail_sc.set_offsets(np.c_[np.array(xs), np.array(zs)]); trail_sc.set_facecolors(np.array(cols))
                else:
                    trail_sc.set_offsets(np.empty((0, 2))); trail_sc.set_facecolors([])
            else:
                trail_sc.set_offsets(np.empty((0, 2))); trail_sc.set_facecolors([])

            tlabel.set_text(ts.tz_convert("Asia/Tokyo").strftime("%Y-%m-%d %H:%M:%S JST"))
            # tz-naive の場合でも安全に JST 表示
            ts_jst = ts.tz_convert("Asia/Tokyo") if ts.tzinfo else ts.tz_localize("UTC").tz_convert("Asia/Tokyo")
            tlabel.set_text(ts_jst.strftime("%Y-%m-%d %H:%M:%S JST"))
            return curr, trail_sc, title, tlabel

        # フレーム列として「時刻リスト」を渡すことで IndexError を根絶
        anim = animation.FuncAnimation(
            fig, _frame, frames=times, interval=1000 / self.movie.fps,
            blit=False, cache_frame_data=False
        )
        info = {
            "frames": total_frames,
            "sec_per_frame": round(sec_per_frame, 6),
            "t0_jst": str(t0),
            "tmax_jst": str(tmax),
            "xlim": (xmn, xmx),
            "zlim": (zmn, zmx),
        }
        logger.info(
            f"frame_plan frames={total_frames} sec_per_frame={sec_per_frame:.6f} real_span={real_span:.3f}s"
        )
        return anim, info

    # --- 保存（tqdm進捗・fps間引き） ---
    def save_mp4(self, anim: animation.FuncAnimation, out_path: str, logger: logging.Logger) -> str:
        try:
            br = _resolve_bitrate_kbps(self.movie)
            writer = animation.FFMpegWriter(
                fps=self.movie.fps, bitrate=br,
                codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y"],
            )

            # 総フレーム数の安全取得
            total_frames = (getattr(anim, "save_count", None) or getattr(anim, "_save_count", None) or int(self.movie.duration_sec * self.movie.fps))
            stride = max(1, int(self.movie.fps))  # 1秒ごと

            _ensure_dir(out_path, logger)

            with tqdm(total=total_frames, desc="Encoding", unit="frame") as pbar:
                def _progress(i, n):
                    if n != pbar.total:
                        pbar.total = n; pbar.refresh()
                    if (i + 1) % stride == 0 or (i + 1) == n:
                        delta = (i + 1) - pbar.n
                        if delta > 0:
                            pbar.update(delta)

                anim.save(out_path, writer=writer, progress_callback=_progress)
            return out_path

        except FileNotFoundError as e:
            logger.error(f"EC={EC_STORAGE_DST_INVALID} ffmpeg_or_path_missing err={e}")
            raise PipelineError(EC_STORAGE_DST_INVALID, "ffmpeg未導入、または出力先が不正") from e
        except PermissionError as e:
            logger.error(f"EC={EC_STORAGE_PERM} perm err={e}")
            raise PipelineError(EC_STORAGE_PERM, "書込権限不足") from e
        except OSError as e:
            logger.error(f"EC={EC_STORAGE_IO} io err={e}")
            raise PipelineError(EC_STORAGE_IO, "I/O例外") from e

    # --- 一括実行 ---
    def run(self, df: pd.DataFrame, output_basename: Optional[str] = None) -> Dict[str, str | int]:
        dt_jst = datetime.now(TZ_JST).strftime("%Y%m%d_%H%M%S")
        mpaths = meta_paths(dt_jst, self.ver)
        logger = get_logger(run_id=dt_jst, log_path=mpaths["log_path"])

        tracemalloc.start(); snap0 = tracemalloc.take_snapshot()
        try:
            df_prep = self.prepare(df, logger)
            # --- 素材長から動画尺を自動決定（必要な場合） ---
            # 実長（解析窓に制限済み）を算出
            t0 = df_prep["sec_floor"].min()
            tmax = df_prep["sec_floor"].max()
            real_span = min((tmax - t0).total_seconds(), self.movie.duration_real)
            # duration_sec が未指定/無効なら自動決定
            if not self.movie.duration_sec or self.movie.duration_sec <= 0:
                # 長すぎる素材は auto_max_sec に圧縮、短い素材は auto_min_sec を確保
                # 例: real_span=3600s → duration=60s（タイムラプス圧縮）
                #     real_span=6s    → duration=10s（最低尺を確保）
                target = max(self.movie.auto_min_sec,
                             min(self.movie.auto_max_sec, int(round(real_span))))
                self._duration_sec_eff = target
            else:
                self._duration_sec_eff = int(self.movie.duration_sec)

            basename = output_basename or build_basename(
                event_day=self._event_day_str, filename=self.io.output_filename, dt=dt_jst,
                ver=self.ver, duration=self._duration_sec_eff  # ← 有効な動画尺で命名
            )
            out_path = result_path("movie", basename)
            if (not self.io.overwrite) and os.path.exists(out_path):
                logger.error(f"EC={EC_STORAGE_DST_INVALID} already_exists path={out_path}")
                raise PipelineError(EC_STORAGE_DST_INVALID, f"既存ファイルあり: {out_path}")

            anim, info = self.render(df_prep, logger)
            saved = self.save_mp4(anim, out_path, logger)

            snap1 = tracemalloc.take_snapshot()
            mem_diff_kb = sum(st.size_diff for st in snap1.compare_to(snap0, "lineno")) / 1024.0
            stats = {"mp4_path": saved, "log_path": mpaths["log_path"], "frames": info["frames"], "mem_kb_diff": round(mem_diff_kb, 1)}
            log_summary(logger, stats)
            return stats
        except PipelineError:
            raise
        except Exception as e:
            logger.exception(f"EC={EC_STATS_UNKNOWN} unexpected err={e}")
            raise PipelineError(EC_STATS_UNKNOWN, "不明エラー") from e
        finally:
            plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
            plt.close("all"); gc.collect(); tracemalloc.stop()

# Colab/外部からの簡易ラッパ（設計 Want を意識）
def run_movie_xz(
    df: pd.DataFrame, *, boundary: Optional[Dict[str, float]] = None,
    theme: Theme = Theme(), movie: MovieParams = MovieParams(),
    point: PointParams = PointParams(), trail: TrailParams = TrailParams(),
    io: IOParams = IOParams(), ver: str = "c2.0",
) -> Dict[str, str | int]:
    gen = MovieGenerator(boundary=boundary, theme=theme, movie=movie, point=point, trail=trail, io=io, ver=ver)
    return gen.run(df)
