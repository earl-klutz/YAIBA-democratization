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
    length_real_seconds: int = 30
    alpha_start: float = 1.0
    alpha_end: float = 0.1

@dataclass
class IOParams:
    output_filename: str = "movie"
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

        # --- 色は一度だけ決めて配列で参照（dictルックアップを毎フレームやらない） ---
        users = pd.Index(df["user_id"].unique())
        cmap = plt.get_cmap(self.theme.palette, max(10, len(users)))
        uid_to_rgba = {uid: cmap(i % cmap.N) for i, uid in enumerate(users)}
        # code化（0..n-1）しておくと配列参照が速い
        uid_cat = pd.Categorical(df["user_id"], categories=users, ordered=False)
        uid_codes = uid_cat.codes.astype(np.int32)  # -1 は欠損を意味する
        color_table = np.array([cmap(i % cmap.N) for i in range(len(users))], dtype=np.float32)
        # 行並びに合わせて色コードを保持
        df = df.assign(_uid_code=uid_codes)

        t0 = df["sec_floor"].min()
        tmax = df["sec_floor"].max()
        real_span = min((tmax - t0).total_seconds(), self.movie.duration_real)
        dur = getattr(self, "_duration_sec_eff", None) or self.movie.duration_sec or self.movie.auto_min_sec
        total_frames = max(1, int(round(dur * self.movie.fps)))
        sec_per_frame = (real_span / (total_frames - 1)) if total_frames > 1 else 0.0

        #   「秒→行範囲のインデックス表」に置き換えると更に速い
        by_sec: Dict[pd.Timestamp, pd.DataFrame] = dict(tuple(df.groupby("sec_floor", sort=True)))
        trail_frames: int = max(1, int(math.ceil(self.trail.length_real_seconds / max(1e-9, sec_per_frame))))
        trail_buf: Deque[pd.DataFrame] = deque(maxlen=trail_frames)

        dpi = 120
        fig, ax = plt.subplots(figsize=(960/dpi, 720/dpi), dpi=dpi)
        fig.patch.set_facecolor(self.theme.bg_color)
        ax.set_facecolor("white")
        ax.set_xlim(xmn, xmx); ax.set_ylim(zmn, zmx)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Z [m]")
        
        # --- タイトル・凡例は焼き付け（更新しない） ---
        ax.text(0.5, 1.02, "YAIBA: ユーザー位置 2Dプロット",
                transform=ax.transAxes, ha="center", va="bottom")
        # 動的テキストだけ animated=True
        tlabel = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top", animated=True)

        # --- 毎フレーム更新するArtistは animated=True で一度だけ作る ---
        curr = ax.scatter([], [], s=self.point.radius_px**2, alpha=self.point.alpha, animated=True)
        trail_sc = ax.scatter([], [], s=(self.point.radius_px * 0.35) ** 2, animated=True)

        times: list[pd.Timestamp] = []
        for i in range(total_frames):
            ti = t0 if total_frames == 1 else t0 + pd.to_timedelta(i * sec_per_frame, unit="s")
            times.append(min(ti, tmax))

        # --- 時刻ラベルは前計算（JST文字列） ---
        def _to_jst_str(ti: pd.Timestamp) -> str:
            # tz-naive → UTC扱いでJSTへ、tzあり → JSTへ
            if ti.tzinfo:
                return ti.tz_convert("Asia/Tokyo").strftime("%Y-%m-%d %H:%M:%S JST")
            else:
                return ti.tz_localize("UTC").tz_convert("Asia/Tokyo").strftime("%Y-%m-%d %H:%M:%S JST")
        times_floor = [pd.Timestamp(t).floor("s") for t in times]
        times_str = [_to_jst_str(t) for t in times_floor]

        def _frame(ts):
            # Matplotlib から渡される frame を「時刻」として扱う
            ts = pd.Timestamp(ts).floor("s")

            df_now = by_sec.get(ts)
            if df_now is None or df_now.empty:
                # 現フレームの点なし
                curr.set_offsets(np.empty((0, 2), dtype=np.float32))
                curr.set_facecolors(np.empty((0, 4), dtype=np.float32))
            else:
                # 位置は2列をまとめて一回でセット
                offs_now = np.c_[df_now["location_x"].to_numpy(np.float32),
                                 df_now["location_z"].to_numpy(np.float32)]
                curr.set_offsets(offs_now)
                # 色：コード→テーブル参照で配列一括
                codes_now = df_now["_uid_code"].to_numpy(np.int32, copy=False)
                curr.set_facecolors(color_table[codes_now])
            trail_buf.append(df_now if df_now is not None else pd.DataFrame())

            # --- トレイル更新（配列一括、alphaは等間隔グラデ） ---
            if len(trail_buf) > 0:
                # 空を除外して連結（Pythonループでの点ごとappendは避ける）
                non_empty = [dfk for dfk in trail_buf if dfk is not None and not dfk.empty]
                if non_empty:
                    K = len(trail_buf)
                    # バッファの先頭→末尾で alpha が上がる
                    if K > 1:
                        alphas_levels = np.linspace(self.trail.alpha_start, self.trail.alpha_end, K, dtype=np.float32)
                    else:
                        alphas_levels = np.array([self.trail.alpha_end], dtype=np.float32)
                    xs_list, zs_list, cols_list = [], [], []
                    for k, dfk in enumerate(trail_buf):
                        if dfk is None or dfk.empty:
                            continue
                        xs_list.append(dfk["location_x"].to_numpy(np.float32))
                        zs_list.append(dfk["location_z"].to_numpy(np.float32))
                        codes_k = dfk.get("_uid_code")
                        if codes_k is None or dfk.empty:
                            continue
                        col = color_table[codes_k.to_numpy(np.int32, copy=False)].copy()
                        col[:, 3] = alphas_levels[k]   # alphaだけ上書き
                        cols_list.append(col)
                    if xs_list:
                        xs_cat = np.concatenate(xs_list) if len(xs_list) > 1 else xs_list[0]
                        zs_cat = np.concatenate(zs_list) if len(zs_list) > 1 else zs_list[0]
                        cols_cat = np.concatenate(cols_list) if len(cols_list) > 1 else cols_list[0]
                        trail_sc.set_offsets(np.c_[xs_cat, zs_cat])
                        trail_sc.set_facecolors(cols_cat)
                    else:
                        trail_sc.set_offsets(np.empty((0, 2), dtype=np.float32))
                        trail_sc.set_facecolors(np.empty((0, 4), dtype=np.float32))
                else:
                    trail_sc.set_offsets(np.empty((0, 2), dtype=np.float32))
                    trail_sc.set_facecolors(np.empty((0, 4), dtype=np.float32))
            else:
                trail_sc.set_offsets(np.empty((0, 2), dtype=np.float32))
                trail_sc.set_facecolors(np.empty((0, 4), dtype=np.float32))

            # --- 事前計算したJSTラベルを差し替え（重複処理を排除） ---
            # frames=times を渡しているので floor した時刻のインデックスは同じ順序
            # 直接 ts を検索せず、times_floor と同順である前提で i を使う方法も可
            tlabel.set_text(_to_jst_str(ts))
            # blit=True のため更新対象のみ返す
            return (curr, trail_sc, tlabel)

        # フレーム列として「時刻リスト」を渡すことで IndexError を根絶
        # blit=True で動的Artistのみを再描画（タイトル等は焼き付け）
        interval_ms = max(1, int(round(1000.0 / max(1, int(self.movie.fps)))))
        anim = animation.FuncAnimation(
            fig, _frame, frames=times, interval=interval_ms,
            blit=True, cache_frame_data=False,
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
            # --- エンコーダ選択（CPU/GPU）と高速プリセット ---
            # self.movie.codec に "libx264"/"libx265"/"h264_nvenc"/"hevc_nvenc"/"h264_qsv"/"h264_amf" 等を想定
            codec = getattr(self.movie, "codec", None) or "libx264"
            extra = ["-movflags", "+faststart", "-y"]
            if codec in ("libx264", "libx265"):
                extra = ["-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", str(getattr(self.movie, "crf", 23))] + extra
            elif codec in ("h264_nvenc", "hevc_nvenc"):
                cq = str(getattr(self.movie, "cq", 23))
                preset = str(getattr(self.movie, "preset", "p5"))  # p1..p7
                tune = str(getattr(self.movie, "tune", "hq"))
                extra = ["-pix_fmt", "yuv420p", "-rc", "vbr", "-cq", cq, "-b:v", "0", "-preset", preset, "-tune", tune] + extra
            elif codec == "h264_qsv":
                gq = str(getattr(self.movie, "global_quality", 23))
                extra = ["-pix_fmt", "nv12", "-global_quality", gq, "-look_ahead", "1", "-preset", "veryfast"] + extra
            elif codec == "h264_amf":
                q = str(getattr(self.movie, "qvbr", 23))
                extra = ["-pix_fmt", "yuv420p", "-quality", "speed", "-rc", "vbr", "-qvbr", q] + extra
            # 任意: キーフレーム間隔（シーク改善）
            gop = int(getattr(self.movie, "gop", self.movie.fps * 2))
            extra += ["-g", str(gop)]

            writer = animation.FFMpegWriter(
                fps=self.movie.fps, bitrate=br, codec=codec, extra_args=extra
            )

            # 総フレーム数の安全取得
            total_frames = (getattr(anim, "save_count", None) or getattr(anim, "_save_count", None) or int(self.movie.duration_sec * self.movie.fps))
            stride = max(1, int(self.movie.fps))  # 1秒ごと（進捗更新のみ）
 
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
