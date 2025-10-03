from __future__ import annotations
import os, gc, logging, tracemalloc
from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
from matplotlib import rcParams
rcParams["font.family"] = "Meiryo"
rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt

from .logging_util import get_logger, log_summary
from .naming import build_basename, result_path, meta_paths
from .validation import (
    PipelineError,  # EC を伴う例外
    require_columns, drop_invalid_types, enforce_min_seconds
)

# ---- エラーコード（抜粋） ----
EC_STATS_UNKNOWN = -2400
EC_STORAGE_DST_INVALID = -2701
EC_STORAGE_PERM = -2702
EC_STORAGE_IO = -2704

TZ_JST = ZoneInfo("Asia/Tokyo")

@dataclass
class HistParams:
    bins: int | str = "fd"       # Freedman–Diaconis / "auto" / 明示個数
    range_min: Optional[int] = None
    range_max: Optional[int] = None
    dpi: int = 120
    width: int = 960
    height: int = 720

@dataclass
class IOParams:
    output_filename: str = "hist"
    overwrite: bool = False

class HistogramGenerator:
    """
    同時接続数（秒単位）の分布をヒストグラム化し、PNGとCSVを出力する。
    - 入力 second は JST（tz-aware/naive どちらでも可。内部で正規化）
    - 必須列: second, user_id, location_x, location_y, location_z, event_day
    """
    def __init__(
        self,
        hist: HistParams = HistParams(),
        io: IOParams = IOParams(),
        ver: str = "c1.0",
    ) -> None:
        self.hist = hist
        self.io = io
        self.ver = ver

    # --------- 前処理 & 同時接続数算出 ----------
    def compute_concurrency(self, df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
        require_columns(df)
        df = drop_invalid_types(df, logger)  # second→JST正規化、数値型整備、欠損drop

        # 秒丸め（"s"：小文字）後、同一秒×同一ユーザの重複は後勝ちで1件に
        df = df.sort_values(["second", "user_id"])
        df["sec_floor"] = df["second"].dt.floor("s")
        df = df.drop_duplicates(subset=["sec_floor", "user_id"], keep="last")

        # 最低 180 秒は確保（設計の動画要件と整合。短すぎる分布は品質担保のため拒否）
        enforce_min_seconds(df, 180)

        # 1秒ごとの同時接続数
        s = (
            df.groupby("sec_floor")["user_id"]
              .nunique()
              .rename("concurrency")
              .reset_index()
        )
        # event_day（命名用）をJST最頻日から決定
        try:
            ed = pd.to_datetime(df["event_day"], errors="coerce")
            ed = ed.dt.tz_localize("Asia/Tokyo") if getattr(ed.dtype, "tz", None) is None else ed.dt.tz_convert("Asia/Tokyo")
            mode = pd.Series(pd.to_datetime(ed, errors="coerce")).dt.date.mode()
            self._event_day_str = str(mode.iat[0]) if not mode.empty else datetime.now(TZ_JST).date().isoformat()
        except Exception:
            self._event_day_str = datetime.now(TZ_JST).date().isoformat()

        return s  # columns: [sec_floor, concurrency]

    # --------- 描画 ----------
    def plot_hist(self, conc_df: pd.DataFrame) -> plt.Figure:
        values = conc_df["concurrency"].to_numpy()

        # 指標
        peak = int(values.max())
        mean = float(values.mean())
        median = float(np.median(values))
        p95 = float(np.percentile(values, 95))

        # 図
        fig_w, fig_h = self.hist.width / self.hist.dpi, self.hist.height / self.hist.dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=self.hist.dpi)
        fig.patch.set_facecolor("#eeeeee")
        ax.set_facecolor("white")

        # ヒストレンジ
        rmin = self.hist.range_min if self.hist.range_min is not None else int(values.min())
        rmax = self.hist.range_max if self.hist.range_max is not None else int(values.max())

        ax.hist(
            values,
            bins=self.hist.bins,
            range=(rmin, rmax),
            edgecolor="black",
            alpha=0.85
        )

        # 目安線
        ax.axvline(peak,   color="#d62728", linestyle="--", linewidth=2, label=f"Peak={peak}")
        ax.axvline(mean,   color="#1f77b4", linestyle=":",  linewidth=2, label=f"Mean={mean:.1f}")
        ax.axvline(median, color="#2ca02c", linestyle="-.", linewidth=2, label=f"Median={median:.1f}")
        ax.axvline(p95,    color="#9467bd", linestyle="-",  linewidth=1.8, label=f"P95={p95:.1f}")

        ax.set_title("YAIBA: 同時接続数の分布（JST, 1秒分解能）")
        ax.set_xlabel("同時接続数 [users]")
        ax.set_ylabel("度数 [counts]")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.3)

        # サマリ注釈
        ax.text(
            0.02, 0.98,
            f"peak={peak}  mean={mean:.1f}  median={median:.1f}  p95={p95:.1f}",
            transform=ax.transAxes, ha="left", va="top"
        )
        return fig

    # --------- 保存 ----------
    def save_png(self, fig: plt.Figure, out_path: str) -> str:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        return out_path

    def save_csv(self, conc_df: pd.DataFrame, out_path: str) -> str:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        conc_df.to_csv(out_path, index=False)
        return out_path

    # --------- 一括実行 ----------
    def run(self, df: pd.DataFrame, output_basename: Optional[str] = None) -> Dict[str, str]:
        dt_jst = datetime.now(TZ_JST).strftime("%Y%m%d_%H%M%S")
        mpaths = meta_paths(dt_jst, self.ver)
        logger = get_logger(run_id=dt_jst, log_path=mpaths["log_path"])

        tracemalloc.start(); snap0 = tracemalloc.take_snapshot()
        try:
            conc_df = self.compute_concurrency(df, logger)
            basename = output_basename or build_basename(
                event_day=self._event_day_str,
                filename=self.io.output_filename,
                dt=dt_jst,
                ver=self.ver,
                duration=None,   # 静止画は duration を付けない運用
            )
            png_path = result_path("image", basename)
            csv_path = result_path("table", basename)

            fig = self.plot_hist(conc_df)
            self.save_png(fig, png_path)
            self.save_csv(conc_df.rename(columns={"sec_floor":"second_jst"}), csv_path)

            snap1 = tracemalloc.take_snapshot()
            mem_kb = sum(st.size_diff for st in snap1.compare_to(snap0, "lineno")) / 1024.0

            stats = {
                "png_path": png_path,
                "csv_path": csv_path,
                "log_path": mpaths["log_path"],
                "peak": int(conc_df["concurrency"].max()),
                "mean": round(float(conc_df["concurrency"].mean()), 2),
                "median": round(float(np.median(conc_df["concurrency"].to_numpy())), 2),
                "p95": round(float(np.percentile(conc_df["concurrency"].to_numpy(), 95)), 2),
                "mem_kb_diff": round(mem_kb, 1),
            }
            log_summary(logger, stats)
            plt.close(fig)
            return stats

        except PipelineError:
            raise
        except PermissionError as e:
            logger.error(f"EC={EC_STORAGE_PERM} perm err={e}")
            raise PipelineError(EC_STORAGE_PERM, "書込権限不足") from e
        except OSError as e:
            logger.error(f"EC={EC_STORAGE_IO} io err={e}")
            raise PipelineError(EC_STORAGE_IO, "I/O例外") from e
        except Exception as e:
            logger.exception(f"EC={EC_STATS_UNKNOWN} unexpected err={e}")
            raise PipelineError(EC_STATS_UNKNOWN, "不明エラー") from e
        finally:
            gc.collect(); tracemalloc.stop()
