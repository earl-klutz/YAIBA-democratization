from __future__ import annotations
"""
YAIBA 可視化: 滞在時間（在室時間）ヒストグラム — 設計準拠 完全版

要点:
- 仕様に合わせ run(self, df, output_basename: str) -> dict を提供
- 出力先は <YAIBA_RESULTS_DIR>/histograms（未設定時は /content/YAIBA_data/output/results/histograms）
- 既存 demo.py 互換: HistParams(dpi, width, height) / IOParams(overwrite) / ver="v1" などを受理
- 集計はユーザ別の在室秒数（unique sec_floor）→ 分に変換してヒスト化
- ラッパー関数 run_histogram_mvp(...) も提供（output_basename 未指定なら IOParams.output_filename）

依存: matplotlib, numpy, pandas（任意: pyarrow で Parquet 読取）
"""
import os
import gc
import json
import math
import logging
import tracemalloc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from .naming import RESULT_ROOT
from typing import Optional, Dict, Tuple, Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====== エラーコード（抜粋） ======
EC_STORAGE_PERM = -2702
EC_STORAGE_IO   = -2704
EC_STATS_INPUT  = -2301
EC_STATS_EMPTY  = -2302
EC_STATS_UNKNOWN= -2399

JST = ZoneInfo("Asia/Tokyo")

# ====== パラメータ定義 ======
@dataclass
class IOParams:
    # out_dir を未指定(None)なら、設計書既定の <YAIBA_RESULTS_DIR>/histograms を用いる
    out_dir: Optional[str] = None
    output_filename: str = "hist_dwell"  # 出力ベース名（ラッパーで未指定時に使用）
    png_dpi: int = 144
    csv_encoding: str = "utf-8"
    overwrite: bool = False  # 将来拡張用（現状は上書き保存の既定挙動のまま）
HistIOParams = IOParams

@dataclass
class HistParams:
    # demo.py 互換パラメータ
    bins: Optional[int] = None        # None→matplotlib の "auto"
    dpi: Optional[int] = None         # fig.set_dpi / savefig の優先候補
    width: Optional[float] = None     # dpi 指定時は px、未指定時は inch
    height: Optional[float] = None    # 上に同じ

    # 直接指定系
    figsize: Tuple[float, float] = (10, 6)  # width/height があれば上書き
    edgecolor: str = "black"
    title: str = "YAIBA: 滞在時間の分布（JST, 1秒分解能）"
    x_label: str = "在室時間 [minutes]"
    y_label: str = "人数 [counts]"

@dataclass
class VerParams:
    version: str = "v1"

__all__ = [
    "IOParams", "HistParams", "VerParams",
    "HistogramGenerator", "run_histogram_mvp"
]

# ====== ユーティリティ ======
REQUIRED_COLS = {"user_id", "second"}


def require_columns(df: pd.DataFrame):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")


def _default_hist_out_dir() -> str:
    """
    設計書準拠: YAIBA_RESULT_ROOT/histograms
    ※ 環境変数が無ければ naming.RESULT_ROOT を既定値に
    """
    base = Path(os.getenv("YAIBA_RESULT_ROOT", RESULT_ROOT))
    return str(base / "histograms")


def build_paths(io: IOParams, base: str) -> Tuple[str, str]:
    out_dir = io.out_dir or _default_hist_out_dir()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    png_path = str(Path(out_dir) / f"{base}.png")
    csv_path = str(Path(out_dir) / f"{base}.csv")
    return png_path, csv_path


def to_jst_floor_seconds(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """second列をJSTに正規化し、sec_floor(秒床)を付与。重複(user_id×sec_floor)は後勝ち1件。
    返却: [second(JST), sec_floor(JST), user_id, event_day(YYYY-MM-DD)]
    """
    df = df.copy()
    sec = pd.to_datetime(df["second"], errors="coerce")
    if sec.dt.tz is None:
        sec = sec.dt.tz_localize(JST)
    else:
        sec = sec.dt.tz_convert(JST)
    df["second"] = sec

    df = df.sort_values(["second", "user_id"])  # 後勝ち安定ソート
    df["sec_floor"] = df["second"].dt.floor("s")
    df = df.drop_duplicates(subset=["sec_floor", "user_id"], keep="last")

    if "event_day" not in df.columns:
        df["event_day"] = df["sec_floor"].dt.date.astype(str)
    return df


def log_summary(logger: logging.Logger, stats: Dict):
    logger.info("SUMMARY " + json.dumps(stats, ensure_ascii=False))

# ====== 本体 ======
class HistogramGenerator:
    """滞在時間ヒストグラムを生成するクラス（設計準拠）。"""
    def __init__(self, io: IOParams, hist: HistParams, ver: Union[VerParams, str, None]):
        self.io = io
        self.hist = hist
        # ver は VerParams / str / None / dict いずれも許容
        if isinstance(ver, VerParams):
            self.ver = ver
        elif isinstance(ver, str) or ver is None:
            self.ver = VerParams(version=(ver or "v1"))
        elif isinstance(ver, dict):
            self.ver = VerParams(**ver)
        else:
            self.ver = VerParams()
        self.logger = logging.getLogger("yaiba_bi.core.histogram")

    # --- 在室時間サマリ作成 ---
    def compute_dwell_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """ユーザ別の在室秒/分サマリ DataFrame を返す。
        入力: 必須列 user_id, second（任意: event_day）
        出力列: user_id, dwell_seconds, dwell_minutes, event_day
        """
        require_columns(df)
        dfj = to_jst_floor_seconds(df, self.logger)

        dwell_sec = dfj.groupby("user_id")["sec_floor"].nunique().rename("dwell_seconds")
        out = dwell_sec.to_frame()
        out["dwell_minutes"] = out["dwell_seconds"] / 60.0

        # 代表 event_day を選定（ユーザ×日で出現数が最大の日）
        by_user_day = dfj.groupby(["user_id", "event_day"]).size().rename("cnt").reset_index()
        rep = by_user_day.loc[by_user_day.groupby("user_id")["cnt"].idxmax(), ["user_id", "event_day"]]
        out = out.merge(rep, on="user_id", how="left")
        return out.reset_index()[["user_id", "dwell_seconds", "dwell_minutes", "event_day"]]

    # --- 描画 ---
    def draw_histogram(self, data: np.ndarray) -> Tuple[plt.Figure, plt.Axes, Dict[str, float]]:
        # ピクセル基準に統一（width/height は px と解釈）
        dpi_base = float(self.hist.dpi or self.io.png_dpi or 144)
        if self.hist.width and self.hist.height:
            figsize = (float(self.hist.width) / dpi_base,
                       float(self.hist.height) / dpi_base)
        else:
            figsize = self.hist.figsize

        fig, ax = plt.subplots(figsize=figsize)
        try:
            fig.set_dpi(int(dpi_base))
        except Exception:
            pass

        bins = self.hist.bins if self.hist.bins is not None else "auto"
        ax.hist(data, bins=bins, edgecolor=self.hist.edgecolor)
        ax.set_xlabel(self.hist.x_label)
        ax.set_ylabel(self.hist.y_label)
        ax.set_title(self.hist.title)

        stats = {
            "mean": float(np.mean(data)) if data.size else float("nan"),
            "median": float(np.median(data)) if data.size else float("nan"),
            "p95": float(np.quantile(data, 0.95)) if data.size else float("nan"),
        }
        for v, label in [(stats["mean"], "Mean"), (stats["median"], "Median"), (stats["p95"], "P95")]:
            if not (isinstance(v, float) and math.isnan(v)):
                ax.axvline(v, linestyle="--")
                ymax = ax.get_ylim()[1]
                ax.text(v, ymax * 0.95, label, rotation=90, va="top")
        return fig, ax, stats

    # --- 実行フロー（設計準拠シグネチャ） ---
    def run(self, df: pd.DataFrame, output_basename: str) -> Dict:
        tracemalloc.start()
        try:
            if df is None or len(df) == 0:
                raise ValueError("empty df")

            self.logger.info("[Dwell] prepare start")
            df_summary = self.compute_dwell_summary(df)
            if df_summary.empty:
                raise ValueError("no dwell data")

            data = df_summary["dwell_minutes"].to_numpy()
            self.logger.info(f"[Dwell] rows={len(df)} users={df_summary.shape[0]}")

            fig, ax, stats = self.draw_histogram(data)

            now_str = datetime.now(JST).strftime("%Y%m%d-%H%M%S")
            base = f"{output_basename}-{now_str}_{self.ver.version}"
            png_path, csv_path = build_paths(self.io, base)

            save_dpi = self.io.png_dpi or self.hist.dpi or 144
            fig.savefig(png_path, dpi=save_dpi, bbox_inches="tight")
            df_summary.to_csv(csv_path, index=False, encoding=self.io.csv_encoding)

            current, peak = tracemalloc.get_traced_memory()
            mem_kb = (peak - current) / 1024.0

            result = {
                "png": png_path,
                "csv": csv_path,
                "users": int(df_summary.shape[0]),
                "mean": round(stats["mean"], 2) if not math.isnan(stats["mean"]) else float("nan"),
                "median": round(stats["median"], 2) if not math.isnan(stats["median"]) else float("nan"),
                "p95": round(stats["p95"], 2) if not math.isnan(stats["p95"]) else float("nan"),
                "mem_kb_diff": round(mem_kb, 1),
            }
            log_summary(self.logger, result)
            plt.close(fig)
            return result

        except PermissionError as e:
            self.logger.error(f"EC={EC_STORAGE_PERM} perm err={e}")
            raise
        except OSError as e:
            self.logger.error(f"EC={EC_STORAGE_IO} io err={e}")
            raise
        except ValueError as e:
            self.logger.error(f"EC={EC_STATS_INPUT} value err={e}")
            raise
        except Exception as e:
            self.logger.exception(f"EC={EC_STATS_UNKNOWN} unexpected err={e}")
            raise
        finally:
            gc.collect()
            tracemalloc.stop()


# ====== シンプルAPI（既存呼び出しの利便性を維持） ======
def run_histogram_mvp(
    df: Optional[pd.DataFrame] = None,
    csv_path: Optional[Union[str, os.PathLike]] = None,
    io: Optional[IOParams] = None,
    hist: Optional[HistParams] = None,
    ver: Optional[Union[VerParams, str]] = None,
    output_basename: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """MVP: 滞在時間ヒストグラムを生成して PNG/CSV を出力し、要約を返す。

    Args:
        df: 入力 DataFrame（列: second, user_id, (任意)event_day）。csv_path とどちらか必須。
        csv_path: 入力 CSV/Parquet のパス。
        io: 出力やファイル名設定。未指定でも OK（設計既定パスに保存）。
        hist: ヒスト設定（bins 等）。未指定ならデフォルト。
        ver: バージョン付与。VerParams でも "v1" のような文字列でも可。
        output_basename: ファイル名のベース。未指定なら io.output_filename を使用。
        logger: 任意のロガー。

    Returns:
        dict: { png, csv, users, mean, median, p95, mem_kb_diff }
    """
    if logger is None:
        logger = logging.getLogger("yaiba_bi.core.histogram")
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if df is None and csv_path is None:
        raise ValueError("df or csv_path must be provided")

    if df is None:
        p = str(csv_path)
        logger.info(f"[Dwell] read: {p}")
        if p.lower().endswith((".parquet", ".parq")):
            # 要: pyarrow
            df = pd.read_parquet(p, engine="pyarrow")
        else:
            df = pd.read_csv(p)

    io = io or IOParams()
    hist = hist or HistParams()
    gen = HistogramGenerator(io=io, hist=hist, ver=ver)
    return gen.run(df, output_basename or io.output_filename)


# ====== ローカル単体動作用（任意） ======
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    # ダミーデータ例: 3ユーザ（A/B/C）
    rng = pd.date_range("2025-10-06 23:59:00+09:00", periods=301, freq="S")
    df_demo = pd.DataFrame({
        "second": np.concatenate([rng, rng, rng[:120]]),
        "user_id": ["A"]*301 + ["B"]*301 + ["C"]*120,
    })
    res = run_histogram_mvp(df=df_demo, output_basename="demo_hist")
    print(json.dumps(res, ensure_ascii=False, indent=2))
