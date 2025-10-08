# -*- coding: utf-8 -*-
"""
demo.py
- CSV を yaiba_loader（core 内最優先）で読み込み → Movie/Histogram 出力
- CSV 未指定ならダミーデータ生成（JST）
- 返り値が LogData の場合は get_position() を使って中間DFを構築
- 依存: matplotlib, numpy, pandas, tzdata, tqdm, imageio-ffmpeg（+ OS側 ffmpeg）
"""
from __future__ import annotations
import os
import sys
import math
import argparse
import importlib
import importlib.util
from types import ModuleType
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import platform

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---- パス調整（src を import 可能に） ----
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---- YAIBA Core API ----
from yaiba_bi.core.movie import MovieGenerator, MovieParams, IOParams as MIO
from yaiba_bi.core.histogram import HistogramGenerator, HistParams, HistIOParams as HIO
from yaiba_bi.core.validation import PipelineError
from yaiba_bi.core.logging_util import get_logger, log_summary
from yaiba_bi.core.naming import meta_paths

TZ_JST = ZoneInfo("Asia/Tokyo")


# ===============================================================
# 環境依存フォント設定
# ===============================================================
def setup_fonts() -> None:
    if platform.system() == "Windows":
        plt.rcParams["font.family"] = "Meiryo"
    else:
        colab_font_path = "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"
        if os.path.exists(colab_font_path):
            font_prop = fm.FontProperties(fname=colab_font_path)
            plt.rcParams["font.family"] = font_prop.get_name()
        else:
            plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


# ===============================================================
# yaiba_loader の自動 import（core 内を最優先）
# ===============================================================
def _import_loader_module(loader_path: str | None) -> ModuleType:
    """
    yaiba_loader を import して返す。
    優先順:
      1) yaiba_bi.core.yaiba_loader
      2) --loader で指定された .py ファイル
      3) 通常 import（PYTHONPATH）
    """
    # 1) core 内
    try:
        return importlib.import_module("yaiba_bi.core.yaiba_loader")
    except Exception:
        pass

    # 2) 明示パス指定
    if loader_path:
        lp = Path(loader_path)
        if not lp.exists():
            raise FileNotFoundError(f"loader not found: {lp}")
        spec = importlib.util.spec_from_file_location("yaiba_loader", str(lp))
        if spec is None or spec.loader is None:
            raise ImportError(f"failed to build spec for {lp}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["yaiba_loader"] = mod
        spec.loader.exec_module(mod)
        return mod

    # 3) 通常 import（プロジェクト直下など）
    return importlib.import_module("yaiba_loader")


def load_obj_via_yaiba_loader(csv_path: str, loader_path: str | None = None):
    """
    yaiba_loader から“生オブジェクト”を取得。
    優先: load_csv_to_df(csv_path [, logger]) → read_csv(csv_path) → load(csv_path)
    DataFrame で返る場合も LogData 等で返る場合もある。
    """
    try:
        ymod = _import_loader_module(loader_path)
    except Exception as e:
        raise ImportError("yaiba_loader の import に失敗しました") from e

    if hasattr(ymod, "load_csv_to_df"):
        fn = getattr(ymod, "load_csv_to_df")
        try:
            return fn(csv_path)
        except TypeError:
            import logging
            return fn(csv_path, logging.getLogger("yaiba_loader.adapter"))
    if hasattr(ymod, "read_csv"):
        return getattr(ymod, "read_csv")(csv_path)
    if hasattr(ymod, "load"):
        return getattr(ymod, "load")(csv_path)

    raise ImportError("yaiba_loader に利用可能な関数がありません（load_csv_to_df / read_csv / load）")


# ===============================================================
# LogData → 中間DFを構築（get_position を利用）
# ===============================================================
def build_df_from_logdata(obj) -> pd.DataFrame:
    """
    yaiba_loader.LogData が提供する API を使って
    中間形式 DataFrame（second, user_id, location_x, location_y, location_z, event_day）を作る。
    """
    # 必須API
    if not hasattr(obj, "get_position"):
        raise TypeError("LogData 互換ではありません（get_position() がありません）")

    pos = obj.get_position()
    if not isinstance(pos, pd.DataFrame) or pos.empty:
        raise TypeError("get_position() が DataFrame ではないか、空です")

    # 欲しい列名の候補群
    def _pick(df: pd.DataFrame, cands: list[str]) -> str | None:
        for c in cands:
            if c in df.columns:
                return c
        lower_map = {str(c).lower(): c for c in df.columns}
        for c in cands:
            lc = c.lower()
            if lc in lower_map:
                return lower_map[lc]
        return None

    # 列マッピング
    tcol = _pick(pos, ["second", "timestamp", "time", "ts", "datetime", "date"])
    uid = _pick(pos, ["user_id", "uid", "user", "id"])
    xcol = _pick(pos, ["location_x", "x", "pos_x", "lon", "lng"])
    ycol = _pick(pos, ["location_y", "y", "pos_y", "lat"])
    zcol = _pick(pos, ["location_z", "z", "pos_z", "alt", "height", "depth"])

    if tcol is None or uid is None or xcol is None:
        raise TypeError(
            f"get_position() から必要列を見つけられません: "
            f"time={tcol}, user_id={uid}, x={xcol}（y/z は無ければ 0 扱い）"
        )

    # ベースDF
    df = pd.DataFrame({
        "second": pd.to_datetime(pos[tcol], errors="coerce"),
        "user_id": pos[uid].astype("string"),
        "location_x": pd.to_numeric(pos[xcol], errors="coerce"),
        "location_y": pd.to_numeric(pos[ycol], errors="coerce") if ycol else 0.0,
        "location_z": pd.to_numeric(pos[zcol], errors="coerce") if zcol else 0.0,
    })

    # JST日付
    sec = df["second"]
    if getattr(sec.dtype, "tz", None) is not None:
        ed = sec.dt.tz_convert("Asia/Tokyo").dt.date
    else:
        ed = sec.dt.tz_localize("Asia/Tokyo").dt.date
    df["event_day"] = ed.astype("string")

    # 必須の欠損除去
    df = df.dropna(subset=["second", "user_id", "location_x"]).reset_index(drop=True)
    return df


# ===============================================================
# 汎用：どんなオブジェクトでも可能な限り DataFrame にする
# ===============================================================
def coerce_to_dataframe(obj) -> pd.DataFrame:
    """
    できるだけ DataFrame に変換する。
    まず LogData API（get_position 等）に対応し、ダメなら汎用の探索/変換にフォールバック。
    """
    # 1) LogData 直対応
    if hasattr(obj, "get_position"):
        try:
            return build_df_from_logdata(obj)
        except Exception:
            pass  # 汎用ルートへ

    # 2) 既に DataFrame
    if isinstance(obj, pd.DataFrame):
        return obj

    # 3) pandas interchange protocol
    try:
        if hasattr(obj, "__dataframe__"):
            import pandas as _pd
            try:
                return _pd.api.interchange.from_dataframe(obj)  # type: ignore[attr-defined]
            except Exception:
                proto = obj.__dataframe__()  # pyright: ignore
                if hasattr(proto, "to_pandas"):
                    return proto.to_pandas()  # type: ignore
    except Exception:
        pass

    # 4) 代表メソッド
    for m in ["to_dataframe", "to_pandas", "as_dataframe", "get_dataframe", "as_pandas", "to_df"]:
        if hasattr(obj, m):
            try:
                val = getattr(obj, m)()
                if isinstance(val, pd.DataFrame):
                    return val
                try:
                    return pd.DataFrame(val)
                except Exception:
                    pass
            except Exception:
                pass

    # 5) 代表属性
    for a in ["df", "dataframe", "data", "table", "frame", "payload", "content", "records", "rows", "items", "body", "result", "value", "values"]:
        if hasattr(obj, a):
            try:
                val = getattr(obj, a)
                if isinstance(val, pd.DataFrame):
                    return val
                try:
                    return pd.DataFrame(val)
                except Exception:
                    pass
            except Exception:
                pass

    # 6) どうしてもダメなら最後の手段
    try:
        return pd.DataFrame(obj)
    except Exception as e:
        raise TypeError(
            f"yaiba_loader からの返り値を DataFrame に変換できませんでした: type={type(obj)}\n"
            "対応: LogData(get_position) / __dataframe__ / to_dataframe / to_pandas / as_dataframe / get_dataframe / to_df / "
            "df / dataframe / data / table / frame / payload / content / records / rows / items / body / result / value / values"
        ) from e


# ===============================================================
# ダミーデータ生成（JST）
# ===============================================================
def build_df_jst(seconds: int = 200, n_users: int = 8) -> pd.DataFrame:
    now = datetime.now(TZ_JST).replace(microsecond=0)
    rows = []
    rng = np.random.default_rng(42)
    for s in range(seconds):
        t = now + timedelta(seconds=s)
        for uid in range(n_users):
            x = float(math.sin(0.01 * s + uid) * 10 + rng.normal(0, 0.1))
            z = float(math.cos(0.008 * s + uid) * 8 + rng.normal(0, 0.1))
            rows.append({
                "second": t,
                "user_id": f"u{uid}",
                "location_x": x,
                "location_y": 0.0,
                "location_z": z,
                "event_day": now.date().isoformat(),
            })
    return pd.DataFrame(rows)


# ===============================================================
# 出力ルート整備
# ===============================================================
def ensure_output_roots(result_root: str, meta_root: str) -> None:
    os.makedirs(result_root, exist_ok=True)
    os.makedirs(meta_root, exist_ok=True)
    os.environ.setdefault("YAIBA_RESULT_ROOT", result_root)
    os.environ.setdefault("YAIBA_META_ROOT", meta_root)


# ===============================================================
# メインパイプライン
# ===============================================================
def run_pipeline(
    df: pd.DataFrame,
    *,
    outfile_base: str,
    overwrite: bool,
    duration_sec: int,
    fps: int,
    bitrate: str | int,
    duration_real: int,
    hist_bins: int | str = "fd",
    hist_dpi: int = 120,
    hist_size: tuple[int, int] = (960, 720),
    ver_movie: str = "demo",
    ver_hist: str = "demo",
) -> dict:
    dt_jst = datetime.now(TZ_JST).strftime("%Y%m%d_%H%M%S")
    logger = get_logger(run_id=dt_jst, log_path=meta_paths(dt_jst, ver_movie)["log_path"])

    setup_fonts()

    results = {"movie": {}, "hist": {}}

    # Movie
    try:
        mv = MovieParams(duration_sec=duration_sec, fps=fps, bitrate=bitrate, duration_real=duration_real)
        mgen = MovieGenerator(movie=mv, io=MIO(output_filename=outfile_base, overwrite=overwrite), ver=ver_movie)
        results["movie"] = mgen.run(df)
    except PipelineError as e:
        logger.error(f"[MOVIE] EC={e.code} msg={e.msg}")
    except Exception as e:
        logger.exception(f"[MOVIE] unexpected err={e}")

    # Histogram
    try:
        hp = HistParams(bins=hist_bins, dpi=hist_dpi, width=hist_size[0], height=hist_size[1])
        hgen = HistogramGenerator(hist=hp, io=HIO(output_filename=f"{outfile_base}_hist", overwrite=overwrite), ver=ver_hist)
        results["hist"] = hgen.run(df, output_basename=f"{outfile_base}_hist")
    except PipelineError as e:
        logger.error(f"[HIST] EC={e.code} msg={e.msg}")
    except Exception as e:
        logger.exception(f"[HIST] unexpected err={e}")

    log_summary(logger, {"results": results})
    return results


# ===============================================================
# CLI
# ===============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YAIBA demo: CSV or synthetic data → movie & histogram")
    p.add_argument("--csv", type=str, default=None, help="CSV ファイルを yaiba_loader で読み込み（未指定ならダミーデータ生成）")
    p.add_argument("--loader", type=str, default=None, help="yaiba_loader.py のパス（省略時は core 内のものを使用）")
    p.add_argument("--seconds", type=int, default=200, help="ダミーデータ総秒数")
    p.add_argument("--n-users", type=int, default=8, help="ダミーユーザ数")
    p.add_argument("--duration-sec", type=int, default=10, help="動画の長さ（秒）")
    p.add_argument("--fps", type=int, default=15, help="動画FPS")
    p.add_argument("--bitrate", type=str, default="1500", help="動画ビットレート（kbps または文字列）")
    p.add_argument("--duration-real", type=int, default=3600, help="解析対象期間（秒）")
    p.add_argument("--outfile", type=str, default="demo", help="出力ファイルベース名")
    p.add_argument("--overwrite", action="store_true", help="既存ファイルを上書き")
    p.add_argument("--result-root", type=str, default=str(ROOT / "results"), help="成果物ルート")
    p.add_argument("--meta-root", type=str, default=str(ROOT / "meta"), help="メタ情報ルート")
    p.add_argument("--hist-bins", default="fd", help="ヒストグラム bins 設定")
    p.add_argument("--hist-dpi", type=int, default=120, help="ヒストDPI")
    p.add_argument("--hist-width", type=int, default=960)
    p.add_argument("--hist-height", type=int, default=720)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_roots(args.result_root, args.meta_root)

    if args.csv:
        raw = load_obj_via_yaiba_loader(args.csv, loader_path=args.loader)
        df = coerce_to_dataframe(raw)
    else:
        df = build_df_jst(seconds=args.seconds, n_users=args.n_users)

    stats = run_pipeline(
        df,
        outfile_base=args.outfile,
        overwrite=args.overwrite,
        duration_sec=args.duration_sec,
        fps=args.fps,
        bitrate=args.bitrate,
        duration_real=args.duration_real,
        hist_bins=args.hist_bins,
        hist_dpi=args.hist_dpi,
        hist_size=(args.hist_width, args.hist_height),
    )
    print(stats)


if __name__ == "__main__":
    main()
