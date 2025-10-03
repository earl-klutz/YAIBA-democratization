# -*- coding: utf-8 -*-
"""
generate_demo.py
JSTのダミーデータを生成して、動画（movie_xz）とヒストグラム（PNG/CSV）を出力するユーティリティ。

前提:
- プロジェクト構成:
  ./src/yaiba_bi/core/...
- 依存:
  pip install pandas numpy matplotlib tqdm

使い方（例）:
  python generate_demo.py
  python generate_demo.py --seconds 300 --n-users 8 --duration-sec 10 --fps 15 --bitrate 1500 --overwrite
"""

from __future__ import annotations
import os
import sys
import math
import logging
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# --- src を import パスに追加 ---
ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# --- YAIBA Core の読み込み ---
from yaiba_bi.core import run_movie_xz
from yaiba_bi.core.movie import MovieParams, IOParams as MIO
from yaiba_bi.core.histogram import HistogramGenerator, HistParams, IOParams as HIO
from yaiba_bi.core.validation import PipelineError

# --- ロガー ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("generate_demo")

TZ_JST = ZoneInfo("Asia/Tokyo")


def build_df_jst(seconds: int = 200, n_users: int = 8) -> pd.DataFrame:
    """
    JSTで 'second' 列を持つダミーデータを生成。
    - seconds >= 180 を推奨（動画要件）
    - 必須列: second, user_id, location_x, location_y, location_z, event_day
    """
    now = datetime.now(TZ_JST).replace(microsecond=0)
    rows = []
    rng = np.random.default_rng(42)
    for s in range(seconds):
        t = now + timedelta(seconds=s)
        for uid in range(n_users):
            # 適当に波形＋ノイズ
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


def ensure_output_roots(result_root: str, meta_root: str) -> None:
    """出力ルート（環境変数）を設定。"""
    os.makedirs(result_root, exist_ok=True)
    os.makedirs(meta_root, exist_ok=True)
    os.environ.setdefault("YAIBA_RESULT_ROOT", result_root)
    os.environ.setdefault("YAIBA_META_ROOT", meta_root)
    logger.info(f"RESULT_ROOT={os.environ['YAIBA_RESULT_ROOT']}")
    logger.info(f"META_ROOT={os.environ['YAIBA_META_ROOT']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate demo movie and histogram from synthetic JST data.")
    p.add_argument("--seconds", type=int, default=200, help="ダミーデータの総秒数（>=180推奨）")
    p.add_argument("--n-users", type=int, default=8, help="ユーザ数")
    p.add_argument("--duration-sec", type=int, default=10, help="動画の再生長（秒）")
    p.add_argument("--fps", type=int, default=15, help="動画fps")
    p.add_argument("--bitrate", type=str, default="1500", help="動画ビットレート（kbps数値 or '2M' など文字列も可）")
    p.add_argument("--duration-real", type=int, default=3600, help="解析の上限窓（秒）")
    p.add_argument("--outfile", type=str, default="demo", help="出力ファイル名のベース（拡張子や接頭辞は自動）")
    p.add_argument("--overwrite", action="store_true", help="既存ファイルがあっても上書きする")
    p.add_argument("--result-root", type=str, default=os.path.join(ROOT, "results"), help="成果物のルート出力先")
    p.add_argument("--meta-root", type=str, default=os.path.join(ROOT, "meta"), help="メタ(ログ等)のルート出力先")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_output_roots(args.result_root, args.meta_root)

    # --- ダミーデータ生成（JST） ---
    df = build_df_jst(seconds=args.seconds, n_users=args.n_users)
    logger.info(f"Generated dummy DataFrame: rows={len(df)} seconds={args.seconds} users={args.n_users}")

    # --- 動画（C-2） ---
    try:
        movie_params = MovieParams(
            duration_sec=args.duration_sec,
            fps=args.fps,
            bitrate=args.bitrate,       # 文字列/数値どちらでもOK（内部で解決）
            duration_real=args.duration_real,
        )
        movie_io = MIO(output_filename=args.outfile, overwrite=args.overwrite)
        mstats = run_movie_xz(df, movie=movie_params, io=movie_io, ver="demo")
        logger.info(f"[MOVIE] mp4_path={mstats['mp4_path']} frames={mstats['frames']} mem_kb_diff={mstats['mem_kb_diff']}")
    except PipelineError as e:
        logger.error(f"[MOVIE] PipelineError: EC={e.code} msg={e.msg}")
        mstats = {}

    # --- ヒストグラム（C-1） ---
    try:
        hist = HistogramGenerator(
            hist=HistParams(bins="fd", dpi=120, width=960, height=720),
            io=HIO(output_filename=f"{args.outfile}_hist", overwrite=True),
            ver="demo"
        )
        hstats = hist.run(df)
        logger.info(f"[HIST] png_path={hstats['png_path']} csv_path={hstats['csv_path']} peak={hstats['peak']} p95={hstats['p95']}")
    except PipelineError as e:
        logger.error(f"[HIST] PipelineError: EC={e.code} msg={e.msg}")
        hstats = {}

    # --- まとめ表示 ---
    print({
        "movie": mstats,
        "hist": hstats,
    })


if __name__ == "__main__":
    main()
