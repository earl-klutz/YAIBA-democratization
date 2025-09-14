# -*- coding: utf-8 -*-
# src/engine/core/validate_position_attendance.py
# 目的: 開発者が単体テストとして yaiba_loader.load(...) を呼び、
#       position / attendance を <DATA_ROOT>/output/meta/valid/ にCSV保存する

from __future__ import annotations
import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

def _import_loader():
    try:
        import yaiba_loader as yl
        return yl
    except Exception:
        sys.path.insert(0, str(Path(__file__).parent))
        import yaiba_loader as yl  # noqa: F401
        return yl

def main():
    ap = argparse.ArgumentParser(description="YAIBA 単体検証: position/attendance をCSV保存")
    ap.add_argument("--user-path", required=True,
                    help=r'ユーザのローカル風パス（例: "C:\Users\me\Downloads\log.txt" や "~/Downloads/log.txt"）')
    ap.add_argument("--span", type=int, default=1, help="リサンプリング間隔（秒, default=1）")
    ap.add_argument("--is-pseudo", action="store_true", default=True,
                    help="匿名化を有効にする（default=True, --no-is-pseudoで無効化）")
    ap.add_argument("--no-is-pseudo", dest="is_pseudo", action="store_false")
    ap.add_argument("--time-sync", type=str, default=None,
                    help="時間同期基準 (例: '2025-09-06 22:55:27')")
    args = ap.parse_args()

    yl = _import_loader()
    data_root = Path(yl.DEFAULT_DATA_ROOT).resolve()

    # 入力ファイル存在チェック
    input_path = Path(args.user_path).expanduser().resolve()
    if not input_path.exists():
        print(f"[ERROR] 入力ファイルが存在しません: {input_path}")
        sys.exit(1)

    # time_sync の変換
    time_sync_dt = None
    if args.time_sync:
        try:
            time_sync_dt = datetime.strptime(args.time_sync, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print("[WARN] time-sync の形式が不正です。無視します。")

    # YAIBA ローダ呼び出し
    try:
        ld = yl.load(str(input_path),
                     span=args.span,
                     is_pseudo=args.is_pseudo,
                     time_sync=time_sync_dt)
    except Exception:
        print("[ERROR] yaiba_loader.load で例外が発生しました。")
        traceback.print_exc()
        sys.exit(2)

    # 出力先: <DATA_ROOT>/output/meta/valid/
    outdir = data_root / "output" / "meta" / "valid"
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # position を保存
    try:
        if ld.position is not None:
            pos_csv = outdir / f"position_{ts}.csv"
            ld.position.to_csv(pos_csv, index=False, encoding="utf-8-sig")
            print(f"[VALID] position -> {pos_csv}")
    except Exception:
        print("[ERROR] position の保存で例外が発生しました。")
        traceback.print_exc()

    # attendance を保存（任意）
    try:
        if ld.attendance is not None and not ld.attendance.empty:
            att_csv = outdir / f"attendance_{ts}.csv"
            ld.attendance.to_csv(att_csv, index=False, encoding="utf-8-sig")
            print(f"[VALID] attendance -> {att_csv}")
        else:
            print("[VALID] attendance は None/empty のため保存スキップ")
    except Exception:
        print("[ERROR] attendance の保存で例外が発生しました。")
        traceback.print_exc()

if __name__ == "__main__":
    main()
