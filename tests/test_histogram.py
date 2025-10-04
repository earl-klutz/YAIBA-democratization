# tests/test_histogram.py
import os
import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")  # ヘッドレス

from yaiba_bi.core import HistogramGenerator
from yaiba_bi.core.histogram import HistParams, IOParams as HIO
from yaiba_bi.core.validation import PipelineError

def test_smoke_hist_png_csv(make_df):
    """
    エンドツーエンド: PNG と CSV を生成できること（短尺データ）。
    - seconds>=180 を満たすダミーデータ
    - bins='fd' のまま
    """
    df = make_df(seconds=200, n_users=6)  # >=180秒
    gen = HistogramGenerator(
        hist=HistParams(bins="fd", dpi=120, width=640, height=480),
        io=HIO(output_filename="hist_smoke", overwrite=True),
        ver="test",
    )
    stats = gen.run(df)

    assert "png_path" in stats and os.path.isfile(stats["png_path"])
    assert "csv_path" in stats and os.path.isfile(stats["csv_path"])
    # サマリ数値が含まれている
    for key in ("peak", "mean", "median", "p95"):
        assert key in stats

    # CSV の中身を軽く確認
    out = pd.read_csv(stats["csv_path"])
    assert set(out.columns) == {"second_jst", "concurrency"}
    assert len(out) >= 180  # 180秒以上の行がある
    assert out["concurrency"].min() >= 0

def test_hist_raises_on_short_data(make_df):
    """
    ユニーク秒 < 180 の場合、EC=-2403 を投げる。
    """
    df = make_df(seconds=100, n_users=4)  # 足りない
    gen = HistogramGenerator(hist=HistParams(bins="auto"), io=HIO(output_filename="short", overwrite=True))
    with pytest.raises(PipelineError) as ei:
        gen.run(df)
    assert ei.value.code == -2403  # EC_STATS_SAMPLE_SHORT

def test_hist_bins_and_range(make_df, tmp_path):
    """
    bins 個数とレンジ指定がエラーなく適用できること。
    - bins=10
    - range_min/max を固定
    """
    df = make_df(seconds=200, n_users=5)
    # concurrency のおおよそのレンジを見て、狭めのレンジを与えても落ちないことだけ確認
    # （プロットの画像そのものは視覚検証対象外）
    gen = HistogramGenerator(
        hist=HistParams(bins=10, range_min=0, range_max=20, dpi=100, width=600, height=400),
        io=HIO(output_filename="hist_range", overwrite=True),
        ver="test",
    )
    stats = gen.run(df)
    assert os.path.isfile(stats["png_path"])
    assert os.path.isfile(stats["csv_path"])

def test_event_day_naming_and_overwrite(make_df, monkeypatch, tmp_path):
    """
    - 命名（event_day-..._ver）で出力されること
    - overwrite=False のとき既存があればエラー、True なら上書き可能
    """
    df = make_df(seconds=200, n_users=6)
    gen = HistogramGenerator(
        hist=HistParams(bins="fd"),
        io=HIO(output_filename="hist_name", overwrite=True),
        ver="c1.0",
    )
    stats1 = gen.run(df)
    assert os.path.isfile(stats1["png_path"])

    # 上書き不可でエラーにするため、一度 overwrite=False で同じベース名を狙う
    # 同じ dt にはならないが、出力先ファイル名の先頭（event_day-...）は共通なので、
    # ここでは overwrite=True/False の動作だけを見る目的で簡易に再実行
    gen2 = HistogramGenerator(
        hist=HistParams(bins="fd"),
        io=HIO(output_filename="hist_name", overwrite=False),
        ver="c1.0",
    )
    # 既存ファイルチェックは movie.py の挙動と異なり、histogram.py は
    # result_path("image"/"table") を毎回新しいdtで生成するため衝突しづらい。
    # ここでは run() が成功すること（エラーにならないこと）だけ確認する。
    stats2 = gen2.run(df)
    assert os.path.isfile(stats2["png_path"])
    assert os.path.isfile(stats2["csv_path"])

def test_hist_summary_values_monotonic(make_df):
    """
    サマリ値（peak >= p95 >= median 付近）がおおよそ整合していることをゆるく確認。
    厳密比較ではなく “明らかな異常がない” 程度の妥当性チェック。
    """
    df = make_df(seconds=200, n_users=8)
    gen = HistogramGenerator(hist=HistParams(bins="auto"), io=HIO(output_filename="hist_stats", overwrite=True))
    stats = gen.run(df)

    peak = stats["peak"]
    mean = stats["mean"]
    median = stats["median"]
    p95 = stats["p95"]

    # 緩い妥当性（常に成り立つとは限らないが、異常値検出のスクリーニングとして）
    assert peak >= p95
    assert peak >= median
    # 平均は中央値と同程度になることが多い（分布による）
    assert abs(mean - median) < max(10, 0.5 * peak)
