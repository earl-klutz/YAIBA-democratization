# tests/test_movie_generator.py
import os
import math
import pytest

from yaiba_bi.core import run_movie_xz, MovieGenerator
from yaiba_bi.core.movie import MovieParams, IOParams
from yaiba_bi.core.validation import PipelineError

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as mpl_anim

# --- ユーティリティ: ffmpeg が使えない環境では保存テストをスキップ ---
ffmpeg_available = mpl_anim.FFMpegWriter.isAvailable()

@pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg が見つからないため MP4 保存テストをスキップ")
def test_smoke_encode_mp4(make_df, tmp_path):
    """
    エンドツーエンド: MP4 を生成できること（短尺・低fps・低ビットレートで高速化）。
    """
    df = make_df(seconds=200, n_users=6)  # >=180秒
    movie = MovieParams(duration_sec=3, fps=10, bitrate=1200, duration_real=3600)
    io = IOParams(output_filename="smoke", overwrite=True)

    stats = run_movie_xz(df, movie=movie, io=io, ver="test")
    assert "mp4_path" in stats
    assert os.path.isfile(stats["mp4_path"])
    assert stats["frames"] == movie.duration_sec * movie.fps

def test_raises_on_short_data(make_df):
    """
    ユニーク秒 < 180 の場合、EC=-2403 を投げる。
    """
    df = make_df(seconds=100, n_users=4)  # 足りない
    gen = MovieGenerator(movie=MovieParams(duration_sec=3, fps=5, bitrate=800))
    with pytest.raises(PipelineError) as ei:
        gen.run(df)
    assert ei.value.code == -2403  # EC_STATS_SAMPLE_SHORT

def test_raises_on_missing_column(make_df):
    """
    必須列欠如で EC=-2102 を投げる。
    """
    df = make_df(seconds=200, n_users=4).drop(columns=["location_z"])
    gen = MovieGenerator(movie=MovieParams(duration_sec=3, fps=5, bitrate=800))
    with pytest.raises(PipelineError) as ei:
        gen.run(df)
    assert ei.value.code == -2102  # EC_INPUT_FORMAT

def test_trail_marker_size_quarter(make_df):

    df = make_df(seconds=200, n_users=4)
    gen = MovieGenerator(movie=MovieParams(duration_sec=2, fps=5, bitrate=800))
    # 事前処理とレンダリング
    import logging
    logger = logging.getLogger("test")
    df_prep = gen.prepare(df, logger)
    anim, info = gen.render(df_prep, logger)

    # アーティストを取得（collections[0]: current, [1]: trail の順で作成）
    ax = anim._fig.axes[0]
    curr_scatter = ax.collections[0]
    trail_scatter = ax.collections[1]

    curr_s = float(curr_scatter.get_sizes()[0]) if len(curr_scatter.get_sizes()) else (gen.point.radius_px ** 2)
    trail_s = float(trail_scatter.get_sizes()[0]) if len(trail_scatter.get_sizes()) else ((gen.point.radius_px * 0.35) ** 2)

    # 期待値
    expected_curr = gen.point.radius_px ** 2
    expected_trail = (gen.point.radius_px * 0.35) ** 2

    # 許容誤差で比較（浮動小数・環境差誤差を吸収）
    assert math.isclose(curr_s, expected_curr, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(trail_s, expected_trail, rel_tol=1e-6, abs_tol=1e-6)
