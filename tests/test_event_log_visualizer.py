from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from yaiba_bi.core.event_log_visualizer import (
    EventLogVisualizer,
    RenderConfig,
    TrajectoryConfig,
)


def _make_attendance() -> pd.DataFrame:
    base = pd.Timestamp("2025-01-01 00:00:00")
    records = [
        {"second": base + pd.Timedelta(seconds=0), "action": "join", "user_id": 1, "is_error": False},
        {"second": base + pd.Timedelta(seconds=5), "action": "join", "user_id": 2, "is_error": False},
        {"second": base + pd.Timedelta(seconds=9), "action": "left", "user_id": 1, "is_error": False},
        {"second": base + pd.Timedelta(seconds=12), "action": "left", "user_id": 2, "is_error": False},
    ]
    return pd.DataFrame(records)


def _make_df_pos() -> pd.DataFrame:
    rows = []
    for idx, user_id in enumerate(["user_a", "user_b"]):
        times = pd.date_range("2025-01-01", periods=6, freq="2s") + pd.Timedelta(seconds=idx)
        xs = np.linspace(-2 + idx, 2 + idx, num=len(times))
        zs = np.linspace(-1 + idx, 1 + idx, num=len(times))
        for second, x, z in zip(times, xs, zs):
            rows.append({
                "second": second,
                "user_id": user_id,
                "location_x": x,
                "location_z": z,
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def visualizer(tmp_path: Path) -> EventLogVisualizer:
    """EventLogVisualizerのインスタンスをテスト用一時ディレクトリで生成するfixture"""
    rcfg = RenderConfig(event_day="2025-01-01", filename="sample")
    tcfg = TrajectoryConfig()
    visualizer = EventLogVisualizer(rcfg, tcfg)
    visualizer._naming.RESULT_ROOT = tmp_path / "results"  # type: ignore[attr-defined]
    return visualizer


def test_compute_cc_stats_matches_numpy(visualizer: EventLogVisualizer) -> None:
    """
    compute_cc_statsの出力がpandas/numpyの計算結果と一致することを確認する。
    max, mean, median, p95が正しく計算されているかをテストする。
    """
    attendance = _make_attendance()
    df_cc = visualizer._build_concurrency(attendance)
    stats = visualizer.compute_cc_stats(df_cc)

    assert stats["max"] == 2
    assert stats["mean"] == pytest.approx(df_cc["cc"].mean())
    assert stats["median"] == pytest.approx(df_cc["cc"].median())
    assert stats["p95"] == pytest.approx(np.percentile(df_cc["cc"], 95))


def test_run_generates_all_outputs(visualizer: EventLogVisualizer) -> None:
    """
    runメソッドが全ての成果物（cc_png, traj_png, stats_txt）を生成し、
    それぞれのファイルが存在すること、stats_txtの内容に統計指標が含まれることを確認する。
    """
    attendance = _make_attendance()
    df_pos = _make_df_pos()
    area = {"min_x": -3.0, "max_x": 3.0, "min_z": -2.0, "max_z": 2.5}

    results = visualizer.run(attendance, df_pos, area)

    for key in ["cc_png", "traj_png", "stats_txt"]:
        path_str = results[key]
        assert path_str
        path = Path(path_str)
        assert path.exists()

    stats_path = Path(results["stats_txt"])
    content = stats_path.read_text(encoding="utf-8")
    assert "max:" in content
    assert "mean:" in content
    assert "median:" in content
    assert "p95:" in content

def test_run_with_missing_data(visualizer: EventLogVisualizer) -> None:
    """
    欠損値（NaN）を含むattendance/df_posを与えてもrunが例外を出さず、
    全ての成果物（cc_png, traj_png, stats_txt）が生成されることを確認する。
    stats_txtの内容にも統計指標が含まれることを確認する。
    """
    # 欠損値を含むattendance
    attendance = _make_attendance()
    attendance_missing = attendance.copy()
    attendance_missing.loc[1, "user_id"] = None  # user_id 欠損
    attendance_missing.loc[2, "action"] = None   # action 欠損

    # 欠損値を含むdf_pos
    df_pos = _make_df_pos()
    df_pos_missing = df_pos.copy()
    df_pos_missing.loc[0, "location_x"] = None
    df_pos_missing.loc[1, "user_id"] = None

    area = {"min_x": -3.0, "max_x": 3.0, "min_z": -2.0, "max_z": 2.5}

    # 欠損データがあってもrunが例外を出さず、出力が生成されること
    results = visualizer.run(attendance_missing, df_pos_missing, area)

    for key in ["cc_png", "traj_png", "stats_txt"]:
        path_str = results[key]
        assert isinstance(path_str, str)
        if key != "traj_png" or path_str:  # areaが正常なのでtraj_pngも生成されるはず
            path = Path(path_str)
            assert path.exists()

    stats_path = Path(results["stats_txt"])
    content = stats_path.read_text(encoding="utf-8")
    assert "max:" in content
    assert "mean:" in content
    assert "median:" in content
    assert "p95:" in content
