from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from yaiba_bi.core import logging_util, naming
from yaiba_bi.core.engine import EngineConfig, VisualizationEngine
from yaiba_bi.core.stats_basic import compute_cc_stats
from yaiba_bi.core.validation import validate_df_cc, validate_df_pos
from yaiba_bi.core.spec_errors import SpecError


@pytest.fixture()
def patched_paths(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    meta_root = tmp_path / "meta"
    monkeypatch.setattr(naming, "RESULT_ROOT", results_root)
    monkeypatch.setattr(naming, "META_ROOT", meta_root)
    monkeypatch.setattr(logging_util, "META_ROOT", meta_root)
    return results_root, meta_root


def make_df_cc():
    times = pd.date_range("2025-01-01", periods=10, freq="S", tz="UTC")
    return pd.DataFrame({"t": times, "cc": np.linspace(10, 19, num=len(times))})


def make_df_pos():
    rows = []
    for idx, uid in enumerate(["user_a", "user_b"]):
        times = pd.date_range("2025-01-01", periods=6, freq="2S", tz="UTC") + pd.Timedelta(seconds=idx)
        xs = np.linspace(-2 + idx, 2 + idx, num=len(times))
        zs = np.linspace(-1 + idx, 1 + idx, num=len(times))
        for t, x, z in zip(times, xs, zs):
            rows.append({"t": t, "user_id": uid, "x": x, "z": z})
    return pd.DataFrame(rows)


def test_validation_errors():
    with pytest.raises(SpecError):
        validate_df_cc(pd.DataFrame({"cc": [1, 2]}))
    with pytest.raises(SpecError):
        validate_df_pos(pd.DataFrame({"t": [], "x": [], "z": []}))


def test_compute_cc_stats_basic():
    df_cc = make_df_cc()
    stats = compute_cc_stats(df_cc)
    assert stats["max"] == 19
    assert stats["mean"] == pytest.approx(df_cc["cc"].mean())
    assert stats["median"] == pytest.approx(df_cc["cc"].median())
    assert stats["p95"] == pytest.approx(np.percentile(df_cc["cc"], 95))


def test_engine_run_generates_outputs(tmp_path, patched_paths):
    results_root, meta_root = patched_paths
    df_cc = make_df_cc()
    df_pos = make_df_pos()
    area = {"min_x": -3.0, "max_x": 4.0, "min_z": -2.0, "max_z": 3.0}

    config = EngineConfig(
        render={"event_day": "2025-01-01", "filename": "sample"},
        trajectory_style={"dpi": 144, "width_px": 1280, "height_px": 720},
        io={"event_day": "2025-01-01", "filename": "sample", "ver": "b1.0", "dt": "20250101_010101"},
    )

    engine = VisualizationEngine(config)
    results = engine.run(df_cc, df_pos, area)

    assert results["cc_png"].startswith(str(results_root))
    assert results["traj_png"].startswith(str(results_root))
    assert results["stats_txt"].startswith(str(results_root))

    for path_str in results.values():
        assert path_str is not None
        path = Path(path_str)
        assert path.exists()

    log_path = meta_root / "logs" / "run_20250101_010101.log"
    assert log_path.exists()

    stats_content = Path(results["stats_txt"]).read_text(encoding="utf-8")
    assert "max:" in stats_content
    assert "mean:" in stats_content
