# tests/conftest.py
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

# プロジェクトの src/ を import パスへ
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

TZ_JST = ZoneInfo("Asia/Tokyo")

@pytest.fixture(autouse=True)
def patch_output_dirs(monkeypatch, tmp_path):
    """
    /content 固定を避け、テスト毎に一時ディレクトリへ出力させる。
    """
    monkeypatch.setenv("YAIBA_RESULT_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("YAIBA_META_ROOT", str(tmp_path / "meta"))

@pytest.fixture
def make_df():
    """
    JSTで 'second' を生成するダミーDataFrameファクトリ。
    seconds>=180 を満たすデータを作る。
    """
    def _make(seconds: int = 200, n_users: int = 8) -> pd.DataFrame:
        now = datetime.now(TZ_JST).replace(microsecond=0)
        rows = []
        rng = np.random.default_rng(42)
        for s in range(seconds):
            t = now + timedelta(seconds=s)
            for uid in range(n_users):
                x = float(np.sin(0.01 * s + uid) * 10 + rng.normal(0, 0.1))
                z = float(np.cos(0.008 * s + uid) * 8 + rng.normal(0, 0.1))
                rows.append({
                    "second": t, "user_id": f"u{uid}",
                    "location_x": x, "location_y": 0.0, "location_z": z,
                    "event_day": now.date().isoformat(),
                })
        return pd.DataFrame(rows)
    return _make
