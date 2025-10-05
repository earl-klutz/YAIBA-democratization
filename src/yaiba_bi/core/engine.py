"""人数推移・軌跡・統計を統合生成する可視化エンジン。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .concurrency import ConcurrencyPlotter
from .trajectory import TrajectoryPlotter
from .stats_basic import compute_cc_stats, save_stats_txt
from .validation import validate_df_cc, validate_df_pos
from .naming import NamingHelper, NamingParameters
from .logging_util import get_logger, log_summary
from .spec_errors import SpecError


@dataclass
class EngineConfig:
    """可視化エンジンの挙動を制御する設定値群。"""

    render: Dict[str, object]
    trajectory_style: Dict[str, object]
    io: Dict[str, object]


class VisualizationEngine:
    """人数推移・軌跡・統計出力を統合運用する実行エントリーポイント。"""

    def __init__(self, config: EngineConfig):
        """設定を受け取り命名・ロガーを初期化する。

        Args:
            config: 各種設定を束ねた `EngineConfig`。
        """
        self.config = config
        dt = config.io.get("dt") or NamingHelper.now_jst()
        naming_params = NamingParameters(
            event_day=str(config.io["event_day"]),
            filename=str(config.io["filename"]),
            dt=str(dt),
            ver=str(config.io.get("ver", "b1.0")),
        )
        self.naming = NamingHelper(naming_params)
        self.logger = get_logger(naming_params.dt)

    def run(
        self,
        df_cc: pd.DataFrame,
        df_pos: pd.DataFrame,
        area: Dict[str, float],
    ) -> Dict[str, Optional[str]]:
        """人数推移・軌跡・統計TXTを生成し保存パスを返す。

        Args:
            df_cc: 同時接続数DataFrame。
            df_pos: 位置ログDataFrame。
            area: 描画エリア境界辞書。

        Returns:
            成果物3種の保存パスを持つ辞書。

        Raises:
            SpecError: 入力検証で致命的エラーが発生した場合。
        """
        results: Dict[str, Optional[str]] = {
            "cc_png": None,
            "traj_png": None,
            "stats_txt": None,
        }
        try:
            validate_df_cc(df_cc)
            validate_df_pos(df_pos)
        except SpecError as exc:
            self.logger.error("%s input validation failed: %s", exc.code, exc.message)
            raise

        stats: Optional[Dict[str, float]] = None
        try:
            stats = compute_cc_stats(df_cc)
            self.logger.info("computed stats: %s", stats)
        except SpecError as exc:
            self.logger.warning("%s stats failed: %s", exc.code, exc.message)

        cc_plotter = ConcurrencyPlotter(self.config.render, self.config.io)
        try:
            fig = cc_plotter.plot(df_cc)
            path = self.naming.build_output_path("cc_line", "png")
            results["cc_png"] = cc_plotter.save_png(fig, path)
        except SpecError as exc:
            self.logger.error("%s cc render failed: %s", exc.code, exc.message)

        traj_plotter = TrajectoryPlotter(area, self.config.trajectory_style, self.config.io)
        try:
            fig = traj_plotter.plot(df_pos)
            path = self.naming.build_output_path("traj2D", "png")
            results["traj_png"] = traj_plotter.save_png(fig, path)
        except SpecError as exc:
            self.logger.error("%s trajectory render failed: %s", exc.code, exc.message)

        if stats is not None:
            try:
                path = self.naming.build_output_path("stats", "txt")
                results["stats_txt"] = save_stats_txt(stats, path)
            except SpecError as exc:
                self.logger.error("%s save stats failed: %s", exc.code, exc.message)

        log_summary(self.logger, {k: v for k, v in results.items() if v})
        return results
