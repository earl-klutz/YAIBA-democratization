"""成果物ファイル名およびメタデータのパス生成を司るモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import pytz

from .spec_errors import SpecError


RESULT_ROOT = Path("/content/YAIBA_data/output/results")
META_ROOT = Path("/content/YAIBA_data/output/meta")


@dataclass
class NamingParameters:
    """命名に必要なコンテキストパラメータ。"""

    event_day: str
    filename: str
    dt: str
    ver: str


class NamingHelper:
    """設計書準拠の命名・フォルダ構成を提供するヘルパー。"""

    def __init__(self, params: NamingParameters):
        """パラメータを受け取りヘルパーを初期化する。

        Args:
            params: 命名に必要なイベント日・ファイル名など。
        """
        self.params = params

    def build_basename(self, prefix: str) -> str:
        """成果物のベース名を生成する。

        Args:
            prefix: 成果物種別を表す接頭辞。

        Returns:
            `<prefix>-{event_day}-{filename}-{dt}{ver}` 形式の文字列。
        """
        event_day = self.params.event_day
        filename = self.params.filename
        dt = self.params.dt
        ver = self.params.ver
        return f"{prefix}-{event_day}-{filename}-{dt}{ver}"

    def build_output_path(self, prefix: str, extension: str) -> str:
        """成果物配置先 `/results` に拡張子付きファイルパスを生成する。

        Args:
            prefix: 成果物種別を表す接頭辞。
            extension: 保存時の拡張子。

        Returns:
            絶対パスの保存先文字列。
        """
        ext = extension.lstrip(".")
        basename = self.build_basename(prefix)
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        return str(RESULT_ROOT / f"{basename}.{ext}")

    def meta_paths(self) -> Dict[str, str]:
        """設定保存・ログ出力のためのメタデータパスを返す。

        Returns:
            config/logファイルの絶対パスを含む辞書。
        """
        dt = self.params.dt
        ver = self.params.ver
        configs = META_ROOT / "configs"
        logs = META_ROOT / "logs"
        configs.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        return {
            "config_path": str(configs / f"run_{ver}_params.yaml"),
            "log_path": str(logs / f"run_{dt}.log"),
        }

    @staticmethod
    def now_jst() -> str:
        """JST現在時刻を `YYYYMMDD_HHMMSS` 形式で返す。

        Returns:
            日時文字列。
        """
        jst = pytz.timezone("Asia/Tokyo")
        return datetime.now(jst).strftime("%Y%m%d_%H%M%S")
