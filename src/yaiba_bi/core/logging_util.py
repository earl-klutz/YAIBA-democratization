"""成果物生成処理のためのロギング初期化とサマリ出力。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from .naming import META_ROOT


def get_logger(run_id: str) -> logging.Logger:
    """指定された実行IDに紐づくロガーを初期化して返す。

    Args:
        run_id: ログファイル命名に使用する実行ID。

    Returns:
        初期化済みの `logging.Logger` インスタンス。
    """
    logger = logging.getLogger(f"yaiba.bi.{run_id}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    log_dir = META_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{run_id}.log"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def log_summary(logger: logging.Logger, stats: Dict[str, object]) -> None:
    """処理サマリをシンプルな `key=value` 形式でログ出力する。

    Args:
        logger: 出力先ロガー。
        stats: ログ出力するサマリ情報。
    """
    summary = " | ".join(f"{k}={v}" for k, v in stats.items())
    logger.info("summary: %s", summary)
