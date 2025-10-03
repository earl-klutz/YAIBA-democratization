import logging
import os

def get_logger(run_id: str, log_path: str) -> logging.Logger:
    logger = logging.getLogger(f"yaiba_bi.{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

def log_summary(logger: logging.Logger, stats: dict) -> None:
    logger.info("summary %s", {k: v for k, v in stats.items()})
