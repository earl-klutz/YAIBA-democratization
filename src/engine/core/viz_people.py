from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal, Dict, Any

import os
import math
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config dataclasses
# =========================


@dataclass(frozen=True)
class RenderConfig:
    event_day: str
    filename: str
    dpi: int = 144
    width_px: int = 1280
    height_px: int = 720


@dataclass(frozen=True)
class TrajectoryConfig:
    color_scheme: Literal["by_user", "by_speed", "by_time"] = "by_user"
    break_gap_factor: float = 3.0
    start_marker_size_px: int = 6
    end_marker_size_px: int = 10
    filter_user_ids: Optional[List[str]] = None
    bounds: Optional[dict] = None
    fit_mode: Literal["fit", "fill", "stretch"] = "fit"
    margin_px: int = 0
    clip_oob: bool = True


# =========================
# Validation
# =========================


class SpecError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _ensure_out_dir() -> str:
    out_dir = os.path.join("out")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def validate_df_cc(df_cc: pd.DataFrame) -> None:
    # 必須列 t, cc
    if df_cc is None or not isinstance(df_cc, pd.DataFrame):
        raise SpecError(-2101, "df_cc is missing or not a DataFrame")
    required = {"t", "cc"}
    if not required.issubset(df_cc.columns):
        raise SpecError(-2102, f"df_cc missing required columns: {required - set(df_cc.columns)}")
    # 型/NaN/非負
    if df_cc["cc"].isna().all():
        raise SpecError(-2204, "concurrency series is empty")
    if (df_cc["cc"] < 0).any():
        raise SpecError(-2204, "cc must be non-negative")
    # 時間昇順
    if not df_cc["t"].is_monotonic_increasing:
        df_cc.sort_values("t", inplace=True, kind="mergesort")


def validate_df_pos(df_pos: pd.DataFrame) -> None:
    if df_pos is None or not isinstance(df_pos, pd.DataFrame):
        raise SpecError(-2101, "df_pos is missing or not a DataFrame")
    required = {"t", "user_id", "x", "z"}
    if not required.issubset(df_pos.columns):
        raise SpecError(-2102, f"df_pos missing required columns: {required - set(df_pos.columns)}")
    if df_pos[["x", "z"]].isna().all(axis=None):
        raise SpecError(-2301, "trajectory positions are entirely null")
    if not df_pos["t"].is_monotonic_increasing:
        df_pos.sort_values(["user_id", "t"], inplace=True, kind="mergesort")


# =========================
# Stats
# =========================


def compute_cc_stats(df_cc: pd.DataFrame) -> Dict[str, float]:
    try:
        series = pd.to_numeric(df_cc["cc"], errors="coerce").dropna()
        if series.empty:
            raise SpecError(-2403, "no samples for stats")
        stats = {
            "max": int(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "p95": float(np.percentile(series.to_numpy(), 95)),
        }
        return stats
    except SpecError:
        raise
    except Exception as exc:
        raise SpecError(-2401, f"failed to compute stats: {exc}")


# =========================
# Rendering
# =========================


def _px_to_inches(px: int, dpi: int) -> float:
    return px / dpi


def render_concurrency_png(df_cc: pd.DataFrame, rcfg: RenderConfig) -> str:
    validate_df_cc(df_cc)
    out_dir = _ensure_out_dir()
    width_in = _px_to_inches(rcfg.width_px, rcfg.dpi)
    height_in = _px_to_inches(rcfg.height_px, rcfg.dpi)

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=rcfg.dpi)
    try:
        # 折れ線（欠測長大区間は線分断は簡易実装：NaNに分断）
        t = pd.to_datetime(df_cc["t"], errors="coerce")
        y = pd.to_numeric(df_cc["cc"], errors="coerce")
        ax.plot(t, y, linewidth=1.5, color="#1f77b4")
        ax.set_title("Concurrent users")
        ax.set_xlabel("time")
        ax.set_ylabel("users [count]")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = os.path.join(out_dir, f"cc_line_{rcfg.event_day}_{rcfg.filename}.png")
        fig.savefig(path, dpi=rcfg.dpi)
        return path
    except Exception as exc:
        raise SpecError(-2202, f"failed to render cc line: {exc}")
    finally:
        plt.close(fig)


def _fit_bounds(ax, bounds: dict, rcfg: RenderConfig, tcfg: TrajectoryConfig) -> None:
    min_x = bounds.get("min_x", -10)
    max_x = bounds.get("max_x", 10)
    min_z = bounds.get("min_z", -10)
    max_z = bounds.get("max_z", 10)
    width = max_x - min_x
    height = max_z - min_z
    if width <= 0 or height <= 0:
        raise SpecError(-2301, "invalid area bounds")
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_z, max_z)
    ax.set_aspect("equal")


def render_trajectory2d_png(df_pos: pd.DataFrame, rcfg: RenderConfig, tcfg: TrajectoryConfig) -> str:
    validate_df_pos(df_pos)
    out_dir = _ensure_out_dir()
    width_in = _px_to_inches(rcfg.width_px, rcfg.dpi)
    height_in = _px_to_inches(rcfg.height_px, rcfg.dpi)

    # bounds: optional in config; when None, infer from data
    if tcfg.bounds is None:
        bounds = {
            "min_x": float(pd.to_numeric(df_pos["x"], errors="coerce").min()),
            "max_x": float(pd.to_numeric(df_pos["x"], errors="coerce").max()),
            "min_z": float(pd.to_numeric(df_pos["z"], errors="coerce").min()),
            "max_z": float(pd.to_numeric(df_pos["z"], errors="coerce").max()),
        }
    else:
        bounds = dict(tcfg.bounds)

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=rcfg.dpi)
    try:
        _fit_bounds(ax, bounds, rcfg, tcfg)
        df = df_pos.copy()
        if tcfg.filter_user_ids:
            df = df[df["user_id"].isin(set(tcfg.filter_user_ids))]
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
        df.sort_values(["user_id", "t"], inplace=True, kind="mergesort")

        # 線分断: 大きな時間ギャップ
        df["dt"] = df.groupby("user_id")["t"].diff().dt.total_seconds()
        gap_threshold = tcfg.break_gap_factor
        # NaN は新系列の始まり
        df["break"] = (df["dt"].isna()) | (df["dt"] > gap_threshold)

        # ユーザー毎に描画
        for user_id, g in df.groupby("user_id"):
            # break で分割
            segment_indices = g.index[g["break"]].tolist()
            # 先頭を必ず追加
            if len(segment_indices) == 0 or segment_indices[0] != g.index.min():
                segment_indices = [g.index.min()] + segment_indices
            segment_indices.append(g.index.max() + 1)
            for i in range(len(segment_indices) - 1):
                start_idx = segment_indices[i]
                end_idx = segment_indices[i + 1]
                seg = g.loc[(g.index >= start_idx) & (g.index < end_idx)]
                if len(seg) < 2:
                    continue
                ax.plot(seg["x"], seg["z"], linewidth=1.5, alpha=0.8)
            # 始点/終点
            first = g.iloc[0]
            last = g.iloc[-1]
            ax.scatter(
                [first["x"]],
                [first["z"]],
                s=tcfg.start_marker_size_px**2,
                c="#2ca02c",
                marker="o",
                zorder=3,
                clip_on=tcfg.clip_oob,
            )
            ax.scatter(
                [last["x"]],
                [last["z"]],
                s=tcfg.end_marker_size_px**2,
                c="#d62728",
                marker="^",
                zorder=3,
                clip_on=tcfg.clip_oob,
            )

        ax.set_title("Trajectories (x-z)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = os.path.join(out_dir, f"traj_{rcfg.event_day}_{rcfg.filename}.png")
        fig.savefig(path, dpi=rcfg.dpi)
        return path
    except SpecError:
        raise
    except Exception as exc:
        raise SpecError(-2303, f"failed to render trajectories: {exc}")
    finally:
        plt.close(fig)


# =========================
# Export
# =========================


def save_stats_txt(stats: Dict[str, float], rcfg: RenderConfig) -> str:
    try:
        out_dir = _ensure_out_dir()
        path = os.path.join(out_dir, f"stats_{rcfg.event_day}_{rcfg.filename}.txt")
        lines = [
            f"max: {int(round(stats['max']))}\n",
            f"mean: {stats['mean']:.1f}\n",
            f"median: {stats['median']:.0f}\n",
            f"p95: {stats['p95']:.0f}\n",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return path
    except Exception as exc:
        raise SpecError(-2704, f"failed to save stats: {exc}")


# =========================
# Entry point
# =========================


def run(
    df_cc: pd.DataFrame,
    df_pos: pd.DataFrame,
    area: Dict[str, float],
    rcfg: RenderConfig,
    tcfg: TrajectoryConfig,
) -> Dict[str, Optional[str]]:
    _ensure_out_dir()
    logger = logging.getLogger("yaiba.bi.viz_people")
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join("out", "run.log"), encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

    results: Dict[str, Optional[str]] = {"cc_png": None, "traj_png": None, "stats_txt": None}

    # 入力検証（致命的。失敗で中断）
    try:
        validate_df_cc(df_cc)
        validate_df_pos(df_pos)
    except SpecError as e:
        logger.error(f"{e.code} input validation failed: {e.message}")
        raise

    # 統計
    stats: Optional[Dict[str, float]] = None
    try:
        stats = compute_cc_stats(df_cc)
    except SpecError as e:
        logger.error(f"{e.code} stats failed: {e.message}")

    # 同時接続数PNG
    try:
        results["cc_png"] = render_concurrency_png(df_cc, rcfg)
    except SpecError as e:
        logger.error(f"{e.code} cc render failed: {e.message}")

    # 軌跡PNG（areaをboundsに流用）
    try:
        tcfg_dict = dict(tcfg.__dict__)
        if tcfg_dict.get("bounds") is None:
            tcfg_dict["bounds"] = area
        tcfg2 = TrajectoryConfig(**tcfg_dict)
        results["traj_png"] = render_trajectory2d_png(df_pos, rcfg, tcfg2)
    except SpecError as e:
        logger.error(f"{e.code} trajectory render failed: {e.message}")

    # 統計txt
    if stats is not None:
        try:
            results["stats_txt"] = save_stats_txt(stats, rcfg)
        except SpecError as e:
            logger.error(f"{e.code} save stats failed: {e.message}")

    return results


