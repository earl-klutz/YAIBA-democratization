import os
import logging
from dataclasses import dataclass
from datetime import timedelta, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.ndimage import gaussian_filter

from .yaiba_loader import Area


JST = timezone(timedelta(hours=9))

fonts = fm.findSystemFonts()
font_list = [font for font in fonts if "NotoSansCJK-Regular.ttc" in font]

if not font_list:
    raise FileNotFoundError("NotoSansCJK-Regular.ttc not found")

font_property = fm.FontProperties(fname=font_list[0])

plt.rcParams["font.family"] = font_property.get_name()


@dataclass
class Theme:
    cmap: str = "viridis"
    image_size_px: Tuple[int, int] = (1280, 720)
    dpi: int = 144

    @property
    def size(self) -> Tuple[float, float]:
        return self.image_size_px[0] / self.dpi, self.image_size_px[1] / self.dpi


class HeatmapGenerator:
    gaussian_sigma_ratio: float = 0.05
    percentile_clip: Tuple[int, int] = (1, 99)
    min_unique_seconds: int = 10
    normalize_method: str = "minmax"

    def __init__(
        self,
        boundary: Area,
        resolution: int = 64,
        overwrite: bool = False,
        theme: Optional[Theme] = None
    ) -> None:
        self.boundary = boundary

        # resolution: grid_resolution 5-100、範囲外は自動補正
        if resolution < 5 or resolution > 100:
            resolution = 64
            self._resolution_adjusted = True
        else:
            self._resolution_adjusted = False
        self.grid_resolution = resolution

        # テーマ（内部固定に準拠）
        self.theme = Theme() if theme is None else theme

        # overwrite 設定
        self.overwrite = overwrite

        # ロガー
        self.logger = get_logger("heatmap")

    def clip_outliers(
        self,
        df: pd.DataFrame,
        lower_percentile: int = 1,
        upper_percentile: int = 99
    ) -> pd.DataFrame:
        # データ点が少ない・指定不正は安全スキップ
        if not (0 <= lower_percentile < upper_percentile <= 100):
            self.logger.warning("[-2301] percentile_clip が不正のためスキップ")
            return df

        if len(df) < max(30, self.min_unique_seconds):
            self.logger.warning("[-2403] サンプル数不足のためアウトライヤクリップをスキップ")
            return df

        dfv = df[["location_x", "location_z"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(dfv) < max(30, self.min_unique_seconds):
            self.logger.warning("[-2403] 有効サンプル不足のためアウトライヤクリップをスキップ")
            return df

        pl, ph = lower_percentile / 100.0, upper_percentile / 100.0
        x_low, x_high = dfv["location_x"].quantile([pl, ph]).values
        z_low, z_high = dfv["location_z"].quantile([pl, ph]).values

        if not (np.isfinite([x_low, x_high, z_low, z_high]).all() and x_low < x_high and z_low < z_high):
            self.logger.warning("[-2403] パーセンタイル範囲が成立しないためスキップ")
            return df

        mask = (
            df["location_x"].between(x_low, x_high, inclusive="both") &
            df["location_z"].between(z_low, z_high, inclusive="both")
        )

        return df.loc[mask].copy()

    @staticmethod
    def calculate_grid(
        df: pd.DataFrame,
        resolution: int,
        bounds: Area
    ) -> np.ndarray:
        x_edges = np.linspace(bounds.x_min, bounds.x_max, resolution + 1)
        z_edges = np.linspace(bounds.z_min, bounds.z_max, resolution + 1)
        h, xe, ze = np.histogram2d(
            df["location_z"].to_numpy(),  # y軸相当
            df["location_x"].to_numpy(),  # x軸相当
            bins=[z_edges, x_edges]
        )
        # 返却は [Z, X] 形状（imshow の行=Z, 列=X に合わせる）
        return h.astype(float)

    def apply_gaussian_smoothing(
        self,
        grid_data: np.ndarray,
        sigma_bins: Optional[int] = None
    ) -> np.ndarray:
        if sigma_bins is None:
            sigma_bins = max(1, round(self.grid_resolution * self.gaussian_sigma_ratio))
        if sigma_bins <= 0:
            return grid_data
        return gaussian_filter(grid_data, sigma=sigma_bins)

    @staticmethod
    def normalize_data(
        grid_data: np.ndarray,
        method: str = "minmax"
    ) -> np.ndarray:
        if method != "minmax":
            return grid_data
        g_min = np.nanmin(grid_data)
        g_max = np.nanmax(grid_data)
        if not np.isfinite([g_min, g_max]).all() or g_max <= g_min:
            return np.zeros_like(grid_data)
        return (grid_data - g_min) / (g_max - g_min)

    @staticmethod
    def compute_metric(
        grid_counts: np.ndarray,
        metric: str = "density"
    ) -> np.ndarray:
        if metric == "count":
            return grid_counts
        # density: 正規化は後段で行うためそのまま返す（スムージング前の値）
        return grid_counts

    @staticmethod
    def generate_heatmap(
        grid_data: np.ndarray,
        bounds: Area,
        theme: Theme,
        metric: str
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=theme.size, dpi=theme.dpi)
        ax.set_aspect("equal")

        # カラーバーの v_min/v_max はスムージング後配列の 1–99% で決定
        flat = grid_data.ravel()
        if np.all(flat == 0):
            v_min, v_max = 0.0, 1.0
        else:
            v_min, v_max = np.percentile(flat, [1, 99])
            if v_max <= v_min:
                v_min, v_max = float(np.min(flat)), float(np.max(flat))
                if v_max == v_min:
                    v_max = v_min + 1.0

        im = ax.imshow(
            grid_data,
            cmap=theme.cmap,
            origin="lower",
            extent=(bounds.x_min, bounds.x_max, bounds.z_min, bounds.z_max),
            vmin=v_min,
            vmax=v_max,
            interpolation="nearest",
            aspect="equal",
        )

        # 軸ラベル・タイトル
        ax.set_xlabel("X座標")
        ax.set_ylabel("Z座標")
        ax.set_title("イベントの混雑エリア")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("密度(0-1)" if metric == "density" else "観測数(人・秒相当)")

        fig.tight_layout()
        return fig

    def save_png(
        self,
        fig: plt.Figure,
        path: str
    ) -> None:
        if os.path.exists(path) and not self.overwrite:
            raise FileExistsError(f"[-2701] 既存ファイルあり（overwrite=False）: {path}")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    # 実行
    def run(
        self,
        df: pd.DataFrame,
        output_basename: str,
        save_dir: str = "",
        metric: str = "density"
    ) -> None:
        try:
            # 入力検証
            df1 = clip_by_boundary(df, self.boundary)

            # 外れ値クリップ（bounds には影響しない）
            lp, up = self.percentile_clip
            df2 = self.clip_outliers(df1, lp, up)

            # グリッド集計
            h = self.calculate_grid(df2, self.grid_resolution, self.boundary)

            # メトリクス適用
            m = self.compute_metric(h, metric=metric)

            # スムージング
            sigma_bins = max(1, round(self.grid_resolution * self.gaussian_sigma_ratio))
            ms = self.apply_gaussian_smoothing(m, sigma_bins=sigma_bins)

            # 正規化（density のとき）
            if metric == "density":
                mn = self.normalize_data(ms, method=self.normalize_method)
            else:
                mn = ms

            # 画像生成
            fig = self.generate_heatmap(mn, self.boundary, self.theme, metric)

            # パス命名・保存
            pd_time = df["second"].min().to_pydatetime()
            now = pd_time.strftime("%Y%m%d_%H%M%S")
            ver = "v1.0"
            basename = f"heatmap_2D-{output_basename}-{now}_{ver}.png"
            self.save_png(fig, os.path.join(save_dir, basename))

            if self._resolution_adjusted:
                self.logger.warning("[-2301] パラメータ範囲外: grid_resolution を 64 に自動補正")
        except Exception as e:
            self.logger.error(f"[ERROR] 実行失敗: {e}")
            raise


def clip_by_boundary(
    df: pd.DataFrame,
    boundary: Area
) -> pd.DataFrame:
    mask = (
            (df["location_x"] >= boundary.x_min) & (df["location_x"] <= boundary.x_max) &
            (df["location_z"] >= boundary.z_min) & (df["location_z"] <= boundary.z_max)
    )
    return df.loc[mask].copy()


def get_logger(
    run_id: str
) -> logging.Logger:
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger
