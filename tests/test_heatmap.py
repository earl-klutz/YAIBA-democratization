import pytest
import numpy as np
import pandas as pd

from src.yaiba_bi.core.heatmap import HeatmapGenerator, Theme, clip_by_boundary
from src.yaiba_bi.core.yaiba_loader import Area


@pytest.fixture
def area() -> Area:
    # A simple square boundary covering [0, 10] x [0, 10]
    return Area(x_min=0, x_max=10, y_min=0, y_max=5, z_min=0, z_max=10)


@pytest.fixture
def df_inside_outside() -> pd.DataFrame:
    # Build a mix of inside and outside points
    now = pd.to_datetime("2024-01-01 00:00:00")
    xs = [1, 5, 9, -1, 11]  # last two are outside in X
    zs = [1, 5, 9, 1, 5]    # chosen so some still outside
    seconds = [now + pd.Timedelta(seconds=i) for i in range(len(xs))]
    return pd.DataFrame({
        "second": seconds,
        "location_x": xs,
        "location_z": zs,
    })


@pytest.fixture
def dense_df(area: Area) -> pd.DataFrame:
    # Create dense enough (>= 30 rows) data within the area
    # to let clip_outliers run its percentile logic when called.
    start = pd.to_datetime("2024-01-01 12:00:00")
    n = 60
    rng = np.random.default_rng(42)
    xs = rng.uniform(area.x_min + 0.5, area.x_max - 0.5, size=n)
    zs = rng.uniform(area.z_min + 0.5, area.z_max - 0.5, size=n)
    seconds = [start + pd.Timedelta(seconds=i) for i in range(n)]
    return pd.DataFrame({
        "second": seconds,
        "location_x": xs,
        "location_z": zs,
    })


def test_clip_by_boundary_filters(df_inside_outside: pd.DataFrame, area: Area):
    clipped = clip_by_boundary(df_inside_outside, area)
    # Points within [0,10] for both x and z are kept
    assert (clipped["location_x"].between(area.x_min, area.x_max).all())
    assert (clipped["location_z"].between(area.z_min, area.z_max).all())
    # At least one row should have been removed (outside points)
    assert len(clipped) < len(df_inside_outside)


def test_calculate_grid_counts_and_shape(dense_df: pd.DataFrame, area: Area):
    gen = HeatmapGenerator(boundary=area, resolution=8)
    h = gen.calculate_grid(dense_df, gen.grid_resolution, area)
    # shape should be (resolution, resolution)
    assert h.shape == (8, 8)
    # Total counts should equal number of input rows
    assert int(h.sum()) == len(dense_df)


def test_apply_gaussian_smoothing_shape(dense_df: pd.DataFrame, area: Area):
    gen = HeatmapGenerator(boundary=area, resolution=8)
    grid = gen.calculate_grid(dense_df, gen.grid_resolution, area)
    smoothed = gen.apply_gaussian_smoothing(grid, sigma_bins=1)
    assert smoothed.shape == grid.shape
    # Smoothed values should be non-negative
    assert np.all(smoothed >= 0)


def test_normalize_data_minmax_and_constant():
    # Normal case
    data = np.array([[0.0, 1.0], [2.0, 3.0]])
    norm = HeatmapGenerator.normalize_data(data, method="minmax")
    assert np.isclose(norm.min(), 0.0)
    assert np.isclose(norm.max(), 1.0)

    # Constant case -> zeros
    data2 = np.full((2, 2), 5.0)
    norm2 = HeatmapGenerator.normalize_data(data2, method="minmax")
    assert np.all(norm2 == 0.0)

    # Unknown method -> unchanged
    data3 = np.array([[1.0, 2.0]])
    same = HeatmapGenerator.normalize_data(data3, method="unknown")
    assert np.array_equal(same, data3)


def test_generate_heatmap_returns_figure(dense_df: pd.DataFrame, area: Area):
    gen = HeatmapGenerator(boundary=area, resolution=8)
    grid = gen.calculate_grid(dense_df, gen.grid_resolution, area)
    smoothed = gen.apply_gaussian_smoothing(grid, sigma_bins=1)
    normalized = gen.normalize_data(smoothed, method="minmax")
    fig = gen.generate_heatmap(normalized, area, Theme(cmap="viridis", image_size_px=(320, 240), dpi=80), metric="density")
    try:
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_save_png_overwrite_behavior(tmp_path, area: Area):
    gen = HeatmapGenerator(boundary=area, resolution=8, overwrite=False)
    # Create a tiny empty figure
    theme = Theme(image_size_px=(160, 120), dpi=80)
    fig = gen.generate_heatmap(np.zeros((8, 8)), area, theme, metric="density")
    path = tmp_path / "test.png"

    # First save should succeed
    gen.save_png(fig, str(path))

    # Second save without overwrite should raise
    theme2 = Theme(image_size_px=(160, 120), dpi=80)
    fig2 = gen.generate_heatmap(np.zeros((8, 8)), area, theme2, metric="density")
    with pytest.raises(FileExistsError):
        gen.save_png(fig2, str(path))

    # With overwrite=True it should succeed
    gen_over = HeatmapGenerator(boundary=area, resolution=8, overwrite=True)
    fig3 = gen_over.generate_heatmap(np.zeros((8, 8)), area, theme2, metric="density")
    gen_over.save_png(fig3, str(path))
