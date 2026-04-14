"""表面粗さ・光学指標の計算テスト。"""

import numpy as np
import pytest

from bsdf_sim.models.base import HeightMap
from bsdf_sim.metrics.surface import compute_rq, compute_ra, compute_rz, compute_sdq, compute_all_surface_metrics
from bsdf_sim.metrics.optical import compute_log_rmse, compute_haze, compute_gloss, compute_doi


@pytest.fixture
def flat_height_map():
    """平坦な高さマップ（Rq=0）。"""
    return HeightMap(data=np.zeros((64, 64), dtype=np.float32), pixel_size_um=0.25)


@pytest.fixture
def sine_height_map():
    """正弦波高さマップ（Rq = amplitude / sqrt(2)）。"""
    x = np.linspace(0, 4 * np.pi, 64)
    amplitude = 0.01  # μm
    h = np.tile((amplitude * np.sin(x))[:, np.newaxis], (1, 64)).astype(np.float32)
    return HeightMap(data=h, pixel_size_um=0.25)


@pytest.fixture
def bsdf_grid():
    """テスト用 BSDF グリッド（法線方向に集中したピーク）。

    N=65（奇数）を使うことでグリッドに u=v=0 の点が含まれる。
    狭幅ガウシアン（σ²=0.0001）で DOI の直進光領域（0.1°）内に電力が集中する。
    """
    N = 65
    u_axis = np.linspace(-1.0, 1.0, N)
    v_axis = np.linspace(-1.0, 1.0, N)
    u, v = np.meshgrid(u_axis, v_axis, indexing="ij")
    uv_r2 = u**2 + v**2
    bsdf = np.exp(-uv_r2 / 0.0001) * 1e4
    bsdf[uv_r2 > 1.0] = 0.0
    return u, v, bsdf.astype(np.float32)


class TestSurfaceMetrics:
    def test_rq_flat(self, flat_height_map):
        assert compute_rq(flat_height_map) == pytest.approx(0.0, abs=1e-10)

    def test_rq_sine(self, sine_height_map):
        """正弦波の Rq = amplitude / sqrt(2)。"""
        amplitude = 0.01
        expected = amplitude / np.sqrt(2)
        assert compute_rq(sine_height_map) == pytest.approx(expected, rel=0.05)

    def test_ra_positive(self, sine_height_map):
        assert compute_ra(sine_height_map) >= 0.0

    def test_rz_flat(self, flat_height_map):
        assert compute_rz(flat_height_map) == pytest.approx(0.0, abs=1e-10)

    def test_rz_positive(self, sine_height_map):
        assert compute_rz(sine_height_map) > 0.0

    def test_sdq_flat(self, flat_height_map):
        assert compute_sdq(flat_height_map) == pytest.approx(0.0, abs=1e-6)

    def test_sdq_sine(self, sine_height_map):
        assert compute_sdq(sine_height_map) > 0.0

    def test_all_metrics_keys(self, sine_height_map):
        metrics = compute_all_surface_metrics(sine_height_map)
        assert set(metrics.keys()) == {"rq_um", "ra_um", "rz_um", "sdq_rad"}


class TestLogRMSE:
    def test_perfect_match(self):
        bsdf = np.array([0.1, 0.01, 0.001])
        rmse = compute_log_rmse(bsdf, bsdf, bsdf_floor=1e-6)
        assert rmse == pytest.approx(0.0, abs=1e-8)

    def test_floor_masking(self):
        """フロア以下の実測値は誤差計算に含まれない。"""
        simulated = np.array([0.1, 0.01, 0.001, 1e-8])
        measured  = np.array([0.1, 0.01, 0.001, 1e-8])
        rmse = compute_log_rmse(simulated, measured, bsdf_floor=1e-6)
        assert rmse == pytest.approx(0.0, abs=1e-8)

    def test_order_of_magnitude_error(self):
        """1桁（10倍）の差は Log-RMSE = 1.0 になる。"""
        simulated = np.array([0.1])
        measured  = np.array([1.0])
        rmse = compute_log_rmse(simulated, measured, bsdf_floor=1e-6)
        assert rmse == pytest.approx(1.0, rel=1e-4)

    def test_all_below_floor(self):
        """全点がフロア以下の場合は inf を返す。"""
        simulated = np.array([1e-8, 1e-9])
        measured  = np.array([1e-8, 1e-9])
        rmse = compute_log_rmse(simulated, measured, bsdf_floor=1e-6)
        assert rmse == float("inf")


class TestOpticalMetrics:
    def test_haze_range(self, bsdf_grid):
        u, v, bsdf = bsdf_grid
        haze = compute_haze(u, v, bsdf, half_angle_deg=2.5)
        assert 0.0 <= haze <= 1.0

    def test_haze_narrow_beam(self, bsdf_grid):
        """法線方向に集中したビームはヘイズが低い。"""
        u, v, bsdf = bsdf_grid
        haze = compute_haze(u, v, bsdf, half_angle_deg=2.5)
        assert haze < 0.5

    def test_gloss_positive(self, bsdf_grid):
        u, v, bsdf = bsdf_grid
        gloss = compute_gloss(u, v, bsdf, gloss_angle_deg=0.0, acceptance_deg=5.0)
        assert gloss >= 0.0

    def test_doi_range(self, bsdf_grid):
        u, v, bsdf = bsdf_grid
        doi = compute_doi(u, v, bsdf)
        assert 0.0 <= doi <= 1.0

    def test_doi_narrow_beam(self, bsdf_grid):
        """法線方向に集中したビームは DOI が高い。"""
        u, v, bsdf = bsdf_grid
        doi = compute_doi(u, v, bsdf)
        assert doi > 0.5
