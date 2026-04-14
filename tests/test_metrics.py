"""表面粗さ・光学指標の計算テスト。"""

import numpy as np
import pytest

from bsdf_sim.models.base import HeightMap
from bsdf_sim.metrics.surface import (
    compute_rq, compute_ra, compute_rz, compute_sdq,
    compute_sq, compute_sa, compute_sp, compute_sv, compute_sz,
    compute_ssk, compute_sku, compute_sdr, compute_sal, compute_str,
    compute_rp, compute_rv, compute_rsk, compute_rku, compute_rsm, compute_rc,
    compute_all_surface_metrics,
)
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
        """正弦波（列方向変動）のプロファイル Rq。

        sine_height_map は axis=0（列方向）に正弦変化し、行内は定数。
        compute_rq は行・列両プロファイルを平均するため：
          - 行プロファイル（定数）: Rq_row = 0
          - 列プロファイル（正弦）: Rq_col = amplitude / sqrt(2)
          - 平均 Rq = amplitude / (2*sqrt(2))
        """
        amplitude = 0.01
        expected = amplitude / (2 * np.sqrt(2))
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
        expected_keys = {
            # ISO 25178-2 S-パラメータ
            "sq_um", "sa_um", "sp_um", "sv_um", "sz_um",
            "ssk", "sku", "sdq_rad", "sdr_pct", "sal_um", "str",
            # JIS B 0601 R-パラメータ
            "rq_um", "ra_um", "rz_um", "rp_um", "rv_um",
            "rsk", "rku", "rsm_um", "rc_um",
        }
        assert set(metrics.keys()) == expected_keys


class TestISO25178Metrics:
    """ISO 25178-2 面パラメータのテスト。"""

    def test_sq_flat(self, flat_height_map):
        assert compute_sq(flat_height_map) == pytest.approx(0.0, abs=1e-10)

    def test_sq_sine(self, sine_height_map):
        """正弦波の Sq = amplitude / sqrt(2)。"""
        amplitude = 0.01
        assert compute_sq(sine_height_map) == pytest.approx(amplitude / np.sqrt(2), rel=0.05)

    def test_sa_flat(self, flat_height_map):
        assert compute_sa(flat_height_map) == pytest.approx(0.0, abs=1e-10)

    def test_sa_positive(self, sine_height_map):
        assert compute_sa(sine_height_map) > 0.0

    def test_sp_positive(self, sine_height_map):
        assert compute_sp(sine_height_map) > 0.0

    def test_sv_positive(self, sine_height_map):
        assert compute_sv(sine_height_map) > 0.0

    def test_sz_equals_sp_plus_sv(self, sine_height_map):
        """Sz = Sp + Sv が成立する。"""
        sp = compute_sp(sine_height_map)
        sv = compute_sv(sine_height_map)
        sz = compute_sz(sine_height_map)
        assert sz == pytest.approx(sp + sv, rel=1e-5)

    def test_ssk_sine_near_zero(self, sine_height_map):
        """正弦波は対称分布なので Ssk ≈ 0。"""
        ssk = compute_ssk(sine_height_map)
        assert abs(ssk) < 0.1

    def test_ssk_flat(self, flat_height_map):
        assert compute_ssk(flat_height_map) == pytest.approx(0.0, abs=1e-8)

    def test_sku_sine(self, sine_height_map):
        """正弦波の Sku = 1.5（理論値）。"""
        sku = compute_sku(sine_height_map)
        assert sku == pytest.approx(1.5, rel=0.05)

    def test_sdr_flat(self, flat_height_map):
        """平坦面の Sdr = 0%。"""
        assert compute_sdr(flat_height_map) == pytest.approx(0.0, abs=1e-6)

    def test_sdr_positive(self, sine_height_map):
        """起伏のある面の Sdr > 0%。"""
        assert compute_sdr(sine_height_map) > 0.0

    def test_sal_positive(self, sine_height_map):
        assert compute_sal(sine_height_map) > 0.0

    def test_str_range(self, sine_height_map):
        """Str は 0〜1 の範囲。"""
        assert 0.0 <= compute_str(sine_height_map) <= 1.0


class TestJISB0601Metrics:
    """JIS B 0601 / ISO 4287 プロファイルパラメータのテスト。"""

    def test_rp_positive(self, sine_height_map):
        assert compute_rp(sine_height_map) > 0.0

    def test_rv_positive(self, sine_height_map):
        assert compute_rv(sine_height_map) > 0.0

    def test_rp_rv_relation(self, sine_height_map):
        """Rp と Rv の和は Rz 以下（プロファイル平均の性質）。"""
        rp = compute_rp(sine_height_map)
        rv = compute_rv(sine_height_map)
        rz = compute_rz(sine_height_map)
        assert rp + rv <= rz * 1.05  # プロファイル平均と全体値の差を考慮

    def test_rsk_sine_near_zero(self, sine_height_map):
        """正弦波は対称分布なので Rsk ≈ 0。"""
        rsk = compute_rsk(sine_height_map)
        assert abs(rsk) < 0.1

    def test_rku_sine(self, sine_height_map):
        """正弦波の Rku ≈ 1.5（理論値）。"""
        rku = compute_rku(sine_height_map)
        assert rku == pytest.approx(1.5, rel=0.1)

    def test_rsm_positive(self, sine_height_map):
        assert compute_rsm(sine_height_map) > 0.0

    def test_rsm_flat(self, flat_height_map):
        """平坦面では Rsm は物理サイズ（要素なし）。"""
        rsm = compute_rsm(flat_height_map)
        assert rsm == pytest.approx(flat_height_map.physical_size_um, rel=0.01)

    def test_rc_positive(self, sine_height_map):
        assert compute_rc(sine_height_map) > 0.0

    def test_rc_flat(self, flat_height_map):
        assert compute_rc(flat_height_map) == pytest.approx(0.0, abs=1e-10)


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
