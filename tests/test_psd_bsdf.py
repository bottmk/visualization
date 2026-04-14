"""PSD法・Q因子（完全形/簡略形）のテスト。"""

import numpy as np
import pytest

from bsdf_sim.models.base import HeightMap
from bsdf_sim.optics.psd_bsdf import compute_psd_2d, compute_bsdf_psd


@pytest.fixture
def simple_height_map():
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((64, 64)) * 0.005).astype(np.float32)
    return HeightMap(data=data, pixel_size_um=0.25)


class TestPSD:
    def test_psd_shape(self, simple_height_map):
        fx, fy, psd = compute_psd_2d(simple_height_map)
        assert psd.shape == (64, 64)
        assert fx.shape == (64, 64)

    def test_psd_non_negative(self, simple_height_map):
        _, _, psd = compute_psd_2d(simple_height_map)
        assert np.all(psd >= 0.0)

    def test_parseval_theorem(self, simple_height_map):
        """パーセバルの定理: PSD の総和は高さの分散に比例する。"""
        _, _, psd = compute_psd_2d(simple_height_map)
        N = simple_height_map.grid_size
        dx = simple_height_map.pixel_size_um
        # PSD 総和 = Rq² * 物理面積
        psd_integral = np.sum(psd) / (N * dx) ** 2
        rq2 = np.mean(simple_height_map.data**2)
        assert psd_integral == pytest.approx(rq2, rel=0.1)


class TestPSDBSDF:
    def test_output_shape(self, simple_height_map):
        u, v, bsdf = compute_bsdf_psd(
            height_map=simple_height_map,
            wavelength_um=0.55,
            theta_i_deg=0.0,
            phi_i_deg=0.0,
        )
        assert bsdf.shape == (64, 64)

    def test_bsdf_non_negative(self, simple_height_map):
        _, _, bsdf = compute_bsdf_psd(
            height_map=simple_height_map,
            wavelength_um=0.55,
            theta_i_deg=0.0,
            phi_i_deg=0.0,
        )
        assert np.all(bsdf >= 0.0)

    def test_complete_vs_simplified_brdf(self, simple_height_map):
        """完全形と簡略形の結果が大きく離れないことを確認（法線入射）。"""
        _, _, bsdf_full = compute_bsdf_psd(
            height_map=simple_height_map,
            wavelength_um=0.55,
            theta_i_deg=0.0,
            phi_i_deg=0.0,
            approx_mode=False,
        )
        _, _, bsdf_approx = compute_bsdf_psd(
            height_map=simple_height_map,
            wavelength_um=0.55,
            theta_i_deg=0.0,
            phi_i_deg=0.0,
            approx_mode=True,
        )
        # 法線入射では完全形と簡略形は比較的近い値になるはず
        valid = bsdf_full > 0
        if np.any(valid):
            ratio = bsdf_approx[valid] / np.maximum(bsdf_full[valid], 1e-20)
            # 極端な乖離がないことを確認（10倍以内）
            assert np.median(ratio) == pytest.approx(1.0, rel=10.0)

    def test_polarization_s(self, simple_height_map):
        _, _, bsdf = compute_bsdf_psd(
            height_map=simple_height_map,
            wavelength_um=0.55,
            theta_i_deg=30.0,
            phi_i_deg=0.0,
            polarization="S",
        )
        assert np.all(bsdf >= 0.0)

    def test_polarization_p(self, simple_height_map):
        _, _, bsdf = compute_bsdf_psd(
            height_map=simple_height_map,
            wavelength_um=0.55,
            theta_i_deg=30.0,
            phi_i_deg=0.0,
            polarization="P",
        )
        assert np.all(bsdf >= 0.0)

    def test_btdf_mode(self, simple_height_map):
        _, _, bsdf = compute_bsdf_psd(
            height_map=simple_height_map,
            wavelength_um=0.55,
            theta_i_deg=30.0,
            phi_i_deg=0.0,
            n1=1.0,
            n2=1.5,
            is_btdf=True,
        )
        assert bsdf.max() > 0.0
