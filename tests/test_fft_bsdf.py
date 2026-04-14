"""FFT法・フレネル係数・BRDF/BTDF のテスト。"""

import numpy as np
import pytest

from bsdf_sim.models.base import HeightMap
from bsdf_sim.models.random_rough import RandomRoughSurface
from bsdf_sim.optics.fresnel import fresnel_rs, fresnel_rp, fresnel_ts, fresnel_tp, snell_angle
from bsdf_sim.optics.fft_bsdf import compute_bsdf_fft, sample_bsdf_at_angles


class TestFresnel:
    def test_normal_incidence_rs(self):
        """法線入射での r_s = (n1-n2)/(n1+n2)。"""
        rs = fresnel_rs(0.0, 1.0, 1.5)
        expected = (1.0 - 1.5) / (1.0 + 1.5)
        assert abs(rs.real - expected) < 1e-6

    def test_normal_incidence_rp(self):
        """法線入射での r_p = -(n2-n1)/(n2+n1)（符号反転あり）。"""
        rp = fresnel_rp(0.0, 1.0, 1.5)
        expected = -(1.5 - 1.0) / (1.5 + 1.0)
        assert abs(rp.real - expected) < 1e-6

    def test_energy_conservation(self):
        """エネルギー保存: |r_s|² + T_s = 1（T_s はフレネル透過率）。"""
        theta_i = 30.0
        n1, n2 = 1.0, 1.5
        rs = fresnel_rs(theta_i, n1, n2)
        ts = fresnel_ts(theta_i, n1, n2)
        theta_t = snell_angle(theta_i, n1, n2)
        cos_i = np.cos(np.deg2rad(theta_i))
        cos_t = np.cos(np.deg2rad(theta_t))
        # R + T = 1: R = |rs|², T = (n2*cos_t)/(n1*cos_i) * |ts|²
        R = abs(rs) ** 2
        T = (n2 * cos_t) / (n1 * cos_i) * abs(ts) ** 2
        assert R + T == pytest.approx(1.0, rel=1e-4)

    def test_snell_angle(self):
        """スネルの法則の検証: sin(θ_i)*n1 = sin(θ_t)*n2。"""
        theta_i = 45.0
        n1, n2 = 1.0, 1.5
        theta_t = snell_angle(theta_i, n1, n2)
        lhs = n1 * np.sin(np.deg2rad(theta_i))
        rhs = n2 * np.sin(np.deg2rad(theta_t))
        assert lhs == pytest.approx(rhs, rel=1e-6)

    def test_total_reflection(self):
        """臨界角超えで ValueError が発生する。"""
        with pytest.raises(ValueError, match="全反射"):
            snell_angle(60.0, 1.5, 1.0)  # 内部から外部、臨界角≈41.8°


class TestFFTBSDF:
    @pytest.fixture
    def simple_height_map(self):
        """テスト用の小さな高さマップ。"""
        rng = np.random.default_rng(42)
        data = (rng.standard_normal((64, 64)) * 0.005).astype(np.float32)
        return HeightMap(data=data, pixel_size_um=0.25)

    def test_output_shape(self, simple_height_map):
        u, v, bsdf = compute_bsdf_fft(
            height_map=simple_height_map,
            wavelength_um=0.55,
            theta_i_deg=0.0,
            phi_i_deg=0.0,
        )
        assert u.shape == (64, 64)
        assert v.shape == (64, 64)
        assert bsdf.shape == (64, 64)

    def test_bsdf_non_negative(self, simple_height_map):
        """BSDF 値は非負でなければならない。"""
        _, _, bsdf = compute_bsdf_fft(
            height_map=simple_height_map,
            wavelength_um=0.55,
            theta_i_deg=0.0,
            phi_i_deg=0.0,
        )
        assert np.all(bsdf >= 0.0)

    def test_brdf_mode(self, simple_height_map):
        """is_btdf=False で正常に動作する。"""
        u, v, bsdf = compute_bsdf_fft(
            height_map=simple_height_map,
            wavelength_um=0.55,
            theta_i_deg=30.0,
            phi_i_deg=0.0,
            is_btdf=False,
        )
        assert bsdf.max() > 0.0

    def test_btdf_mode(self, simple_height_map):
        """is_btdf=True で正常に動作する。"""
        u, v, bsdf = compute_bsdf_fft(
            height_map=simple_height_map,
            wavelength_um=0.55,
            theta_i_deg=30.0,  # 表面側換算済み角度
            phi_i_deg=0.0,
            n1=1.0,
            n2=1.5,
            is_btdf=True,
        )
        assert bsdf.max() > 0.0

    def test_sample_at_angles(self, simple_height_map):
        """指定角度でのサンプリングが動作する。"""
        u, v, bsdf = compute_bsdf_fft(
            height_map=simple_height_map,
            wavelength_um=0.55,
            theta_i_deg=0.0,
            phi_i_deg=0.0,
        )
        theta_query = np.array([0.0, 10.0, 20.0])
        phi_query = np.array([0.0, 0.0, 0.0])
        result = sample_bsdf_at_angles(u, v, bsdf, theta_query, phi_query)
        assert result.shape == (3,)
        assert np.all(result >= 0.0)

    def test_uv_range(self, simple_height_map):
        """UV グリッドの最大半径が 1 以内（半球内）に収まる。"""
        u, v, bsdf = compute_bsdf_fft(
            height_map=simple_height_map,
            wavelength_um=0.55,
            theta_i_deg=0.0,
            phi_i_deg=0.0,
        )
        # 有効範囲（bsdf > 0）は UV 半径 1 以内
        valid = bsdf > 0
        uv_r = np.sqrt(u[valid]**2 + v[valid]**2)
        assert np.all(uv_r <= 1.0 + 1e-5)
