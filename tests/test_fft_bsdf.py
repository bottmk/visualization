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


class TestFFTModes:
    """fft_mode = 'tilt' / 'output_shift' / 'zero' の 3 モードのテスト。"""

    @pytest.fixture
    def rough_surface(self):
        """中粗さのランダムラフ表面（漏れの効果が可視化できる）。"""
        surface = RandomRoughSurface(
            grid_size=256, pixel_size_um=0.2,  # λ/(2dx)=1.375 で output_shift も OK
            rq_um=0.02, lc_um=2.0, fractal_dim=2.5, seed=42,
        )
        return surface.get_height_map()

    def test_invalid_mode_raises(self, rough_surface):
        with pytest.raises(ValueError, match="fft_mode"):
            compute_bsdf_fft(
                height_map=rough_surface, wavelength_um=0.55,
                theta_i_deg=0.0, phi_i_deg=0.0, fft_mode="unknown",
            )

    def test_tilt_default_backward_compat(self, rough_surface):
        """fft_mode 省略時は 'tilt' と同じ挙動（既存 API 互換）。"""
        u1, v1, b1 = compute_bsdf_fft(
            height_map=rough_surface, wavelength_um=0.55,
            theta_i_deg=20.0, phi_i_deg=0.0,
        )
        u2, v2, b2 = compute_bsdf_fft(
            height_map=rough_surface, wavelength_um=0.55,
            theta_i_deg=20.0, phi_i_deg=0.0, fft_mode="tilt",
        )
        np.testing.assert_array_equal(u1, u2)
        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(b1, b2)

    def test_tilt_vs_output_shift_peak_location(self, rough_surface):
        """tilt / output_shift どちらも specular が (sin θ_i, 0) に出る。"""
        ti = 20.0
        u_spec = np.sin(np.deg2rad(ti))
        for mode in ("tilt", "output_shift"):
            u, v, b = compute_bsdf_fft(
                height_map=rough_surface, wavelength_um=0.55,
                theta_i_deg=ti, phi_i_deg=0.0, fft_mode=mode,
            )
            idx = np.unravel_index(np.argmax(b), b.shape)
            assert abs(u[idx] - u_spec) < 0.05, f"{mode}: u_peak={u[idx]} vs {u_spec}"
            assert abs(v[idx]) < 0.05, f"{mode}: v_peak={v[idx]}"

    def test_zero_mode_peak_at_origin(self, rough_surface):
        """zero モード: θ_i によらず specular は (0, 0) にとどまる。"""
        for ti in (0.0, 20.0, 45.0):
            u, v, b = compute_bsdf_fft(
                height_map=rough_surface, wavelength_um=0.55,
                theta_i_deg=ti, phi_i_deg=0.0, fft_mode="zero",
            )
            idx = np.unravel_index(np.argmax(b), b.shape)
            assert abs(u[idx]) < 0.01, f"ti={ti}: u_peak={u[idx]}"
            assert abs(v[idx]) < 0.01, f"ti={ti}: v_peak={v[idx]}"

    def test_zero_mode_theta_i_independent(self, rough_surface):
        """zero モード: 結果が θ_i に依存しない（BSDF グリッドが一致）。"""
        _, _, b0 = compute_bsdf_fft(
            height_map=rough_surface, wavelength_um=0.55,
            theta_i_deg=0.0, phi_i_deg=0.0, fft_mode="zero",
        )
        _, _, b30 = compute_bsdf_fft(
            height_map=rough_surface, wavelength_um=0.55,
            theta_i_deg=30.0, phi_i_deg=0.0, fft_mode="zero",
        )
        _, _, b60 = compute_bsdf_fft(
            height_map=rough_surface, wavelength_um=0.55,
            theta_i_deg=60.0, phi_i_deg=0.0, fft_mode="zero",
        )
        np.testing.assert_allclose(b0, b30, rtol=1e-5)
        np.testing.assert_allclose(b0, b60, rtol=1e-5)

    def test_output_shift_no_leakage_at_flat_surface(self):
        """h=0 の平面で output_shift モードは clean delta（漏れなし）。"""
        flat_hm = HeightMap(data=np.zeros((256, 256), dtype=np.float32), pixel_size_um=0.2)
        u, v, b = compute_bsdf_fft(
            height_map=flat_hm, wavelength_um=0.55,
            theta_i_deg=20.0, phi_i_deg=0.0, fft_mode="output_shift",
        )
        b_sorted = np.sort(b.ravel())[::-1]
        assert b_sorted[0] / max(b_sorted[1], 1e-30) > 1e5

    def test_tilt_has_leakage_at_flat_surface(self):
        """h=0 でも tilt モードは非整数 θ_i でスペクトル漏れを生じる（記録用）。"""
        flat_hm = HeightMap(data=np.zeros((256, 256), dtype=np.float32), pixel_size_um=0.2)
        # θ_i=20°: sin(20°)·N·dx/λ = 0.342·256·0.2/0.55 ≒ 31.83（非整数）→ 漏れる
        u, v, b = compute_bsdf_fft(
            height_map=flat_hm, wavelength_um=0.55,
            theta_i_deg=20.0, phi_i_deg=0.0, fft_mode="tilt",
        )
        b_sorted = np.sort(b.ravel())[::-1]
        # tilt では 2 番目以降にもエネルギーが残る（output_shift より悪い）
        assert b_sorted[0] / max(b_sorted[1], 1e-30) < 1e4

    def test_output_shift_grid_shifted(self, rough_surface):
        """output_shift の u_grid は u_spec だけ正方向にオフセット。"""
        ti = 30.0
        u_spec = np.sin(np.deg2rad(ti))
        u_t, _, _ = compute_bsdf_fft(
            height_map=rough_surface, wavelength_um=0.55,
            theta_i_deg=ti, phi_i_deg=0.0, fft_mode="tilt",
        )
        u_s, _, _ = compute_bsdf_fft(
            height_map=rough_surface, wavelength_um=0.55,
            theta_i_deg=ti, phi_i_deg=0.0, fft_mode="output_shift",
        )
        # tilt は中心 0 の格子、output_shift は中心 u_spec にシフト
        np.testing.assert_allclose(u_s - u_t, u_spec, rtol=1e-5)
