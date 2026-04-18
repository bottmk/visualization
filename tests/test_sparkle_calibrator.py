"""Sparkle Cs 校正フレームワークのテスト。

対象: src/bsdf_sim/metrics/sparkle_calibrator.py
"""

import numpy as np
import pytest

from bsdf_sim.metrics.sparkle_calibrator import (
    apply_calibration,
    fit_polynomial,
    fit_scale,
)


class TestApplyCalibration:
    def test_none_passthrough(self):
        """calibration=None で raw Cs をそのまま返す。"""
        assert apply_calibration(0.5, None) == 0.5
        assert apply_calibration(0.5, {}) == 0.5

    def test_mode_none_passthrough(self):
        """mode='none' / 'null' でも passthrough。"""
        assert apply_calibration(0.5, {"mode": "none"}) == 0.5
        assert apply_calibration(0.5, {"mode": "null"}) == 0.5

    def test_mode_missing_passthrough(self):
        """mode 省略時は passthrough。"""
        assert apply_calibration(0.5, {"scale": 0.1}) == 0.5  # mode 無しなら無視

    def test_scale_linear(self):
        """scale モードで線形スケーリング。"""
        assert apply_calibration(0.5, {"mode": "scale", "scale": 0.1}) == pytest.approx(0.05)
        assert apply_calibration(2.0, {"mode": "scale", "scale": 0.5}) == pytest.approx(1.0)

    def test_scale_missing_raises(self):
        """scale 値が欠けているとエラー。"""
        with pytest.raises(ValueError, match="scale"):
            apply_calibration(0.5, {"mode": "scale"})

    def test_polynomial_basic(self):
        """polynomial モードで a*x^b + c を計算。"""
        # a=2, b=1, c=0 → 線形
        result = apply_calibration(0.5, {"mode": "polynomial", "polynomial": [2.0, 1.0, 0.0]})
        assert result == pytest.approx(1.0)
        # a=1, b=2, c=0.1 → 0.5^2 + 0.1 = 0.35
        result = apply_calibration(0.5, {"mode": "polynomial", "polynomial": [1.0, 2.0, 0.1]})
        assert result == pytest.approx(0.35)

    def test_polynomial_missing_raises(self):
        """polynomial 値が欠けているとエラー。"""
        with pytest.raises(ValueError, match="polynomial"):
            apply_calibration(0.5, {"mode": "polynomial"})

    def test_polynomial_wrong_length_raises(self):
        """polynomial が 3 要素でないとエラー。"""
        with pytest.raises(ValueError, match="3 要素"):
            apply_calibration(0.5, {"mode": "polynomial", "polynomial": [1.0, 2.0]})

    def test_unknown_mode_raises(self):
        """未知 mode でエラー。"""
        with pytest.raises(ValueError, match="未知"):
            apply_calibration(0.5, {"mode": "invalid"})

    def test_negative_cs_zero_clamp(self):
        """polynomial に負値を渡しても 0 にクランプして計算される。"""
        result = apply_calibration(-0.1, {"mode": "polynomial", "polynomial": [1.0, 0.5, 0.0]})
        # 0.0^0.5 = 0
        assert result == pytest.approx(0.0)


class TestFitScale:
    def test_simple_scale(self):
        """完全比例のデータから正確に k が求まる。"""
        sim = [0.1, 0.2, 0.3, 0.4, 0.5]
        meas = [0.01, 0.02, 0.03, 0.04, 0.05]
        k = fit_scale(sim, meas)
        assert k == pytest.approx(0.1)

    def test_noisy_scale(self):
        """ノイズ混入でも k が近似的に求まる。"""
        rng = np.random.default_rng(42)
        sim = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        k_true = 0.1
        meas = k_true * sim + rng.normal(0, 0.001, size=5)
        k = fit_scale(sim.tolist(), meas.tolist())
        assert k == pytest.approx(k_true, rel=0.05)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="長さ"):
            fit_scale([0.1, 0.2], [0.01, 0.02, 0.03])

    def test_all_zero_sim_raises(self):
        """cs_sim が全てゼロならエラー。"""
        with pytest.raises(ValueError, match="有効なサンプル"):
            fit_scale([0.0, 0.0], [0.1, 0.2])


class TestFitPolynomial:
    def test_polynomial_fit(self):
        """a*x^b + c のデータから (a, b, c) が近似的に求まる。"""
        sim = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
        a_true, b_true, c_true = 0.15, 1.2, 0.01
        meas = a_true * sim**b_true + c_true
        a, b, c = fit_polynomial(sim.tolist(), meas.tolist())
        assert a == pytest.approx(a_true, rel=0.05)
        assert b == pytest.approx(b_true, rel=0.05)
        assert c == pytest.approx(c_true, abs=0.005)

    def test_insufficient_samples_raises(self):
        """サンプル数 < 3 でエラー。"""
        with pytest.raises(ValueError, match="3 サンプル以上"):
            fit_polynomial([0.1, 0.2], [0.01, 0.02])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="長さ"):
            fit_polynomial([0.1, 0.2, 0.3], [0.01, 0.02])


class TestCalibrationIntegration:
    """compute_all_optical_metrics との統合テスト。"""

    @pytest.fixture
    def sparkle_config(self):
        return {
            "viewing": {"distance_mm": 300.0, "pupil_diameter_mm": 3.0},
            "display": {"pixel_pitch_mm": 0.062, "subpixel_layout": "rgb_stripe"},
        }

    def test_l1_with_scale_calibration(self, sparkle_config):
        """L1 + scale 校正が compute_all_optical_metrics に反映される。"""
        from bsdf_sim.metrics.optical import compute_all_optical_metrics
        from bsdf_sim.models.base import HeightMap
        from bsdf_sim.optics.fft_bsdf import compute_bsdf_fft

        rng = np.random.default_rng(42)
        hm = HeightMap(data=rng.normal(0, 0.05, size=(128, 128)), pixel_size_um=0.25)
        u, v, bsdf = compute_bsdf_fft(
            hm, 0.525, 0.0, 0.0, is_btdf=True, fft_mode="zero"
        )

        # 校正なし
        cfg_raw = {"sparkle": {**sparkle_config, "enabled": True, "level": "L1"}}
        r_raw = compute_all_optical_metrics(
            u_grid=u, v_grid=v, bsdf=bsdf,
            method_name="fft", wavelength_nm=525,
            config=cfg_raw, sparkle_only=True, height_map=hm,
        )
        cs_raw = r_raw["sparkle_l1_fft_525_0_t"]

        # 校正あり (scale=0.001)
        cfg_cal = {
            "sparkle": {
                **sparkle_config, "enabled": True, "level": "L1",
                "calibration": {"mode": "scale", "scale": 0.001},
            }
        }
        r_cal = compute_all_optical_metrics(
            u_grid=u, v_grid=v, bsdf=bsdf,
            method_name="fft", wavelength_nm=525,
            config=cfg_cal, sparkle_only=True, height_map=hm,
        )
        cs_cal = r_cal["sparkle_l1_fft_525_0_t"]

        assert cs_cal == pytest.approx(cs_raw * 0.001, rel=1e-9)
