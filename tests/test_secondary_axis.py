"""Tests for visualization.secondary_axis (unit transforms + helpers)."""

from __future__ import annotations

import numpy as np
import pytest

from bsdf_sim.visualization.secondary_axis import (
    AXIS_UNITS,
    DEFAULT_SECONDARY_X_UNIT,
    get_axis_unit_spec,
    make_secondary_xaxis_hook,
)


class TestAxisUnits:
    def test_default_is_lambda_scale(self):
        assert DEFAULT_SECONDARY_X_UNIT == "lambda_scale"

    def test_all_units_registered(self):
        for key in ("theta_s", "lambda_scale", "u", "f", "k_x"):
            assert key in AXIS_UNITS

    def test_get_unknown_unit_raises(self):
        with pytest.raises(ValueError):
            get_axis_unit_spec("not_a_unit")

    @pytest.mark.parametrize("unit", ["theta_s", "lambda_scale", "u", "f", "k_x"])
    def test_roundtrip_consistency(self, unit):
        """from_theta → to_theta roundtrip preserves θ_s."""
        spec = AXIS_UNITS[unit]
        lam = 0.55
        thetas = np.array([1.0, 10.0, 30.0, 60.0, 89.9])
        vals = spec.from_theta(thetas, lam)
        back = spec.to_theta(vals, lam)
        np.testing.assert_allclose(back, thetas, rtol=1e-5)

    def test_lambda_scale_at_30deg(self):
        """Λ = λ/sin(30°) = 0.55/0.5 = 1.1 μm."""
        spec = AXIS_UNITS["lambda_scale"]
        val = spec.from_theta(np.array([30.0]), 0.55)
        np.testing.assert_allclose(val, 1.1, rtol=1e-6)

    def test_u_at_90deg_is_one(self):
        spec = AXIS_UNITS["u"]
        val = spec.from_theta(np.array([90.0]), 0.55)
        np.testing.assert_allclose(val, 1.0, rtol=1e-6)

    def test_f_equals_u_over_lambda(self):
        """f = sin θ_s / λ = u / λ."""
        spec_f = AXIS_UNITS["f"]
        spec_u = AXIS_UNITS["u"]
        thetas = np.array([1.0, 30.0, 60.0])
        lam = 0.55
        f_val = spec_f.from_theta(thetas, lam)
        u_val = spec_u.from_theta(thetas, lam)
        np.testing.assert_allclose(f_val, u_val / lam, rtol=1e-6)

    def test_k_x_equals_2pi_over_lambda_times_sin(self):
        """k_x = (2π/λ) sin θ_s."""
        spec = AXIS_UNITS["k_x"]
        thetas = np.array([30.0, 60.0])
        lam = 0.55
        val = spec.from_theta(thetas, lam)
        expected = (2 * np.pi / lam) * np.sin(np.deg2rad(thetas))
        np.testing.assert_allclose(val, expected, rtol=1e-6)

    def test_english_labels_present(self):
        """matplotlib 用の英語ラベルがすべて定義されている（tofu 回避）。"""
        for key, spec in AXIS_UNITS.items():
            assert spec.label_en, f"{key}: label_en empty"
            # 英語ラベルは ASCII-only（tofu 回避）
            assert spec.label_en.isascii(), f"{key}: non-ASCII in label_en"

    def test_lambda_scale_recommends_log(self):
        assert AXIS_UNITS["lambda_scale"].log_scale_recommended is True
        assert AXIS_UNITS["u"].log_scale_recommended is False


class TestSecondaryAxisHook:
    def test_hook_is_callable(self):
        hook = make_secondary_xaxis_hook("lambda_scale", 0.55)
        assert callable(hook)

    def test_hook_unknown_unit_raises(self):
        with pytest.raises(ValueError):
            make_secondary_xaxis_hook("not_a_unit", 0.55)

    def test_hook_no_crash_with_none_plot(self):
        """Hook should be resilient to plot.handles missing."""
        class DummyPlot:
            handles: dict = {}
        hook = make_secondary_xaxis_hook("lambda_scale", 0.55)
        hook(DummyPlot(), None)  # should not raise
