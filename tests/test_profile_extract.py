"""Tests for visualization.profile_extract (slice_phi0, sort_and_floor)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bsdf_sim.visualization.constants import BSDF_LOG_FLOOR_DEFAULT
from bsdf_sim.visualization.profile_extract import slice_phi0, sort_and_floor


def _make_fft_grid(N: int, dx: float, wavelength_um: float):
    """Return (u_grid, v_grid) in fftfreq ordering."""
    freq = np.fft.fftfreq(N, d=dx)
    fx, fy = np.meshgrid(freq, freq, indexing="ij")
    u = fx * wavelength_um
    v = fy * wavelength_um
    return u, v


class TestSlicePhi0:
    def test_positive_mode_monotonic_u(self):
        """positive mode: returned u axis is strictly ascending in [0, 1]."""
        u, v = _make_fft_grid(32, dx=0.15, wavelength_um=0.55)
        bsdf = np.ones_like(u)
        u_out, b_out = slice_phi0(u, v, bsdf, mode="positive")
        assert u_out.min() >= 0.0
        assert u_out.max() <= 1.0
        assert np.all(np.diff(u_out) > 0)
        assert b_out.shape == u_out.shape

    def test_signed_mode_covers_both_sides(self):
        """signed mode: both negative and positive u are included."""
        u, v = _make_fft_grid(32, dx=0.15, wavelength_um=0.55)
        bsdf = np.ones_like(u)
        u_out, _ = slice_phi0(u, v, bsdf, mode="signed")
        assert u_out.min() < 0.0
        assert u_out.max() > 0.0
        assert np.all(np.diff(u_out) > 0)

    def test_floor_clipping(self):
        """Values below floor are replaced with floor."""
        u, v = _make_fft_grid(16, dx=0.15, wavelength_um=0.55)
        bsdf = np.full_like(u, 1e-30)
        _, b_out = slice_phi0(u, v, bsdf, floor=1e-10)
        assert np.all(b_out >= 1e-10)

    def test_v_band_averaging_changes_values(self):
        """v_band_bins > 0 averages over multiple v columns."""
        u, v = _make_fft_grid(32, dx=0.15, wavelength_um=0.55)
        # Make bsdf with distinct v=0 value vs surrounding
        bsdf = np.ones_like(u)
        bsdf[:, 0] = 10.0
        _, b_single = slice_phi0(u, v, bsdf, v_band_bins=0)
        _, b_band = slice_phi0(u, v, bsdf, v_band_bins=3)
        # Single slice reflects the v=0 row → 10
        assert np.allclose(b_single, 10.0)
        # Band average dilutes 10 → ~10/7 at most
        assert b_band.max() < 10.0

    def test_no_double_line_artifact(self):
        """positive mode output has no return-trip in u axis.

        Simulates BUG-009: with fftfreq ordering, a naive |u| produces a
        folded-back curve. slice_phi0 must guarantee strictly ascending u.
        """
        N = 64
        u, v = _make_fft_grid(N, dx=0.15, wavelength_um=0.55)
        bsdf = np.arange(N * N, dtype=float).reshape(N, N)
        u_out, _ = slice_phi0(u, v, bsdf, mode="positive")
        assert np.all(np.diff(u_out) > 0)

    def test_invalid_mode_raises(self):
        u, v = _make_fft_grid(16, dx=0.15, wavelength_um=0.55)
        bsdf = np.ones_like(u)
        with pytest.raises(ValueError):
            slice_phi0(u, v, bsdf, mode="bogus")


class TestSortAndFloor:
    def test_sorts_by_theta_s(self):
        df = pd.DataFrame({
            "theta_s_deg": [30.0, 10.0, 20.0],
            "bsdf":        [3.0,  1.0,  2.0],
        })
        x, y = sort_and_floor(df)
        np.testing.assert_array_equal(x, [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(y, [1.0, 2.0, 3.0])

    def test_applies_floor(self):
        df = pd.DataFrame({
            "theta_s_deg": [0.0, 45.0],
            "bsdf":        [1e-30, 1.0],
        })
        _, y = sort_and_floor(df, floor=1e-6)
        assert y[0] == 1e-6
        assert y[1] == 1.0

    def test_empty_returns_empty_arrays(self):
        df = pd.DataFrame({"theta_s_deg": [], "bsdf": []})
        x, y = sort_and_floor(df)
        assert len(x) == 0
        assert len(y) == 0

    def test_none_returns_empty_arrays(self):
        x, y = sort_and_floor(None)
        assert len(x) == 0
        assert len(y) == 0

    def test_default_floor_is_module_constant(self):
        df = pd.DataFrame({
            "theta_s_deg": [0.0],
            "bsdf":        [0.0],
        })
        _, y = sort_and_floor(df)
        assert y[0] == BSDF_LOG_FLOOR_DEFAULT
