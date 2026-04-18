"""Tests for visualization.profile_extract.slice_phi0."""

from __future__ import annotations

import numpy as np
import pytest

from bsdf_sim.visualization.profile_extract import slice_phi0


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
