"""Tests for _bsdf_1d_to_2d_binned and df_to_2d_grid (bincount-based binning)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bsdf_sim.visualization.holoviews_plots import (
    _bsdf_1d_to_2d_binned,
    df_to_2d_grid,
)


class TestBsdf1dTo2dBinned:
    def test_shape_matches_n_grid(self):
        u = np.array([0.0])
        v = np.array([0.0])
        b = np.array([1.0])
        u_g, v_g, bsdf_2d = _bsdf_1d_to_2d_binned(u, v, b, n_grid=32)
        assert u_g.shape == (32, 32)
        assert v_g.shape == (32, 32)
        assert bsdf_2d.shape == (32, 32)

    def test_single_point_lands_on_nearest_bin(self):
        """A single point at (u=0, v=0) lands on the center bin."""
        u = np.array([0.0])
        v = np.array([0.0])
        b = np.array([5.0])
        u_g, v_g, bsdf_2d = _bsdf_1d_to_2d_binned(u, v, b, n_grid=33)  # odd→center exists
        center = 33 // 2
        assert bsdf_2d[center, center] == pytest.approx(5.0, rel=1e-5)
        # all other bins are 0
        mask = np.ones_like(bsdf_2d, dtype=bool)
        mask[center, center] = False
        assert np.all(bsdf_2d[mask] == 0.0)

    def test_multiple_points_in_same_bin_are_averaged(self):
        """Points that map to the same bin are averaged (not summed)."""
        u = np.array([0.0, 0.0, 0.0])
        v = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0])
        _, _, bsdf_2d = _bsdf_1d_to_2d_binned(u, v, b, n_grid=33)
        center = 33 // 2
        assert bsdf_2d[center, center] == pytest.approx(2.0, rel=1e-5)

    def test_outside_hemisphere_zeroed(self):
        """Bins with u²+v²>1 are forced to 0 even if points fell there."""
        # A point at (u=0.95, v=0.95) → u²+v² = 1.805 > 1 (outside hemisphere)
        u = np.array([0.95])
        v = np.array([0.95])
        b = np.array([99.0])
        u_g, v_g, bsdf_2d = _bsdf_1d_to_2d_binned(u, v, b, n_grid=64)
        outside = u_g ** 2 + v_g ** 2 > 1.0
        assert np.all(bsdf_2d[outside] == 0.0)

    def test_grid_axis_spans_minus1_to_plus1(self):
        """u_grid and v_grid span [-1, 1] uniformly."""
        u = np.array([0.0])
        v = np.array([0.0])
        b = np.array([1.0])
        u_g, v_g, _ = _bsdf_1d_to_2d_binned(u, v, b, n_grid=11)
        assert u_g[0, 0] == pytest.approx(-1.0)
        assert u_g[-1, 0] == pytest.approx(1.0)
        assert v_g[0, 0] == pytest.approx(-1.0)
        assert v_g[0, -1] == pytest.approx(1.0)

    def test_empty_input_returns_zero_grid(self):
        u = np.array([])
        v = np.array([])
        b = np.array([])
        _, _, bsdf_2d = _bsdf_1d_to_2d_binned(u, v, b, n_grid=16)
        assert bsdf_2d.shape == (16, 16)
        assert np.all(bsdf_2d == 0.0)

    def test_clip_out_of_range_to_boundary(self):
        """Points with |u|>1 or |v|>1 are clipped to the boundary bin (no crash)."""
        u = np.array([1.5, -2.0])
        v = np.array([0.5, -0.5])
        b = np.array([7.0, 8.0])
        # Must not raise; clipping to edge bins.
        _, _, bsdf_2d = _bsdf_1d_to_2d_binned(u, v, b, n_grid=8)
        assert bsdf_2d.shape == (8, 8)


class TestDfTo2dGrid:
    def test_delegates_to_binning(self):
        """df_to_2d_grid is a thin DataFrame adapter — same result as direct call."""
        df = pd.DataFrame({
            "u":    [0.0, 0.3, -0.2],
            "v":    [0.0, 0.0,  0.1],
            "bsdf": [1.0, 2.0,  3.0],
        })
        u_ref, v_ref, bsdf_ref = _bsdf_1d_to_2d_binned(
            df["u"].values, df["v"].values, df["bsdf"].values, n_grid=16,
        )
        u_df, v_df, bsdf_df = df_to_2d_grid(df, n_grid=16)
        np.testing.assert_array_equal(u_df, u_ref)
        np.testing.assert_array_equal(v_df, v_ref)
        np.testing.assert_array_equal(bsdf_df, bsdf_ref)
