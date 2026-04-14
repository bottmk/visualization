"""表面形状モデルのテスト。"""

import numpy as np
import pytest

from bsdf_sim.models.base import HeightMap, BaseSurfaceModel
from bsdf_sim.models.random_rough import RandomRoughSurface
from bsdf_sim.models.spherical_array import SphericalArraySurface


class TestHeightMap:
    def test_basic_creation(self):
        data = np.zeros((64, 64), dtype=np.float32)
        hm = HeightMap(data=data, pixel_size_um=0.25)
        assert hm.grid_size == 64
        assert hm.physical_size_um == pytest.approx(16.0)
        assert hm.pixel_size_um == 0.25

    def test_invalid_non_square(self):
        with pytest.raises(ValueError, match="正方形"):
            HeightMap(data=np.zeros((32, 64)), pixel_size_um=0.25)

    def test_invalid_pixel_size(self):
        with pytest.raises(ValueError, match="正の値"):
            HeightMap(data=np.zeros((64, 64)), pixel_size_um=-1.0)

    def test_rq_property(self):
        data = np.ones((64, 64), dtype=np.float32) * 0.01
        hm = HeightMap(data=data, pixel_size_um=0.25)
        assert hm.rq_um == pytest.approx(0.01, rel=1e-4)

    def test_resample(self):
        data = np.random.default_rng(0).random((128, 128)).astype(np.float32)
        hm = HeightMap(data=data, pixel_size_um=0.25)
        hm_small = hm.resample(64)
        assert hm_small.grid_size == 64
        assert hm_small.pixel_size_um == 0.25


class TestRandomRoughSurface:
    def test_output_shape(self):
        model = RandomRoughSurface(rq_um=0.005, lc_um=2.0, grid_size=64, pixel_size_um=0.25, seed=0)
        hm = model.get_height_map()
        assert hm.data.shape == (64, 64)

    def test_rq_normalization(self):
        """生成した面の RMS粗さが指定値に一致する。"""
        target_rq = 0.005
        model = RandomRoughSurface(rq_um=target_rq, lc_um=2.0, grid_size=256, pixel_size_um=0.25, seed=42)
        hm = model.get_height_map()
        actual_rq = np.sqrt(np.mean(hm.data**2))
        assert actual_rq == pytest.approx(target_rq, rel=0.05)

    def test_preview_reduced_area(self):
        """reduced_area モードでは pixel_size_um が固定される。"""
        model = RandomRoughSurface(rq_um=0.005, lc_um=2.0, grid_size=512, pixel_size_um=0.25, seed=0)
        hm_prev = model.get_preview_height_map(mode="reduced_area", preview_grid_size=64)
        assert hm_prev.grid_size == 64
        assert hm_prev.pixel_size_um == pytest.approx(0.25)

    def test_preview_reduced_resolution(self):
        """reduced_resolution モードでは physical_size_um が固定される。"""
        model = RandomRoughSurface(rq_um=0.005, lc_um=2.0, grid_size=512, pixel_size_um=0.25, seed=0)
        hm_prev = model.get_preview_height_map(mode="reduced_resolution", preview_grid_size=64)
        assert hm_prev.grid_size == 64
        expected_pixel = model.physical_size_um / 64
        assert hm_prev.pixel_size_um == pytest.approx(expected_pixel)

    def test_invalid_mode(self):
        model = RandomRoughSurface(rq_um=0.005, lc_um=2.0, grid_size=64, pixel_size_um=0.25)
        with pytest.raises(ValueError):
            model.get_preview_height_map(mode="invalid")

    def test_invalid_rq(self):
        with pytest.raises(ValueError):
            RandomRoughSurface(rq_um=-0.001, lc_um=2.0)

    def test_fractal_dim_range(self):
        with pytest.raises(ValueError):
            RandomRoughSurface(rq_um=0.005, lc_um=2.0, fractal_dim=1.5)


class TestSphericalArraySurface:
    def test_grid_placement(self):
        model = SphericalArraySurface(
            radius_um=10.0, pitch_um=20.0, placement="Grid",
            grid_size=64, pixel_size_um=0.5, seed=0
        )
        hm = model.get_height_map()
        assert hm.data.shape == (64, 64)
        assert hm.data.min() >= 0.0

    def test_hexagonal_placement(self):
        model = SphericalArraySurface(
            radius_um=10.0, pitch_um=20.0, placement="Hexagonal",
            grid_size=64, pixel_size_um=0.5, seed=0
        )
        hm = model.get_height_map()
        assert hm.data.min() >= 0.0

    def test_maximum_overlap(self):
        """Maximum モードでは高さが負にならない。"""
        model = SphericalArraySurface(
            radius_um=10.0, pitch_um=10.0, placement="Grid",
            overlap_mode="Maximum", grid_size=64, pixel_size_um=0.5
        )
        hm = model.get_height_map()
        assert hm.data.min() >= 0.0

    def test_additive_overlap(self):
        """Additive モードでは重なった領域が加算される。"""
        model_max = SphericalArraySurface(
            radius_um=10.0, pitch_um=5.0, placement="Grid",
            overlap_mode="Maximum", grid_size=64, pixel_size_um=0.5, seed=0
        )
        model_add = SphericalArraySurface(
            radius_um=10.0, pitch_um=5.0, placement="Grid",
            overlap_mode="Additive", grid_size=64, pixel_size_um=0.5, seed=0
        )
        hm_max = model_max.get_height_map()
        hm_add = model_add.get_height_map()
        # Additive の最大値は Maximum より大きいはず
        assert hm_add.data.max() >= hm_max.data.max()
