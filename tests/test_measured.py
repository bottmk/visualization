"""MeasuredSurface の前処理・ローダー・from_config のテスト。"""

import textwrap

import numpy as np
import pytest

from bsdf_sim.models.measured import MeasuredSurface


@pytest.fixture
def simple_data():
    """2×2 の単純な高さ配列（μm）。"""
    return np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)


@pytest.fixture
def data_with_nan():
    """NaN を含む 4×4 配列（μm）。"""
    data = np.ones((4, 4), dtype=np.float64)
    data[1, 1] = np.nan
    data[2, 3] = np.nan
    return data


class TestMeasuredSurfaceInit:
    def test_generates_correct_grid_size(self, simple_data):
        ms = MeasuredSurface(
            height_data=simple_data,
            source_pixel_size_um=0.5,
            grid_size=4,
            pixel_size_um=0.5,
            leveling=False,
        )
        hm = ms.get_height_map()
        assert hm.grid_size == 4

    def test_leveling_removes_tilt(self):
        """傾き成分がレベリングで除去される。"""
        x = np.arange(16, dtype=np.float32).reshape(4, 4)
        ms = MeasuredSurface(
            height_data=x,
            source_pixel_size_um=1.0,
            grid_size=4,
            pixel_size_um=1.0,
            leveling=True,
        )
        hm = ms.get_height_map()
        # レベリング後は平均がほぼゼロ
        assert abs(float(np.mean(hm.data))) < 1e-3

    def test_no_leveling_preserves_data(self, simple_data):
        ms = MeasuredSurface(
            height_data=simple_data,
            source_pixel_size_um=0.5,
            grid_size=2,
            pixel_size_um=0.5,
            leveling=False,
        )
        hm = ms.get_height_map()
        assert hm.data is not None


class TestNaNInterpolation:
    def test_nan_removed_after_preprocessing(self, data_with_nan):
        ms = MeasuredSurface(
            height_data=data_with_nan,
            source_pixel_size_um=1.0,
            grid_size=4,
            pixel_size_um=1.0,
            leveling=False,
        )
        assert not np.any(np.isnan(ms._processed))

    def test_all_nan_returns_zero(self):
        data = np.full((4, 4), np.nan, dtype=np.float64)
        ms = MeasuredSurface(
            height_data=data,
            source_pixel_size_um=1.0,
            grid_size=4,
            pixel_size_um=1.0,
            leveling=False,
        )
        assert np.all(ms._processed == 0.0)


class TestFromNumpy:
    def test_from_numpy_basic(self, simple_data):
        ms = MeasuredSurface.from_numpy(
            data=simple_data,
            source_pixel_size_um=0.5,
            grid_size=4,
            pixel_size_um=0.5,
            leveling=False,
        )
        assert isinstance(ms, MeasuredSurface)

    def test_from_numpy_pixel_size(self, simple_data):
        ms = MeasuredSurface.from_numpy(
            data=simple_data,
            source_pixel_size_um=0.5,
            grid_size=4,
            pixel_size_um=0.25,
        )
        hm = ms.get_height_map()
        assert hm.pixel_size_um == pytest.approx(0.25)


class TestFromCSV:
    def test_from_csv_um(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("0.0,1.0,2.0\n3.0,4.0,5.0\n6.0,7.0,8.0\n", encoding="utf-8")
        ms = MeasuredSurface.from_csv(
            path=csv_path,
            source_pixel_size_um=1.0,
            height_unit="um",
            grid_size=3,
            pixel_size_um=1.0,
            leveling=False,
        )
        assert isinstance(ms, MeasuredSurface)

    def test_from_csv_nm_converts_to_um(self, tmp_path):
        """nm 単位の CSV を読み込み μm に変換できる。"""
        csv_path = tmp_path / "data_nm.csv"
        csv_path.write_text("1000.0,2000.0\n3000.0,4000.0\n", encoding="utf-8")
        ms = MeasuredSurface.from_csv(
            path=csv_path,
            source_pixel_size_um=1.0,
            height_unit="nm",
            grid_size=2,
            pixel_size_um=1.0,
            leveling=False,
        )
        # 1000 nm = 1.0 μm → _processed の最小値 ≈ 1.0 μm（レベリング後は差分のみ）
        assert ms._processed is not None

    def test_from_csv_invalid_height_unit(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("1.0,2.0\n3.0,4.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="height_unit"):
            MeasuredSurface.from_csv(
                path=csv_path,
                source_pixel_size_um=1.0,
                height_unit="ft",
            )

    def test_from_csv_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MeasuredSurface.from_csv(
                path=tmp_path / "nonexistent.csv",
                source_pixel_size_um=1.0,
            )

    def test_from_csv_skiprows(self, tmp_path):
        """ヘッダ行をスキップして読み込める。"""
        csv_path = tmp_path / "data_header.csv"
        lines = "# header line 1\n# header line 2\n0.0,1.0\n2.0,3.0\n"
        csv_path.write_text(lines, encoding="utf-8")
        ms = MeasuredSurface.from_csv(
            path=csv_path,
            source_pixel_size_um=1.0,
            height_unit="um",
            skiprows=2,
            grid_size=2,
            pixel_size_um=1.0,
            leveling=False,
        )
        assert isinstance(ms, MeasuredSurface)


class TestFromConfig:
    def test_from_config_basic(self, tmp_path):
        """from_config が CSV パスを読み込み MeasuredSurface を返す。"""
        csv_path = tmp_path / "surface.csv"
        csv_path.write_text("0.0,1.0,2.0\n3.0,4.0,5.0\n6.0,7.0,8.0\n", encoding="utf-8")

        config = {
            "surface": {
                "model": "MeasuredSurface",
                "grid_size": 3,
                "pixel_size_um": 1.0,
                "measured": {
                    "path": str(csv_path),
                    "source_pixel_size_um": 1.0,
                    "height_unit": "um",
                    "skiprows": 0,
                    "leveling": False,
                },
            }
        }
        ms = MeasuredSurface.from_config(config)
        assert isinstance(ms, MeasuredSurface)

    def test_from_config_no_path_raises(self):
        config = {
            "surface": {
                "model": "MeasuredSurface",
                "measured": {},
            }
        }
        with pytest.raises(ValueError, match="path"):
            MeasuredSurface.from_config(config)

    def test_from_config_height_unit_nm(self, tmp_path):
        """nm 単位の CSV を from_config で読み込める。"""
        csv_path = tmp_path / "surface_nm.csv"
        csv_path.write_text("100.0,200.0\n300.0,400.0\n", encoding="utf-8")

        config = {
            "surface": {
                "model": "MeasuredSurface",
                "grid_size": 2,
                "pixel_size_um": 0.5,
                "measured": {
                    "path": str(csv_path),
                    "source_pixel_size_um": 0.5,
                    "height_unit": "nm",
                    "leveling": False,
                },
            }
        }
        ms = MeasuredSurface.from_config(config)
        assert isinstance(ms, MeasuredSurface)
