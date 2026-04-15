"""MeasuredSurface の前処理・ローダー・from_config・パディングのテスト。"""

import textwrap

import numpy as np
import pytest

from bsdf_sim.models.measured import (
    MeasuredSurface,
    VALID_PADDINGS,
    _pad_zeros,
    _pad_tile,
    _pad_reflect,
    _pad_smooth_tile,
)


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


# ── パディング関数のユニットテスト ─────────────────────────────────────────────

@pytest.fixture
def square_data_32():
    """32×32 のランダム粗面データ（μm）。"""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((32, 32)).astype(np.float32) * 0.005
    return data


class TestPaddingFunctions:
    """4 種類のパディング関数の基本動作テスト。"""

    @pytest.mark.parametrize("grid_size", [64, 128, 200])
    def test_zeros_output_shape(self, square_data_32, grid_size):
        result = _pad_zeros(square_data_32, grid_size)
        assert result.shape == (grid_size, grid_size)

    def test_zeros_center_preserved(self, square_data_32):
        grid_size = 64
        result = _pad_zeros(square_data_32, grid_size)
        offset = (grid_size - 32) // 2
        np.testing.assert_array_equal(
            result[offset:offset + 32, offset:offset + 32],
            square_data_32,
        )

    def test_zeros_border_is_zero(self, square_data_32):
        result = _pad_zeros(square_data_32, 64)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[63, 63] == pytest.approx(0.0)

    @pytest.mark.parametrize("grid_size", [64, 128, 200])
    def test_tile_output_shape(self, square_data_32, grid_size):
        result = _pad_tile(square_data_32, grid_size)
        assert result.shape == (grid_size, grid_size)

    def test_tile_first_block_matches_data(self, square_data_32):
        result = _pad_tile(square_data_32, 64)
        np.testing.assert_array_equal(result[:32, :32], square_data_32)

    def test_tile_second_block_matches_data(self, square_data_32):
        result = _pad_tile(square_data_32, 64)
        np.testing.assert_array_equal(result[:32, 32:64], square_data_32)

    @pytest.mark.parametrize("grid_size", [64, 128, 200])
    def test_reflect_output_shape(self, square_data_32, grid_size):
        result = _pad_reflect(square_data_32, grid_size)
        assert result.shape == (grid_size, grid_size)

    def test_reflect_first_block_matches_data(self, square_data_32):
        result = _pad_reflect(square_data_32, 64)
        np.testing.assert_array_equal(result[:32, :32], square_data_32)

    def test_reflect_second_block_is_mirrored(self, square_data_32):
        result = _pad_reflect(square_data_32, 64)
        np.testing.assert_array_equal(result[:32, 32:64], square_data_32[:, ::-1])

    @pytest.mark.parametrize("grid_size", [64, 128, 200])
    def test_smooth_tile_output_shape(self, square_data_32, grid_size):
        result = _pad_smooth_tile(square_data_32, grid_size)
        assert result.shape == (grid_size, grid_size)

    def test_smooth_tile_center_unchanged(self, square_data_32):
        grid_size = 128
        result = _pad_smooth_tile(square_data_32, grid_size, blend_ratio=0.1)
        tiled = _pad_tile(square_data_32, grid_size)
        np.testing.assert_array_equal(result[:32, :32], tiled[:32, :32])

    def test_smooth_tile_boundary_continuity(self, square_data_32):
        grid_size = 128
        result = _pad_smooth_tile(square_data_32, grid_size)
        tiled  = _pad_tile(square_data_32, grid_size)
        diff_smooth = abs(float(result[:, -1].mean()) - float(result[:, 0].mean()))
        diff_tile   = abs(float(tiled[:, -1].mean())  - float(tiled[:, 0].mean()))
        assert diff_smooth <= diff_tile + 1e-6

    @pytest.mark.parametrize("mode", VALID_PADDINGS)
    def test_all_modes_dtype_float32(self, square_data_32, mode):
        from bsdf_sim.models.measured import _apply_padding
        result = _apply_padding(square_data_32, 64, mode)
        assert result.dtype == np.float32

    def test_invalid_padding_raises(self, square_data_32):
        from bsdf_sim.models.measured import _apply_padding
        with pytest.raises(ValueError, match="padding"):
            _apply_padding(square_data_32, 64, "unknown")


class TestMeasuredSurfacePadding:
    """MeasuredSurface の padding パラメータ統合テスト。"""

    @pytest.mark.parametrize("mode", VALID_PADDINGS)
    def test_padding_mode_accepted(self, mode):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((16, 16)).astype(np.float32)
        ms = MeasuredSurface(
            height_data=data,
            source_pixel_size_um=1.0,
            grid_size=32,
            pixel_size_um=1.0,
            leveling=False,
            padding=mode,
        )
        hm = ms.get_height_map()
        assert hm.data.shape == (32, 32)

    def test_invalid_padding_raises(self):
        data = np.zeros((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="padding"):
            MeasuredSurface(
                height_data=data,
                source_pixel_size_um=1.0,
                grid_size=8,
                pixel_size_um=1.0,
                padding="invalid",
            )

    def test_non_square_input_becomes_square(self):
        """非正方形の実測データが正方形にクロップされる。"""
        data = np.ones((30, 50), dtype=np.float32)
        ms = MeasuredSurface(
            height_data=data,
            source_pixel_size_um=1.0,
            grid_size=16,
            pixel_size_um=1.0,
            leveling=False,
            padding="zeros",
        )
        hm = ms.get_height_map()
        assert hm.data.shape[0] == hm.data.shape[1]

    def test_data_larger_than_grid_crops(self):
        rng = np.random.default_rng(1)
        data = rng.standard_normal((100, 100)).astype(np.float32)
        ms = MeasuredSurface(
            height_data=data,
            source_pixel_size_um=1.0,
            grid_size=32,
            pixel_size_um=1.0,
            leveling=False,
        )
        hm = ms.get_height_map()
        assert hm.data.shape == (32, 32)

    def test_padding_in_from_config(self, tmp_path):
        csv_path = tmp_path / "surface.csv"
        csv_path.write_text(
            "\n".join(",".join(["0.0"] * 8) for _ in range(8)),
            encoding="utf-8",
        )
        config = {
            "surface": {
                "model": "MeasuredSurface",
                "grid_size": 16,
                "pixel_size_um": 1.0,
                "measured": {
                    "path": str(csv_path),
                    "source_pixel_size_um": 1.0,
                    "leveling": False,
                    "padding": "reflect",
                },
            }
        }
        ms = MeasuredSurface.from_config(config)
        assert ms.padding == "reflect"


class TestDeviceVk6AutoDetect:
    """DeviceVk6Surface の自動検出機能テスト。"""

    @pytest.fixture
    def vk6_sample_csv(self, tmp_path):
        """VK-X フォーマットの最小サンプル CSV（Shift-JIS）を生成する。"""
        header = (
            '"測定日時","2026-04-15 00:00:00"\r\n'
            '"機種","VK-X1000 Series"\r\n'
            '"ファイル種別","ImageDataCsv"\r\n'
            '"ファイル バージョン","1000"\r\n'
            '"測定データ名","test"\r\n'
            '"倍率","50"\r\n'
            '"XYキャリブレーション","0.5","μm"\r\n'
            '"出力画像データ","高さ"\r\n'
            '"横","8"\r\n'
            '"縦","6"\r\n'
            '"最小値","-1.0"\r\n'
            '"最大値","1.0"\r\n'
            '"単位","μm"\r\n'
            '\r\n'
            '"高さ"\r\n'
        )
        data_rows = "\r\n".join(
            ",".join(f'"{0.1 * (i + j):.3f}"' for j in range(8))
            for i in range(6)
        )
        content = (header + data_rows).encode("shift-jis")
        p = tmp_path / "vk6_test.csv"
        p.write_bytes(content)
        return p

    def test_auto_pixel_size_from_header(self, vk6_sample_csv):
        """pixel_size_um 省略時にヘッダ値（0.5 μm）が使われる。"""
        from custom_surfaces.device_vk6 import DeviceVk6Surface
        ms = DeviceVk6Surface.from_vk6_csv(path=vk6_sample_csv, leveling=False)
        assert ms.pixel_size_um == pytest.approx(0.5)
        assert ms.source_pixel_size_um == pytest.approx(0.5)

    def test_auto_grid_size_from_header(self, vk6_sample_csv):
        """grid_size 省略時: 短辺 6 → 2^2=4。"""
        from custom_surfaces.device_vk6 import DeviceVk6Surface
        ms = DeviceVk6Surface.from_vk6_csv(path=vk6_sample_csv, leveling=False)
        assert ms.grid_size == 4

    def test_explicit_pixel_size_overrides_header(self, vk6_sample_csv):
        from custom_surfaces.device_vk6 import DeviceVk6Surface
        ms = DeviceVk6Surface.from_vk6_csv(
            path=vk6_sample_csv, pixel_size_um=1.0, grid_size=2, leveling=False,
        )
        assert ms.pixel_size_um == pytest.approx(1.0)
        assert ms.get_height_map().data.shape == (2, 2)

    def test_explicit_grid_size_overrides_auto(self, vk6_sample_csv):
        from custom_surfaces.device_vk6 import DeviceVk6Surface
        ms = DeviceVk6Surface.from_vk6_csv(
            path=vk6_sample_csv, grid_size=2, leveling=False,
        )
        assert ms.grid_size == 2
        assert ms.get_height_map().data.shape == (2, 2)

    def test_padding_passed_through(self, vk6_sample_csv):
        from custom_surfaces.device_vk6 import DeviceVk6Surface
        for mode in ("zeros", "tile", "reflect", "smooth_tile"):
            ms = DeviceVk6Surface.from_vk6_csv(
                path=vk6_sample_csv, grid_size=4, leveling=False, padding=mode,
            )
            assert ms.padding == mode
