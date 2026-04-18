"""L3'/L4/L5 拡張 sparkle 計算のテスト。

対象: src/bsdf_sim/metrics/sparkle_extended.py
"""

import numpy as np
import pytest

from bsdf_sim.metrics.sparkle_extended import (
    _COLOR_WAVELENGTHS_UM,
    _cs_from_luminance,
    _generate_subpixel_mask,
    _v_lambda,
    compute_sparkle_l3prime,
    compute_sparkle_l4,
    compute_sparkle_l5,
)
from bsdf_sim.models.base import HeightMap


# ── フィクスチャ ──


@pytest.fixture
def sparkle_config():
    """smartphone + FHD プリセット相当の config dict。"""
    return {
        "viewing": {"distance_mm": 300.0, "pupil_diameter_mm": 3.0},
        "display": {"pixel_pitch_mm": 0.062, "subpixel_layout": "rgb_stripe"},
    }


@pytest.fixture
def sparkle_config_small_pixel():
    """L5 テスト用の小ピクセル設定（計算量削減のため dx を大きく取り、
    pixel_pitch_um / dx の比率を低減）。"""
    return {
        "viewing": {"distance_mm": 300.0, "pupil_diameter_mm": 3.0},
        "display": {"pixel_pitch_mm": 0.020, "subpixel_layout": "rgb_stripe"},  # 高 PPI 相当
    }


@pytest.fixture
def flat_height_map():
    """平坦 (h=0) の HeightMap。"""
    N = 128
    return HeightMap(data=np.zeros((N, N), dtype=np.float64), pixel_size_um=0.25)


@pytest.fixture
def random_height_map():
    """ランダム粗面（シード固定）。sparkle > 0 を期待。"""
    rng = np.random.default_rng(42)
    N = 128
    h = rng.normal(0.0, 0.05, size=(N, N))  # Rq ≈ 0.05 μm
    return HeightMap(data=h, pixel_size_um=0.25)


# ── ヘルパー関数テスト ──


class TestSubpixelMask:
    def test_rgb_stripe_coverage(self):
        """RGB ストライプで R/G/B それぞれ画素の 1/3 を占めることを確認。"""
        N = 60
        pixel_size_um = 1.0
        pixel_pitch_um = 15.0  # 15 サンプル = 1 ディスプレイ画素
        masks = {
            c: _generate_subpixel_mask(N, pixel_size_um, pixel_pitch_um, "rgb_stripe", c)
            for c in ("R", "G", "B")
        }
        total = masks["R"] + masks["G"] + masks["B"]
        # 全体がほぼ 1.0（各画素が R+G+B で埋まる）
        np.testing.assert_allclose(total, np.ones_like(total))
        # 各色の占有率が 1/3（整数サンプル分割ならちょうど）
        for c in ("R", "G", "B"):
            assert abs(masks[c].mean() - 1.0 / 3) < 0.02

    def test_rgb_stripe_order(self):
        """RGB ストライプの順序が R→G→B であることを確認。"""
        N = 12
        mask_r = _generate_subpixel_mask(N, 1.0, 3.0, "rgb_stripe", "R")
        mask_g = _generate_subpixel_mask(N, 1.0, 3.0, "rgb_stripe", "G")
        mask_b = _generate_subpixel_mask(N, 1.0, 3.0, "rgb_stripe", "B")
        # 最初の画素（x=0,1,2）の 1 行目
        assert mask_r[0, 0] == 1.0 and mask_g[0, 0] == 0.0 and mask_b[0, 0] == 0.0
        assert mask_r[0, 1] == 0.0 and mask_g[0, 1] == 1.0 and mask_b[0, 1] == 0.0
        assert mask_r[0, 2] == 0.0 and mask_g[0, 2] == 0.0 and mask_b[0, 2] == 1.0

    def test_bgr_stripe_order(self):
        """BGR ストライプは B→G→R の順であることを確認。"""
        N = 12
        mask_r = _generate_subpixel_mask(N, 1.0, 3.0, "bgr_stripe", "R")
        mask_b = _generate_subpixel_mask(N, 1.0, 3.0, "bgr_stripe", "B")
        assert mask_b[0, 0] == 1.0 and mask_r[0, 0] == 0.0
        assert mask_b[0, 2] == 0.0 and mask_r[0, 2] == 1.0

    def test_invalid_color_raises(self):
        with pytest.raises(ValueError, match="color"):
            _generate_subpixel_mask(10, 1.0, 3.0, "rgb_stripe", "X")  # type: ignore[arg-type]

    def test_invalid_layout_raises(self):
        with pytest.raises(ValueError, match="subpixel_layout"):
            _generate_subpixel_mask(10, 1.0, 3.0, "invalid", "R")  # type: ignore[arg-type]


class TestVLambda:
    def test_peak_at_555nm(self):
        """V(λ) のピークは 555 nm 付近で 1.0。"""
        assert _v_lambda(0.555) == pytest.approx(1.0, abs=0.01)

    def test_rgb_values_reasonable(self):
        """R=630, G=525, B=465 nm での V(λ) が既知の範囲にあるか。"""
        assert 0.25 < _v_lambda(0.630) < 0.40  # R ≈ 0.27
        assert 0.75 < _v_lambda(0.525) < 0.85  # G ≈ 0.79
        assert 0.07 < _v_lambda(0.465) < 0.15  # B ≈ 0.10

    def test_outside_visible_returns_zero(self):
        assert _v_lambda(0.300) == 0.0
        assert _v_lambda(0.900) == 0.0


class TestCsFromLuminance:
    def test_uniform_returns_zero(self):
        """全画素が同輝度ならば σ/μ = 0。"""
        assert _cs_from_luminance(np.ones(100)) == 0.0

    def test_zero_mean_returns_zero(self):
        """平均がゼロに近い場合は 0 を返す（数値安定性）。"""
        assert _cs_from_luminance(np.zeros(100)) == 0.0

    def test_known_variance(self):
        """既知の分布で σ/μ の値を検証。"""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = arr.std() / arr.mean()
        assert _cs_from_luminance(arr) == pytest.approx(expected)

    def test_short_array_returns_zero(self):
        assert _cs_from_luminance(np.array([1.0])) == 0.0


# ── L3' ──


class TestSparkleL3Prime:
    def test_flat_surface_finite_cs(self, flat_height_map, sparkle_config):
        """平坦面でも Cs は有限値を返す（DC 集中 + サブピクセル Moiré 起因の高い
        apparent Cs になるが、NaN や inf にはならない）。"""
        cs = compute_sparkle_l3prime(flat_height_map, "G", sparkle_config)
        assert cs >= 0.0
        assert np.isfinite(cs)

    def test_random_surface_positive_sparkle(self, random_height_map, sparkle_config):
        """ランダム粗面では sparkle > 0。"""
        cs = compute_sparkle_l3prime(random_height_map, "G", sparkle_config)
        assert cs > 0.0

    def test_default_wavelength_by_color(self, random_height_map, sparkle_config):
        """wavelength_um=None で色別デフォルト波長が使われる。"""
        cs_g_default = compute_sparkle_l3prime(random_height_map, "G", sparkle_config)
        cs_g_explicit = compute_sparkle_l3prime(
            random_height_map, "G", sparkle_config, wavelength_um=_COLOR_WAVELENGTHS_UM["G"]
        )
        assert cs_g_default == pytest.approx(cs_g_explicit)

    def test_different_colors_different_sparkle(self, random_height_map, sparkle_config):
        """R/G/B で通常異なる sparkle 値が得られる（波長依存 + マスク位相依存）。"""
        values = {
            c: compute_sparkle_l3prime(random_height_map, c, sparkle_config)
            for c in ("R", "G", "B")
        }
        # 少なくとも 2 色で異なる値
        vs = list(values.values())
        assert max(vs) - min(vs) > 1e-6

    def test_invalid_color_raises(self, random_height_map, sparkle_config):
        with pytest.raises(KeyError):
            compute_sparkle_l3prime(random_height_map, "X", sparkle_config)  # type: ignore[arg-type]


# ── L4 ──


class TestSparkleL4:
    def test_returns_finite_positive(self, random_height_map, sparkle_config):
        """L4 白点灯で有限の非負値が得られる。"""
        cs = compute_sparkle_l4(random_height_map, sparkle_config)
        assert np.isfinite(cs)
        assert cs >= 0.0

    def test_white_differs_from_single_color(self, random_height_map, sparkle_config):
        """L4 白点灯 Cs は L3' 単色 Cs と一般に異なる（σ/μ は非線形）。"""
        cs_white = compute_sparkle_l4(random_height_map, sparkle_config)
        cs_g = compute_sparkle_l3prime(random_height_map, "G", sparkle_config)
        # 明示的な等式不成立を確認
        assert abs(cs_white - cs_g) > 1e-6

    def test_only_green_equivalent_to_l3prime(self, random_height_map, sparkle_config):
        """R,B の光源強度を 0 にすれば L4 は L3' (G) と数値的に一致する。"""
        cs_white_g_only = compute_sparkle_l4(
            random_height_map, sparkle_config,
            source_intensity={"R": 0.0, "G": 1.0, "B": 0.0},
        )
        cs_g = compute_sparkle_l3prime(random_height_map, "G", sparkle_config)
        # 重み（V(λ_G)）は σ/μ を変えないためキャンセル
        assert cs_white_g_only == pytest.approx(cs_g, rel=1e-9)

    def test_zero_source_returns_zero(self, random_height_map, sparkle_config):
        """全光源 0 なら Cs=0 を返す。"""
        cs = compute_sparkle_l4(
            random_height_map, sparkle_config,
            source_intensity={"R": 0.0, "G": 0.0, "B": 0.0},
        )
        assert cs == 0.0


# ── L5 ──


class TestSparkleL5:
    @pytest.fixture
    def large_height_map(self):
        """L5 は窓サイズ確保のため大きめグリッドが必要。dx=0.25μm, N=512 で
        small_pixel config (p=20μm) の window 3x = 240 サンプルが収まる。"""
        rng = np.random.default_rng(42)
        N = 512
        h = rng.normal(0.0, 0.05, size=(N, N))
        return HeightMap(data=h, pixel_size_um=0.25)

    def test_returns_finite_positive(self, large_height_map, sparkle_config_small_pixel):
        """L5 で有限の非負値が得られる。"""
        cs = compute_sparkle_l5(large_height_map, "G", sparkle_config_small_pixel)
        assert np.isfinite(cs)
        assert cs >= 0.0

    def test_flat_surface_low_sparkle(self, sparkle_config_small_pixel):
        """平坦面で L5 は小さい値（窓内で位相が均一 → 画素間で同等 → σ≈0）。
        L5 は L1/L3' と違い局所 FFT で DC 集中アーティファクトを回避するため、
        平坦面では本来的に Cs ≈ 0 となる。"""
        N = 512
        hm = HeightMap(data=np.zeros((N, N)), pixel_size_um=0.25)
        cs = compute_sparkle_l5(hm, "G", sparkle_config_small_pixel)
        # 平坦面は全画素で同じローカル BSDF → σ/μ は数値精度レベル
        assert cs < 1e-6

    def test_small_grid_raises(self, sparkle_config):
        """グリッドサイズが窓幅に満たない場合にエラー（smartphone デフォルト）。"""
        hm = HeightMap(data=np.zeros((32, 32)), pixel_size_um=0.25)
        with pytest.raises(ValueError, match="窓幅"):
            compute_sparkle_l5(hm, "G", sparkle_config)

    def test_window_size_factor_affects_result(
        self, large_height_map, sparkle_config_small_pixel
    ):
        """窓幅を変えると sparkle 値が変わる（収束性テストの基礎）。"""
        cs_3 = compute_sparkle_l5(
            large_height_map, "G", sparkle_config_small_pixel, window_size_factor=3.0
        )
        cs_5 = compute_sparkle_l5(
            large_height_map, "G", sparkle_config_small_pixel, window_size_factor=5.0
        )
        # 窓幅変更で値が変わるはず（完全一致はしない）
        assert abs(cs_3 - cs_5) > 1e-9

    def test_pupil_integration_equals_dc_for_small_pupil(
        self, large_height_map, sparkle_config_small_pixel
    ):
        """窓 FFT 分解能より pupil が小さい場合、pupil 積分は DC 1 点と一致する。
        smartphone 3×20μm 窓で u_pupil=0.005 < du=0.008 となり 1 サンプルに退化。"""
        cs_pup = compute_sparkle_l5(
            large_height_map, "G", sparkle_config_small_pixel,
            window_size_factor=3.0, pupil_integration=True,
        )
        cs_dc = compute_sparkle_l5(
            large_height_map, "G", sparkle_config_small_pixel,
            window_size_factor=3.0, pupil_integration=False,
        )
        # 小さい pupil では数値的にほぼ一致（自動 fallback が働く）
        assert cs_pup == pytest.approx(cs_dc, rel=1e-6)

    def test_pupil_integration_differs_for_large_window(self, large_height_map):
        """窓幅を十分大きくして FFT 分解能を pupil より細かくすると、pupil 積分と
        DC 1 点で値が異なることを確認する。"""
        # p=100μm、d_p=3mm、D=300mm → u_pupil=0.005
        # 窓 window_size_factor=5 → W=500μm → du=λ/W ≈ 0.001 → pupil 内に ~5x5 サンプル
        cfg = {
            "viewing": {"distance_mm": 300.0, "pupil_diameter_mm": 3.0},
            "display": {"pixel_pitch_mm": 0.100, "subpixel_layout": "rgb_stripe"},
        }
        # large_height_map (N=512, dx=0.25μm) → 500μm 窓 = 2000 samples > N → 不可
        # 代わりに dx=1μm でスケールアップ
        hm = HeightMap(
            data=large_height_map.data, pixel_size_um=1.0
        )  # dx=1μm, 物理サイズ 512μm、pitch=100μm なら 5 画素
        cs_pup = compute_sparkle_l5(
            hm, "G", cfg, window_size_factor=3.0, pupil_integration=True
        )
        cs_dc = compute_sparkle_l5(
            hm, "G", cfg, window_size_factor=3.0, pupil_integration=False
        )
        # pupil 積分と DC 1 点で異なる値になる
        assert abs(cs_pup - cs_dc) > 1e-6
