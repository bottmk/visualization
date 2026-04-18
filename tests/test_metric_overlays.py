"""光学指標の BSDF オーバーレイ描画テスト（Phase 1）。

tests the functions in src/bsdf_sim/visualization/metric_overlays.py:
- overlay_haze_2d
- overlay_gloss_2d
- overlay_doi_nser_2d
- overlay_all_metrics_2d
"""

import numpy as np
import pytest

try:
    import holoviews as hv
    hv.extension("bokeh")
    _HV_OK = True
except ImportError:
    _HV_OK = False

pytestmark = pytest.mark.skipif(not _HV_OK, reason="HoloViews 未インストール")


@pytest.fixture
def bsdf_heatmap():
    """テスト用の BSDF 2D ヒートマップ（hv.Image）。"""
    N = 33
    u_axis = np.linspace(-1.0, 1.0, N)
    v_axis = np.linspace(-1.0, 1.0, N)
    u, v = np.meshgrid(u_axis, v_axis, indexing="ij")
    uv_r2 = u**2 + v**2
    bsdf = np.exp(-uv_r2 / 0.1) * 1.0
    bsdf[uv_r2 > 1.0] = 0.0
    # log scale を想定した Image
    data = np.log10(np.maximum(bsdf, 1e-6))
    img = hv.Image(
        (u_axis, v_axis, data.T),
        kdims=["u", "v"],
        vdims=["log10 BSDF"],
    )
    return img


class TestOverlayHaze2D:
    def test_returns_overlay(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_haze_2d
        result = overlay_haze_2d(bsdf_heatmap, u_center=0.0, v_center=0.0, half_angle_deg=2.5)
        assert isinstance(result, hv.Overlay)
        # Image + Ellipse の 2 要素
        assert len(list(result)) == 2

    def test_custom_style(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_haze_2d
        # スタイル上書きが例外なく通ること
        result = overlay_haze_2d(
            bsdf_heatmap, style={"color": "red", "line_width": 3}, label="Custom",
        )
        assert isinstance(result, hv.Overlay)


class TestOverlayGloss2D:
    @pytest.mark.parametrize("angle", [20, 60, 85])
    def test_standard_angles(self, bsdf_heatmap, angle):
        from bsdf_sim.visualization.metric_overlays import overlay_gloss_2d
        result = overlay_gloss_2d(bsdf_heatmap, gloss_angle_deg=float(angle))
        assert isinstance(result, hv.Overlay)

    def test_u_center_auto(self, bsdf_heatmap):
        """u_center を省略すると sin(gloss_angle) になる。"""
        from bsdf_sim.visualization.metric_overlays import overlay_gloss_2d
        # 自動計算されること（例外を出さない）
        result = overlay_gloss_2d(bsdf_heatmap, gloss_angle_deg=60.0)
        assert isinstance(result, hv.Overlay)

    def test_aperture_override(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_gloss_2d
        result = overlay_gloss_2d(
            bsdf_heatmap, gloss_angle_deg=45.0,
            aperture_override={"in_plane_deg": 2.0, "cross_plane_deg": 4.0},
        )
        assert isinstance(result, hv.Overlay)


class TestOverlayDOINSER2D:
    def test_two_circles(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_doi_nser_2d
        result = overlay_doi_nser_2d(bsdf_heatmap)
        assert isinstance(result, hv.Overlay)
        # Image + 2 Ellipse = 3 要素
        assert len(list(result)) == 3

    def test_custom_angles(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_doi_nser_2d
        result = overlay_doi_nser_2d(
            bsdf_heatmap, direct_half_angle_deg=0.2, halo_half_angle_deg=3.0,
        )
        assert isinstance(result, hv.Overlay)


class TestOverlayDOIComb2D:
    def test_basic(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_doi_comb_2d
        result = overlay_doi_comb_2d(bsdf_heatmap)
        assert isinstance(result, hv.Overlay)
        # Image + band + 最大 4 本の pitch バー（±1, ±2 周期）
        assert len(list(result)) >= 2

    def test_no_pitch_bars(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_doi_comb_2d
        result = overlay_doi_comb_2d(bsdf_heatmap, show_pitch_bars=False)
        # Image + band のみ
        assert len(list(result)) == 2


class TestOverlayDOIAstm2D:
    def test_three_circles(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_doi_astm_2d
        result = overlay_doi_astm_2d(bsdf_heatmap)
        # Image + center + offset+ + offset- = 4 要素
        assert len(list(result)) == 4

    def test_custom_offset(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_doi_astm_2d
        result = overlay_doi_astm_2d(
            bsdf_heatmap, offset_deg=0.5, aperture_half_deg=0.1,
        )
        assert isinstance(result, hv.Overlay)


class TestOverlayAllMetrics2D:
    def test_all_enabled(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_all_metrics_2d
        cfg = {
            "haze": {"enabled": True, "half_angle_deg": 2.5},
            "gloss": {"enabled": True, "enabled_angles": [20, 60, 85]},
            "doi_nser": {"enabled": True},
        }
        result = overlay_all_metrics_2d(
            bsdf_heatmap, metrics_config=cfg, theta_i_deg=0.0, mode="BTDF",
        )
        assert isinstance(result, hv.Overlay)
        # Image + Haze(1) + Gloss 3 色(3) + NSER(2) = 7 要素
        assert len(list(result)) == 7

    def test_with_comb_and_astm(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_all_metrics_2d
        cfg = {
            "haze": {"enabled": True},
            "gloss": {"enabled": True, "enabled_angles": [60]},
            "doi_nser": {"enabled": True},
            "doi_comb": {"enabled": True, "scan_half_angle_deg": 4.0},
            "doi_astm": {"enabled": True, "offset_deg": 0.3},
        }
        result = overlay_all_metrics_2d(
            bsdf_heatmap, metrics_config=cfg, theta_i_deg=0.0, mode="BTDF",
        )
        assert isinstance(result, hv.Overlay)
        # 指標の積分領域が全て追加されていること（具体数は COMB の pitch 本数に依存）
        assert len(list(result)) >= 8

    def test_initially_shown_applies_visibility(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_all_metrics_2d
        cfg = {
            "haze": {"enabled": True},
            "gloss": {"enabled": True, "enabled_angles": [20, 60, 85]},
            "doi_nser": {"enabled": True},
        }
        result = overlay_all_metrics_2d(
            bsdf_heatmap, metrics_config=cfg,
            initially_shown=["haze", "gloss_60"],
        )
        # visible=False が 4 レイヤー（gloss_20 / gloss_85 / NSER inner / NSER outer）に
        # 適用されていることを確認
        hidden = 0
        for item in result:
            opts = item.opts.get()
            if "visible" in opts.kwargs and opts.kwargs["visible"] is False:
                hidden += 1
        assert hidden == 4

    def test_partial_enabled(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_all_metrics_2d
        cfg = {
            "haze": {"enabled": True},
            "gloss": {"enabled": False},
            "doi_nser": {"enabled": False},
        }
        result = overlay_all_metrics_2d(bsdf_heatmap, metrics_config=cfg)
        # Image + Haze = 2
        assert len(list(result)) == 2

    def test_click_policy_option(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_all_metrics_2d
        cfg = {"haze": {"enabled": True}}
        result = overlay_all_metrics_2d(
            bsdf_heatmap, metrics_config=cfg, click_policy="mute",
        )
        assert isinstance(result, hv.Overlay)

    def test_brdf_mode(self, bsdf_heatmap):
        """BRDF モードで specular u 中心がシフトする。"""
        from bsdf_sim.visualization.metric_overlays import overlay_all_metrics_2d
        cfg = {"doi_nser": {"enabled": True}}
        result = overlay_all_metrics_2d(
            bsdf_heatmap, metrics_config=cfg,
            theta_i_deg=30.0, mode="BRDF",
        )
        assert isinstance(result, hv.Overlay)


class TestOverlayFromConfig:
    def test_show_overlay_false(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_from_config
        full_cfg = {
            "visualization": {"metric_overlay": {"show_overlay": False}},
            "metrics": {"haze": {"enabled": True}},
        }
        result = overlay_from_config(bsdf_heatmap, full_cfg)
        # heatmap そのまま（Overlay に包まれていない）
        assert result is bsdf_heatmap

    def test_show_overlay_true(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_from_config
        full_cfg = {
            "visualization": {
                "metric_overlay": {
                    "show_overlay": True,
                    "initially_shown": ["haze"],
                    "click_policy": "mute",
                }
            },
            "metrics": {
                "haze": {"enabled": True},
                "gloss": {"enabled": True, "enabled_angles": [60]},
            },
        }
        result = overlay_from_config(bsdf_heatmap, full_cfg)
        assert isinstance(result, hv.Overlay)
        # Image + Haze + Gloss60 = 3 要素
        assert len(list(result)) == 3


def _ellipses(overlay):
    """Overlay から Ellipse 要素のリストを返す。"""
    return [item for item in overlay if isinstance(item, hv.Ellipse)]


def _rectangles(overlay):
    """Overlay から Rectangles 要素のリストを返す。"""
    return [item for item in overlay if isinstance(item, hv.Rectangles)]


class TestOverlayGeometry:
    """Phase 3: 描画要素の座標・寸法が仕様通りか検証。"""

    def test_haze_circle_radius(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_haze_2d
        result = overlay_haze_2d(bsdf_heatmap, u_center=0.0, v_center=0.0, half_angle_deg=2.5)
        els = _ellipses(result)
        assert len(els) == 1
        e = els[0]
        r_expected = float(np.sin(np.deg2rad(2.5)))
        assert e.x == pytest.approx(0.0)
        assert e.y == pytest.approx(0.0)
        # hv.Ellipse の width/height は直径
        assert e.width == pytest.approx(2 * r_expected)
        assert e.height == pytest.approx(2 * r_expected)

    @pytest.mark.parametrize("angle", [20, 60, 85])
    def test_gloss_rectangle_dims(self, bsdf_heatmap, angle):
        from bsdf_sim.metrics.optical import _GLOSS_APERTURES_DEG
        from bsdf_sim.visualization.metric_overlays import overlay_gloss_2d
        result = overlay_gloss_2d(bsdf_heatmap, gloss_angle_deg=float(angle))
        rects = _rectangles(result)
        assert len(rects) == 1
        row = rects[0].data.iloc[0]
        u_c = float(np.sin(np.deg2rad(angle)))
        ap = _GLOSS_APERTURES_DEG[angle]
        du_half = np.cos(np.deg2rad(angle)) * np.deg2rad(ap["in_plane_deg"] / 2.0)
        dv_half = np.deg2rad(ap["cross_plane_deg"] / 2.0)
        assert row["x0"] == pytest.approx(u_c - du_half)
        assert row["x1"] == pytest.approx(u_c + du_half)
        assert row["y0"] == pytest.approx(-dv_half)
        assert row["y1"] == pytest.approx(+dv_half)

    def test_nser_inner_outer_radii(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_doi_nser_2d
        result = overlay_doi_nser_2d(
            bsdf_heatmap, direct_half_angle_deg=0.1, halo_half_angle_deg=2.0,
        )
        els = _ellipses(result)
        assert len(els) == 2
        # 半径の小さい方が inner
        els_sorted = sorted(els, key=lambda e: e.width)
        r_in = float(np.sin(np.deg2rad(0.1)))
        r_out = float(np.sin(np.deg2rad(2.0)))
        assert els_sorted[0].width == pytest.approx(2 * r_in)
        assert els_sorted[1].width == pytest.approx(2 * r_out)

    def test_comb_band_rectangle(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_doi_comb_2d
        result = overlay_doi_comb_2d(
            bsdf_heatmap, u_center=0.0, v_center=0.0,
            scan_half_angle_deg=4.0, v_band_half_deg=0.2,
            show_pitch_bars=False,
        )
        rects = _rectangles(result)
        assert len(rects) == 1
        row = rects[0].data.iloc[0]
        u_half = float(np.sin(np.deg2rad(4.0)))
        v_half = float(np.sin(np.deg2rad(0.2)))
        assert row["x0"] == pytest.approx(-u_half)
        assert row["x1"] == pytest.approx(+u_half)
        assert row["y0"] == pytest.approx(-v_half)
        assert row["y1"] == pytest.approx(+v_half)

    def test_comb_pitch_period(self, bsdf_heatmap):
        """pitch バーは最小くし幅の周期位置 ±1, ±2 に出る（走査範囲内のみ）。"""
        from bsdf_sim.visualization.metric_overlays import overlay_doi_comb_2d
        result = overlay_doi_comb_2d(
            bsdf_heatmap, u_center=0.0, scan_half_angle_deg=4.0,
            comb_widths_mm=[0.125, 1.0], distance_mm=280.0, show_pitch_bars=True,
        )
        rects = _rectangles(result)
        # 1 本目 = band, 残りが pitch バー
        assert len(rects) >= 2
        period_u = 2.0 * 0.125 / 280.0
        u_half = np.sin(np.deg2rad(4.0))
        # 走査範囲内の k: |k·period_u| <= u_half → |k| <= ~895、よって ±1, ±2 全部入る
        pitch_rects = rects[1:]  # index 0 は band
        assert len(pitch_rects) == 4
        centers = sorted((r.data.iloc[0]["x0"] + r.data.iloc[0]["x1"]) / 2 for r in pitch_rects)
        expected = sorted([k * period_u for k in (-2, -1, 1, 2)])
        for c, e in zip(centers, expected):
            assert c == pytest.approx(e)
            assert abs(c) <= u_half

    def test_astm_three_circles(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_doi_astm_2d
        result = overlay_doi_astm_2d(
            bsdf_heatmap, u_center=0.0, v_center=0.0,
            offset_deg=0.3, aperture_half_deg=0.05,
        )
        els = _ellipses(result)
        assert len(els) == 3
        centers_u = sorted(e.x for e in els)
        sin_off = float(np.sin(np.deg2rad(0.3)))
        assert centers_u[0] == pytest.approx(-sin_off)
        assert centers_u[1] == pytest.approx(0.0)
        assert centers_u[2] == pytest.approx(+sin_off)
        # 全円の直径が aperture_half_deg に対応
        r_ap = float(np.sin(np.deg2rad(0.05)))
        for e in els:
            assert e.width == pytest.approx(2 * r_ap)

    def test_oblique_incidence_shifts_center(self, bsdf_heatmap):
        """BRDF θ_i=30° で Haze/NSER の中心が sin(30°) にシフトする。"""
        from bsdf_sim.visualization.metric_overlays import overlay_all_metrics_2d
        cfg = {"haze": {"enabled": True}, "doi_nser": {"enabled": True}}
        result = overlay_all_metrics_2d(
            bsdf_heatmap, metrics_config=cfg, theta_i_deg=30.0, mode="BRDF",
        )
        els = _ellipses(result)
        # 全要素（Haze 1 + NSER 2）が u=sin(30°) 中心
        u_expected = float(np.sin(np.deg2rad(30.0)))
        for e in els:
            assert e.x == pytest.approx(u_expected, abs=1e-9)
            assert e.y == pytest.approx(0.0)


class TestHazeBoundaryIn1DOverlay:
    """plot_bsdf_1d_overlay の show_haze_boundary オプションの挙動確認。"""

    def test_show_haze_boundary_adds_curve(self):
        import pandas as pd
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_1d_overlay

        df = pd.DataFrame({
            "method": ["FFT"] * 10,
            "wavelength_um": [0.555] * 10,
            "theta_i_deg": [0.0] * 10,
            "mode": ["BTDF"] * 10,
            "phi_s_deg": [0.0] * 10,
            "theta_s_deg": np.linspace(0, 45, 10),
            "bsdf": np.linspace(1.0, 0.001, 10),
        })
        # 境界なし
        plot_no = plot_bsdf_1d_overlay(df, show_haze_boundary=False)
        # 境界あり
        plot_yes = plot_bsdf_1d_overlay(df, show_haze_boundary=True)
        # 要素数が増えている（境界 Curve が追加）
        assert len(list(plot_yes)) > len(list(plot_no))
