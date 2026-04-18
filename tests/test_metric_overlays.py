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
