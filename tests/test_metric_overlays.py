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
        # Image + beam window + 5 幅分 Imax 縞 = 7
        assert len(list(result)) == 7

    def test_no_stripes(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_doi_comb_2d
        result = overlay_doi_comb_2d(bsdf_heatmap, show_stripes=False)
        # Image + beam window のみ
        assert len(list(result)) == 2

    def test_custom_widths(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_doi_comb_2d
        result = overlay_doi_comb_2d(
            bsdf_heatmap, comb_widths_mm=[0.5, 1.0], distance_mm=280.0,
        )
        # Image + beam window + 2 幅 = 4
        assert len(list(result)) == 4

    def test_imin_phase(self, bsdf_heatmap):
        from bsdf_sim.visualization.metric_overlays import overlay_doi_comb_2d
        result = overlay_doi_comb_2d(
            bsdf_heatmap, comb_widths_mm=[1.0], distance_mm=280.0,
            show_stripes=True, show_imin_phase=True,
        )
        # Image + window + Imax + Imin = 4
        assert len(list(result)) == 4


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
            "doi_comb": {
                "enabled": True, "scan_half_angle_deg": 4.0,
                "comb_widths_mm": [0.125, 0.25, 0.5, 1.0, 2.0],
            },
            "doi_astm": {"enabled": True, "offset_deg": 0.3},
        }
        result = overlay_all_metrics_2d(
            bsdf_heatmap, metrics_config=cfg, theta_i_deg=0.0, mode="BTDF",
        )
        assert isinstance(result, hv.Overlay)
        # Image(1) + Haze(1) + Gloss60(1) + NSER(2) + COMB band(1) + 縞 5 幅(5) + ASTM(3) = 14
        assert len(list(result)) == 14

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
            show_stripes=False,
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

    def test_comb_stripe_period(self, bsdf_heatmap):
        """各くし幅の明スリット数と周期が理論値と一致する。"""
        from bsdf_sim.visualization.metric_overlays import overlay_doi_comb_2d
        # d=1.0mm, distance=280mm: period_u = 2/280 ≈ 0.00714
        # u_half = sin(4°) ≈ 0.0698 → k_max = ceil(0.0698/0.00714)+1 = 11 → 明 23 本相当
        # （端のクリップで幅 0 になるものは除外されうる）
        result = overlay_doi_comb_2d(
            bsdf_heatmap, u_center=0.0, scan_half_angle_deg=4.0,
            comb_widths_mm=[1.0], distance_mm=280.0, show_stripes=True,
        )
        rects = _rectangles(result)
        assert len(rects) == 2  # band + 1 幅分
        stripe_rect = rects[1]
        period_u = 2.0 * 1.0 / 280.0
        # 中心間隔が period_u と一致（連続 2 スリットの中心差）
        centers = sorted(
            [(row["x0"] + row["x1"]) / 2 for _, row in stripe_rect.data.iterrows()]
        )
        diffs = np.diff(centers)
        # 端のクリップ区間を除き、内部は period_u
        interior_diffs = diffs[1:-1] if len(diffs) > 2 else diffs
        assert np.allclose(interior_diffs, period_u, atol=1e-9)

    def test_comb_imin_phase_offset(self, bsdf_heatmap):
        """Imin 位相 = Imax 位相から half period ずれた位置に明スリット中心がある。"""
        from bsdf_sim.visualization.metric_overlays import overlay_doi_comb_2d
        u_half = float(np.sin(np.deg2rad(4.0)))
        period_u = 2.0 * 1.0 / 280.0
        result = overlay_doi_comb_2d(
            bsdf_heatmap, u_center=0.0, scan_half_angle_deg=4.0,
            comb_widths_mm=[1.0], distance_mm=280.0,
            show_stripes=True, show_imin_phase=True,
        )
        rects = _rectangles(result)
        # [window, Imax, Imin]
        assert len(rects) == 3
        # 端のクリップを避けるため、|center| < u_half - period_u の内部スリットのみ比較
        interior_bound = u_half - period_u
        imax_centers = sorted(
            c for _, row in rects[1].data.iterrows()
            if abs((c := (row["x0"] + row["x1"]) / 2)) < interior_bound
        )
        imin_centers = sorted(
            c for _, row in rects[2].data.iterrows()
            if abs((c := (row["x0"] + row["x1"]) / 2)) < interior_bound
        )
        # いずれの Imin 中心も最寄りの Imax 中心から period/2 ずれている
        half = period_u / 2.0
        for c_min in imin_centers:
            nearest_max = min(imax_centers, key=lambda c: abs(c - c_min))
            assert abs(abs(c_min - nearest_max) - half) < 1e-9

    def test_comb_stripe_duty_50pct(self, bsdf_heatmap):
        """明スリット幅 = period/2（duty 50%）。"""
        from bsdf_sim.visualization.metric_overlays import overlay_doi_comb_2d
        result = overlay_doi_comb_2d(
            bsdf_heatmap, u_center=0.0, scan_half_angle_deg=4.0,
            comb_widths_mm=[2.0], distance_mm=280.0, show_stripes=True,
        )
        stripe_rect = _rectangles(result)[1]
        period_u = 2.0 * 2.0 / 280.0
        # 非クリップの内部区間を検出（幅 = period_u/2）
        for _, row in stripe_rect.data.iterrows():
            width = row["x1"] - row["x0"]
            # クリップされた端の区間は幅が小さいので、内部の全幅スリットだけチェック
            if width > period_u / 2.0 * 0.99:
                assert width == pytest.approx(period_u / 2.0, abs=1e-9)

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


class TestPlotBsdf2DHeatmapWithOverlay:
    """plot_bsdf_2d_heatmap の metrics_config / metric_overlay_config 引数の統合テスト。"""

    def _uv_bsdf(self, n=33):
        u_axis = np.linspace(-1.0, 1.0, n)
        v_axis = np.linspace(-1.0, 1.0, n)
        U, V = np.meshgrid(u_axis, v_axis, indexing="ij")
        bsdf = np.exp(-(U**2 + V**2) / 0.1)
        return U, V, bsdf

    def test_no_overlay_when_configs_absent(self):
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_2d_heatmap
        U, V, bsdf = self._uv_bsdf()
        result = plot_bsdf_2d_heatmap(U, V, bsdf)
        # Image + 半球境界円 = 2 のまま（overlay 追加なし）
        assert len(list(result)) == 2

    def test_overlay_applied_when_show_true(self):
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_2d_heatmap
        U, V, bsdf = self._uv_bsdf()
        result = plot_bsdf_2d_heatmap(
            U, V, bsdf,
            metrics_config={"haze": {"enabled": True}},
            metric_overlay_config={"show_overlay": True, "click_policy": "hide"},
        )
        # 2 (base) + Haze (1) = 3
        assert len(list(result)) == 3

    def test_overlay_disabled_when_show_false(self):
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_2d_heatmap
        U, V, bsdf = self._uv_bsdf()
        result = plot_bsdf_2d_heatmap(
            U, V, bsdf,
            metrics_config={"haze": {"enabled": True}},
            metric_overlay_config={"show_overlay": False},
        )
        # overlay が効かず 2 要素のまま
        assert len(list(result)) == 2

    def test_theta_i_shifts_haze_center(self):
        """BRDF θ_i=30° のとき Haze 中心が sin(30°) に移動する。"""
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_2d_heatmap
        U, V, bsdf = self._uv_bsdf()
        result = plot_bsdf_2d_heatmap(
            U, V, bsdf,
            metrics_config={"haze": {"enabled": True}},
            metric_overlay_config={"show_overlay": True},
            theta_i_deg=30.0, mode="BRDF",
        )
        ellipses = [item for item in result if isinstance(item, hv.Ellipse)]
        # Haze 円の中心が sin(30°) ≈ 0.5 に移動
        assert len(ellipses) == 1
        assert ellipses[0].x == pytest.approx(float(np.sin(np.deg2rad(30.0))))


class TestPlotBsdf2DHeatmapClim:
    """plot_bsdf_2d_heatmap の clim 引数（カラーバー範囲固定）の統合テスト。"""

    def _uv_bsdf(self, n=17):
        u_axis = np.linspace(-1.0, 1.0, n)
        v_axis = np.linspace(-1.0, 1.0, n)
        U, V = np.meshgrid(u_axis, v_axis, indexing="ij")
        bsdf = np.exp(-(U**2 + V**2) / 0.1)
        return U, V, bsdf

    def _extract_image_opts(self, result):
        """Overlay/Image から hv.Image を取り出し opts 辞書を返す。"""
        img = None
        for item in result if hasattr(result, "__iter__") else [result]:
            if isinstance(item, hv.Image):
                img = item
                break
        assert img is not None, "hv.Image が見つからない"
        return img.opts.get("plot").kwargs | img.opts.get("style").kwargs

    def test_clim_none_leaves_image_without_fixed_range(self):
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_2d_heatmap
        U, V, bsdf = self._uv_bsdf()
        result = plot_bsdf_2d_heatmap(U, V, bsdf, clim=None, log_scale=True)
        opts = self._extract_image_opts(result)
        assert "clim" not in opts

    def test_clim_linear_passes_values_directly(self):
        """log_scale=False のとき clim は raw 値がそのまま渡る。"""
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_2d_heatmap
        U, V, bsdf = self._uv_bsdf()
        result = plot_bsdf_2d_heatmap(
            U, V, bsdf, log_scale=False, clim=(0.0, 1.0),
        )
        opts = self._extract_image_opts(result)
        assert opts["clim"] == pytest.approx((0.0, 1.0))

    def test_clim_log_applies_log10_conversion(self):
        """log_scale=True のとき clim は log10 変換されて適用される。"""
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_2d_heatmap
        U, V, bsdf = self._uv_bsdf()
        result = plot_bsdf_2d_heatmap(
            U, V, bsdf, log_scale=True, clim=(1e-6, 1e2),
        )
        opts = self._extract_image_opts(result)
        vmin, vmax = opts["clim"]
        assert vmin == pytest.approx(-6.0)
        assert vmax == pytest.approx(2.0)

    def test_clim_log_floors_non_positive(self):
        """log_scale=True で clim に 0 / 負値が来ても BSDF_LOG_FLOOR_DEFAULT で
        クリップされ log10 が NaN や -inf にならない。"""
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_2d_heatmap
        U, V, bsdf = self._uv_bsdf()
        result = plot_bsdf_2d_heatmap(
            U, V, bsdf, log_scale=True, clim=(0.0, 1.0),
        )
        opts = self._extract_image_opts(result)
        vmin, vmax = opts["clim"]
        assert np.isfinite(vmin) and np.isfinite(vmax)
        # 0.0 → log10(1e-10) = -10 にクリップ
        assert vmin == pytest.approx(-10.0)
        assert vmax == pytest.approx(0.0)


class TestOverlayDOIComb1D:
    def test_stripes_added(self):
        from bsdf_sim.visualization.metric_overlays import overlay_doi_comb_1d
        curve = hv.Curve([(0, 1), (1, 0.5), (2, 0.1)], kdims=["θ_s"], vdims=["BSDF"])
        result = overlay_doi_comb_1d(
            curve, theta_axis_deg=0.0, scan_half_angle_deg=4.0,
            comb_widths_mm=[0.125, 0.25, 0.5, 1.0, 2.0],
        )
        # curve + 5 幅 = 6 要素
        assert len(list(result)) == 6

    def test_stripe_period_in_deg(self):
        from bsdf_sim.visualization.metric_overlays import overlay_doi_comb_1d
        curve = hv.Curve([(0, 1)], kdims=["θ_s"], vdims=["BSDF"])
        result = overlay_doi_comb_1d(
            curve, theta_axis_deg=0.0, scan_half_angle_deg=4.0,
            comb_widths_mm=[1.0], distance_mm=280.0,
        )
        rects = _rectangles(result)
        assert len(rects) == 1
        period_deg = float(np.rad2deg(2.0 * 1.0 / 280.0))
        centers = sorted(
            [(row["x0"] + row["x1"]) / 2 for _, row in rects[0].data.iterrows()]
        )
        diffs = np.diff(centers)
        # 内部の周期が period_deg と一致
        interior = diffs[1:-1] if len(diffs) > 2 else diffs
        assert np.allclose(interior, period_deg, atol=1e-9)


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
