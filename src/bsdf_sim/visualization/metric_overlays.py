"""光学指標の積分領域を BSDF 2D ヒートマップに重ね書きするオーバーレイ関数群。

docs/metric_bsdf_relation.md の可視化提案に基づく Phase 1 実装。
Phase 1 の対象: Haze / Gloss / DOI-NSER の 2D オーバーレイと統合関数。
（DOI-COMB / DOI-ASTM は Phase 2 で追加予定）

基本方針:
- 各指標の積分領域を色付き幾何要素（円・長方形）としてヒートマップに重ねる
- HoloViews + Bokeh 構成で `legend_click_policy` による表示/非表示切替に対応
- 色は `docs/metric_bsdf_relation.md` の配色スキームに準拠

API:
- overlay_haze_2d / overlay_gloss_2d / overlay_doi_nser_2d (個別)
- overlay_all_metrics_2d (config から自動展開)
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import holoviews as hv
    _HV_AVAILABLE = True
except ImportError:
    _HV_AVAILABLE = False

from ..metrics.optical import _GLOSS_APERTURES_DEG, _specular_u_center


# ── 配色設定 ─────────────────────────────────────────────────────────────────
# docs/metric_bsdf_relation.md の推奨スタイル（重ね書き時も識別可能に）

_DEFAULT_STYLES: dict[str, dict[str, Any]] = {
    "haze":     {"color": "white",   "line_dash": "dashed", "line_width": 2},
    "gloss_20": {"color": "cyan",    "line_dash": "solid",  "line_width": 2},
    "gloss_60": {"color": "yellow",  "line_dash": "solid",  "line_width": 2},
    "gloss_85": {"color": "magenta", "line_dash": "solid",  "line_width": 2},
    "doi_nser_inner": {"color": "#4fc3ff", "line_dash": "solid",  "line_width": 2},
    "doi_nser_outer": {"color": "#4fc3ff", "line_dash": "dashed", "line_width": 2},
}


def _check_hv() -> None:
    if not _HV_AVAILABLE:
        raise ImportError(
            "holoviews が必要。`pip install holoviews bokeh` でインストールしてください。"
        )


def _ellipse(u_c: float, v_c: float, half_angle_deg: float, **opts: Any) -> Any:
    """円（Ellipse）要素を作成。half_angle_deg は角度半径 [deg]。"""
    r = np.sin(np.deg2rad(half_angle_deg))
    # hv.Ellipse(x, y, (width, height)) で楕円。円なら width=height=2r
    return hv.Ellipse(u_c, v_c, (2 * r, 2 * r)).opts(**opts)


def _rectangle(
    u_c: float, v_c: float, du_half: float, dv_half: float, **opts: Any,
) -> Any:
    """長方形（Rectangles）要素を作成。du_half/dv_half は u,v 空間の半幅。"""
    return hv.Rectangles([(u_c - du_half, v_c - dv_half, u_c + du_half, v_c + dv_half)]).opts(
        fill_alpha=0.0,
        **opts,
    )


# ── 個別オーバーレイ ─────────────────────────────────────────────────────────


def overlay_haze_2d(
    heatmap: Any,
    u_center: float = 0.0,
    v_center: float = 0.0,
    half_angle_deg: float = 2.5,
    style: dict[str, Any] | None = None,
    label: str = "Haze 2.5°",
) -> Any:
    """Haze の 2.5° 境界円を重ねる。

    Args:
        heatmap: 既存の hv.Image / Overlay
        u_center, v_center: 直進光方向（規格 θ_i=0 なら 0, 0）
        half_angle_deg: 境界角 [deg]（デフォルト 2.5°）
        style: カスタムスタイル（color / line_dash / line_width）
        label: 凡例ラベル

    Returns:
        heatmap に Haze 境界円を重ねた Overlay
    """
    _check_hv()
    s = dict(_DEFAULT_STYLES["haze"])
    if style:
        s.update(style)
    circle = _ellipse(u_center, v_center, half_angle_deg, **s).relabel(label)
    return heatmap * circle


def overlay_gloss_2d(
    heatmap: Any,
    gloss_angle_deg: float,
    u_center: float | None = None,
    v_center: float = 0.0,
    aperture_override: dict[str, float] | None = None,
    style: dict[str, Any] | None = None,
    label: str | None = None,
) -> Any:
    """Gloss の規格長方形絞りを重ねる。

    Args:
        heatmap: 既存の hv.Image / Overlay
        gloss_angle_deg: グロス測定角（規格 20/60/85、その他は 60° の絞りを流用）
        u_center: None のとき sin(gloss_angle_deg) を使用（規格 BRDF 正反射位置）
        v_center: v 座標中心（通常 0）
        aperture_override: {"in_plane_deg":, "cross_plane_deg":} で絞り仕様上書き
        style: カスタムスタイル
        label: 凡例ラベル（デフォルト "Gloss <deg>°"）

    Returns:
        heatmap に Gloss 長方形を重ねた Overlay
    """
    _check_hv()
    if u_center is None:
        u_center = float(np.sin(np.deg2rad(gloss_angle_deg)))
    angle_int = int(round(gloss_angle_deg))
    ap = (
        aperture_override
        if aperture_override is not None
        else _GLOSS_APERTURES_DEG.get(angle_int, _GLOSS_APERTURES_DEG[60])
    )
    cos_sp = np.cos(np.deg2rad(gloss_angle_deg))
    du_half = cos_sp * np.deg2rad(ap["in_plane_deg"] / 2.0)
    dv_half = np.deg2rad(ap["cross_plane_deg"] / 2.0)

    # 角度別のデフォルト色
    default_key = f"gloss_{angle_int}" if angle_int in (20, 60, 85) else "gloss_60"
    s = dict(_DEFAULT_STYLES[default_key])
    if style:
        s.update(style)
    rect = _rectangle(u_center, v_center, du_half, dv_half, **s)
    if label is None:
        label = f"Gloss {angle_int}°"
    return heatmap * rect.relabel(label)


def overlay_doi_nser_2d(
    heatmap: Any,
    u_center: float = 0.0,
    v_center: float = 0.0,
    direct_half_angle_deg: float = 0.1,
    halo_half_angle_deg: float = 2.0,
    style_inner: dict[str, Any] | None = None,
    style_outer: dict[str, Any] | None = None,
) -> Any:
    """DOI-NSER の 2 重円（直進光 + ハロー）を重ねる。

    Args:
        heatmap: 既存の hv.Image / Overlay
        u_center, v_center: 直進光方向
        direct_half_angle_deg: 直進光コーン半角 [deg]
        halo_half_angle_deg: ハローコーン半角 [deg]
        style_inner, style_outer: 内/外円の個別スタイル

    Returns:
        heatmap に 2 重円を重ねた Overlay
    """
    _check_hv()
    s_in = dict(_DEFAULT_STYLES["doi_nser_inner"])
    if style_inner:
        s_in.update(style_inner)
    s_out = dict(_DEFAULT_STYLES["doi_nser_outer"])
    if style_outer:
        s_out.update(style_outer)

    inner = _ellipse(u_center, v_center, direct_half_angle_deg, **s_in).relabel(
        f"DOI-NSER inner {direct_half_angle_deg}°"
    )
    outer = _ellipse(u_center, v_center, halo_half_angle_deg, **s_out).relabel(
        f"DOI-NSER outer {halo_half_angle_deg}°"
    )
    return heatmap * inner * outer


# ── 統合オーバーレイ ─────────────────────────────────────────────────────────


def overlay_all_metrics_2d(
    heatmap: Any,
    metrics_config: dict[str, Any] | None,
    theta_i_deg: float = 0.0,
    mode: str = "BTDF",
    n1: float = 1.0,
    n2: float = 1.5,
    click_policy: str = "hide",
    initially_shown: list[str] | None = None,
    legend_position: str = "right",
) -> Any:
    """config.metrics で有効な指標を全て重ねる。

    Phase 1 対象: haze / gloss (enabled_angles) / doi_nser
    （doi_comb / doi_astm は Phase 2 で追加）

    Args:
        heatmap: 既存の hv.Image / Overlay
        metrics_config: config.yaml の metrics セクション辞書
        theta_i_deg, mode, n1, n2: 窓中心決定用の条件（`_specular_u_center` と同じ）
        click_policy: Bokeh の凡例クリック動作 ("hide" / "mute" / "none")
        initially_shown: 初期表示する指標キーのリスト（空/None なら全表示）
            使えるキー: "haze", "gloss_20", "gloss_60", "gloss_85", "doi_nser"
        legend_position: 凡例位置 ("right" / "top_right" / "bottom" 等)

    Returns:
        全指標オーバーレイ済みの Overlay（legend_click_policy 適用済み）
    """
    _check_hv()
    cfg = metrics_config or {}
    u_c = _specular_u_center(theta_i_deg, mode, n1, n2)
    v_c = 0.0

    result = heatmap
    shown = set(initially_shown) if initially_shown else None

    # Haze
    haze_cfg = cfg.get("haze")
    if haze_cfg is not None and haze_cfg.get("enabled", True):
        result = overlay_haze_2d(
            result, u_center=u_c, v_center=v_c,
            half_angle_deg=haze_cfg.get("half_angle_deg", 2.5),
        )

    # Gloss (各有効角度ごとに長方形を追加)
    gloss_cfg = cfg.get("gloss")
    if gloss_cfg is not None and gloss_cfg.get("enabled", True):
        for ang in gloss_cfg.get("enabled_angles", [20, 60, 85]):
            # Gloss は BRDF 規格、u_center は sin(ang)
            result = overlay_gloss_2d(
                result, gloss_angle_deg=float(ang),
                u_center=float(np.sin(np.deg2rad(ang))), v_center=0.0,
            )

    # DOI-NSER
    nser_cfg = cfg.get("doi_nser")
    if nser_cfg is not None and nser_cfg.get("enabled", True):
        result = overlay_doi_nser_2d(
            result, u_center=u_c, v_center=v_c,
            direct_half_angle_deg=nser_cfg.get("direct_half_angle_deg", 0.1),
            halo_half_angle_deg=nser_cfg.get("halo_half_angle_deg", 2.0),
        )

    # 初期表示制御（shown に無い系列を visible=False にする）は HoloViews 側で
    # 個別に opts をかけるのが難しいため、凡例クリックでの切替に委ねる。
    # （Phase 2 で layer ごとに opts(visible=...) を適用する拡張を検討）
    _ = shown  # 未使用（将来の拡張で initially_shown を反映）

    opts_kwargs: dict[str, Any] = {
        "show_legend": True,
        "legend_position": legend_position,
    }
    if click_policy != "none":
        # HoloViews (Bokeh backend) の Overlay オプション名は click_policy
        opts_kwargs["click_policy"] = click_policy

    return result.opts(**opts_kwargs)
