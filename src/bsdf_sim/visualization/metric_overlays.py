"""光学指標の積分領域を BSDF 2D ヒートマップに重ね書きするオーバーレイ関数群。

docs/metric_bsdf_relation.md の可視化提案に基づく Phase 1+2 実装。
対象: Haze / Gloss / DOI-NSER / DOI-COMB / DOI-ASTM の 2D オーバーレイと統合関数。

基本方針:
- 各指標の積分領域を色付き幾何要素（円・長方形）としてヒートマップに重ねる
- HoloViews + Bokeh 構成で `click_policy` による表示/非表示切替に対応
- `initially_shown` で各レイヤーの初期可視性を制御（visible=False）
- 色は `docs/metric_bsdf_relation.md` の配色スキームに準拠

API:
- overlay_haze_2d / overlay_gloss_2d / overlay_doi_nser_2d (個別)
- overlay_doi_comb_2d / overlay_doi_astm_2d (Phase 2)
- overlay_all_metrics_2d (config から自動展開、initially_shown 反映)
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
    "doi_comb_band":  {"color": "#ffb347", "line_dash": "solid",  "line_width": 2},
    "doi_comb_scan":  {"color": "#ffb347", "line_dash": "dashed", "line_width": 1},
    "doi_comb_pitch": {"color": "#ffb347", "line_dash": "dotted", "line_width": 1},
    "doi_astm_center": {"color": "#ff6b6b", "line_dash": "solid",  "line_width": 2},
    "doi_astm_offset": {"color": "#ff6b6b", "line_dash": "dashed", "line_width": 2},
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


def overlay_doi_comb_2d(
    heatmap: Any,
    u_center: float = 0.0,
    v_center: float = 0.0,
    scan_half_angle_deg: float = 4.0,
    v_band_half_deg: float = 0.2,
    comb_widths_mm: list[float] | None = None,
    distance_mm: float = 280.0,
    show_pitch_bars: bool = True,
    style_band: dict[str, Any] | None = None,
    style_scan: dict[str, Any] | None = None,
    style_pitch: dict[str, Any] | None = None,
) -> Any:
    """DOI-COMB（JIS K 7374 光学くし）の走査帯とくし周期を重ねる。

    Args:
        heatmap: 既存の hv.Image / Overlay
        u_center, v_center: 走査中心（通常 specular u_c, v=0）
        scan_half_angle_deg: 走査範囲の半角 [deg]（u 走査幅）
        v_band_half_deg: P(θ) 抽出時の v 軸バンド半角 [deg]
        comb_widths_mm: くし幅リスト [mm]（None で JIS 標準 5 値）
        distance_mm: 試料〜くし距離 [mm]
        show_pitch_bars: 最大くし幅の周期位置に破線バーを描画するか
        style_band / style_scan / style_pitch: 個別スタイル

    Returns:
        heatmap に COMB 解析領域を重ねた Overlay
    """
    _check_hv()
    s_band = dict(_DEFAULT_STYLES["doi_comb_band"])
    if style_band:
        s_band.update(style_band)
    s_scan = dict(_DEFAULT_STYLES["doi_comb_scan"])
    if style_scan:
        s_scan.update(style_scan)
    s_pitch = dict(_DEFAULT_STYLES["doi_comb_pitch"])
    if style_pitch:
        s_pitch.update(style_pitch)

    u_half = np.sin(np.deg2rad(scan_half_angle_deg))
    v_half = np.sin(np.deg2rad(v_band_half_deg))
    # 走査帯 (v バンド × u 走査幅) の長方形
    band = _rectangle(u_center, v_center, u_half, v_half, **s_band).relabel(
        f"DOI-COMB band ±{scan_half_angle_deg}°×{v_band_half_deg}°"
    )

    result = heatmap * band

    # くし周期バー（最小くし幅で 2〜3 本、中心から ±n·period）
    if show_pitch_bars:
        widths = comb_widths_mm or [0.125, 0.25, 0.5, 1.0, 2.0]
        d_min = float(min(widths))
        period_u = 2.0 * d_min / distance_mm  # 明暗 1 組
        # ±1, ±2 周期位置に縦破線（hv.VLine が使えない環境もあるため Rectangles で細線）
        for k in (-2, -1, 1, 2):
            x = u_center + k * period_u
            if abs(x - u_center) > u_half:
                continue
            tick = _rectangle(x, v_center, period_u * 0.02, v_half, **s_pitch)
            label = f"DOI-COMB pitch ({d_min} mm)" if k == 1 else ""
            result = result * (tick.relabel(label) if label else tick)

    return result


def overlay_doi_astm_2d(
    heatmap: Any,
    u_center: float = 0.0,
    v_center: float = 0.0,
    offset_deg: float = 0.3,
    aperture_half_deg: float = 0.05,
    style_center: dict[str, Any] | None = None,
    style_offset: dict[str, Any] | None = None,
) -> Any:
    """DOI-ASTM（ASTM E430 Dorigon）の 0° + ±offset 位置を 3 円で重ねる。

    Args:
        heatmap: 既存の hv.Image / Overlay
        u_center, v_center: 0° 位置（通常 specular）
        offset_deg: オフセット角 [deg]（規格 0.3°）
        aperture_half_deg: 受光絞り半角 [deg]
        style_center / style_offset: 個別スタイル

    Returns:
        heatmap に 3 円を重ねた Overlay
    """
    _check_hv()
    s_c = dict(_DEFAULT_STYLES["doi_astm_center"])
    if style_center:
        s_c.update(style_center)
    s_o = dict(_DEFAULT_STYLES["doi_astm_offset"])
    if style_offset:
        s_o.update(style_offset)

    sin_off = float(np.sin(np.deg2rad(offset_deg)))
    center = _ellipse(u_center, v_center, aperture_half_deg, **s_c).relabel(
        f"DOI-ASTM 0° ({aperture_half_deg}°)"
    )
    off_p = _ellipse(u_center + sin_off, v_center, aperture_half_deg, **s_o).relabel(
        f"DOI-ASTM +{offset_deg}°"
    )
    off_n = _ellipse(u_center - sin_off, v_center, aperture_half_deg, **s_o)
    return heatmap * center * off_p * off_n


# ── 統合オーバーレイ ─────────────────────────────────────────────────────────


_ALL_METRIC_KEYS = (
    "haze", "gloss_20", "gloss_60", "gloss_85",
    "doi_nser", "doi_comb", "doi_astm",
)


def _apply_visibility(overlay: Any, shown: set[str] | None) -> Any:
    """Overlay 内の各レイヤーに対し、shown に含まれないキーの visible=False を適用。

    各 overlay レイヤーの label 先頭トークンをキーとして判定する（例:
    "Haze 2.5°" → "haze", "Gloss 20°" → "gloss_20", "DOI-NSER inner..." → "doi_nser"）。
    """
    if shown is None:
        return overlay

    def _layer_key(label: str) -> str | None:
        lb = label.lower()
        if lb.startswith("haze"):
            return "haze"
        if lb.startswith("gloss"):
            parts = lb.split()
            # "gloss 20°" → "gloss_20"
            if len(parts) >= 2:
                deg = parts[1].rstrip("°")
                return f"gloss_{deg}"
            return "gloss_60"
        if lb.startswith("doi-nser") or lb.startswith("nser"):
            return "doi_nser"
        if lb.startswith("doi-comb") or lb.startswith("comb"):
            return "doi_comb"
        if lb.startswith("doi-astm") or lb.startswith("astm"):
            return "doi_astm"
        return None

    new_items = []
    for item in overlay:
        label = getattr(item, "label", "") or ""
        key = _layer_key(label) if label else None
        if key is None or key in shown:
            new_items.append(item)
        else:
            new_items.append(item.opts(visible=False))
    return hv.Overlay(new_items)


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

    対象: haze / gloss (enabled_angles) / doi_nser / doi_comb / doi_astm

    Args:
        heatmap: 既存の hv.Image / Overlay
        metrics_config: config.yaml の metrics セクション辞書
        theta_i_deg, mode, n1, n2: 窓中心決定用の条件（`_specular_u_center` と同じ）
        click_policy: Bokeh の凡例クリック動作 ("hide" / "mute" / "none")
        initially_shown: 初期表示する指標キーのリスト（空/None なら全表示）
            使えるキー: "haze", "gloss_20", "gloss_60", "gloss_85",
                        "doi_nser", "doi_comb", "doi_astm"
        legend_position: 凡例位置 ("right" / "top_right" / "bottom" 等)

    Returns:
        全指標オーバーレイ済みの Overlay（click_policy / visible 適用済み）
    """
    _check_hv()
    cfg = metrics_config or {}
    u_c = _specular_u_center(theta_i_deg, mode, n1, n2)
    v_c = 0.0

    result = heatmap

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

    # DOI-COMB
    comb_cfg = cfg.get("doi_comb")
    if comb_cfg is not None and comb_cfg.get("enabled", True):
        result = overlay_doi_comb_2d(
            result, u_center=u_c, v_center=v_c,
            scan_half_angle_deg=comb_cfg.get("scan_half_angle_deg", 4.0),
            v_band_half_deg=comb_cfg.get("v_band_half_deg", 0.2),
            comb_widths_mm=comb_cfg.get("comb_widths_mm"),
            distance_mm=comb_cfg.get("distance_mm", 280.0),
        )

    # DOI-ASTM
    astm_cfg = cfg.get("doi_astm")
    if astm_cfg is not None and astm_cfg.get("enabled", True):
        # ASTM は BRDF 規格だが overlay 表示は与えられた u_c 基準に任せる
        result = overlay_doi_astm_2d(
            result, u_center=u_c, v_center=v_c,
            offset_deg=astm_cfg.get("offset_deg", 0.3),
            aperture_half_deg=astm_cfg.get("aperture_half_deg", 0.05),
        )

    # 初期表示制御（shown に無い系列を visible=False にする）
    shown = set(initially_shown) if initially_shown else None
    result = _apply_visibility(result, shown)

    opts_kwargs: dict[str, Any] = {
        "show_legend": True,
        "legend_position": legend_position,
    }
    if click_policy != "none":
        # HoloViews (Bokeh backend) の Overlay オプション名は click_policy
        opts_kwargs["click_policy"] = click_policy

    return result.opts(**opts_kwargs)


def overlay_from_config(
    heatmap: Any,
    full_config: dict[str, Any],
    theta_i_deg: float = 0.0,
    mode: str = "BTDF",
    n1: float = 1.0,
    n2: float = 1.5,
) -> Any:
    """config.yaml ルート辞書から `visualization.metric_overlay` を読んで適用する。

    `visualization.metric_overlay.show_overlay` が False なら heatmap をそのまま返す。
    `initially_shown` / `click_policy` / `legend_position` はそのまま
    `overlay_all_metrics_2d` に渡す。

    Args:
        heatmap: 既存の hv.Image / Overlay
        full_config: config.yaml をパースした辞書（metrics / visualization セクションを含む）
        theta_i_deg, mode, n1, n2: 条件（specular 位置決定用）

    Returns:
        overlay 適用済み Overlay、または show_overlay=False の heatmap そのまま
    """
    vis_cfg = (full_config.get("visualization") or {}).get("metric_overlay") or {}
    if not vis_cfg.get("show_overlay", True):
        return heatmap

    return overlay_all_metrics_2d(
        heatmap,
        metrics_config=full_config.get("metrics"),
        theta_i_deg=theta_i_deg, mode=mode, n1=n1, n2=n2,
        click_policy=vis_cfg.get("click_policy", "hide"),
        initially_shown=vis_cfg.get("initially_shown"),
        legend_position=vis_cfg.get("legend_position", "right"),
    )
