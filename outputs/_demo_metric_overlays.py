"""Phase 1 光学指標オーバーレイのデモ（HTML + PNG 出力）。

ダミー BSDF ヒートマップ（ガウス + 弱い散乱成分）に overlay_all_metrics_2d で
Haze / Gloss 20°,60°,85° / DOI-NSER を重ねて書き出す。

出力:
- demo_metric_overlays.html  (凡例クリックで表示切替可能な Bokeh)
- demo_metric_overlays.png   (matplotlib 経由の静的画像、存在すれば)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import holoviews as hv

hv.extension("bokeh")

from bsdf_sim.visualization.metric_overlays import overlay_all_metrics_2d

OUT_HTML = Path(__file__).parent / "demo_metric_overlays.html"
OUT_PNG = Path(__file__).parent / "demo_metric_overlays.png"
OUT_PNG_ZOOM = Path(__file__).parent / "demo_metric_overlays_zoom.png"


def _make_dummy_bsdf(n: int = 257) -> hv.Image:
    """specular + 広角散乱のダミー BSDF を log10 で返す。"""
    u = np.linspace(-1.0, 1.0, n)
    v = np.linspace(-1.0, 1.0, n)
    U, V = np.meshgrid(u, v, indexing="ij")
    r2 = U**2 + V**2
    # 正透過（u=0 近傍）+ 60° 近傍の弱いハイライト（u≈sin 60°）+ ブロード散乱
    specular = np.exp(-r2 / 0.001) * 1e3
    spec60 = np.exp(-((U - np.sin(np.deg2rad(60))) ** 2 + V**2) / 0.003) * 5.0
    broad = np.exp(-r2 / 0.5) * 0.05
    bsdf = specular + spec60 + broad
    bsdf[r2 > 1.0] = 0.0
    data = np.log10(np.maximum(bsdf, 1e-6))
    return hv.Image(
        (u, v, data.T), kdims=["u", "v"], vdims=["log10 BSDF"],
    ).opts(cmap="viridis", width=600, height=600, colorbar=True, tools=["hover"])


def _make_dummy_bsdf_array(n: int = 257) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(-1.0, 1.0, n)
    v = np.linspace(-1.0, 1.0, n)
    U, V = np.meshgrid(u, v, indexing="ij")
    r2 = U**2 + V**2
    specular = np.exp(-r2 / 0.001) * 1e3
    spec60 = np.exp(-((U - np.sin(np.deg2rad(60))) ** 2 + V**2) / 0.003) * 5.0
    broad = np.exp(-r2 / 0.5) * 0.05
    bsdf = specular + spec60 + broad
    bsdf[r2 > 1.0] = 0.0
    return u, v, np.log10(np.maximum(bsdf, 1e-6))


def main() -> None:
    u, v, data = _make_dummy_bsdf_array(n=257)
    heatmap = hv.Image(
        (u, v, data.T), kdims=["u", "v"], vdims=["log10 BSDF"],
    ).opts(cmap="viridis", width=600, height=600, colorbar=True, tools=["hover"])
    cfg = {
        "haze": {"enabled": True, "half_angle_deg": 2.5},
        "gloss": {"enabled": True, "enabled_angles": [20, 60, 85]},
        "doi_nser": {
            "enabled": True, "direct_half_angle_deg": 0.1, "halo_half_angle_deg": 2.0,
        },
        "doi_comb": {
            "enabled": True, "scan_half_angle_deg": 4.0, "v_band_half_deg": 0.2,
            "comb_widths_mm": [0.125, 0.25, 0.5, 1.0, 2.0], "distance_mm": 280.0,
        },
        "doi_astm": {
            "enabled": True, "offset_deg": 0.3, "aperture_half_deg": 0.05,
        },
    }
    overlay = overlay_all_metrics_2d(
        heatmap, metrics_config=cfg, theta_i_deg=0.0, mode="BTDF",
        click_policy="hide", legend_position="right",
        initially_shown=["haze", "gloss_60", "doi_nser", "doi_comb", "doi_astm"],
    )

    hv.save(overlay, str(OUT_HTML), backend="bokeh")
    print(f"saved: {OUT_HTML}")

    # PNG は matplotlib で直接同等図を描画（overlay 要素を再構成）
    _save_png_matplotlib(data, cfg, OUT_PNG, xlim=(-1, 1), ylim=(-1, 1))
    print(f"saved: {OUT_PNG}")
    _save_png_matplotlib(data, cfg, OUT_PNG_ZOOM, xlim=(-0.1, 0.1), ylim=(-0.1, 0.1))
    print(f"saved: {OUT_PNG_ZOOM}")


def _save_png_matplotlib(
    data: np.ndarray, cfg: dict, path: Path,
    xlim: tuple[float, float] = (-1, 1), ylim: tuple[float, float] = (-1, 1),
) -> None:
    """HoloViews overlay と同等の重ね描きを matplotlib で PNG 出力。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    from bsdf_sim.metrics.optical import _GLOSS_APERTURES_DEG

    fig, ax = plt.subplots(figsize=(7, 6))
    # data[u_idx, v_idx] → imshow は (rows=y, cols=x) なので T して v→row, u→col
    im = ax.imshow(
        data.T, origin="lower", extent=(-1, 1, -1, 1),
        cmap="viridis", aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="log10 BSDF")

    # Haze 円
    r_haze = np.sin(np.deg2rad(cfg["haze"]["half_angle_deg"]))
    ax.add_patch(Circle(
        (0, 0), r_haze, fill=False, edgecolor="white",
        linestyle="--", linewidth=2, label=f"Haze {cfg['haze']['half_angle_deg']}°",
    ))
    # Gloss 長方形
    gloss_colors = {20: "cyan", 60: "yellow", 85: "magenta"}
    for ang in cfg["gloss"]["enabled_angles"]:
        ap = _GLOSS_APERTURES_DEG.get(int(ang), _GLOSS_APERTURES_DEG[60])
        u_c = float(np.sin(np.deg2rad(ang)))
        du = np.cos(np.deg2rad(ang)) * np.deg2rad(ap["in_plane_deg"] / 2.0)
        dv = np.deg2rad(ap["cross_plane_deg"] / 2.0)
        ax.add_patch(Rectangle(
            (u_c - du, -dv), 2 * du, 2 * dv, fill=False,
            edgecolor=gloss_colors.get(int(ang), "yellow"),
            linewidth=2, label=f"Gloss {ang}°",
        ))
    # DOI-NSER 2 重円
    nser = cfg["doi_nser"]
    ax.add_patch(Circle(
        (0, 0), np.sin(np.deg2rad(nser["direct_half_angle_deg"])),
        fill=False, edgecolor="#4fc3ff", linewidth=2,
        label=f"NSER inner {nser['direct_half_angle_deg']}°",
    ))
    ax.add_patch(Circle(
        (0, 0), np.sin(np.deg2rad(nser["halo_half_angle_deg"])),
        fill=False, edgecolor="#4fc3ff", linestyle="--", linewidth=2,
        label=f"NSER outer {nser['halo_half_angle_deg']}°",
    ))
    # DOI-COMB 走査帯 + pitch バー
    if cfg.get("doi_comb", {}).get("enabled"):
        comb = cfg["doi_comb"]
        u_half = np.sin(np.deg2rad(comb["scan_half_angle_deg"]))
        v_half = np.sin(np.deg2rad(comb["v_band_half_deg"]))
        ax.add_patch(Rectangle(
            (-u_half, -v_half), 2 * u_half, 2 * v_half, fill=False,
            edgecolor="#ffb347", linewidth=2,
            label=f"COMB band ±{comb['scan_half_angle_deg']}°×{comb['v_band_half_deg']}°",
        ))
        # 最小くし幅の周期位置（代表として ±1 周期のみ）
        d_min = float(min(comb["comb_widths_mm"]))
        period_u = 2.0 * d_min / comb["distance_mm"]
        for sign in (+1, -1):
            ax.axvline(
                sign * period_u, ymin=0.48, ymax=0.52,
                color="#ffb347", linestyle=":", linewidth=1,
            )
    # DOI-ASTM 3 円
    if cfg.get("doi_astm", {}).get("enabled"):
        astm = cfg["doi_astm"]
        sin_off = np.sin(np.deg2rad(astm["offset_deg"]))
        r_ap = np.sin(np.deg2rad(astm["aperture_half_deg"]))
        ax.add_patch(Circle(
            (0, 0), r_ap, fill=False, edgecolor="#ff6b6b",
            linewidth=2, label=f"ASTM 0° (±{astm['aperture_half_deg']}°)",
        ))
        for sign in (+1, -1):
            ax.add_patch(Circle(
                (sign * sin_off, 0), r_ap, fill=False, edgecolor="#ff6b6b",
                linestyle="--", linewidth=2,
                label=f"ASTM ±{astm['offset_deg']}°" if sign == 1 else None,
            ))
    ax.set_xlabel("u = sin θ cos φ")
    ax.set_ylabel("v = sin θ sin φ")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    title_suffix = " (zoom)" if (xlim[1] - xlim[0]) < 1.5 else ""
    ax.set_title(f"Metric overlays{title_suffix} (Haze/Gloss/NSER/COMB/ASTM)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
