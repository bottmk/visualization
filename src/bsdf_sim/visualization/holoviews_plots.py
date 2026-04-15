"""HoloViews による BSDF 可視化。

spec_main.md Section 7:
- FFT/PSD/実測の3者オーバーレイ比較
- リニア/片対数/両対数の軸スケール切替 UI（Panel ウィジェット）
- UV 座標空間での 2D ヒートマップ
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import holoviews as hv
    import panel as pn
    hv.extension("bokeh")
    _HV_AVAILABLE = True
except ImportError:
    _HV_AVAILABLE = False


def _check_holoviews() -> None:
    if not _HV_AVAILABLE:
        raise ImportError(
            "holoviews と panel が必要。"
            "`pip install holoviews panel bokeh` でインストールしてください。"
        )


# ── 1次元 BSDF プロット（角度 vs BSDF）───────────────────────────────────────

def plot_bsdf_1d_overlay(
    df: pd.DataFrame,
    phi_s_deg: float = 0.0,
    theta_i_deg: float = 0.0,
    wavelength_um: float = 0.55,
    scale: str = "log",
    title: str = "BSDF 比較",
) -> Any:
    """FFT/PSD/実測 BSDF の1次元オーバーレイプロットを生成する。

    指定した入射面（phi_s = 定数）に沿ったBSDFプロファイルを比較する。

    Args:
        df: BSDF Parquet DataFrame（long format）
        phi_s_deg: プロットする散乱方位角 [deg]（デフォルト: 0°）
        theta_i_deg: 入射天頂角 [deg]でフィルタ
        wavelength_um: 波長 [μm]でフィルタ
        scale: 'linear' / 'log'（デフォルト: 'log'）
        title: プロットタイトル

    Returns:
        HoloViews Overlay オブジェクト
    """
    _check_holoviews()

    # 指定条件でフィルタ
    mask = (
        (np.abs(df["theta_i_deg"] - theta_i_deg) < 0.5)
        & (np.abs(df["wavelength_um"] - wavelength_um) < 0.01)
        & (np.abs(df["phi_s_deg"] - phi_s_deg) < 6.0)  # ±5° の許容範囲
    )
    filtered = df[mask].copy()

    if filtered.empty:
        return hv.Text(0, 0, "データなし")

    # スタイル定義
    style_map = {
        "FFT":      {"color": "blue",   "line_dash": "solid",  "size": 4, "label": "FFT計算"},
        "PSD":      {"color": "orange", "line_dash": "dashed", "size": 4, "label": "PSD計算"},
        "measured": {"color": "black",  "line_dash": "solid",  "size": 6, "label": "実測データ"},
        "MultiLayer": {"color": "green", "line_dash": "dotted", "size": 4, "label": "多層合成"},
    }

    curves = []
    for method, style in style_map.items():
        sub = filtered[filtered["method"] == method].sort_values("theta_s_deg")
        if sub.empty:
            continue

        x = sub["theta_s_deg"].values
        y = np.maximum(sub["bsdf"].values, 1e-10)

        if method == "measured":
            curve = hv.Scatter(
                (x, y), kdims=["散乱角 θ_s [deg]"], vdims=["BSDF [sr⁻¹]"],
                label=style["label"],
            ).opts(color=style["color"], size=style["size"])
        else:
            curve = hv.Curve(
                (x, y), kdims=["散乱角 θ_s [deg]"], vdims=["BSDF [sr⁻¹]"],
                label=style["label"],
            ).opts(color=style["color"], line_dash=style["line_dash"], line_width=2)

        curves.append(curve)

    if not curves:
        return hv.Text(0, 0, "表示できるデータなし")

    overlay = hv.Overlay(curves).opts(
        title=title,
        width=700,
        height=450,
        legend_position="top_right",
        logy=(scale == "log"),
    )
    return overlay


def plot_bsdf_2d_heatmap(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    bsdf: np.ndarray,
    title: str = "BSDF 2D ヒートマップ",
    log_scale: bool = True,
) -> Any:
    """BSDF の2次元UV空間ヒートマップを生成する。

    Args:
        u_grid: 方向余弦 u（2D）
        v_grid: 方向余弦 v（2D）
        bsdf: BSDF 値 [sr⁻¹]（2D）
        title: プロットタイトル
        log_scale: True の場合 log10 スケールで表示

    Returns:
        HoloViews Image オブジェクト
    """
    _check_holoviews()

    u_axis = u_grid[:, 0]
    v_axis = v_grid[0, :]

    data = np.log10(np.maximum(bsdf, 1e-10)) if log_scale else bsdf
    label = "log₁₀(BSDF)" if log_scale else "BSDF [sr⁻¹]"

    img = hv.Image(
        (u_axis, v_axis, data.T),
        kdims=["u = sin θ_s cos φ_s", "v = sin θ_s sin φ_s"],
        vdims=[label],
    ).opts(
        title=title,
        width=500,
        height=500,
        colorbar=True,
        cmap="viridis",
        aspect="equal",
    )

    # 半球境界円のオーバーレイ
    theta_boundary = np.linspace(0, 2 * np.pi, 200)
    boundary = hv.Curve(
        list(zip(np.cos(theta_boundary), np.sin(theta_boundary))),
        kdims=["u = sin θ_s cos φ_s"],
        vdims=["v = sin θ_s sin φ_s"],
    ).opts(color="white", line_dash="dashed", line_width=1)

    return (img * boundary).opts(title=title)


def create_scale_toggle_panel(
    df: pd.DataFrame,
    **plot_kwargs: Any,
) -> Any:
    """軸スケール切替 UI 付きパネルを生成する。

    Panel ウィジェットでリニア/片対数/両対数を切り替えられる。

    Args:
        df: BSDF Parquet DataFrame
        **plot_kwargs: plot_bsdf_1d_overlay に渡す引数

    Returns:
        Panel Layout オブジェクト
    """
    _check_holoviews()

    scale_selector = pn.widgets.RadioButtonGroup(
        name="軸スケール",
        options=["linear", "log"],
        value="log",
        button_type="default",
    )

    @pn.depends(scale=scale_selector)
    def update_plot(scale: str) -> Any:
        return plot_bsdf_1d_overlay(df, scale=scale, **plot_kwargs)

    return pn.Column(
        pn.Row(pn.pane.Markdown("**軸スケール:**"), scale_selector),
        pn.panel(update_plot),
    )


def plot_heightmap(
    hm: "HeightMap",
    title: str = "表面形状",
    colormap: str = "RdYlBu_r",
    unit: str = "nm",
) -> Any:
    """高さマップを 2D カラーマップ・ヒストグラム・断面プロファイルで可視化する。

    Args:
        hm: HeightMap オブジェクト
        title: タイトル
        colormap: カラーマップ名（Bokeh/Matplotlib 互換）
        unit: 表示単位 'nm' または 'um'

    Returns:
        HoloViews Layout（2D マップ + ヒストグラム + X/Y 断面）
    """
    _check_holoviews()

    scale = 1000.0 if unit == "nm" else 1.0
    unit_label = f"[{unit}]"

    data = hm.data.astype(np.float64) * scale  # μm → 表示単位
    N = hm.grid_size
    phys = hm.physical_size_um  # μm

    x_axis = np.linspace(0, phys, N)
    y_axis = np.linspace(0, phys, N)

    # ── 2D カラーマップ ───────────────────────────────────────────────────────
    img = hv.Image(
        (x_axis, y_axis, data.T),
        kdims=["x [μm]", "y [μm]"],
        vdims=[f"高さ {unit_label}"],
    ).opts(
        title=title,
        width=480,
        height=460,
        colorbar=True,
        cmap=colormap,
        aspect="equal",
        toolbar="above",
    )

    # ── 高さ分布ヒストグラム ──────────────────────────────────────────────────
    counts, edges = np.histogram(data.ravel(), bins=80)
    hist = hv.Histogram(
        (edges, counts),
        kdims=[f"高さ {unit_label}"],
        vdims=["頻度"],
    ).opts(
        title="高さ分布",
        width=280,
        height=220,
        color="steelblue",
        alpha=0.8,
        toolbar=None,
    )

    # ── X 断面プロファイル（y 中心） ──────────────────────────────────────────
    mid = N // 2
    prof_x = hv.Curve(
        (x_axis, data[:, mid]),
        kdims=["x [μm]"],
        vdims=[f"高さ {unit_label}"],
    ).opts(
        title=f"断面プロファイル（y={phys/2:.1f}μm）",
        width=480,
        height=200,
        color="royalblue",
        line_width=1.2,
        toolbar=None,
    )

    # ── Y 断面プロファイル（x 中心） ──────────────────────────────────────────
    prof_y = hv.Curve(
        (y_axis, data[mid, :]),
        kdims=["y [μm]"],
        vdims=[f"高さ {unit_label}"],
    ).opts(
        title=f"断面プロファイル（x={phys/2:.1f}μm）",
        width=280,
        height=200,
        color="tomato",
        line_width=1.2,
        toolbar=None,
    )

    # ── レイアウト組み立て ────────────────────────────────────────────────────
    layout = (img + hist + prof_x + prof_y).cols(2).opts(
        title=title,
    )
    return layout


def save_heightmap_png(
    hm: "HeightMap",
    path: str | Path,
    title: str = "表面形状",
    colormap: str = "RdYlBu_r",
    unit: str = "nm",
    dpi: int = 150,
) -> None:
    """高さマップを PNG ファイルとして保存する（matplotlib 使用）。

    Args:
        hm: HeightMap オブジェクト
        path: 保存先パス
        title: タイトル
        colormap: matplotlib カラーマップ名
        unit: 表示単位 'nm' または 'um'
        dpi: 解像度
    """
    import matplotlib
    matplotlib.use("Agg")  # ヘッドレス環境でも動作
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    scale = 1000.0 if unit == "nm" else 1.0
    unit_label = unit
    data = hm.data.astype(np.float64) * scale
    N = hm.grid_size
    phys = hm.physical_size_um

    x_axis = np.linspace(0, phys, N)
    y_axis = np.linspace(0, phys, N)

    fig = plt.figure(figsize=(12, 9))
    rq_nm = float(np.sqrt(np.mean((hm.data - hm.data.mean()) ** 2))) * 1000
    fig.suptitle(
        f"{title}  (grid={N}x{N}, pixel={hm.pixel_size_um}um, Rq={rq_nm:.2f}nm)",
        fontsize=12,
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.38)

    # 2D color map
    ax_map = fig.add_subplot(gs[:, :2])
    im = ax_map.imshow(
        data.T,
        origin="lower",
        extent=[0, phys, 0, phys],
        cmap=colormap,
        aspect="equal",
    )
    ax_map.set_xlabel("x [um]")
    ax_map.set_ylabel("y [um]")
    ax_map.set_title(f"Height map [{unit_label}]")
    plt.colorbar(im, ax=ax_map, label=f"Height [{unit_label}]", fraction=0.046)

    # Height histogram
    ax_hist = fig.add_subplot(gs[0, 2])
    ax_hist.hist(data.ravel(), bins=80, color="steelblue", alpha=0.8, edgecolor="none")
    ax_hist.set_xlabel(f"Height [{unit_label}]")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Height distribution")

    # X/Y cross-section profiles
    ax_px = fig.add_subplot(gs[1, 2])
    mid = N // 2
    ax_px.plot(x_axis, data[:, mid], color="royalblue", linewidth=0.8, label="X profile")
    ax_px.plot(y_axis, data[mid, :], color="tomato",    linewidth=0.8, label="Y profile")
    ax_px.set_xlabel("[um]")
    ax_px.set_ylabel(f"Height [{unit_label}]")
    ax_px.set_title(f"Cross-section (center)")
    ax_px.legend(fontsize=8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_html(plot: Any, path: str | Path, title: str = "BSDF Report") -> None:
    """HoloViews / Panel オブジェクトを HTML ファイルとして保存する。

    Args:
        plot: HoloViews または Panel オブジェクト
        path: 保存先パス
        title: HTML タイトル
    """
    _check_holoviews()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(plot, pn.viewable.Viewable):
        plot.save(str(path))
    else:
        hv.save(plot, str(path), backend="bokeh")
