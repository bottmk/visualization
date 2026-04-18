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
    theta_i_deg: float | None = None,
    wavelength_um: float | None = None,
    mode: str | None = None,
    scale: str = "log",
    title: str = "BSDF 比較",
) -> Any:
    """FFT/PSD/実測 BSDF の1次元オーバーレイプロットを生成する。

    指定した入射面（phi_s = 定数）に沿ったBSDFプロファイルを比較する。
    `theta_i_deg` / `wavelength_um` / `mode` を省略すると DataFrame 内の
    先頭条件を自動採用する（多条件 Parquet でも "データなし" にならない）。

    Args:
        df: BSDF Parquet DataFrame（long format）
        phi_s_deg: プロットする散乱方位角 [deg]（デフォルト: 0°）
        theta_i_deg: 入射天頂角 [deg]でフィルタ。None → df から自動選択
        wavelength_um: 波長 [μm]でフィルタ。None → df から自動選択
        mode: 'BRDF' / 'BTDF' でフィルタ。None → df から自動選択
        scale: 'linear' / 'log'（デフォルト: 'log'）
        title: プロットタイトル

    Returns:
        HoloViews Overlay オブジェクト
    """
    _check_holoviews()

    if df.empty:
        return hv.Text(0, 0, "データなし")

    # ── 条件が未指定なら df から自動選択 ──
    if wavelength_um is None:
        wavelength_um = float(df["wavelength_um"].iloc[0])
    if theta_i_deg is None:
        theta_i_deg = float(df["theta_i_deg"].iloc[0])
    if mode is None and "mode" in df.columns:
        mode = str(df["mode"].iloc[0])

    # 指定条件でフィルタ
    mask = (
        (np.abs(df["theta_i_deg"] - theta_i_deg) < 0.5)
        & (np.abs(df["wavelength_um"] - wavelength_um) < 0.01)
        & (np.abs(df["phi_s_deg"] - phi_s_deg) < 6.0)  # ±5° の許容範囲
    )
    if mode is not None and "mode" in df.columns:
        mask = mask & (df["mode"].astype(str) == mode)
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
        frame_width=400,
        frame_height=400,
        colorbar=True,
        cmap="viridis",
    )

    # 半球境界円のオーバーレイ
    theta_boundary = np.linspace(0, 2 * np.pi, 200)
    boundary = hv.Curve(
        list(zip(np.cos(theta_boundary), np.sin(theta_boundary))),
        kdims=["u = sin θ_s cos φ_s"],
        vdims=["v = sin θ_s sin φ_s"],
    ).opts(color="white", line_dash="dashed", line_width=1)

    return (img * boundary).opts(title=title, frame_width=400, frame_height=400)


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
    title: str = "Surface height",
    colormap: str = "RdYlBu_r",
    unit: str = "nm",
    dpi: int = 150,
) -> None:
    """高さマップを PNG ファイルとして保存する（matplotlib 使用）。

    Note:
        matplotlib に日本語フォントが無い環境（CI/Codespace 等）では
        日本語タイトルが豆腐（□）化するため、title は ASCII のみ推奨。
        日本語 UI は HoloViews/Panel 側（HTML）で提供する。

    Args:
        hm: HeightMap オブジェクト
        path: 保存先パス
        title: タイトル（ASCII 推奨）
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

    # MLflow UI で縦スクロール不要になる縦サイズに抑える（元 12x9 → 6x4.5）
    fig = plt.figure(figsize=(6, 4.5))
    rq_nm = float(np.sqrt(np.mean((hm.data - hm.data.mean()) ** 2))) * 1000
    fig.suptitle(
        f"{title}  (grid={N}x{N}, pixel={hm.pixel_size_um}um, Rq={rq_nm:.2f}nm)",
        fontsize=9,
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.45)

    # 2D color map
    ax_map = fig.add_subplot(gs[:, :2])
    im = ax_map.imshow(
        data.T,
        origin="lower",
        extent=[0, phys, 0, phys],
        cmap=colormap,
        aspect="equal",
    )
    ax_map.set_xlabel("x [um]", fontsize=8)
    ax_map.set_ylabel("y [um]", fontsize=8)
    ax_map.set_title(f"Height map [{unit_label}]", fontsize=9)
    ax_map.tick_params(labelsize=7)
    cbar = plt.colorbar(im, ax=ax_map, label=f"Height [{unit_label}]", fraction=0.046)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(f"Height [{unit_label}]", fontsize=8)

    # Height histogram
    ax_hist = fig.add_subplot(gs[0, 2])
    ax_hist.hist(data.ravel(), bins=80, color="steelblue", alpha=0.8, edgecolor="none")
    ax_hist.set_xlabel(f"Height [{unit_label}]", fontsize=8)
    ax_hist.set_ylabel("Count", fontsize=8)
    ax_hist.set_title("Height distribution", fontsize=9)
    ax_hist.tick_params(labelsize=7)

    # X/Y cross-section profiles
    ax_px = fig.add_subplot(gs[1, 2])
    mid = N // 2
    ax_px.plot(x_axis, data[:, mid], color="royalblue", linewidth=0.8, label="X profile")
    ax_px.plot(y_axis, data[mid, :], color="tomato",    linewidth=0.8, label="Y profile")
    ax_px.set_xlabel("[um]", fontsize=8)
    ax_px.set_ylabel(f"Height [{unit_label}]", fontsize=8)
    ax_px.set_title("Cross-section (center)", fontsize=9)
    ax_px.tick_params(labelsize=7)
    ax_px.legend(fontsize=7)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _bsdf_1d_to_2d_binned(
    u_pts: np.ndarray,
    v_pts: np.ndarray,
    bsdf_pts: np.ndarray,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D (u, v, bsdf) 点列を np.bincount でビン集計して 2D グリッドに変換する。

    scipy.griddata を使わず O(N) で完全ベクトル化。大規模点群でも高速。
    """
    axis = np.linspace(-1.0, 1.0, n_grid, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(axis, axis, indexing="ij")

    # 各点を最近傍グリッドセルにマップ
    u_idx = np.clip(
        np.round((u_pts.astype(np.float32) + 1.0) * (n_grid - 1) / 2.0).astype(np.int32),
        0, n_grid - 1,
    )
    v_idx = np.clip(
        np.round((v_pts.astype(np.float32) + 1.0) * (n_grid - 1) / 2.0).astype(np.int32),
        0, n_grid - 1,
    )

    flat_idx = u_idx * n_grid + v_idx
    bsdf_sum = np.bincount(flat_idx, weights=bsdf_pts.astype(np.float64),
                           minlength=n_grid * n_grid)
    count    = np.bincount(flat_idx, minlength=n_grid * n_grid).astype(np.float64)

    bsdf_2d = np.where(count > 0, bsdf_sum / np.maximum(count, 1), 0.0)
    bsdf_2d = bsdf_2d.reshape(n_grid, n_grid).astype(np.float32)
    bsdf_2d[u_grid ** 2 + v_grid ** 2 > 1.0] = 0.0

    return u_grid, v_grid, bsdf_2d


def save_bsdf_2d_png(
    u: np.ndarray,
    v: np.ndarray,
    bsdf: np.ndarray,
    path: str | Path,
    title: str = "BSDF 2D Heatmap",
    method: str = "",
    log_scale: bool = True,
    dpi: int = 150,
    n_grid: int = 256,
) -> None:
    """2D BSDF ヒートマップを PNG ファイルとして保存する（matplotlib 使用）。

    2D 入力（simulate）: 直接ダウンサンプリングで高速処理。
    1D 入力（scattered）: np.bincount ビン集計で高速処理。

    Args:
        u, v: 方向余弦（1D または 2D）
        bsdf: BSDF 値（u/v と同形状）
        path: 保存先パス
        title: タイトル
        method: 手法名（タイトルに付加）
        log_scale: True の場合 log10 スケール
        dpi: 解像度
        n_grid: 出力グリッドサイズ
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if u.ndim == 2:
        # simulate から渡された正規 2D グリッド（fftfreq 順：DC が [0,0]）
        # fftshift で DC をセンター [N//2, N//2] に移動してからダウンサンプリング
        u_s    = np.fft.fftshift(u)
        v_s    = np.fft.fftshift(v)
        bsdf_s = np.fft.fftshift(bsdf)
        N = u_s.shape[0]
        step = max(1, N // n_grid)
        bsdf_2d = bsdf_s[::step, ::step][:n_grid, :n_grid].astype(np.float64)
        u_ds    = u_s[::step, ::step][:n_grid, :n_grid]
        v_ds    = v_s[::step, ::step][:n_grid, :n_grid]
        outside = u_ds ** 2 + v_ds ** 2 > 1.0
        bsdf_2d[outside] = np.nan
    else:
        # 1D 散布点 → ビン集計
        _, _, bsdf_2d_f = _bsdf_1d_to_2d_binned(
            u.ravel(), v.ravel(), bsdf.ravel(), n_grid
        )
        bsdf_2d = bsdf_2d_f.astype(np.float64)
        bsdf_2d[bsdf_2d == 0.0] = np.nan

    data  = np.log10(np.maximum(bsdf_2d, 1e-10)) if log_scale else bsdf_2d
    label = "log10(BSDF [sr-1])" if log_scale else "BSDF [sr-1]"

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(
        data.T, origin="lower", extent=[-1, 1, -1, 1],
        cmap="viridis", aspect="equal",
    )
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "w--", linewidth=0.8)
    plt.colorbar(im, ax=ax, label=label, fraction=0.046)
    ax.set_xlabel("u = sin(ts)cos(ps)")
    ax.set_ylabel("v = sin(ts)sin(ps)")
    method_str = f" [{method}]" if method else ""
    ax.set_title(f"{title}{method_str}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def df_to_2d_grid(
    df_method: "pd.DataFrame",
    n_grid: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """long-format DataFrame の (u, v, bsdf) から 2D グリッドを再構築する。

    np.bincount ビン集計を使用（scipy 不要・大規模点群でも高速）。

    Args:
        df_method: method でフィルタ済みの BSDF DataFrame
        n_grid: 出力グリッドサイズ

    Returns:
        (u_grid, v_grid, bsdf_2d) — いずれも (n_grid, n_grid) の ndarray
    """
    return _bsdf_1d_to_2d_binned(
        df_method["u"].values,
        df_method["v"].values,
        df_method["bsdf"].values,
        n_grid=n_grid,
    )


def _build_condition_panel(
    df_cond: "pd.DataFrame",
    scale: str,
    n_grid: int,
    wl_um: float,
    theta_i: float,
    mode: str,
    log_rmse_by_method: dict[str, float],
) -> Any:
    """1 条件分の 1D オーバーレイ＋2D ヒートマップ Panel を生成する。"""
    _check_holoviews()

    # 1D オーバーレイ（FFT/PSD/MultiLayer/measured を自動重ね描き）
    title_1d = f"1D BSDF — λ={wl_um * 1000:.0f}nm, θ_i={theta_i:.0f}°, {mode}"
    plot_1d = plot_bsdf_1d_overlay(
        df_cond, wavelength_um=wl_um, theta_i_deg=theta_i, mode=mode,
        scale=scale, title=title_1d,
    )

    # 2D ヒートマップ（手法別）
    method_order = ["FFT", "PSD", "MultiLayer", "measured"]
    methods_in_df = [m for m in method_order if m in df_cond["method"].unique()]

    heatmap_plots = []
    for method in methods_in_df:
        df_m = df_cond[df_cond["method"] == method]
        if df_m.empty:
            continue
        u_g, v_g, bsdf_2d = df_to_2d_grid(df_m, n_grid=n_grid)
        hm = plot_bsdf_2d_heatmap(
            u_g, v_g, bsdf_2d,
            title=f"2D [{method}]",
            log_scale=(scale == "log"),
        )
        heatmap_plots.append(hm)

    components: list[Any] = [plot_1d]
    if heatmap_plots:
        components.append(pn.Row(*heatmap_plots))

    # Log-RMSE 表示（計算値 vs 実測）
    if log_rmse_by_method:
        rmse_lines = " · ".join(
            f"**{m}**: log_rmse={v:.3f}" for m, v in log_rmse_by_method.items()
        )
        components.append(pn.pane.Markdown(f"**比較誤差**: {rmse_lines}"))

    return pn.Column(*components)


def plot_bsdf_report(
    df: "pd.DataFrame",
    metrics: dict[str, float] | None = None,
    scale: str = "log",
    title: str = "BSDF Report",
    n_grid: int = 256,
) -> Any:
    """1D プロファイル・2D ヒートマップ・指標テーブルをまとめた Panel レポートを生成する。

    多条件 Parquet の場合は (wavelength, theta_i, mode) ごとに Panel Tab を生成する。
    各 Tab 内で FFT/PSD/MultiLayer と実測（`method='measured'`）の 1D 重ね描き、
    2D ヒートマップ、Log-RMSE が表示される。

    Args:
        df: BSDF Parquet DataFrame（long format、`method='measured'` 行を含んでよい）
        metrics: MLflow から取得した指標辞書（省略可）
        scale: 'linear' / 'log'
        title: レポートタイトル
        n_grid: 2D ヒートマップのグリッドサイズ

    Returns:
        Panel Column レイアウト
    """
    _check_holoviews()

    # ── 条件の一覧を抽出 ────────────────────────────────────────────────────────
    if "mode" not in df.columns:
        df = df.copy()
        df["mode"] = "BRDF"

    conditions = (
        df[["wavelength_um", "theta_i_deg", "mode"]]
        .drop_duplicates()
        .sort_values(["wavelength_um", "theta_i_deg", "mode"])
        .reset_index(drop=True)
    )

    # 条件ごとの Log-RMSE を事前集計
    log_rmse_map: dict[tuple, dict[str, float]] = {}
    if "log_rmse" in df.columns:
        sim_rows = df[df["method"] != "measured"]
        for (wl, theta, mode_v, method_v), grp in sim_rows.groupby(
            ["wavelength_um", "theta_i_deg", "mode", "method"], observed=True
        ):
            vals = grp["log_rmse"].dropna().unique()
            if len(vals) == 0:
                continue
            rmse = float(vals[0])
            if np.isnan(rmse):
                continue
            log_rmse_map.setdefault((wl, theta, mode_v), {})[str(method_v)] = rmse

    # ── 条件ごとのパネル ───────────────────────────────────────────────────────
    components: list[Any] = [pn.pane.Markdown(f"## {title}")]

    has_measured = "measured" in df["method"].unique()
    if has_measured:
        components.append(pn.pane.Markdown(
            "✱ 実測データ（`method='measured'`）を 1D プロットに黒点でオーバーレイ表示。"
        ))

    if len(conditions) <= 1:
        # 単条件：従来通りフラットに表示
        if len(conditions) == 1:
            wl = float(conditions["wavelength_um"].iloc[0])
            theta_i = float(conditions["theta_i_deg"].iloc[0])
            mode_v = str(conditions["mode"].iloc[0])
            rmse_dict = log_rmse_map.get((wl, theta_i, mode_v), {})
            cond_panel = _build_condition_panel(
                df, scale, n_grid, wl, theta_i, mode_v, rmse_dict,
            )
            components.append(cond_panel)
    else:
        # 多条件：Tabs で切替
        tabs = []
        for _, row in conditions.iterrows():
            wl = float(row["wavelength_um"])
            theta_i = float(row["theta_i_deg"])
            mode_v = str(row["mode"])
            cond_mask = (
                (np.abs(df["wavelength_um"] - wl) < 1e-6)
                & (np.abs(df["theta_i_deg"] - theta_i) < 1e-6)
                & (df["mode"].astype(str) == mode_v)
            )
            df_cond = df[cond_mask]
            rmse_dict = log_rmse_map.get((wl, theta_i, mode_v), {})
            tab_name = (
                f"λ{int(round(wl * 1000))}nm θ{int(round(theta_i))}° {mode_v}"
            )
            cond_panel = _build_condition_panel(
                df_cond, scale, n_grid, wl, theta_i, mode_v, rmse_dict,
            )
            tabs.append((tab_name, cond_panel))
        components.append(pn.Tabs(*tabs, tabs_location="left"))

    # ── 指標テーブル ─────────────────────────────────────────────────────────────
    if metrics:
        rows = []
        for k in sorted(metrics.keys()):
            # 新命名規則 <name>_<method>_<deg>_<mode>: method は中央にある
            method_token = next(
                (s for s in ("_fft_", "_psd_", "_ml_") if s in k),
                next(
                    (s for s in ("_fft", "_psd", "_ml") if k.endswith(s)),
                    None,
                ),
            )
            if k.startswith("log_rmse"):
                category = "Comparison"
                metric_name = k
            elif method_token:
                category = f"Optical ({method_token.strip('_').upper()})"
                metric_name = k
            else:
                category = "Surface"
                metric_name = k
            rows.append({
                "Category": category,
                "Metric": metric_name,
                "Key": k,
                "Value": f"{metrics[k]:.6g}",
            })
        metrics_df = pd.DataFrame(rows)[["Category", "Metric", "Value"]]
        components.append(pn.pane.Markdown("### Metrics"))
        components.append(pn.pane.DataFrame(metrics_df, index=False, width=500))

    return pn.Column(*components)


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
