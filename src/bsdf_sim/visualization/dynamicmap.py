"""DynamicMap によるリアルタイム BSDF ダッシュボード（多モデル対応）。

spec_main.md Section 7:
- スライダー操作に連動してリアルタイム BSDF 再計算
- 3段階解像度: drag=128 / idle=512 / 本計算=4096
- LRU キャッシュ + Panel throttle による操作性と精度の両立
- 実測 BSDF ファイル（.bsdf 等）を config に指定すると、1D プロットに
  黒点 Scatter で自動オーバーレイされる

対応する表面モデル:
- RandomRoughSurface: rq_um / lc_um / fractal_dim スライダー
- SphericalArraySurface: radius_um / pitch_um / placement / overlap_mode
- MeasuredSurface 系（DeviceVk6Surface 等）: スライダー無し、固定表示
"""

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
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

from ..io.bsdf_reader import load_bsdf_readers, read_bsdf_file, select_block
from ..io.config_loader import BSDFConfig
from ..models import create_model_from_config, load_plugins
from ..models.base import BaseSurfaceModel, HeightMap
from ..models.random_rough import RandomRoughSurface
from ..models.spherical_array import SphericalArraySurface
from ..optics.fft_bsdf import compute_bsdf_fft

logger = logging.getLogger(__name__)


def _check_holoviews() -> None:
    if not _HV_AVAILABLE:
        raise ImportError(
            "holoviews と panel が必要。"
            "`pip install holoviews panel bokeh` でインストールしてください。"
        )


# ── 共通ヘルパー ─────────────────────────────────────────────────────────────


def _extract_measured_profile(
    measured_df: pd.DataFrame,
    phi_s_tolerance_deg: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """実測 DataFrame から phi_s≈0 の 1D プロファイルを抽出する。

    Returns:
        (theta_s_deg, bsdf) の 1D 配列ペア
    """
    if measured_df is None or len(measured_df) == 0:
        return np.array([]), np.array([])
    mask = np.abs(measured_df["phi_s_deg"].values) < phi_s_tolerance_deg
    sub = measured_df[mask]
    if len(sub) == 0:
        return np.array([]), np.array([])
    order = np.argsort(sub["theta_s_deg"].values)
    return (
        sub["theta_s_deg"].values[order],
        np.maximum(sub["bsdf"].values[order], 1e-10),
    )


def _make_1d_overlay(
    u: np.ndarray,
    v: np.ndarray,
    bsdf: np.ndarray,
    scale: str,
    title: str,
    measured_profile: tuple[np.ndarray, np.ndarray] | None = None,
) -> Any:
    """UV グリッドから phi=0° プロファイルを抽出し、実測と重ね描きする。"""
    # sim: phi≈0 スライス（u 軸方向）
    # fftfreq では index=0 が v=0（phi=0 方向）。
    # N//2 は Nyquist 周波数で |v|>1（半球外）になり BSDF=0 になるため 0 を使う。
    u_axis = u[:, 0]
    bsdf_slice = bsdf[:, 0]
    theta_s = np.rad2deg(np.arcsin(np.clip(np.abs(u_axis), 0, 1)))
    valid = np.abs(u_axis) <= 1.0
    y_sim = np.maximum(bsdf_slice[valid], 1e-10)
    x_sim = theta_s[valid]

    sim_curve = hv.Curve(
        (x_sim, y_sim),
        kdims=["散乱角 θ_s [deg]"],
        vdims=["BSDF [sr⁻¹]"],
        label="FFT 計算",
    ).opts(color="blue", line_width=2)

    elements: list[Any] = [sim_curve]

    if measured_profile is not None:
        x_meas, y_meas = measured_profile
        if len(x_meas) > 0:
            meas_scatter = hv.Scatter(
                (x_meas, y_meas),
                kdims=["散乱角 θ_s [deg]"],
                vdims=["BSDF [sr⁻¹]"],
                label="実測データ",
            ).opts(color="black", size=6, marker="circle")
            elements.append(meas_scatter)

    overlay = hv.Overlay(elements).opts(
        title=title,
        width=700,
        height=450,
        legend_position="top_right",
        logy=(scale == "log"),
    )
    return overlay


def _format_surface_metrics_md(hm: HeightMap) -> str:
    """高さマップから ISO/JIS 指標を算出して Markdown 文字列にする。"""
    from ..metrics.surface import compute_all_surface_metrics
    try:
        m = compute_all_surface_metrics(hm)
        return (
            "**ISO 25178-2 S-パラメータ**\n"
            f"- Sq = {m['sq_um']*1000:.2f} nm\n"
            f"- Sa = {m['sa_um']*1000:.2f} nm\n"
            f"- Sp = {m['sp_um']*1000:.2f} nm\n"
            f"- Sz = {m['sz_um']*1000:.2f} nm\n"
            f"- Ssk = {m['ssk']:.3f}\n"
            f"- Sku = {m['sku']:.3f}\n"
            f"- Sdq = {m['sdq_rad']:.4f} rad\n"
            f"- Sdr = {m['sdr_pct']:.4f} %\n"
        )
    except Exception as e:
        return f"指標計算中...（{e}）"


# ── 基底クラス ────────────────────────────────────────────────────────────────


class _BaseBSDFDashboard(ABC):
    """BSDF リアルタイムダッシュボードの基底クラス。

    表面モデルに依存しない共通機能（BSDF 計算、1D プロット、実測オーバーレイ、
    サーブ）を提供する。サブクラスは `create_dashboard()` で UI を組み立てる。
    """

    def __init__(
        self,
        wavelength_um: float = 0.55,
        theta_i_deg: float = 0.0,
        phi_i_deg: float = 0.0,
        n1: float = 1.0,
        n2: float = 1.5,
        is_btdf: bool = False,
        pixel_size_um: float = 0.25,
        preview_grid_size_idle: int = 512,
        measured_dfs: list[pd.DataFrame] | None = None,
        measured_tolerance_deg: float = 1.0,
        measured_tolerance_nm: float = 5.0,
    ) -> None:
        _check_holoviews()
        self.wavelength_um = wavelength_um
        self.theta_i_deg = theta_i_deg
        self.phi_i_deg = phi_i_deg
        self.n1 = n1
        self.n2 = n2
        self.is_btdf = is_btdf
        self.pixel_size_um = pixel_size_um
        self.preview_grid_size_idle = preview_grid_size_idle
        self.measured_dfs = measured_dfs or []
        self.measured_tolerance_deg = measured_tolerance_deg
        self.measured_tolerance_nm = measured_tolerance_nm

        # 条件に一致する実測ブロックを事前抽出
        self._matched_meas_df: pd.DataFrame | None = None
        if self.measured_dfs:
            mode = "BTDF" if self.is_btdf else "BRDF"
            self._matched_meas_df = select_block(
                self.measured_dfs,
                wavelength_um=self.wavelength_um,
                theta_i_deg=self.theta_i_deg,
                mode=mode,
                tolerance_deg=self.measured_tolerance_deg,
                tolerance_nm=self.measured_tolerance_nm,
            )
            if self._matched_meas_df is None:
                logger.warning(
                    f"実測ブロックが見つからない: "
                    f"λ={self.wavelength_um * 1000:.0f}nm, "
                    f"θ_i={self.theta_i_deg:.1f}°, {mode}"
                )

    def _compute_bsdf_for_model(
        self, model: BaseSurfaceModel, grid_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """モデルから縮小グリッド HeightMap を生成し BSDF を計算する。"""
        hm = model.get_preview_height_map(
            mode="reduced_area", preview_grid_size=grid_size,
        )
        return compute_bsdf_fft(
            height_map=hm,
            wavelength_um=self.wavelength_um,
            theta_i_deg=self.theta_i_deg,
            phi_i_deg=self.phi_i_deg,
            n1=self.n1, n2=self.n2,
            is_btdf=self.is_btdf,
        )

    def _measured_profile(self) -> tuple[np.ndarray, np.ndarray] | None:
        """条件一致の実測ブロックから phi≈0 プロファイルを抽出。"""
        if self._matched_meas_df is None:
            return None
        return _extract_measured_profile(self._matched_meas_df)

    def _title_suffix(self) -> str:
        mode = "BTDF" if self.is_btdf else "BRDF"
        return (
            f"λ={self.wavelength_um * 1000:.0f}nm · "
            f"θ_i={self.theta_i_deg:.0f}° · {mode}"
        )

    @abstractmethod
    def create_dashboard(self) -> Any:
        """サブクラスが UI を組み立てて Panel Layout を返す。"""

    def serve(self, port: int = 5006, show: bool = True, **kwargs: Any) -> None:
        """ダッシュボードをローカルサーバーで起動する。"""
        dashboard = self.create_dashboard(**kwargs)
        pn.serve(dashboard, port=port, show=show)


# ── キャッシュ付き計算（RandomRough 用 / Spherical 用）──────────────────────


def _compute_bsdf_random_rough(
    rq_um: float,
    lc_um: float,
    fractal_dim: float,
    grid_size: int,
    pixel_size_um: float,
    wavelength_um: float,
    theta_i_deg: float,
    phi_i_deg: float,
    n1: float,
    n2: float,
    is_btdf: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """キャッシュなし BSDF 計算（RandomRoughSurface、内部用）。"""
    model = RandomRoughSurface(
        rq_um=rq_um, lc_um=lc_um, fractal_dim=fractal_dim,
        grid_size=grid_size, pixel_size_um=pixel_size_um, seed=seed,
    )
    hm = model.get_height_map()
    return compute_bsdf_fft(
        height_map=hm, wavelength_um=wavelength_um,
        theta_i_deg=theta_i_deg, phi_i_deg=phi_i_deg,
        n1=n1, n2=n2, is_btdf=is_btdf,
    )


def _compute_bsdf_spherical(
    radius_um: float,
    pitch_um: float,
    base_height_um: float,
    placement: str,
    overlap_mode: str,
    grid_size: int,
    pixel_size_um: float,
    wavelength_um: float,
    theta_i_deg: float,
    phi_i_deg: float,
    n1: float,
    n2: float,
    is_btdf: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """キャッシュなし BSDF 計算（SphericalArraySurface、内部用）。"""
    model = SphericalArraySurface(
        radius_um=radius_um, pitch_um=pitch_um,
        base_height_um=base_height_um,
        placement=placement, overlap_mode=overlap_mode,
        grid_size=grid_size, pixel_size_um=pixel_size_um, seed=seed,
    )
    hm = model.get_height_map()
    return compute_bsdf_fft(
        height_map=hm, wavelength_um=wavelength_um,
        theta_i_deg=theta_i_deg, phi_i_deg=phi_i_deg,
        n1=n1, n2=n2, is_btdf=is_btdf,
    )


# ── RandomRoughDynamicMap ────────────────────────────────────────────────────


class RandomRoughDynamicMap(_BaseBSDFDashboard):
    """RandomRoughSurface パラメータを対話的に探索するダッシュボード。

    Args:
        random_seed: プレビュー用乱数シード（キャッシュ有効化のため固定）
        cache_size: LRU キャッシュサイズ
        preview_grid_size_drag: ドラッグ中のグリッドサイズ
        preview_grid_size_idle: 停止後のグリッドサイズ
        （親クラスの引数は基底クラス参照）
    """

    def __init__(
        self,
        *,
        preview_grid_size_drag: int = 128,
        preview_grid_size_idle: int = 512,
        random_seed: int = 42,
        cache_size: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(preview_grid_size_idle=preview_grid_size_idle, **kwargs)
        self.preview_grid_size_drag = preview_grid_size_drag
        self.random_seed = random_seed
        self._cached_bsdf = functools.lru_cache(maxsize=cache_size)(
            _compute_bsdf_random_rough
        )

    def _compute_preview(
        self, rq_um: float, lc_um: float, fractal_dim: float, grid_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._cached_bsdf(
            rq_um=round(rq_um, 5),
            lc_um=round(lc_um, 4),
            fractal_dim=round(fractal_dim, 3),
            grid_size=grid_size,
            pixel_size_um=self.pixel_size_um,
            wavelength_um=self.wavelength_um,
            theta_i_deg=self.theta_i_deg,
            phi_i_deg=self.phi_i_deg,
            n1=self.n1, n2=self.n2,
            is_btdf=self.is_btdf,
            seed=self.random_seed,
        )

    def create_dashboard(
        self,
        rq_range: tuple[float, float] = (0.001, 0.1),
        lc_range: tuple[float, float] = (0.5, 20.0),
        fractal_range: tuple[float, float] = (2.0, 3.0),
    ) -> Any:
        rq_slider = pn.widgets.FloatSlider(
            name="RMS粗さ Rq [μm]", start=rq_range[0], end=rq_range[1],
            value=0.005, step=0.001,
        )
        lc_slider = pn.widgets.FloatSlider(
            name="相関長 Lc [μm]", start=lc_range[0], end=lc_range[1],
            value=2.0, step=0.1,
        )
        fractal_slider = pn.widgets.FloatSlider(
            name="フラクタル次元", start=fractal_range[0], end=fractal_range[1],
            value=2.5, step=0.05,
        )
        scale_selector = pn.widgets.RadioButtonGroup(
            name="BSDF スケール", options=["linear", "log"], value="log",
        )

        meas_profile = self._measured_profile()

        @pn.depends(
            rq=rq_slider, lc=lc_slider, fractal=fractal_slider, scale=scale_selector,
        )
        def update_plot(rq: float, lc: float, fractal: float, scale: str) -> Any:
            try:
                u, v, bsdf = self._compute_preview(
                    rq, lc, fractal, self.preview_grid_size_idle
                )
                title = (
                    f"BSDF (N={self.preview_grid_size_idle}) · {self._title_suffix()}"
                )
                return _make_1d_overlay(
                    u, v, bsdf, scale, title, measured_profile=meas_profile,
                )
            except Exception as e:
                logger.warning(f"BSDF 計算エラー: {e}")
                return hv.Text(0, 0, f"エラー: {e}")

        @pn.depends(rq=rq_slider, lc=lc_slider, fractal=fractal_slider)
        def update_metrics(rq: float, lc: float, fractal: float) -> str:
            try:
                model = RandomRoughSurface(
                    rq_um=rq, lc_um=lc, fractal_dim=fractal,
                    grid_size=64, pixel_size_um=self.pixel_size_um,
                    seed=self.random_seed,
                )
                return _format_surface_metrics_md(model.get_height_map())
            except Exception:
                return "指標計算中..."

        return pn.Column(
            pn.pane.Markdown("# BSDF ダッシュボード — RandomRoughSurface"),
            pn.Row(
                pn.Column(
                    rq_slider, lc_slider, fractal_slider, scale_selector,
                    pn.pane.Markdown(update_metrics),
                    width=300,
                ),
                pn.panel(update_plot),
            ),
            pn.pane.Markdown(
                f"*プレビュー: N={self.preview_grid_size_idle}（停止後）。"
                f"実測データは {'あり' if meas_profile else 'なし'}。*"
            ),
        )


# ── SphericalArrayDynamicMap ─────────────────────────────────────────────────


class SphericalArrayDynamicMap(_BaseBSDFDashboard):
    """SphericalArraySurface パラメータを対話的に探索するダッシュボード。

    スライダー:
      - radius_um（曲率半径）
      - pitch_um（配置ピッチ）
      - base_height_um（ベース高さ）
    選択:
      - placement: Grid / Hexagonal / Random / PoissonDisk
      - overlap_mode: Maximum / Additive
    """

    def __init__(
        self,
        *,
        preview_grid_size_idle: int = 512,
        random_seed: int = 42,
        cache_size: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(preview_grid_size_idle=preview_grid_size_idle, **kwargs)
        self.random_seed = random_seed
        self._cached_bsdf = functools.lru_cache(maxsize=cache_size)(
            _compute_bsdf_spherical
        )

    def _compute_preview(
        self,
        radius_um: float,
        pitch_um: float,
        base_height_um: float,
        placement: str,
        overlap_mode: str,
        grid_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._cached_bsdf(
            radius_um=round(radius_um, 3),
            pitch_um=round(pitch_um, 3),
            base_height_um=round(base_height_um, 4),
            placement=placement,
            overlap_mode=overlap_mode,
            grid_size=grid_size,
            pixel_size_um=self.pixel_size_um,
            wavelength_um=self.wavelength_um,
            theta_i_deg=self.theta_i_deg,
            phi_i_deg=self.phi_i_deg,
            n1=self.n1, n2=self.n2,
            is_btdf=self.is_btdf,
            seed=self.random_seed,
        )

    def create_dashboard(
        self,
        radius_range: tuple[float, float] = (5.0, 100.0),
        pitch_range: tuple[float, float] = (2.0, 100.0),
        base_height_range: tuple[float, float] = (0.0, 10.0),
    ) -> Any:
        radius_slider = pn.widgets.FloatSlider(
            name="曲率半径 R [μm]", start=radius_range[0], end=radius_range[1],
            value=50.0, step=1.0,
        )
        pitch_slider = pn.widgets.FloatSlider(
            name="配置ピッチ P [μm]", start=pitch_range[0], end=pitch_range[1],
            value=50.0, step=1.0,
        )
        base_slider = pn.widgets.FloatSlider(
            name="ベース高さ [μm]", start=base_height_range[0], end=base_height_range[1],
            value=0.0, step=0.1,
        )
        placement_select = pn.widgets.Select(
            name="配置アルゴリズム",
            options=["Hexagonal", "Grid", "Random", "PoissonDisk"],
            value="Hexagonal",
        )
        overlap_select = pn.widgets.Select(
            name="重なり処理", options=["Maximum", "Additive"], value="Maximum",
        )
        scale_selector = pn.widgets.RadioButtonGroup(
            name="BSDF スケール", options=["linear", "log"], value="log",
        )

        meas_profile = self._measured_profile()

        @pn.depends(
            radius=radius_slider, pitch=pitch_slider, base=base_slider,
            placement=placement_select, overlap=overlap_select, scale=scale_selector,
        )
        def update_plot(
            radius: float, pitch: float, base: float,
            placement: str, overlap: str, scale: str,
        ) -> Any:
            try:
                u, v, bsdf = self._compute_preview(
                    radius, pitch, base, placement, overlap,
                    self.preview_grid_size_idle,
                )
                title = (
                    f"BSDF (N={self.preview_grid_size_idle}) · {self._title_suffix()}"
                )
                return _make_1d_overlay(
                    u, v, bsdf, scale, title, measured_profile=meas_profile,
                )
            except Exception as e:
                logger.warning(f"BSDF 計算エラー: {e}")
                return hv.Text(0, 0, f"エラー: {e}")

        @pn.depends(
            radius=radius_slider, pitch=pitch_slider, base=base_slider,
            placement=placement_select, overlap=overlap_select,
        )
        def update_metrics(
            radius: float, pitch: float, base: float, placement: str, overlap: str,
        ) -> str:
            try:
                model = SphericalArraySurface(
                    radius_um=radius, pitch_um=pitch, base_height_um=base,
                    placement=placement, overlap_mode=overlap,
                    grid_size=128, pixel_size_um=self.pixel_size_um,
                    seed=self.random_seed,
                )
                return _format_surface_metrics_md(model.get_height_map())
            except Exception as e:
                return f"指標計算中...（{e}）"

        return pn.Column(
            pn.pane.Markdown("# BSDF ダッシュボード — SphericalArraySurface"),
            pn.Row(
                pn.Column(
                    radius_slider, pitch_slider, base_slider,
                    placement_select, overlap_select, scale_selector,
                    pn.pane.Markdown(update_metrics),
                    width=320,
                ),
                pn.panel(update_plot),
            ),
            pn.pane.Markdown(
                f"*プレビュー: N={self.preview_grid_size_idle}（停止後）。"
                f"実測データは {'あり' if meas_profile else 'なし'}。*"
            ),
        )


# ── MeasuredSurfaceDynamicMap ────────────────────────────────────────────────


class MeasuredSurfaceDynamicMap(_BaseBSDFDashboard):
    """MeasuredSurface 系（実測高さファイル）の固定表示ダッシュボード。

    スライダーは無し。形状ファイルと padding モードを切替可能。
    """

    def __init__(
        self, *, model: BaseSurfaceModel, **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model

    def create_dashboard(self) -> Any:
        scale_selector = pn.widgets.RadioButtonGroup(
            name="BSDF スケール", options=["linear", "log"], value="log",
        )

        meas_profile = self._measured_profile()

        @pn.depends(scale=scale_selector)
        def update_plot(scale: str) -> Any:
            try:
                u, v, bsdf = self._compute_bsdf_for_model(
                    self.model, self.preview_grid_size_idle,
                )
                title = (
                    f"BSDF (N={self.preview_grid_size_idle}) · {self._title_suffix()}"
                )
                return _make_1d_overlay(
                    u, v, bsdf, scale, title, measured_profile=meas_profile,
                )
            except Exception as e:
                logger.warning(f"BSDF 計算エラー: {e}")
                return hv.Text(0, 0, f"エラー: {e}")

        # 形状指標は固定表示（スライダーなし）
        try:
            hm = self.model.get_preview_height_map(
                mode="reduced_area", preview_grid_size=128,
            )
            metrics_md = _format_surface_metrics_md(hm)
        except Exception as e:
            metrics_md = f"指標計算エラー: {e}"

        model_name = type(self.model).__name__
        padding = getattr(self.model, "padding", "—")
        return pn.Column(
            pn.pane.Markdown(f"# BSDF ダッシュボード — {model_name}"),
            pn.pane.Markdown(
                f"**パディング**: `{padding}`  "
                f"**ピクセル**: {self.pixel_size_um} μm"
            ),
            pn.Row(
                pn.Column(
                    scale_selector,
                    pn.pane.Markdown(metrics_md),
                    width=320,
                ),
                pn.panel(update_plot),
            ),
            pn.pane.Markdown(
                f"*実測形状からの BSDF 計算。実測データは "
                f"{'あり' if meas_profile else 'なし'}。*"
            ),
        )


# ── ファクトリ関数 ───────────────────────────────────────────────────────────


def create_dashboard_from_config(
    config_path: str | Path,
    preview_grid_size_idle: int = 512,
) -> _BaseBSDFDashboard:
    """config.yaml から表面モデル種別を判定して適切なダッシュボードを返す。

    Args:
        config_path: 設定ファイルパス
        preview_grid_size_idle: プレビュー計算のグリッドサイズ

    Returns:
        表面モデルに応じた `_BaseBSDFDashboard` サブクラスのインスタンス

    Raises:
        ValueError: 対応していない表面モデルの場合
    """
    _check_holoviews()
    load_plugins()
    cfg = BSDFConfig.from_file(config_path)

    surface_cfg = cfg.surface
    model_name = surface_cfg.get("model", "RandomRoughSurface")

    # 共通引数（ダッシュボードが固定で使う光学条件）
    common = dict(
        wavelength_um=cfg.wavelength_um,
        theta_i_deg=cfg.theta_i_effective_deg,
        phi_i_deg=cfg.phi_i_deg,
        n1=cfg.n1, n2=cfg.n2,
        is_btdf=cfg.is_btdf,
        pixel_size_um=float(surface_cfg.get("pixel_size_um", 0.25)),
        preview_grid_size_idle=preview_grid_size_idle,
    )

    # 実測 BSDF ファイルの読み込み
    measured_dfs: list[pd.DataFrame] = []
    if cfg.measured_bsdf_path:
        logger.info(f"実測 BSDF ファイル読み込み中: {cfg.measured_bsdf_path}")
        load_bsdf_readers()
        measured_dfs = read_bsdf_file(cfg.measured_bsdf_path)
        logger.info(f"  ブロック数: {len(measured_dfs)}")
    common["measured_dfs"] = measured_dfs
    common["measured_tolerance_deg"] = cfg.match_tolerance_deg
    common["measured_tolerance_nm"] = cfg.match_tolerance_nm

    if model_name == "RandomRoughSurface":
        rr = surface_cfg.get("random_rough", {}) or {}
        return RandomRoughDynamicMap(
            random_seed=int(rr.get("seed", 42)),
            **common,
        )

    if model_name == "SphericalArraySurface":
        sa = surface_cfg.get("spherical_array", {}) or {}
        return SphericalArrayDynamicMap(
            random_seed=int(sa.get("seed", 42)),
            **common,
        )

    # MeasuredSurface / DeviceVk6Surface / DeviceXyzSurface 等の実測系
    # （create_model_from_config でプラグインも含めてインスタンス化）
    try:
        model = create_model_from_config(cfg._resolved)
    except Exception as e:
        raise ValueError(
            f"ダッシュボード非対応の表面モデル: '{model_name}' — {e}"
        ) from e
    return MeasuredSurfaceDynamicMap(model=model, **common)
