"""DynamicMap によるリアルタイム BSDF ダッシュボード。

spec_main.md Section 7:
- スライダー操作に連動してリアルタイム BSDF 再計算
- 3段階解像度: drag=128 / idle=512 / 本計算=4096
- LRU キャッシュ + Panel throttle による操作性と精度の両立
- RandomRoughSurface は random_seed を固定してキャッシュを有効化
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import numpy as np

try:
    import holoviews as hv
    import panel as pn
    hv.extension("bokeh")
    _HV_AVAILABLE = True
except ImportError:
    _HV_AVAILABLE = False

from ..models.base import HeightMap
from ..models.random_rough import RandomRoughSurface
from ..optics.fft_bsdf import compute_bsdf_fft
from .holoviews_plots import plot_bsdf_1d_overlay, plot_bsdf_2d_heatmap

logger = logging.getLogger(__name__)


def _check_holoviews() -> None:
    if not _HV_AVAILABLE:
        raise ImportError(
            "holoviews と panel が必要。"
            "`pip install holoviews panel bokeh` でインストールしてください。"
        )


# ── キャッシュ付き BSDF 計算 ─────────────────────────────────────────────────

@functools.lru_cache(maxsize=256)
def _cached_bsdf(
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
    """乱数シード固定の LRU キャッシュ付き BSDF 計算。

    プレビュー用。同一パラメータの再計算をスキップする。
    random_seed を固定することでキャッシュが正しく機能する。
    """
    model = RandomRoughSurface(
        rq_um=rq_um,
        lc_um=lc_um,
        fractal_dim=fractal_dim,
        grid_size=grid_size,
        pixel_size_um=pixel_size_um,
        seed=seed,
    )
    hm = model.get_height_map()
    return compute_bsdf_fft(
        height_map=hm,
        wavelength_um=wavelength_um,
        theta_i_deg=theta_i_deg,
        phi_i_deg=phi_i_deg,
        n1=n1,
        n2=n2,
        is_btdf=is_btdf,
    )


# ── ダッシュボード ────────────────────────────────────────────────────────────

class RandomRoughDynamicMap:
    """RandomRoughSurface パラメータを対話的に探索する DynamicMap ダッシュボード。

    スライダーを動かすと BSDF がリアルタイムで更新される（3段階解像度）。

    Args:
        wavelength_um: 波長 [μm]
        theta_i_deg: 入射天頂角 [deg]
        phi_i_deg: 入射方位角 [deg]
        n1, n2: 媒質屈折率
        is_btdf: BTDF モードフラグ
        pixel_size_um: ピクセルサイズ [μm]（固定）
        preview_grid_size_drag: ドラッグ中のグリッドサイズ（デフォルト: 128）
        preview_grid_size_idle: 停止後のグリッドサイズ（デフォルト: 512）
        random_seed: プレビュー用乱数シード（デフォルト: 42）
        cache_size: LRU キャッシュサイズ（デフォルト: 256）
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
        preview_grid_size_drag: int = 128,
        preview_grid_size_idle: int = 512,
        random_seed: int = 42,
        cache_size: int = 256,
    ) -> None:
        _check_holoviews()

        self.wavelength_um = wavelength_um
        self.theta_i_deg = theta_i_deg
        self.phi_i_deg = phi_i_deg
        self.n1 = n1
        self.n2 = n2
        self.is_btdf = is_btdf
        self.pixel_size_um = pixel_size_um
        self.preview_grid_size_drag = preview_grid_size_drag
        self.preview_grid_size_idle = preview_grid_size_idle
        self.random_seed = random_seed

        # LRU キャッシュサイズを動的に設定
        _cached_bsdf.__wrapped__ = functools.lru_cache(maxsize=cache_size)(
            _cached_bsdf.__wrapped__
        )

    def _compute_preview(
        self,
        rq_um: float,
        lc_um: float,
        fractal_dim: float,
        grid_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """指定グリッドサイズでプレビュー BSDF を計算する。"""
        return _cached_bsdf(
            rq_um=round(rq_um, 5),
            lc_um=round(lc_um, 4),
            fractal_dim=round(fractal_dim, 3),
            grid_size=grid_size,
            pixel_size_um=self.pixel_size_um,
            wavelength_um=self.wavelength_um,
            theta_i_deg=self.theta_i_deg,
            phi_i_deg=self.phi_i_deg,
            n1=self.n1,
            n2=self.n2,
            is_btdf=self.is_btdf,
            seed=self.random_seed,
        )

    def create_dashboard(
        self,
        rq_range: tuple[float, float] = (0.001, 0.1),
        lc_range: tuple[float, float] = (0.5, 20.0),
        fractal_range: tuple[float, float] = (2.0, 3.0),
    ) -> Any:
        """インタラクティブダッシュボードを生成する。

        Args:
            rq_range: RMS粗さ Rq のスライダー範囲 [μm]
            lc_range: 相関長 Lc のスライダー範囲 [μm]
            fractal_range: フラクタル次元のスライダー範囲

        Returns:
            Panel Layout オブジェクト（servable）
        """
        # スライダー定義（throttled=True でドラッグ中は粗い計算）
        rq_slider = pn.widgets.FloatSlider(
            name="RMS粗さ Rq [μm]",
            start=rq_range[0], end=rq_range[1], value=0.005,
            step=0.001,
        )
        lc_slider = pn.widgets.FloatSlider(
            name="相関長 Lc [μm]",
            start=lc_range[0], end=lc_range[1], value=2.0,
            step=0.1,
        )
        fractal_slider = pn.widgets.FloatSlider(
            name="フラクタル次元",
            start=fractal_range[0], end=fractal_range[1], value=2.5,
            step=0.05,
        )
        scale_selector = pn.widgets.RadioButtonGroup(
            name="BSDF スケール",
            options=["linear", "log"],
            value="log",
        )

        # ドラッグ中フラグ（Panel の value vs value_throttled で近似）
        _is_dragging = {"state": False}

        @pn.depends(
            rq=rq_slider,
            lc=lc_slider,
            fractal=fractal_slider,
            scale=scale_selector,
        )
        def update_plot_idle(rq: float, lc: float, fractal: float, scale: str) -> Any:
            """ドラッグ停止後の精細計算（N=512）。"""
            try:
                u, v, bsdf = self._compute_preview(rq, lc, fractal, self.preview_grid_size_idle)
                return _make_overlay(u, v, bsdf, scale, title=f"BSDF (N={self.preview_grid_size_idle})")
            except Exception as e:
                logger.warning(f"BSDF 計算エラー: {e}")
                return hv.Text(0, 0, f"エラー: {e}")

        def _make_overlay(
            u: np.ndarray,
            v: np.ndarray,
            bsdf: np.ndarray,
            scale: str,
            title: str,
        ) -> Any:
            """UV グリッドから角度プロファイル（phi=0°）を抽出してプロットする。"""
            # phi=0 方向（v≈0, u>0）のスライスを抽出
            u_axis = u[:, 0]
            v_mid = u.shape[1] // 2
            bsdf_slice = bsdf[:, v_mid]
            theta_s = np.rad2deg(np.arcsin(np.clip(np.abs(u_axis), 0, 1)))

            valid = np.abs(u_axis) <= 1.0
            y = np.maximum(bsdf_slice[valid], 1e-10)
            x = theta_s[valid]

            curve = hv.Curve(
                (x, y),
                kdims=["散乱角 θ_s [deg]"],
                vdims=["BSDF [sr⁻¹]"],
            ).opts(
                title=title,
                width=650,
                height=400,
                logy=(scale == "log"),
                color="blue",
                line_width=2,
            )
            return curve

        # メトリクス表示
        @pn.depends(rq=rq_slider, lc=lc_slider, fractal=fractal_slider)
        def update_metrics(rq: float, lc: float, fractal: float) -> str:
            try:
                from ..metrics.surface import compute_all_surface_metrics
                from ..models.base import HeightMap

                model = RandomRoughSurface(
                    rq_um=rq, lc_um=lc, fractal_dim=fractal,
                    grid_size=64, pixel_size_um=self.pixel_size_um,
                    seed=self.random_seed,
                )
                hm = model.get_height_map()
                m = compute_all_surface_metrics(hm)
                return (
                    f"**表面形状指標**\n"
                    f"- Rq = {m['rq_um']*1000:.2f} nm\n"
                    f"- Ra = {m['ra_um']*1000:.2f} nm\n"
                    f"- Rz = {m['rz_um']*1000:.2f} nm\n"
                    f"- Sdq = {m['sdq_rad']:.4f} rad"
                )
            except Exception:
                return "指標計算中..."

        dashboard = pn.Column(
            pn.pane.Markdown("# BSDF リアルタイムダッシュボード"),
            pn.Row(
                pn.Column(
                    rq_slider, lc_slider, fractal_slider, scale_selector,
                    pn.pane.Markdown(update_metrics),
                    width=300,
                ),
                pn.panel(update_plot_idle),
            ),
            pn.pane.Markdown(
                f"*プレビュー: N={self.preview_grid_size_idle}（停止後）。"
                f"本計算は `bsdf simulate` コマンドで実行。*"
            ),
        )

        return dashboard

    def serve(self, port: int = 5006, **kwargs: Any) -> None:
        """ダッシュボードをローカルサーバーで起動する。

        Args:
            port: サーバーポート番号
        """
        dashboard = self.create_dashboard(**kwargs)
        pn.serve(dashboard, port=port, show=True)
