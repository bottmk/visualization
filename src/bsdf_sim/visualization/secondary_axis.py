"""BSDF 1D プロットの副軸（上段 X 軸）ユニット定義と変換ヘルパ。

散乱角 θ_s [deg] を主軸とした 1D プロットに、**波動ベクトル系のアナロジー**
を使って副軸を表示する。単位が違うだけで情報は同一（すべて θ_s の単調関数）。

Unit 一覧:
    - 'theta_s'      : 散乱角 θ_s [deg]（主軸相当、副軸表示は通常しない）
    - 'lambda_scale' : 構造スケール Λ = λ/sin(θ_s) [μm]（BSDF 解析で最も有用）
    - 'u'            : 方向余弦 u = sin(θ_s)
    - 'f'            : 空間周波数 f = sin(θ_s)/λ [μm⁻¹]
    - 'k_x'          : 横方向波数 k_x = 2π·sin(θ_s)/λ [rad/μm]

θ_s=0 では Λ と 1/f は発散するため、プロット時は `θ_s > 0.1°` に制限される
想定（X 軸 log スケールと同じ扱い）。

方言の対応表:
    solid-state / antenna    : k-space, visible region
    imaging optics           : pupil plane, sine-space
    diffraction / crystallogr: spatial frequency, reciprocal space
    BSDF (本プロジェクト)    : 方向余弦 u, v
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


class AxisUnitSpec:
    """副軸ユニットの仕様（ラベル + θ_s ↔ 値の変換関数）。"""

    def __init__(
        self,
        label: str,
        label_en: str,
        from_theta: Callable[[np.ndarray, float], np.ndarray],
        to_theta: Callable[[np.ndarray, float], np.ndarray],
        js_from_theta: str,
        log_scale_recommended: bool = False,
    ) -> None:
        """
        Args:
            label: 軸ラベル（日本語、Bokeh HTML 用）
            label_en: 軸ラベル（英語、matplotlib PNG 用・tofu 回避）
            from_theta: (θ_s_deg, λ_μm) → 副軸の値
            to_theta:   (副軸の値, λ_μm) → θ_s_deg
            js_from_theta: Bokeh FuncTickFormatter 用の JS コード断片。
                `tick` 変数に θ_s_deg が、`lam` 変数に λ_μm が入る前提で、
                副軸の表示文字列を返す式。例:
                `(lam / Math.sin(tick * Math.PI / 180)).toFixed(2)`
            log_scale_recommended: True の場合、matplotlib 副軸は log スケール
                で表示（値が複数桁に跨る場合に推奨）。
        """
        self.label = label
        self.label_en = label_en
        self.from_theta = from_theta
        self.to_theta = to_theta
        self.js_from_theta = js_from_theta
        self.log_scale_recommended = log_scale_recommended


# ── ユニット定義レジストリ ────────────────────────────────────────────────

# 全関数は np.clip でゼロ近傍を抑え、log(0) / div-by-zero を回避する
_EPS_DEG = 1e-6


def _theta_deg_to_sin(theta_deg: np.ndarray | float) -> np.ndarray:
    return np.sin(np.deg2rad(np.clip(np.asarray(theta_deg, dtype=np.float64), _EPS_DEG, 90.0)))


AXIS_UNITS: dict[str, AxisUnitSpec] = {
    "theta_s": AxisUnitSpec(
        label="散乱角 θ_s [deg]",
        label_en="Scattering angle theta_s [deg]",
        from_theta=lambda t, lam: np.asarray(t, dtype=np.float64),
        to_theta=lambda v, lam: np.asarray(v, dtype=np.float64),
        js_from_theta="tick.toFixed(1)",
    ),
    "lambda_scale": AxisUnitSpec(
        label="構造スケール Λ [μm]",
        label_en="Structure scale Lambda [um]",
        from_theta=lambda t, lam: lam / _theta_deg_to_sin(t),
        to_theta=lambda v, lam: np.rad2deg(np.arcsin(np.clip(lam / np.maximum(np.asarray(v, dtype=np.float64), 1e-12), 0, 1))),
        js_from_theta=(
            "var s = Math.sin(tick * Math.PI / 180); "
            "if (s <= 0) { return '∞'; } "
            "var v = lam / s; "
            "return (v >= 100) ? v.toFixed(0) : (v >= 10) ? v.toFixed(1) : v.toFixed(2);"
        ),
        log_scale_recommended=True,
    ),
    "u": AxisUnitSpec(
        label="方向余弦 u = sin θ_s",
        label_en="Direction cosine u = sin(theta_s)",
        from_theta=lambda t, lam: _theta_deg_to_sin(t),
        to_theta=lambda v, lam: np.rad2deg(np.arcsin(np.clip(np.asarray(v, dtype=np.float64), -1.0, 1.0))),
        js_from_theta="Math.sin(tick * Math.PI / 180).toFixed(3)",
    ),
    "f": AxisUnitSpec(
        label="空間周波数 f [μm⁻¹]",
        label_en="Spatial frequency f [1/um]",
        from_theta=lambda t, lam: _theta_deg_to_sin(t) / lam,
        to_theta=lambda v, lam: np.rad2deg(np.arcsin(np.clip(np.asarray(v, dtype=np.float64) * lam, -1.0, 1.0))),
        js_from_theta="(Math.sin(tick * Math.PI / 180) / lam).toFixed(3)",
    ),
    "k_x": AxisUnitSpec(
        label="波数 k_x [rad/μm]",
        label_en="Wavenumber k_x [rad/um]",
        from_theta=lambda t, lam: (2 * np.pi / lam) * _theta_deg_to_sin(t),
        to_theta=lambda v, lam: np.rad2deg(
            np.arcsin(np.clip(np.asarray(v, dtype=np.float64) * lam / (2 * np.pi), -1.0, 1.0))
        ),
        js_from_theta="(2 * Math.PI * Math.sin(tick * Math.PI / 180) / lam).toFixed(2)",
    ),
}


DEFAULT_SECONDARY_X_UNIT: str = "lambda_scale"


def get_axis_unit_spec(unit: str) -> AxisUnitSpec:
    """ユニット名から AxisUnitSpec を取得。未知の値なら ValueError。"""
    if unit not in AXIS_UNITS:
        raise ValueError(
            f"Unknown secondary x-axis unit: {unit!r}. "
            f"Valid: {sorted(AXIS_UNITS.keys())}"
        )
    return AXIS_UNITS[unit]


# ── matplotlib 用ヘルパ ───────────────────────────────────────────────────

def add_secondary_xaxis_mpl(
    ax: Any,
    unit: str,
    wavelength_um: float,
    *,
    prefer_english: bool = True,
    force_log_scale: bool | None = None,
) -> Any:
    """matplotlib Axes に上段副軸を `ax.twiny()` で追加する。

    主軸は θ_s [deg] を想定。副軸は `unit` に対応する量で目盛ラベルを表示。
    `secondary_xaxis(functions=...)` は forward が monotonic-decreasing の
    log 軸で左右が swap されるバグ的挙動があるため、`ax.twiny()` で明示的に
    range を設定して制御する（forward が decreasing でも正しく描画される）。

    Args:
        ax: matplotlib Axes（主軸 x は θ_s [deg]）
        unit: 'lambda_scale' / 'u' / 'f' / 'k_x' / 'theta_s' のいずれか
        wavelength_um: 波長 [μm]。`from_theta` に渡す
        prefer_english: True（既定）で英語ラベルを使う。matplotlib の
            headless Agg バックエンドは日本語フォント未搭載のため、PNG 出力時は
            英語ラベルにしないと tofu（□）になる。
        force_log_scale: 副軸を log スケールにするか。None（既定）で
            spec.log_scale_recommended に従う（Λ は log、u/theta_s は linear）。

    Returns:
        twin Axes オブジェクト（ax.twiny() の戻り値）
    """
    spec = get_axis_unit_spec(unit)
    xmin, xmax = ax.get_xlim()
    thetas = np.array([max(xmin, 1e-6), xmax], dtype=np.float64)
    sec_vals = spec.from_theta(thetas, wavelength_um)
    y_left, y_right = float(sec_vals[0]), float(sec_vals[1])

    ax2 = ax.twiny()
    # primary と secondary の描画範囲を対応付け（forward が decreasing でも OK）
    ax2.set_xlim(y_left, y_right)
    # primary の log/linear を継承（tick 位置の揃いを保つため）
    try:
        ax2.set_xscale(ax.get_xscale())
    except Exception:
        pass
    # 副軸自体のスケール（log / linear）
    use_log = force_log_scale if force_log_scale is not None else spec.log_scale_recommended
    if use_log:
        try:
            ax2.set_xscale("log")
        except Exception:
            pass
    ax2.set_xlabel(spec.label_en if prefer_english else spec.label)
    return ax2


# ── Bokeh (HoloViews hook) 用ヘルパ ────────────────────────────────────────

def make_secondary_xaxis_hook(
    unit: str,
    wavelength_um: float,
) -> Callable[[Any, Any], None]:
    """HoloViews .opts(hooks=[...]) に渡せる Bokeh 副軸追加フックを生成する。

    副軸は**主軸 x_range を共有**し、`FuncTickFormatter` で θ_s_deg → 副軸値
    の変換を JS 側で行う（tick 位置は主軸と同じ）。

    Args:
        unit: 副軸ユニット名
        wavelength_um: 波長 [μm]（JS 側に定数として埋め込む）

    Returns:
        `hook(plot, element)` 形式の callable。HoloViews の opts(hooks=[...])
        に渡すとフック時に Bokeh Plot に副軸を追加する。
    """
    spec = get_axis_unit_spec(unit)
    lam = float(wavelength_um)

    def _hook(plot: Any, element: Any) -> None:
        try:
            from bokeh.models import FuncTickFormatter, LinearAxis
        except ImportError:
            return
        try:
            p = plot.handles.get("plot")
            if p is None:
                return
            # 既に副軸がある（フックが二重実行された）なら何もしない
            for existing in p.above:
                if getattr(existing, "name", "") == "secondary_x_bsdf":
                    return
            formatter = FuncTickFormatter(
                args=dict(lam=lam),
                code=spec.js_from_theta,
            )
            sec = LinearAxis(
                axis_label=spec.label,
                formatter=formatter,
                name="secondary_x_bsdf",
            )
            p.add_layout(sec, "above")
        except Exception:
            # プロット失敗時は副軸追加をスキップ
            pass

    return _hook
