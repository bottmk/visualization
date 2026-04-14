"""FFT法によるBSDF計算（スカラー回折理論）。

spec_main.md Section 3.2 の仕様に従い実装する。
FFT法はスカラー近似であり偏光依存性は扱わない（偏光はPSD法のQ因子に委ねる）。

物理単位は μm 統一。
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from ..models.base import HeightMap

logger = logging.getLogger(__name__)

# 広角警告の閾値 [deg]（スカラー近似の限界）
_WIDE_ANGLE_WARNING_DEG = 30.0


def compute_bsdf_fft(
    height_map: HeightMap,
    wavelength_um: float,
    theta_i_deg: float,
    phi_i_deg: float,
    n1: float = 1.0,
    n2: float = 1.5,
    polarization: str = "Unpolarized",
    is_btdf: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FFT法（スカラー回折理論）でBSDFを計算する。

    Args:
        height_map: 高さマップ（HeightMap dataclass）
        wavelength_um: 波長 [μm]
        theta_i_deg: 入射天頂角 [deg]（BTDF の場合は表面側換算済みの有効角）
        phi_i_deg: 入射方位角 [deg]
        n1: 入射側媒質の屈折率
        n2: 透過側媒質の屈折率（BRDFモードでは使用しない）
        polarization: 'S' / 'P' / 'Unpolarized'（FFT法では偏光依存性なし）
        is_btdf: True の場合 BTDF として位相変換を行う

    Returns:
        u_grid: 方向余弦 u = sin(theta_s)*cos(phi_s) の2次元グリッド
        v_grid: 方向余弦 v = sin(theta_s)*sin(phi_s) の2次元グリッド
        bsdf: BSDF値 [sr⁻¹] の2次元グリッド（u_grid, v_grid と同形状）

    Raises:
        ValueError: theta_i_deg が無効な範囲の場合
    """
    # 広角 × 偏光の警告
    if polarization in ("S", "P"):
        logger.warning(
            f"FFT法（スカラー近似）で偏光（{polarization}）が指定された。"
            f"広角（>{_WIDE_ANGLE_WARNING_DEG}°）では誤差が大きくなる。"
            f"偏光依存BSDFはPSD法を使用すること。"
        )

    h = height_map.data.astype(np.float64)
    N = height_map.grid_size
    dx = height_map.pixel_size_um  # [μm]
    k = 2 * np.pi / wavelength_um  # 波数 [μm⁻¹]

    theta_i_rad = np.deg2rad(theta_i_deg)
    phi_i_rad = np.deg2rad(phi_i_deg)
    cos_i = np.cos(theta_i_rad)
    sin_i = np.sin(theta_i_rad)

    # ── 位相変換 ─────────────────────────────────────────────────────────────
    if is_btdf:
        # BTDFモード: φ(x,y) = (2π/λ)*(n2*cos(θ_t) - n1*cos(θ_i))*h
        from .fresnel import snell_angle
        theta_t_deg = snell_angle(theta_i_deg, n1, n2)
        theta_t_rad = np.deg2rad(theta_t_deg)
        cos_t = np.cos(theta_t_rad)
        phi_surface = k * (n2 * cos_t - n1 * cos_i) * h
    else:
        # BRDFモード: φ(x,y) = (4π/λ)*n1*h*cos(θ_i)
        phi_surface = (4 * np.pi / wavelength_um) * n1 * h * cos_i

    # 斜入射時の x 方向傾き項（シフト不変性の活用）
    # φ_tilt(x) = (2π/λ)*n1*sin(θ_i)*cos(φ_i)*x
    #             + (2π/λ)*n1*sin(θ_i)*sin(φ_i)*y
    x_coords = np.arange(N) * dx
    y_coords = np.arange(N) * dx
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")
    phi_tilt = k * n1 * sin_i * (np.cos(phi_i_rad) * xx + np.sin(phi_i_rad) * yy)

    # 複素振幅
    U = np.exp(1j * (phi_surface + phi_tilt))

    # ── 2次元 FFT ─────────────────────────────────────────────────────────────
    U_fft = np.fft.fft2(U)
    # パワースペクトル（強度）
    I_fft = np.abs(U_fft) ** 2

    # ── 空間周波数グリッド → 方向余弦（UV）空間へマッピング ─────────────────
    freq_x = np.fft.fftfreq(N, d=dx)  # [μm⁻¹]
    freq_y = np.fft.fftfreq(N, d=dx)
    fx, fy = np.meshgrid(freq_x, freq_y, indexing="ij")

    # 方向余弦: u = f_x * λ, v = f_y * λ（スネルの法則より sin(θ_s) = f * λ）
    u_grid = fx * wavelength_um
    v_grid = fy * wavelength_um

    # 物理的に有効な範囲（u² + v² ≤ 1：半球内）
    uv_r2 = u_grid**2 + v_grid**2
    valid_mask = uv_r2 <= 1.0

    # ── BSDF への変換 ─────────────────────────────────────────────────────────
    # BSDF = I(u,v) / (N² * dx² * cos(θ_s))
    # cos(θ_s) = sqrt(1 - u² - v²)
    cos_s = np.where(valid_mask, np.sqrt(np.maximum(1.0 - uv_r2, 0.0)), 1.0)

    # 正規化係数: N² * (物理面積) * dΩ/du/dv = N² * dx²
    # BSDF [sr⁻¹] = I_fft / (N² * dx²) / cos_s * (1/λ²)
    # λ² は UV 空間と角度空間の面積変換に由来
    normalization = (N * dx) ** 2  # 全面積 [μm²]
    bsdf = np.where(
        valid_mask,
        I_fft / normalization / np.maximum(cos_s, 1e-10) / (wavelength_um**2),
        0.0,
    )

    # 広角散乱の警告チェック
    theta_s_max = float(np.rad2deg(np.arcsin(np.clip(np.sqrt(uv_r2[valid_mask].max()), 0, 1))))
    if theta_s_max > _WIDE_ANGLE_WARNING_DEG and polarization in ("S", "P"):
        logger.warning(
            f"散乱角の最大値 {theta_s_max:.1f}° > {_WIDE_ANGLE_WARNING_DEG}°。"
            f"スカラー近似の限界を超えているため偏光依存の誤差が大きくなる。"
        )

    return u_grid, v_grid, bsdf.astype(np.float32)


def sample_bsdf_at_angles(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    bsdf: np.ndarray,
    theta_s_deg: np.ndarray,
    phi_s_deg: np.ndarray,
) -> np.ndarray:
    """計算した BSDF グリッドから指定角度点でサンプリングする（双線形補間）。

    実測データの測定角度と直接比較するために使用する。
    UV 座標空間上で補間を行うことで天頂付近の座標歪みを防ぐ。

    Args:
        u_grid: 方向余弦 u グリッド（2D）
        v_grid: 方向余弦 v グリッド（2D）
        bsdf: BSDF グリッド（2D）
        theta_s_deg: サンプリングする散乱天頂角 [deg]（1D または 2D）
        phi_s_deg: サンプリングする散乱方位角 [deg]（1D または 2D）

    Returns:
        補間された BSDF 値（theta_s_deg と同形状）
    """
    from scipy.interpolate import RegularGridInterpolator

    # u_grid の軸を抽出し昇順にソート（FFT周波数軸は非単調のため）
    u_axis = u_grid[:, 0]
    v_axis = v_grid[0, :]
    u_sort = np.argsort(u_axis)
    v_sort = np.argsort(v_axis)

    interp = RegularGridInterpolator(
        (u_axis[u_sort], v_axis[v_sort]),
        bsdf[np.ix_(u_sort, v_sort)],
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    theta_s_rad = np.deg2rad(theta_s_deg)
    phi_s_rad = np.deg2rad(phi_s_deg)
    u_query = np.sin(theta_s_rad) * np.cos(phi_s_rad)
    v_query = np.sin(theta_s_rad) * np.sin(phi_s_rad)

    query_points = np.column_stack([u_query.ravel(), v_query.ravel()])
    result = interp(query_points).reshape(theta_s_deg.shape)
    return result.astype(np.float32)
