"""PSD法によるBSDF計算（Rayleigh-Rice理論近似）。

spec_main.md Section 3.3 および Appendix A の仕様に従い実装する。

偏光因子 Q:
- 完全形（デフォルト）: Elson-Bennett形式。θ_i と θ_s（θ_t）両方でフレネル評価。
- 簡略形（approx_mode=True）: Stover近似。θ_i のみでフレネル評価。

物理単位は μm 統一。
"""

from __future__ import annotations

import numpy as np

from ..models.base import HeightMap
from .fresnel import fresnel_rp, fresnel_rs, fresnel_tp, fresnel_ts, snell_angle


def compute_psd_2d(height_map: HeightMap) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """高さマップの2次元パワースペクトル密度（PSD）を計算する。

    PSD(f_x, f_y) = |FFT(h)|² * dx² / (N²)

    Args:
        height_map: 高さマップ

    Returns:
        fx: 空間周波数グリッド x 軸 [μm⁻¹]（2D）
        fy: 空間周波数グリッド y 軸 [μm⁻¹]（2D）
        psd: パワースペクトル密度 [μm⁴]（2D）
    """
    h = height_map.data.astype(np.float64)
    N = height_map.grid_size
    dx = height_map.pixel_size_um

    H_fft = np.fft.fft2(h)
    psd = (np.abs(H_fft) ** 2) * (dx**2) / (N**2)

    freq = np.fft.fftfreq(N, d=dx)
    fx, fy = np.meshgrid(freq, freq, indexing="ij")
    return fx, fy, psd.astype(np.float64)


def _Q_complete_brdf(
    theta_i_deg: float,
    theta_s_deg: np.ndarray,
    n1: float,
    n2: float,
    polarization: str,
) -> np.ndarray:
    """BRDFモード・完全形（Elson-Bennett）の偏光因子 Q を計算する。

    Q_s_refl = |r_s(θ_i)*cos(θ_s) + r_s(θ_s)*cos(θ_i)|² / (2*cos(θ_i))²
    Q_p_refl = |r_p(θ_i)*cos(θ_s) + r_p(θ_s)*cos(θ_i)|² / (2*cos(θ_i))²
    Q_u_refl = (Q_s + Q_p) / 2

    Args:
        theta_i_deg: 入射角 [deg]
        theta_s_deg: 散乱角グリッド [deg]（2D array）
        n1, n2: 媒質屈折率
        polarization: 'S' / 'P' / 'Unpolarized'

    Returns:
        Q グリッド（theta_s_deg と同形状）
    """
    theta_i_rad = np.deg2rad(theta_i_deg)
    cos_i = np.cos(theta_i_rad)
    rs_i = complex(fresnel_rs(theta_i_deg, n1, n2))
    rp_i = complex(fresnel_rp(theta_i_deg, n1, n2))

    theta_s_rad = np.deg2rad(theta_s_deg)
    cos_s = np.cos(theta_s_rad)

    # θ_s ごとのフレネル係数
    rs_s = np.vectorize(lambda ts: fresnel_rs(ts, n1, n2))(theta_s_deg)
    rp_s = np.vectorize(lambda ts: fresnel_rp(ts, n1, n2))(theta_s_deg)

    denom = (2 * cos_i) ** 2

    Q_s = np.abs(rs_i * cos_s + rs_s * cos_i) ** 2 / denom
    Q_p = np.abs(rp_i * cos_s + rp_s * cos_i) ** 2 / denom

    if polarization == "S":
        return Q_s.astype(np.float64)
    elif polarization == "P":
        return Q_p.astype(np.float64)
    else:  # Unpolarized
        return ((Q_s + Q_p) / 2).astype(np.float64)


def _Q_complete_btdf(
    theta_i_deg: float,
    theta_s_deg: np.ndarray,
    n1: float,
    n2: float,
    polarization: str,
) -> np.ndarray:
    """BTDFモード・完全形（Elson-Bennett）の偏光因子 Q を計算する。

    E = (n2*cos(θ_t)) / (n1*cos(θ_i))  （エネルギー保存係数）
    Q_s_trans = E * |t_s(θ_i)*cos(θ_t) + t_s(θ_t)*cos(θ_i)|² / (2*cos(θ_i))²
    Q_p_trans = E * |t_p(θ_i)*cos(θ_t) + t_p(θ_t)*cos(θ_i)|² / (2*cos(θ_i))²

    Args:
        theta_i_deg: 入射角（表面側換算済み）[deg]
        theta_s_deg: 散乱角グリッド [deg]（2D array）
        n1, n2: 媒質屈折率
        polarization: 'S' / 'P' / 'Unpolarized'

    Returns:
        Q グリッド（theta_s_deg と同形状）
    """
    theta_i_rad = np.deg2rad(theta_i_deg)
    cos_i = np.cos(theta_i_rad)

    theta_t_i_deg = snell_angle(theta_i_deg, n1, n2)
    theta_t_i_rad = np.deg2rad(theta_t_i_deg)
    cos_t_i = np.cos(theta_t_i_rad)

    E = (n2 * cos_t_i) / (n1 * cos_i)

    ts_i = complex(fresnel_ts(theta_i_deg, n1, n2))
    tp_i = complex(fresnel_tp(theta_i_deg, n1, n2))

    theta_s_rad = np.deg2rad(theta_s_deg)
    cos_s = np.cos(theta_s_rad)

    # θ_t(θ_s) の計算: スネル則で θ_s（散乱側）から θ_t を導出
    # 散乱側を θ_s として、対応する透過側フレネル係数を θ_s の角度で評価
    def safe_snell_and_fresnel_ts(ts_deg: float) -> complex:
        try:
            return fresnel_ts(ts_deg, n1, n2)
        except ValueError:
            return 0.0 + 0j

    def safe_snell_and_fresnel_tp(ts_deg: float) -> complex:
        try:
            return fresnel_tp(ts_deg, n1, n2)
        except ValueError:
            return 0.0 + 0j

    ts_s = np.vectorize(safe_snell_and_fresnel_ts)(theta_s_deg)
    tp_s = np.vectorize(safe_snell_and_fresnel_tp)(theta_s_deg)

    denom = (2 * cos_i) ** 2
    Q_s = E * np.abs(ts_i * cos_s + ts_s * cos_i) ** 2 / denom
    Q_p = E * np.abs(tp_i * cos_s + tp_s * cos_i) ** 2 / denom

    if polarization == "S":
        return Q_s.astype(np.float64)
    elif polarization == "P":
        return Q_p.astype(np.float64)
    else:
        return ((Q_s + Q_p) / 2).astype(np.float64)


def _Q_simplified_brdf(
    theta_i_deg: float,
    theta_s_deg: np.ndarray,
    n1: float,
    n2: float,
    polarization: str,
) -> np.ndarray:
    """BRDFモード・簡略形（Stover近似）の偏光因子 Q を計算する。

    Q_s = |r_s(θ_i)|²
    Q_p = |r_p(θ_i)|² * cos²(θ_i + θ_s)

    注意: 広角散乱（θ_s > 30°）では誤差が増大する。
    """
    rs_i = complex(fresnel_rs(theta_i_deg, n1, n2))
    rp_i = complex(fresnel_rp(theta_i_deg, n1, n2))

    theta_i_rad = np.deg2rad(theta_i_deg)
    theta_s_rad = np.deg2rad(theta_s_deg)

    Q_s = np.full_like(theta_s_deg, abs(rs_i) ** 2, dtype=np.float64)
    Q_p = abs(rp_i) ** 2 * np.cos(theta_i_rad + theta_s_rad) ** 2

    if polarization == "S":
        return Q_s
    elif polarization == "P":
        return Q_p.astype(np.float64)
    else:
        return ((Q_s + Q_p) / 2).astype(np.float64)


def _Q_simplified_btdf(
    theta_i_deg: float,
    theta_s_deg: np.ndarray,
    n1: float,
    n2: float,
    polarization: str,
) -> np.ndarray:
    """BTDFモード・簡略形（Stover近似）の偏光因子 Q を計算する。

    E = (n2*cos(θ_t)) / (n1*cos(θ_i))
    Q_s = E * |t_s(θ_i)|²
    Q_p = E * |t_p(θ_i)|² * cos²(θ_i - θ_t)
    """
    theta_t_deg = snell_angle(theta_i_deg, n1, n2)
    theta_i_rad = np.deg2rad(theta_i_deg)
    theta_t_rad = np.deg2rad(theta_t_deg)
    cos_i = np.cos(theta_i_rad)
    cos_t = np.cos(theta_t_rad)

    E = (n2 * cos_t) / (n1 * cos_i)
    ts_i = complex(fresnel_ts(theta_i_deg, n1, n2))
    tp_i = complex(fresnel_tp(theta_i_deg, n1, n2))

    Q_s = np.full_like(theta_s_deg, E * abs(ts_i) ** 2, dtype=np.float64)
    Q_p = E * abs(tp_i) ** 2 * np.cos(theta_i_rad - theta_t_rad) ** 2

    if polarization == "S":
        return Q_s
    elif polarization == "P":
        return np.full_like(theta_s_deg, Q_p, dtype=np.float64)
    else:
        return ((Q_s + Q_p) / 2).astype(np.float64)


def compute_bsdf_psd(
    height_map: HeightMap,
    wavelength_um: float,
    theta_i_deg: float,
    phi_i_deg: float,
    n1: float = 1.0,
    n2: float = 1.5,
    polarization: str = "Unpolarized",
    is_btdf: bool = False,
    approx_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PSD法（Rayleigh-Rice理論）でBSDFを計算する。

    BSDF(θ_i, θ_s) = (16π²/λ⁴) * cos(θ_i) * cos²(θ_s) * Q * PSD(f_x, f_y)
        f_x = (sin θ_s cos φ_s - sin θ_i cos φ_i) / λ
        f_y = (sin θ_s sin φ_s - sin θ_i sin φ_i) / λ

    返却する `u_grid = sin θ_s cos φ_s`, `v_grid = sin θ_s sin φ_s` は
    上式の f_x, f_y に sin θ_i cos/sin φ_i を加算して得る（FFT の output_shift
    モードと同じ座標規約）。この結果 specular は (sin θ_i cos φ_i, sin θ_i sin φ_i)
    に正しく配置される。

    フル半球カバレッジには `dx ≤ λ / (2·(1 + sin θ_i))` が必要。これを超えると
    後方散乱側の一部が FFT 格子（f_x ∈ [-1/(2dx), +1/(2dx)]）の外に出て欠損する。

    Args:
        height_map: 高さマップ
        wavelength_um: 波長 [μm]
        theta_i_deg: 入射天頂角 [deg]（表面側換算済み）
        phi_i_deg: 入射方位角 [deg]
        n1: 入射側媒質の屈折率
        n2: 透過側媒質の屈折率
        polarization: 'S' / 'P' / 'Unpolarized'
        is_btdf: True の場合 BTDF として Q を計算する
        approx_mode: False=完全形（デフォルト）/ True=簡略形（高速）

    Returns:
        u_grid: 方向余弦 u = sin(θ_s)*cos(φ_s) の2次元グリッド
        v_grid: 方向余弦 v = sin(θ_s)*sin(φ_s) の2次元グリッド
        bsdf: BSDF値 [sr⁻¹] の2次元グリッド
    """
    # PSD 計算
    fx, fy, psd = compute_psd_2d(height_map)

    N = height_map.grid_size
    dx = height_map.pixel_size_um

    # 空間周波数 → 方向余弦 UV（Rayleigh-Rice 標準定義）
    # f_x = (sin θ_s cos φ_s - sin θ_i cos φ_i) / λ
    # f_y = (sin θ_s sin φ_s - sin θ_i sin φ_i) / λ
    # ⇔ u = sin θ_s cos φ_s = f_x · λ + sin θ_i cos φ_i
    #   v = sin θ_s sin φ_s = f_y · λ + sin θ_i sin φ_i
    # θ_i=0 ではオフセット 0、θ_i>0 では specular を (sin θ_i cos φ_i, sin θ_i sin φ_i)
    # に正しく配置する。
    theta_i_rad = np.deg2rad(theta_i_deg)
    phi_i_rad = np.deg2rad(phi_i_deg)
    u_spec = np.sin(theta_i_rad) * np.cos(phi_i_rad)
    v_spec = np.sin(theta_i_rad) * np.sin(phi_i_rad)
    u_grid = fx * wavelength_um + u_spec
    v_grid = fy * wavelength_um + v_spec
    uv_r2 = u_grid**2 + v_grid**2
    valid_mask = uv_r2 <= 1.0

    # 散乱角グリッド
    theta_s_deg = np.where(valid_mask, np.rad2deg(np.arcsin(np.sqrt(np.minimum(uv_r2, 1.0)))), 0.0)
    cos_i = np.cos(theta_i_rad)
    cos_s = np.where(valid_mask, np.sqrt(np.maximum(1.0 - uv_r2, 0.0)), 1.0)

    # 偏光因子 Q の計算
    if is_btdf:
        if approx_mode:
            Q = _Q_simplified_btdf(theta_i_deg, theta_s_deg, n1, n2, polarization)
        else:
            Q = _Q_complete_btdf(theta_i_deg, theta_s_deg, n1, n2, polarization)
    else:
        if approx_mode:
            Q = _Q_simplified_brdf(theta_i_deg, theta_s_deg, n1, n2, polarization)
        else:
            Q = _Q_complete_brdf(theta_i_deg, theta_s_deg, n1, n2, polarization)

    # BSDF = (16π²/λ⁴) * cos(θ_i) * cos²(θ_s) * Q * PSD
    prefactor = 16 * np.pi**2 / wavelength_um**4
    bsdf = np.where(
        valid_mask,
        prefactor * cos_i * cos_s**2 * Q * psd,
        0.0,
    )

    return u_grid, v_grid, bsdf.astype(np.float32)
