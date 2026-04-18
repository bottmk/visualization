"""ギラツキ計算の拡張レベル L3' / L4 / L5 の実装。

近似階層の定義と数式は `docs/sparkle_approximation_levels.md` を参照。

| Level | 分光 | 空間発光 | AG 応答 |
|---|---|---|---|
| L3' | 単波長 (該当色) | サブピクセル限定発光 | グローバル BSDF |
| L4 | V(λ) 重み + 各色代表波長 | サブピクセル限定発光 (色ごと) | グローバル BSDF |
| L5 | V(λ) 重み | サブピクセル限定発光 | 空間分解 BSDF (窓付き FFT) |

物理単位は μm 統一。
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..models.base import HeightMap

Color = Literal["R", "G", "B"]
SubpixelLayout = Literal["rgb_stripe", "bgr_stripe"]

# 各色の代表波長 [μm]（典型的な sRGB / 表示用プライマリ）
_COLOR_WAVELENGTHS_UM: dict[str, float] = {
    "R": 0.630,
    "G": 0.525,
    "B": 0.465,
}

# CIE 1931 明所視 V(λ) のサンプル値（5 nm 刻み、380–780 nm）
# 使用時は線形補間
_CIE_V_LAMBDA_NM = np.arange(380, 785, 5)
_CIE_V_LAMBDA_VALUES = np.array([
    0.000039, 0.000064, 0.000120, 0.000217, 0.000396, 0.000640, 0.001210, 0.002180,
    0.004000, 0.007300, 0.011600, 0.016840, 0.023000, 0.029800, 0.038000, 0.048000,
    0.060000, 0.073900, 0.090980, 0.112600, 0.139020, 0.169300, 0.208020, 0.258600,
    0.323000, 0.407300, 0.503000, 0.608200, 0.710000, 0.793200, 0.862000, 0.914850,
    0.954000, 0.980300, 0.994950, 1.000000, 0.995000, 0.978600, 0.952000, 0.915400,
    0.870000, 0.816300, 0.757000, 0.694900, 0.631000, 0.566800, 0.503000, 0.441200,
    0.381000, 0.321000, 0.265000, 0.217000, 0.175000, 0.138200, 0.107000, 0.081600,
    0.061000, 0.044580, 0.032000, 0.023200, 0.017000, 0.011920, 0.008210, 0.005723,
    0.004102, 0.002929, 0.002091, 0.001484, 0.001047, 0.000740, 0.000520, 0.000361,
    0.000249, 0.000172, 0.000120, 0.000085, 0.000060, 0.000042, 0.000030, 0.000021,
    0.000015,
])


def _v_lambda(wavelength_um: float) -> float:
    """CIE 1931 V(λ) を与えられた波長 [μm] で線形補間して返す。"""
    wl_nm = wavelength_um * 1000.0
    return float(np.interp(wl_nm, _CIE_V_LAMBDA_NM, _CIE_V_LAMBDA_VALUES, left=0.0, right=0.0))


# ── 共通ヘルパー ────────────────────────────────────────────────────────────


def _compute_phase(
    height_map: HeightMap,
    wavelength_um: float,
    theta_i_deg: float = 0.0,
    n1: float = 1.0,
    n2: float = 1.5,
    is_btdf: bool = True,
) -> np.ndarray:
    """AG フィルム表面高さから位相分布 φ(x, y) を計算する。

    `compute_bsdf_fft` の位相生成と同じロジック（垂直入射・'zero' mode 相当の簡易版）。
    Sparkle は通常垂直入射 × BTDF で評価するためそちらに最適化している。

    Args:
        height_map: 高さマップ
        wavelength_um: 波長 [μm]
        theta_i_deg: 入射角 [deg]（現状は 0 のみサポート）
        n1, n2: 屈折率
        is_btdf: True で BTDF、False で BRDF

    Returns:
        位相分布 φ(x, y) [rad]、shape = (N, N)
    """
    if theta_i_deg != 0.0:
        raise NotImplementedError("L3'/L4/L5 は現状 theta_i_deg=0 のみサポート")
    h = height_map.data.astype(np.float64)
    k = 2 * np.pi / wavelength_um
    if is_btdf:
        return k * (n2 - n1) * h
    return 2 * k * n1 * h


def _generate_subpixel_mask(
    grid_size: int,
    pixel_size_um: float,
    pixel_pitch_um: float,
    subpixel_layout: SubpixelLayout,
    color: Color,
) -> np.ndarray:
    """サブピクセル発光マスク M_c(x, y) を生成する。

    rgb_stripe / bgr_stripe レイアウトを x 方向のストライプとしてモデル化する。
    1 ディスプレイ画素内で x 方向に 3 分割し、該当色のサブピクセル領域のみ 1。

    Args:
        grid_size: グリッドサイズ N
        pixel_size_um: シミュレーショングリッドのピクセルサイズ dx [μm]
        pixel_pitch_um: ディスプレイ画素ピッチ [μm]
        subpixel_layout: 'rgb_stripe' / 'bgr_stripe'
        color: 'R' / 'G' / 'B'

    Returns:
        マスク配列 shape=(N, N)、値は {0.0, 1.0}
    """
    if color not in ("R", "G", "B"):
        raise ValueError(f"color must be 'R', 'G', or 'B'. Got {color!r}")
    if subpixel_layout == "rgb_stripe":
        order = {"R": 0, "G": 1, "B": 2}
    elif subpixel_layout == "bgr_stripe":
        order = {"B": 0, "G": 1, "R": 2}
    else:
        raise ValueError(f"unsupported subpixel_layout: {subpixel_layout}")

    # グリッド点の x 座標 [μm]
    x_um = np.arange(grid_size) * pixel_size_um
    # ディスプレイ画素内での相対位置 [0, pitch) を求め、3 分割
    sub_idx = np.floor(3.0 * (x_um % pixel_pitch_um) / pixel_pitch_um).astype(np.int32)
    sub_idx = np.clip(sub_idx, 0, 2)
    mask_row = (sub_idx == order[color]).astype(np.float64)
    # 2D へ拡張（y 方向は一様）
    mask = np.broadcast_to(mask_row[np.newaxis, :], (grid_size, grid_size)).copy()
    return mask


def _geometry_from_config(sparkle_config: dict) -> tuple[float, float, float]:
    """config から観察ジオメトリ (omega_pupil, sin_half, pixel_pitch_um) を取り出す。"""
    viewing = sparkle_config.get("viewing", {})
    display = sparkle_config.get("display", {})
    distance_mm = float(viewing.get("distance_mm", 300.0))
    pupil_mm = float(viewing.get("pupil_diameter_mm", 3.0))
    pixel_pitch_mm = float(display.get("pixel_pitch_mm", 0.062))

    omega_pupil = np.pi * (pupil_mm / 2 / distance_mm) ** 2
    sin_half = np.sin(np.arctan(pixel_pitch_mm / 2 / distance_mm))
    pixel_pitch_um = pixel_pitch_mm * 1000.0
    return omega_pupil, sin_half, pixel_pitch_um


def _pixel_luminance_from_U(
    U: np.ndarray,
    pixel_size_um: float,
    wavelength_um: float,
    omega_pupil: float,
    sin_half: float,
) -> np.ndarray:
    """複素振幅 U(x,y) から画素ごとの輝度配列 {L_k} を計算する。

    L1 の `_compute_sparkle_single` と同じ角度ビニングロジックを BSDF 計算と
    統合したもの。グローバル FFT → 角度ビン振り分け → bincount 加算。

    Args:
        U: 複素振幅 shape=(N, N)
        pixel_size_um: グリッドの dx [μm]
        wavelength_um: 波長 [μm]
        omega_pupil: 瞳孔立体角 [sr]
        sin_half: 1 ディスプレイ画素の方向余弦半幅

    Returns:
        画素ごとの輝度 1D 配列
    """
    N = U.shape[0]
    U_fft = np.fft.fft2(U)
    I_fft = np.abs(U_fft) ** 2

    freq_x = np.fft.fftfreq(N, d=pixel_size_um)
    freq_y = np.fft.fftfreq(N, d=pixel_size_um)
    fx, fy = np.meshgrid(freq_x, freq_y, indexing="ij")
    u_grid = fx * wavelength_um
    v_grid = fy * wavelength_um

    uv_r2 = u_grid**2 + v_grid**2
    valid = uv_r2 <= 1.0
    cos_s = np.sqrt(np.maximum(1.0 - uv_r2, 0.0))

    normalization = (N * pixel_size_um) ** 2
    bsdf = np.zeros_like(I_fft)
    bsdf[valid] = I_fft[valid] / normalization / np.maximum(cos_s[valid], 1e-10) / (wavelength_um**2)

    du = abs(u_grid[1, 0] - u_grid[0, 0]) if N > 1 else 1.0
    dv = abs(v_grid[0, 1] - v_grid[0, 0]) if N > 1 else 1.0

    pix_u = np.round(u_grid / (2.0 * sin_half)).astype(np.int32)
    pix_v = np.round(v_grid / (2.0 * sin_half)).astype(np.int32)

    pu_flat = pix_u[valid]
    pv_flat = pix_v[valid]
    power_flat = (bsdf * cos_s * du * dv * omega_pupil)[valid]

    if len(power_flat) < 2:
        return np.array([], dtype=np.float64)

    offset_u = int(pu_flat.min())
    offset_v = int(pv_flat.min())
    n_cols = int(pv_flat.max()) - offset_v + 1
    pixel_key = (pu_flat - offset_u) * n_cols + (pv_flat - offset_v)

    _, inverse = np.unique(pixel_key, return_inverse=True)
    return np.bincount(inverse, weights=power_flat)


def _cs_from_luminance(L_k: np.ndarray) -> float:
    """画素輝度配列から sparkle コントラスト Cs = σ/μ を計算する。"""
    if len(L_k) < 2:
        return 0.0
    mu = float(np.mean(L_k))
    if mu < 1e-30:
        return 0.0
    sigma = float(np.std(L_k))
    return float(sigma / mu)


# ── L3': 単色表示（サブピクセル限定発光 + 単波長） ───────────────────────────


def compute_sparkle_l3prime(
    height_map: HeightMap,
    color: Color,
    sparkle_config: dict,
    wavelength_um: float | None = None,
    n1: float = 1.0,
    n2: float = 1.5,
    is_btdf: bool = True,
) -> float:
    """L3': 単色表示 (R/G/B いずれか単独点灯) での sparkle Cs = σ/μ を計算する。

    該当色のサブピクセル位置のみ発光、単一代表波長で評価。

    ⚠️ **警告**: 本関数は角度ビニング方式を用いるため、返り値 Cs は SEMI D63 /
    IDMS 実測値と比較して 100–2000× 大きい apparent 値である。**相対比較用途のみ**
    で使用し、絶対値が必要な場合は `compute_sparkle_l5` を使うこと。詳細は
    `docs/sparkle_calculation.md` Section 1.1 の警告ブロック参照。

    Args:
        height_map: AG フィルム高さマップ
        color: 点灯色 'R' / 'G' / 'B'
        sparkle_config: config.yaml の metrics.sparkle セクション
        wavelength_um: 評価波長 [μm]。None の場合は色別デフォルト（R=0.63, G=0.525, B=0.465）
        n1, n2: 屈折率
        is_btdf: BTDF (True) / BRDF (False)

    Returns:
        ギラツキコントラスト Cs（apparent 値、相対比較用）
    """
    if wavelength_um is None:
        wavelength_um = _COLOR_WAVELENGTHS_UM[color]

    omega_pupil, sin_half, pixel_pitch_um = _geometry_from_config(sparkle_config)

    phase = _compute_phase(height_map, wavelength_um, n1=n1, n2=n2, is_btdf=is_btdf)

    display = sparkle_config.get("display", {})
    subpixel_layout = display.get("subpixel_layout") or "rgb_stripe"
    mask = _generate_subpixel_mask(
        grid_size=height_map.grid_size,
        pixel_size_um=height_map.pixel_size_um,
        pixel_pitch_um=pixel_pitch_um,
        subpixel_layout=subpixel_layout,
        color=color,
    )
    U = mask * np.exp(1j * phase)

    L_k = _pixel_luminance_from_U(
        U, height_map.pixel_size_um, wavelength_um, omega_pupil, sin_half
    )
    return _cs_from_luminance(L_k)


# ── L4: 白点灯 + V(λ) 重み + サブピクセル（各色独立 FFT の非線形合成） ────────


def compute_sparkle_l4(
    height_map: HeightMap,
    sparkle_config: dict,
    color_wavelengths_um: dict[str, float] | None = None,
    source_intensity: dict[str, float] | None = None,
    n1: float = 1.0,
    n2: float = 1.5,
    is_btdf: bool = True,
) -> float:
    """L4: 白点灯時の sparkle Cs = σ/μ を計算する。

    R/G/B サブピクセルそれぞれで独立に輝度配列 {L_k^(c)} を計算し、V(λ) × 光源強度
    で重み付け加算してから σ/μ を取る（sparkle は非線形指標なので加算してから
    コントラストを計算する点が重要）。

    narrowband 近似: 各色は代表波長 1 点で評価する（S_c(λ) ≈ δ(λ-λ_c)）。分光幅を
    考慮した完全版は将来拡張とする（docs/sparkle_approximation_levels.md Section 6）。

    ⚠️ **警告**: L3' と同じ角度ビニング方式のため、返り値 Cs は SEMI D63 / IDMS
    実測値より 100–1500× 大きい apparent 値。**相対比較用途のみ** で使用し、絶対値
    が必要な場合は `compute_sparkle_l5` を使うこと。

    Args:
        height_map: AG フィルム高さマップ
        sparkle_config: config.yaml の metrics.sparkle セクション
        color_wavelengths_um: 各色の代表波長 {'R': 0.63, 'G': 0.525, 'B': 0.465}。
            None でデフォルト
        source_intensity: 各色の光源強度（白点の内訳、デフォルトは等輝度 1.0）
        n1, n2, is_btdf: 光学系パラメータ

    Returns:
        ギラツキコントラスト Cs
    """
    if color_wavelengths_um is None:
        color_wavelengths_um = dict(_COLOR_WAVELENGTHS_UM)
    if source_intensity is None:
        source_intensity = {"R": 1.0, "G": 1.0, "B": 1.0}

    omega_pupil, sin_half, pixel_pitch_um = _geometry_from_config(sparkle_config)
    display = sparkle_config.get("display", {})
    subpixel_layout = display.get("subpixel_layout") or "rgb_stripe"

    L_k_total: np.ndarray | None = None
    for color in ("R", "G", "B"):
        lam = color_wavelengths_um[color]
        weight = _v_lambda(lam) * source_intensity[color]
        if weight <= 0.0:
            continue
        phase = _compute_phase(height_map, lam, n1=n1, n2=n2, is_btdf=is_btdf)
        mask = _generate_subpixel_mask(
            grid_size=height_map.grid_size,
            pixel_size_um=height_map.pixel_size_um,
            pixel_pitch_um=pixel_pitch_um,
            subpixel_layout=subpixel_layout,
            color=color,
        )
        U = mask * np.exp(1j * phase)
        L_k = _pixel_luminance_from_U(
            U, height_map.pixel_size_um, lam, omega_pupil, sin_half
        )
        if L_k_total is None:
            L_k_total = weight * L_k
        else:
            n = min(len(L_k_total), len(L_k))
            L_k_total = L_k_total[:n] + weight * L_k[:n]

    if L_k_total is None or len(L_k_total) < 2:
        return 0.0
    return _cs_from_luminance(L_k_total)


# ── L5: 空間分解 BSDF（窓付き FFT） ──────────────────────────────────────────


def _hann_window_2d(size_samples: int) -> np.ndarray:
    """2D Hann 窓を生成する。size_samples × size_samples、中心で 1、端で 0。"""
    w1 = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(size_samples) / (size_samples - 1)))
    return np.outer(w1, w1)


def compute_sparkle_l5(
    height_map: HeightMap,
    color: Color,
    sparkle_config: dict,
    wavelength_um: float | None = None,
    window_size_factor: float = 3.0,
    pupil_integration: bool = True,
    n1: float = 1.0,
    n2: float = 1.5,
    is_btdf: bool = True,
) -> float:
    """L5: 空間分解 BSDF（窓付き FFT）による sparkle Cs = σ/μ を計算する。

    ディスプレイ画素ごとに窓付き FFT で局所 BSDF を計算し、観察者方向付近で
    輝度を積分して L_k を得る。AG フィルムが空間的に不均一な場合や L_c / p が
    小さくない場合に L3'/L4 よりも物理的に厳密な評価を与える。

    単色評価（1 色のみ点灯）とし、窓幅は画素ピッチ × window_size_factor とする。

    pupil_integration=True（既定）のとき、観察者方向の 1 点（DC 成分）ではなく
    瞳孔立体角 $\\Omega_\\text{{pupil}}$ 内のフーリエ成分を合算する（物理的により
    厳密）。計算負荷増は無視できる程度（画素あたり数個のグリッド点加算のみ）。

    Args:
        height_map: AG フィルム高さマップ
        color: 点灯色 'R' / 'G' / 'B'
        sparkle_config: config.yaml の metrics.sparkle セクション
        wavelength_um: 評価波長 [μm]。None で色別デフォルト
        window_size_factor: 窓幅 = pixel_pitch × この係数。3–5 を推奨
        pupil_integration: True で瞳孔立体角内を積分、False で DC 1 点近似
        n1, n2, is_btdf: 光学系パラメータ

    Returns:
        ギラツキコントラスト Cs
    """
    if wavelength_um is None:
        wavelength_um = _COLOR_WAVELENGTHS_UM[color]

    omega_pupil, _sin_half, pixel_pitch_um = _geometry_from_config(sparkle_config)
    viewing = sparkle_config.get("viewing", {})
    distance_mm = float(viewing.get("distance_mm", 300.0))
    pupil_mm = float(viewing.get("pupil_diameter_mm", 3.0))
    # pupil 角度半径（方向余弦空間）: u_pupil = sin(arctan(d_p/2D)) ≈ d_p/(2D)
    u_pupil = np.sin(np.arctan(pupil_mm / 2.0 / distance_mm))

    display = sparkle_config.get("display", {})
    subpixel_layout = display.get("subpixel_layout") or "rgb_stripe"

    phase = _compute_phase(height_map, wavelength_um, n1=n1, n2=n2, is_btdf=is_btdf)
    mask = _generate_subpixel_mask(
        grid_size=height_map.grid_size,
        pixel_size_um=height_map.pixel_size_um,
        pixel_pitch_um=pixel_pitch_um,
        subpixel_layout=subpixel_layout,
        color=color,
    )
    U_full = mask * np.exp(1j * phase)

    # 窓サイズ（サンプル数）
    dx = height_map.pixel_size_um
    W_um = pixel_pitch_um * window_size_factor
    W_samples = int(round(W_um / dx))
    if W_samples < 4:
        raise ValueError(
            f"窓幅が小さすぎる (W_samples={W_samples})。dx={dx} μm に対し "
            f"pixel_pitch={pixel_pitch_um} μm × factor={window_size_factor} が不足。"
        )
    if W_samples >= height_map.grid_size:
        raise ValueError(
            f"窓幅がグリッドサイズ以上 (W_samples={W_samples} >= N={height_map.grid_size})"
        )
    window = _hann_window_2d(W_samples)

    # ディスプレイ画素中心位置のサンプル間隔
    pixel_step_samples = int(round(pixel_pitch_um / dx))
    if pixel_step_samples < 1:
        raise ValueError(f"画素ピッチがグリッド dx より小さい (pixel_step_samples={pixel_step_samples})")

    N = height_map.grid_size
    half_W = W_samples // 2
    # 境界を避けるためのピクセル中心候補
    centers = np.arange(half_W, N - half_W, pixel_step_samples, dtype=np.int32)
    if len(centers) < 2:
        return 0.0

    # 窓 FFT の方向余弦空間グリッド（1 回だけ計算）
    freq = np.fft.fftfreq(W_samples, d=dx)
    fx, fy = np.meshgrid(freq, freq, indexing="ij")
    u_local = fx * wavelength_um
    v_local = fy * wavelength_um
    uv_r2 = u_local**2 + v_local**2

    if pupil_integration:
        # pupil 立体角内のフーリエ成分を積分（観察者方向 = (u=0, v=0) 中心）
        pupil_mask = uv_r2 <= u_pupil**2
        # pupil 内にサンプルが 0 の場合（u_pupil < FFT 分解能）は DC 1 点にフォールバック
        if not np.any(pupil_mask):
            pupil_mask = np.zeros_like(uv_r2, dtype=bool)
            pupil_mask[0, 0] = True
    else:
        # DC 1 点のみ
        pupil_mask = np.zeros_like(uv_r2, dtype=bool)
        pupil_mask[0, 0] = True

    du_local = abs(u_local[1, 0] - u_local[0, 0])
    dv_local = abs(v_local[0, 1] - v_local[0, 0])
    cos_s_pupil = np.sqrt(np.maximum(1.0 - uv_r2[pupil_mask], 0.0))

    L_k_list: list[float] = []
    norm_factor = np.sum(window**2) * dx**2  # 窓のエネルギー

    for cx in centers:
        for cy in centers:
            sub = U_full[cx - half_W : cx + half_W, cy - half_W : cy + half_W]
            if sub.shape != (W_samples, W_samples):
                continue
            U_w = window * sub
            U_fft = np.fft.fft2(U_w)
            I_fft = np.abs(U_fft) ** 2
            # 局所 BSDF = I_fft / 窓エネルギー / λ² / cos_s
            # pupil 内のフーリエ成分を ∫ BSDF·cos·du·dv で積分
            bsdf_in_pupil = I_fft[pupil_mask] / norm_factor / (wavelength_um**2)
            flux = float(np.sum(bsdf_in_pupil * cos_s_pupil) * du_local * dv_local)
            L_k_list.append(flux)

    L_k_arr = np.array(L_k_list, dtype=np.float64)
    return _cs_from_luminance(L_k_arr)
