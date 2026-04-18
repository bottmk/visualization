"""光学指標の計算 (Haze, Gloss, DOI (NSER/COMB/ASTM), Sparkle)。

spec_main.md Section 5 の仕様に従い実装する。
物理単位は μm 統一。角度は度数法 [deg]。
"""

from __future__ import annotations

import numpy as np


# ── 窓中心ヘルパー ───────────────────────────────────────────────────────────

def _specular_u_center(theta_i_deg: float, mode: str, n1: float = 1.0, n2: float = 1.5) -> float:
    """BSDF の (u, v) 空間でスペキュラー方向の u 座標を返す。

    BRDF の場合: u = sin(θ_i)  (鏡面反射方向)
    BTDF の場合: u = sin(θ_t)  (Snell で屈折した方向)

    FFT BSDF の tilt mode では入射側 tilt により θ_i > 0 でも BSDF ピークが
    この u 値の位置に現れる（fft_bsdf.py:120-127 参照）。v_center は常に 0
    （phi_i=0 前提）。

    Args:
        theta_i_deg: 入射角 [deg]
        mode: "BRDF" / "BTDF"
        n1, n2: 屈折率（BTDF 時のみ Snell で使用）

    Returns:
        スペキュラー u 座標（v は常に 0）
    """
    theta_i_rad = np.deg2rad(theta_i_deg)
    if mode.upper() == "BTDF":
        sin_t = n1 * np.sin(theta_i_rad) / n2
        sin_t = np.clip(sin_t, -1.0, 1.0)
        return float(sin_t)
    return float(np.sin(theta_i_rad))


# ── Log-RMSE ─────────────────────────────────────────────────────────────────

def compute_log_rmse(
    simulated: np.ndarray,
    measured: np.ndarray,
    bsdf_floor: float = 1e-6,
) -> float:
    """Log-RMSE（実測データとシミュレーション結果の対数スケール誤差）を計算する。

    spec_main.md Section 3.4 の仕様:
    - 実測データがフロア以上の点のみを誤差計算対象（マスク処理）
    - シミュレーション値のゼロ処理: max(simulated, bsdf_floor) でクリップ

    Args:
        simulated: シミュレーション BSDF 値 [sr⁻¹]（任意形状）
        measured: 実測 BSDF 値 [sr⁻¹]（simulated と同形状）
        bsdf_floor: ノイズフロア [sr⁻¹]（デフォルト: 1e-6）

    Returns:
        Log-RMSE スカラー値
    """
    mask = measured > bsdf_floor
    if not np.any(mask):
        return float("inf")

    log_sim = np.log10(np.maximum(simulated[mask], bsdf_floor))
    log_meas = np.log10(measured[mask])
    return float(np.sqrt(np.mean((log_sim - log_meas) ** 2)))


# ── ヘイズ ───────────────────────────────────────────────────────────────────

def compute_haze(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    bsdf: np.ndarray,
    half_angle_deg: float = 2.5,
    u_center: float = 0.0,
    v_center: float = 0.0,
) -> float:
    """ヘイズ（Haze）を計算する。

    規格（JIS K 7136 / ISO 14782 / ASTM D1003）は θ_i=0° BTDF を前提で、
    直進光方向から `half_angle_deg`（2.5°）以上偏向した光を「散乱」とする。

    斜入射時（θ_i > 0）は (u_center, v_center) を直進光方向に指定する
    （BTDF なら Snell 屈折後の u、BRDF なら鏡面反射方向の u）。

    Args:
        u_grid: 方向余弦 u グリッド（2D）
        v_grid: 方向余弦 v グリッド（2D）
        bsdf: BSDF 値 [sr⁻¹]（2D）
        half_angle_deg: ヘイズ境界角 [deg]（デフォルト: 2.5°）
        u_center: 直進光方向 u 座標（デフォルト 0 = 法線）
        v_center: 直進光方向 v 座標（デフォルト 0）

    Returns:
        ヘイズ値（0〜1）
    """
    uv_r2 = u_grid**2 + v_grid**2
    # 直進光中心からの角距離（小角近似で u,v 空間距離を角度とみなす）
    du_c = u_grid - u_center
    dv_c = v_grid - v_center
    offset_r2 = du_c**2 + dv_c**2
    sin_half = np.sin(np.deg2rad(half_angle_deg))
    threshold_r2 = sin_half**2

    # 微小立体角 dΩ = cos(θ_s) * du * dv（UV空間での積分）
    du = abs(u_grid[1, 0] - u_grid[0, 0]) if u_grid.shape[0] > 1 else 1.0
    dv = abs(v_grid[0, 1] - v_grid[0, 0]) if v_grid.shape[1] > 1 else 1.0
    cos_s = np.sqrt(np.maximum(1.0 - uv_r2, 0.0))
    valid = uv_r2 <= 1.0

    # 全透過光量（半球積分）
    total = np.sum(bsdf[valid] * cos_s[valid]) * du * dv

    # 広角散乱光量（直進光中心から境界角以上）
    wide = offset_r2 >= threshold_r2
    haze_power = np.sum(bsdf[valid & wide] * cos_s[valid & wide]) * du * dv

    if total < 1e-30:
        return 0.0
    return float(haze_power / total)


# ── グロス ───────────────────────────────────────────────────────────────────
#
# 規格: ISO 2813 / ASTM D523 / JIS Z 8741
# 標準測定角は 20° / 60° / 85°。各角度で規定された長方形絞り（受光側）を使う。
# BRDF 計算であり、正反射方向 (θ_s = θ_i = gloss_angle_deg) 周りで積分する。
#
# 受光絞り（ISO 2813）:
#   20° gloss: 1.8° (in-plane) × 3.6° (cross-plane)
#   60° gloss: 4.4° (in-plane) × 11.7° (cross-plane)
#   85° gloss: 4.0° (in-plane) × 6.0° (cross-plane)
#
# 「in-plane」は入射面内（u 軸方向、特異点では cos(θ_sp)·δθ で u スケール）
# 「cross-plane」は入射面と垂直方向（v 軸、角距離 = v 値）
#
# 黒ガラス基準化: 屈折率 n=1.567 の黒ガラスの Fresnel 反射率 R_BG(θ) を 100 GU と
# して規格化。黒ガラスの「黒」（α→∞）は背面反射を排除するためで、現状の FFT BSDF
# が前面反射のみを計算しているため n の情報だけで十分（spec Section 5.5 参照）。


_GLOSS_APERTURES_DEG: dict[int, dict[str, float]] = {
    20: {"in_plane_deg": 1.8, "cross_plane_deg": 3.6},
    60: {"in_plane_deg": 4.4, "cross_plane_deg": 11.7},
    85: {"in_plane_deg": 4.0, "cross_plane_deg": 6.0},
}


def _fresnel_reflectance_unpol(theta_i_deg: float, n1: float = 1.0, n2: float = 1.567) -> float:
    """無偏光の Fresnel パワー反射率 R = (R_s + R_p) / 2 を返す（0〜1）。

    黒ガラス基準化に使用。α→∞（完全吸収）前提で背面反射は考慮しない。
    """
    theta_i_rad = np.deg2rad(theta_i_deg)
    cos_i = np.cos(theta_i_rad)
    sin_t = n1 * np.sin(theta_i_rad) / n2
    if abs(sin_t) > 1.0:
        return 1.0  # 全反射（この構成では起きない）
    cos_t = np.sqrt(1.0 - sin_t**2)
    r_s = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
    r_p = ((n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)) ** 2
    return float(0.5 * (r_s + r_p))


def compute_gloss(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    bsdf: np.ndarray,
    gloss_angle_deg: float = 60.0,
    u_center: float | None = None,
    v_center: float = 0.0,
    aperture_override: dict[str, float] | None = None,
    black_glass_normalization: bool = False,
    black_glass_n: float = 1.567,
) -> float:
    """グロス（Gloss）を計算する。

    規格 (ISO 2813 / ASTM D523 / JIS Z 8741) に基づく長方形絞りで
    正反射方向の反射率を積分する。

    Args:
        u_grid, v_grid: 方向余弦グリッド (2D)
        bsdf: BSDF 値 [sr⁻¹] (2D)
        gloss_angle_deg: グロス測定角 [deg]、規格標準は 20° / 60° / 85°
        u_center: スペキュラー u 中心。None なら sin(gloss_angle_deg) を使用
            （規格の正反射位置、BRDF sim 前提）
        v_center: スペキュラー v 中心（デフォルト 0）
        aperture_override: 絞りサイズを手動指定するとき {"in_plane_deg":, "cross_plane_deg":}
        black_glass_normalization: True で黒ガラス (n=1.567) 基準化 → GU 値 (0〜100)
        black_glass_n: 黒ガラス屈折率（デフォルト 1.567、規格値固定推奨）

    Returns:
        `black_glass_normalization=False`: 受光絞り内の積分フラックス（無次元）
        `black_glass_normalization=True`:  GU 値（0〜100、黒ガラス=100）
    """
    # スペキュラー中心決定
    if u_center is None:
        u_center = float(np.sin(np.deg2rad(gloss_angle_deg)))

    # 絞り仕様決定
    angle_int = int(round(gloss_angle_deg))
    if aperture_override is not None:
        ap = aperture_override
    elif angle_int in _GLOSS_APERTURES_DEG:
        ap = _GLOSS_APERTURES_DEG[angle_int]
    else:
        # 非規格角度 → 60° の絞りを流用
        ap = _GLOSS_APERTURES_DEG[60]

    # 長方形絞りを (u, v) 空間の半幅に変換
    cos_sp = np.cos(np.deg2rad(gloss_angle_deg))
    du_half = cos_sp * np.deg2rad(ap["in_plane_deg"] / 2.0)
    dv_half = np.deg2rad(ap["cross_plane_deg"] / 2.0)

    # 長方形マスク
    in_u = np.abs(u_grid - u_center) <= du_half
    in_v = np.abs(v_grid - v_center) <= dv_half
    mask = in_u & in_v

    uv_r2 = u_grid**2 + v_grid**2
    valid = uv_r2 <= 1.0
    cos_s = np.sqrt(np.maximum(1.0 - uv_r2, 0.0))
    du = abs(u_grid[1, 0] - u_grid[0, 0]) if u_grid.shape[0] > 1 else 1.0
    dv = abs(v_grid[0, 1] - v_grid[0, 0]) if v_grid.shape[1] > 1 else 1.0

    selected = valid & mask
    if not np.any(selected):
        return 0.0

    # 受光絞り内の積分フラックス（BSDF · cosθ · dΩ）
    flux = float(np.sum(bsdf[selected] * cos_s[selected]) * du * dv)

    if black_glass_normalization:
        r_bg = _fresnel_reflectance_unpol(gloss_angle_deg, n1=1.0, n2=black_glass_n)
        if r_bg < 1e-30:
            return 0.0
        return float(100.0 * flux / r_bg)

    return flux


# ── 写像性 (DOI / Clarity) ───────────────────────────────────────────────────
#
# 3 方式を実装する。詳細は spec_main.md Section 5.4 を参照。
#
#   (A) NSER (Near-Specular Energy Ratio)  — 独自方式 (規格非準拠)
#       直進光コーンと小角散乱コーンの BSDF エネルギー比。
#
#   (B) COMB (JIS K 7374 光学くし方式, 通称「くしば方式」)
#       5 種のくし幅でのコントラスト M(d) = (Imax - Imin)/(Imax + Imin) の平均。
#       1D 角度プロファイルに矩形波を畳み込み、走査信号の振幅から算出。
#
#   (C) ASTM (ASTM E430 Dorigon 方式)
#       [R(0°) - R(offset)] / R(0°) × 100。デフォルト offset = 0.3°。


def compute_doi_nser(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    bsdf: np.ndarray,
    direct_half_angle_deg: float = 0.1,
    halo_half_angle_deg: float = 2.0,
    u_center: float = 0.0,
    v_center: float = 0.0,
) -> float:
    """NSER (Near-Specular Energy Ratio) 方式で写像性を計算する。

    DOI_NSER = 直進光エネルギー / (直進光 + 小角散乱光)

    斜入射時は (u_center, v_center) に直進光方向（BRDF: 鏡面反射、BTDF: 屈折）
    を指定する。

    Args:
        u_grid: 方向余弦 u グリッド (2D)
        v_grid: 方向余弦 v グリッド (2D)
        bsdf: BSDF 値 [sr⁻¹] (2D)
        direct_half_angle_deg: 直進光の半角 [deg] (デフォルト: 0.1°)
        halo_half_angle_deg: ハロー領域の半角 [deg] (デフォルト: 2.0°)
        u_center, v_center: 直進光方向の (u, v) 座標（デフォルト 0, 0 = 法線）

    Returns:
        DOI 値 (0〜1)
    """
    uv_r2 = u_grid**2 + v_grid**2
    du_c = u_grid - u_center
    dv_c = v_grid - v_center
    offset_r2 = du_c**2 + dv_c**2

    cos_s = np.sqrt(np.maximum(1.0 - uv_r2, 0.0))
    valid = uv_r2 <= 1.0

    du = abs(u_grid[1, 0] - u_grid[0, 0]) if u_grid.shape[0] > 1 else 1.0
    dv = abs(v_grid[0, 1] - v_grid[0, 0]) if v_grid.shape[1] > 1 else 1.0

    sin_direct = np.sin(np.deg2rad(direct_half_angle_deg))
    sin_halo = np.sin(np.deg2rad(halo_half_angle_deg))

    direct_mask = valid & (offset_r2 <= sin_direct**2)
    halo_mask = valid & (offset_r2 <= sin_halo**2)

    direct_power = np.sum(bsdf[direct_mask] * cos_s[direct_mask]) * du * dv
    total_near = np.sum(bsdf[halo_mask] * cos_s[halo_mask]) * du * dv

    if total_near < 1e-30:
        return 1.0
    return float(direct_power / total_near)


def compute_doi_comb(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    bsdf: np.ndarray,
    comb_widths_mm: list[float] | None = None,
    distance_mm: float = 280.0,
    scan_half_angle_deg: float = 4.0,
    v_band_half_deg: float = 0.2,
    u_center: float = 0.0,
    v_center: float = 0.0,
) -> float:
    """JIS K 7374 光学くし (COMB) 方式で写像性を計算する。

    試料 〜 光学くし間距離 `distance_mm` [mm] を仮定し、角度領域で等価な矩形波パターン
    (周期 = 2·くし幅/距離 [rad]) を 1D 角度プロファイル P(θ) に畳み込む。
    走査信号の Imax/Imin からコントラスト M(d) = (Imax-Imin)/(Imax+Imin) を
    各くし幅で算出し、5 値を算術平均して返す (JIS K 7374 に基づく標準処理)。

    P(θ) は BSDF を v 軸の中心バンド (|v| <= sin(v_band_half_deg)) で平均して得る。

    Args:
        u_grid: 方向余弦 u グリッド (2D)
        v_grid: 方向余弦 v グリッド (2D)
        bsdf: BSDF 値 [sr⁻¹] (2D)
        comb_widths_mm: くし幅リスト [mm]。デフォルト [0.125,0.25,0.5,1.0,2.0] (JIS 準拠)
        distance_mm: 試料〜くし距離 [mm] (Suga ICM 標準 ≈ 280 mm)
        scan_half_angle_deg: 走査評価範囲の半角 [deg]。この範囲の角度プロファイルを使う
        v_band_half_deg: P(θ) 抽出時の v 軸バンド半角 [deg]

    Returns:
        平均コントラスト (0〜1)。値が大きいほど写像性が高い。
        パーセント表示が必要な場合は呼び出し側で ×100 する。
    """
    if comb_widths_mm is None:
        comb_widths_mm = [0.125, 0.25, 0.5, 1.0, 2.0]

    # u 軸等間隔を仮定 (既存 haze/gloss と同じ前提)
    u_axis = u_grid[:, 0]
    v_axis = v_grid[0, :]
    du = abs(u_axis[1] - u_axis[0]) if u_axis.size > 1 else 1.0

    # v バンド平均で 1D プロファイル P(u) を作る（v_center 基準）
    sin_band = np.sin(np.deg2rad(v_band_half_deg))
    v_mask = np.abs(v_axis - v_center) <= sin_band
    if not np.any(v_mask):
        v_mask = np.abs(v_axis - v_center) == np.min(np.abs(v_axis - v_center))
    profile = bsdf[:, v_mask].mean(axis=1)

    # 走査範囲を制限 (u_center ± scan_half_angle_deg)
    sin_scan = np.sin(np.deg2rad(scan_half_angle_deg))
    scan_mask = np.abs(u_axis - u_center) <= sin_scan
    if not np.any(scan_mask):
        return 0.0
    P = profile[scan_mask]
    u_sub = u_axis[scan_mask] - u_center  # 中心基準にシフト

    contrasts: list[float] = []
    for d_mm in comb_widths_mm:
        # 角度領域での明/暗 1 組の周期 [rad] (小角近似: sin θ ≈ θ)
        period_rad = 2.0 * d_mm / distance_mm
        # u 空間での周期 (u = sin θ ≈ θ)
        period_u = period_rad

        # 矩形波くし: duty 50% (明/暗)。走査位置 x を位相として振る
        # I(x) = Σ P(u) · comb(u - x) du
        # comb 値: 明=1 / 暗=0
        n_phase = 32
        phases = np.linspace(0.0, period_u, n_phase, endpoint=False)
        intensities = np.empty(n_phase)
        for i, x in enumerate(phases):
            # (u - x) mod period_u の前半 = 明 (1), 後半 = 暗 (0)
            rel = np.mod(u_sub - x, period_u)
            bright = rel < (period_u / 2.0)
            intensities[i] = np.sum(P[bright]) * du

        i_max = float(np.max(intensities))
        i_min = float(np.min(intensities))
        denom = i_max + i_min
        if denom < 1e-30:
            m = 0.0
        else:
            m = (i_max - i_min) / denom
        contrasts.append(m)

    return float(np.mean(contrasts))


def compute_doi_astm(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    bsdf: np.ndarray,
    offset_deg: float = 0.3,
    aperture_half_deg: float = 0.05,
    u_center: float = 0.0,
    v_center: float = 0.0,
) -> float:
    """ASTM E430 Dorigon 方式で写像性を計算する。

    DOI_ASTM = [R(0°) - R(offset)] / R(0°)

    R(θ) は中心 (u,v)=(0,0) と (sin(offset), 0) 近傍 (半角 aperture_half_deg)
    の BSDF·cosθ 積分値。offset 側は ±方向を平均する。

    Args:
        u_grid: 方向余弦 u グリッド (2D)
        v_grid: 方向余弦 v グリッド (2D)
        bsdf: BSDF 値 [sr⁻¹] (2D)
        offset_deg: オフセット角 [deg] (デフォルト: 0.3°、ASTM E430 標準)
        aperture_half_deg: 受光絞りの半角 [deg]

    Returns:
        DOI 値 (0〜1)。値が大きいほど写像性が高い。負値は 0 にクリップ。
        パーセント表示が必要な場合は呼び出し側で ×100 する
        (ASTM E430 規格値は従来 0〜100 [%] 表記)。
    """
    uv_r2 = u_grid**2 + v_grid**2
    cos_s = np.sqrt(np.maximum(1.0 - uv_r2, 0.0))

    u_axis = u_grid[:, 0]
    v_axis = v_grid[0, :]
    du = abs(u_axis[1] - u_axis[0]) if u_axis.size > 1 else 1.0
    dv = abs(v_axis[1] - v_axis[0]) if v_axis.size > 1 else 1.0

    sin_ap2 = np.sin(np.deg2rad(aperture_half_deg)) ** 2
    sin_off = np.sin(np.deg2rad(offset_deg))

    # R(specular): 中心 (u_center, v_center) 絞り内の積分
    dr0_2 = (u_grid - u_center) ** 2 + (v_grid - v_center) ** 2
    mask0 = dr0_2 <= sin_ap2
    r0 = float(np.sum(bsdf[mask0] * cos_s[mask0]) * du * dv)

    # R(offset): 中心から ±u 方向にオフセットした点での受光の平均
    r_off_values: list[float] = []
    for sign in (+1.0, -1.0):
        u0 = u_center + sign * sin_off
        dr2 = (u_grid - u0) ** 2 + (v_grid - v_center) ** 2
        mask = dr2 <= sin_ap2
        if np.any(mask):
            r_off_values.append(float(np.sum(bsdf[mask] * cos_s[mask]) * du * dv))
    if not r_off_values:
        return 0.0
    r_off = float(np.mean(r_off_values))

    if r0 < 1e-30:
        return 0.0
    val = (r0 - r_off) / r0
    return float(max(0.0, val))


# ── ギラツキ（Sparkle） ──────────────────────────────────────────────────────

# プリセット定義
_VIEWING_PRESETS = {
    "smartphone": {"distance_mm": 300.0, "pupil_diameter_mm": 3.0},
    "tablet":     {"distance_mm": 350.0, "pupil_diameter_mm": 3.0},
    "monitor":    {"distance_mm": 600.0, "pupil_diameter_mm": 3.0},
}

_DISPLAY_PRESETS = {
    "fhd_smartphone": {"pixel_pitch_mm": 0.062, "subpixel_layout": "rgb_stripe"},
    "qhd_monitor":    {"pixel_pitch_mm": 0.124, "subpixel_layout": "rgb_stripe"},
    "4k_monitor":     {"pixel_pitch_mm": 0.160, "subpixel_layout": "rgb_stripe"},
}

def _compute_sparkle_single(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    bsdf: np.ndarray,
    omega_pupil: float,
    sin_half: float,
) -> np.ndarray:
    """1種類のBSDFグリッドについて全画素の輝度配列を計算する（内部用）。

    ベクトル化実装: 各グリッド点を最近傍画素に割り当て、np.bincount で一括積算する。
    計算量 O(N²) — ループ版の O((1/sin_half)² × N²) より大幅に高速。

    典型例: smartphone プリセット（sin_half≈0.0001）でループ版は 3.7億回だが、
    本実装は grid_size² 回（512²=26万回）のみ。
    """
    uv_r2 = u_grid**2 + v_grid**2
    valid = uv_r2 <= 1.0
    du = abs(u_grid[1, 0] - u_grid[0, 0]) if u_grid.shape[0] > 1 else 1.0
    dv = abs(v_grid[0, 1] - v_grid[0, 0]) if v_grid.shape[1] > 1 else 1.0
    cos_s = np.sqrt(np.maximum(1.0 - uv_r2, 0.0))

    # 各グリッド点を最近傍画素インデックスに割り当て（丸め）
    pix_u = np.round(u_grid / (2.0 * sin_half)).astype(np.int32)
    pix_v = np.round(v_grid / (2.0 * sin_half)).astype(np.int32)

    pu_flat = pix_u[valid]
    pv_flat = pix_v[valid]
    power_flat = (bsdf * cos_s * du * dv * omega_pupil)[valid]

    if len(power_flat) < 2:
        return np.array([], dtype=np.float64)

    # 2D 画素インデックス → 1D キー（負インデックスをオフセットで非負化）
    offset_u = int(pu_flat.min())
    offset_v = int(pv_flat.min())
    n_cols = int(pv_flat.max()) - offset_v + 1
    pixel_key = (pu_flat - offset_u) * n_cols + (pv_flat - offset_v)

    # 同一画素に属するグリッド点の輝度を合算（np.unique で連続インデックスに圧縮）
    _, inverse = np.unique(pixel_key, return_inverse=True)
    return np.bincount(inverse, weights=power_flat)


def compute_sparkle(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    bsdf: np.ndarray,
    sparkle_config: dict,
) -> float:
    """ギラツキコントラスト Cs = σ/μ を計算する。

    ディスプレイ画素ごとにBSDFを積分し、輝度のばらつき（σ/μ）を計算する。
    multi-wavelength な比較は simulate() 側の条件ループで波長ごとに独立に
    計算される（サフィックス `sparkle_fft_<nm>_<deg>_<mode>` 等で記録）。

    Args:
        u_grid: 方向余弦 u グリッド（2D）
        v_grid: 方向余弦 v グリッド（2D）
        bsdf: BSDF 値 [sr⁻¹]（2D、1 波長分）
        sparkle_config: 設定辞書（config.yaml の metrics.sparkle セクション。
            viewing と display のプリセットのみ参照。illumination は読まない）

    Returns:
        ギラツキコントラスト Cs = σ/μ
    """
    viewing = sparkle_config.get("viewing", {})
    display = sparkle_config.get("display", {})

    distance_mm = float(viewing.get("distance_mm", 300.0))
    pupil_mm = float(viewing.get("pupil_diameter_mm", 3.0))
    pixel_pitch_mm = float(display.get("pixel_pitch_mm", 0.062))

    # 瞳孔の立体角 [sr]
    omega_pupil = np.pi * (pupil_mm / 2 / distance_mm) ** 2

    # 1画素の UV 空間での半角
    sin_half = np.sin(np.arctan(pixel_pitch_mm / 2 / distance_mm))

    arr = _compute_sparkle_single(u_grid, v_grid, bsdf, omega_pupil, sin_half)
    if len(arr) < 2:
        return 0.0

    mu = float(np.mean(arr))
    if mu < 1e-30:
        return 0.0
    sigma = float(np.std(arr))
    return float(sigma / mu)


# ── 統合計算関数 ──────────────────────────────────────────────────────────────

def compute_all_optical_metrics(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    bsdf: np.ndarray,
    theta_i_deg: float = 0.0,
    mode: str = "BTDF",
    n1: float = 1.0,
    n2: float = 1.5,
    method_name: str = "fft",
    wavelength_nm: int = 555,
    simulated: np.ndarray | None = None,
    measured: np.ndarray | None = None,
    config: dict | None = None,
    bsdf_floor: float = 1e-6,
    standards_only: bool = False,
    sparkle_only: bool = False,
    allow_oblique: bool = False,
) -> dict[str, float]:
    """すべての光学指標を一括計算する。

    返り値のキーは `<name>_<method>_<deg>_<mode>` 形式（波長依存メトリクスは
    `<name>_<method>_<nm>_<deg>_<mode>`）。規格 θ_i/mode に一致しない条件では
    該当指標は計算されない（斜入射対応は `compute_at_sim_angles` で別制御）。

    Args:
        u_grid, v_grid: 方向余弦グリッド
        bsdf: BSDF 値 [sr⁻¹]
        theta_i_deg: 入射角 [deg]
        mode: "BRDF" / "BTDF"
        n1, n2: 屈折率（BTDF の Snell 計算用）
        method_name: "fft" / "psd" / "ml"（メトリクスキーのサフィックス）
        wavelength_nm: 波長 [nm]（Sparkle/Log-RMSE キーに使用）
        simulated, measured: Log-RMSE 用の BSDF 配列
        config: metrics セクションの設定辞書
        bsdf_floor: ノイズフロア [sr⁻¹]
        standards_only: True で Haze/Gloss/DOI のみ計算（代表波長条件向け）
        sparkle_only: True で Sparkle/Log-RMSE のみ計算（全条件向け）
        allow_oblique: True で規格 θ_i/mode に一致しない条件でも計算する
            （compute_at_sim_angles=true 時）

    Returns:
        指標名と値の辞書
    """
    cfg = config or {}
    results: dict[str, float] = {}

    # サフィックス構築
    theta_int = int(round(theta_i_deg))
    mode_char = "r" if mode.upper() == "BRDF" else "t"
    std_suffix = f"_{method_name}_{theta_int}_{mode_char}"
    wl_suffix = f"_{method_name}_{wavelength_nm}_{theta_int}_{mode_char}"

    # 窓中心（スペキュラー方向）
    u_c = _specular_u_center(theta_i_deg, mode, n1, n2)
    v_c = 0.0

    is_brdf = mode.upper() == "BRDF"
    is_btdf = mode.upper() == "BTDF"

    # ── 単波長メトリクス（Haze/Gloss/DOI）: 代表波長でのみ計算 ──────────────
    # 各指標は規格条件と一致する (θ_i, mode) でのみ計算する。
    # allow_oblique=True のときは規格外条件でも計算（compute_at_sim_angles 用）。
    if not sparkle_only:
        # Haze: 規格 (0°, BTDF)
        haze_cfg = cfg.get("haze")
        if haze_cfg is not None and haze_cfg.get("enabled", True):
            is_haze_std = is_btdf and theta_int == 0
            if is_haze_std or allow_oblique:
                results[f"haze{std_suffix}"] = compute_haze(
                    u_grid, v_grid, bsdf,
                    half_angle_deg=haze_cfg.get("half_angle_deg", 2.5),
                    u_center=u_c, v_center=v_c,
                )

        # Gloss: 規格 (enabled_angles, BRDF)
        gloss_cfg = cfg.get("gloss")
        if gloss_cfg is not None and gloss_cfg.get("enabled", True):
            enabled_angles = gloss_cfg.get("enabled_angles", [20, 60, 85])
            is_gloss_std = is_brdf and theta_int in enabled_angles
            if is_gloss_std or allow_oblique:
                results[f"gloss{std_suffix}"] = compute_gloss(
                    u_grid, v_grid, bsdf,
                    gloss_angle_deg=float(theta_int),
                    u_center=u_c, v_center=v_c,
                    black_glass_normalization=gloss_cfg.get("black_glass_normalization", False),
                    black_glass_n=gloss_cfg.get("black_glass_n", 1.567),
                )

        # DOI-NSER: 規格 (0°, BTDF) （独自指標・規格非準拠だが BTDF 0° を基準値とする）
        doi_nser_cfg = cfg.get("doi_nser")
        if doi_nser_cfg is not None and doi_nser_cfg.get("enabled", True):
            is_nser_std = is_btdf and theta_int == 0
            if is_nser_std or allow_oblique:
                results[f"doi_nser{std_suffix}"] = compute_doi_nser(
                    u_grid, v_grid, bsdf,
                    direct_half_angle_deg=doi_nser_cfg.get("direct_half_angle_deg", 0.1),
                    halo_half_angle_deg=doi_nser_cfg.get("halo_half_angle_deg", 2.0),
                    u_center=u_c, v_center=v_c,
                )

        # DOI-COMB: 規格 (0°, enabled_modes) — 0° 以外の sim で得られる値は oblique 扱い
        doi_comb_cfg = cfg.get("doi_comb")
        if doi_comb_cfg is not None and doi_comb_cfg.get("enabled", True):
            enabled_modes = set(doi_comb_cfg.get("enabled_modes", ["t", "r"]))
            is_comb_std = theta_int == 0 and mode_char in enabled_modes
            if is_comb_std or allow_oblique:
                results[f"doi_comb{std_suffix}"] = compute_doi_comb(
                    u_grid, v_grid, bsdf,
                    comb_widths_mm=doi_comb_cfg.get("comb_widths_mm"),
                    distance_mm=doi_comb_cfg.get("distance_mm", 280.0),
                    scan_half_angle_deg=doi_comb_cfg.get("scan_half_angle_deg", 4.0),
                    v_band_half_deg=doi_comb_cfg.get("v_band_half_deg", 0.2),
                    u_center=u_c, v_center=v_c,
                )

        # DOI-ASTM: 規格 (enabled_angles, BRDF)
        doi_astm_cfg = cfg.get("doi_astm")
        if doi_astm_cfg is not None and doi_astm_cfg.get("enabled", True):
            enabled_angles = doi_astm_cfg.get("enabled_angles", [20, 30])
            is_astm_std = is_brdf and theta_int in enabled_angles
            if is_astm_std or allow_oblique:
                results[f"doi_astm{std_suffix}"] = compute_doi_astm(
                    u_grid, v_grid, bsdf,
                    offset_deg=doi_astm_cfg.get("offset_deg", 0.3),
                    aperture_half_deg=doi_astm_cfg.get("aperture_half_deg", 0.05),
                    u_center=u_c, v_center=v_c,
                )

    # ── 波長依存メトリクス（Sparkle/Log-RMSE）──────────────────────────────
    if not standards_only:
        sparkle_cfg = cfg.get("sparkle")
        if sparkle_cfg is not None and sparkle_cfg.get("enabled", True):
            results[f"sparkle{wl_suffix}"] = compute_sparkle(u_grid, v_grid, bsdf, sparkle_cfg)

        if simulated is not None and measured is not None:
            results[f"log_rmse{wl_suffix}"] = compute_log_rmse(simulated, measured, bsdf_floor)

    return results
