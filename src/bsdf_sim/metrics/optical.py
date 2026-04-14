"""光学指標の計算（Haze, Gloss, DOI, Sparkle）。

spec_main.md Section 5 の仕様に従い実装する。
物理単位は μm 統一。角度は度数法 [deg]。
"""

from __future__ import annotations

import numpy as np


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
) -> float:
    """ヘイズ（Haze）を計算する。

    ヘイズ = 2.5°以上の広角へ散乱した透過光の割合。

    Args:
        u_grid: 方向余弦 u グリッド（2D）
        v_grid: 方向余弦 v グリッド（2D）
        bsdf: BSDF 値 [sr⁻¹]（2D）
        half_angle_deg: ヘイズ境界角 [deg]（デフォルト: 2.5°）

    Returns:
        ヘイズ値（0〜1）
    """
    uv_r2 = u_grid**2 + v_grid**2
    half_angle_rad = np.deg2rad(half_angle_deg)
    sin_half = np.sin(half_angle_rad)
    threshold_r2 = sin_half**2

    # 微小立体角 dΩ = cos(θ_s) * du * dv（UV空間での積分）
    du = abs(u_grid[1, 0] - u_grid[0, 0]) if u_grid.shape[0] > 1 else 1.0
    dv = abs(v_grid[0, 1] - v_grid[0, 0]) if v_grid.shape[1] > 1 else 1.0
    cos_s = np.sqrt(np.maximum(1.0 - uv_r2, 0.0))
    valid = uv_r2 <= 1.0

    # 全透過光量（半球積分）
    total = np.sum(bsdf[valid] * cos_s[valid]) * du * dv

    # 広角散乱光量（境界角以上）
    wide = uv_r2 >= threshold_r2
    haze_power = np.sum(bsdf[valid & wide] * cos_s[valid & wide]) * du * dv

    if total < 1e-30:
        return 0.0
    return float(haze_power / total)


# ── グロス ───────────────────────────────────────────────────────────────────

def compute_gloss(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    bsdf: np.ndarray,
    gloss_angle_deg: float = 60.0,
    acceptance_deg: float = 0.9,
) -> float:
    """グロス（Gloss）を計算する。

    正反射方向（gloss_angle_deg）付近のピーク強度を評価する。

    Args:
        u_grid: 方向余弦 u グリッド（2D）
        v_grid: 方向余弦 v グリッド（2D）
        bsdf: BSDF 値 [sr⁻¹]（2D）
        gloss_angle_deg: グロス測定角 [deg]（デフォルト: 60°）
        acceptance_deg: 受光角範囲 ±[deg]（デフォルト: 0.9°）

    Returns:
        グロス値 [sr⁻¹]（正反射方向の平均 BSDF）
    """
    gloss_rad = np.deg2rad(gloss_angle_deg)
    acc_rad = np.deg2rad(acceptance_deg)

    # 正反射方向の u, v（phi_s=0 方向）
    u_spec = np.sin(gloss_rad)
    uv_r2 = u_grid**2 + v_grid**2
    valid = uv_r2 <= 1.0

    # 指定角度付近の点を選択
    u_diff = np.abs(u_grid - u_spec)
    v_diff = np.abs(v_grid)
    angle_diff = np.sqrt(u_diff**2 + v_diff**2)
    in_acceptance = angle_diff <= np.sin(acc_rad)

    selected = valid & in_acceptance
    if not np.any(selected):
        return 0.0
    return float(np.mean(bsdf[selected]))


# ── 写像性（DOI / Clarity） ──────────────────────────────────────────────────

def compute_doi(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    bsdf: np.ndarray,
    direct_half_angle_deg: float = 0.1,
    halo_half_angle_deg: float = 2.0,
) -> float:
    """写像性（DOI / Clarity）を計算する。

    DOI = 直進光エネルギー / (直進光 + 小角散乱光)

    Args:
        u_grid: 方向余弦 u グリッド（2D）
        v_grid: 方向余弦 v グリッド（2D）
        bsdf: BSDF 値 [sr⁻¹]（2D）
        direct_half_angle_deg: 直進光の半角 [deg]（デフォルト: 0.1°）
        halo_half_angle_deg: ハロー領域の半角 [deg]（デフォルト: 2.0°）

    Returns:
        DOI 値（0〜1）
    """
    uv_r2 = u_grid**2 + v_grid**2
    cos_s = np.sqrt(np.maximum(1.0 - uv_r2, 0.0))
    valid = uv_r2 <= 1.0

    du = abs(u_grid[1, 0] - u_grid[0, 0]) if u_grid.shape[0] > 1 else 1.0
    dv = abs(v_grid[0, 1] - v_grid[0, 0]) if v_grid.shape[1] > 1 else 1.0

    sin_direct = np.sin(np.deg2rad(direct_half_angle_deg))
    sin_halo = np.sin(np.deg2rad(halo_half_angle_deg))

    direct_mask = valid & (uv_r2 <= sin_direct**2)
    halo_mask = valid & (uv_r2 <= sin_halo**2)

    direct_power = np.sum(bsdf[direct_mask] * cos_s[direct_mask]) * du * dv
    total_near = np.sum(bsdf[halo_mask] * cos_s[halo_mask]) * du * dv

    if total_near < 1e-30:
        return 1.0
    return float(direct_power / total_near)


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

_ILLUMINATION_PRESETS = {
    "green": {"wavelengths_um": [0.55]},
    "rgb":   {"wavelengths_um": [0.45, 0.55, 0.65]},
}

# RGB 輝度加重（CIE 1931）
_RGB_LUMINANCE_WEIGHTS = [0.0722, 0.7152, 0.2126]  # B, G, R


def _compute_sparkle_single(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    bsdf: np.ndarray,
    omega_pupil: float,
    sin_half: float,
) -> list[float]:
    """1種類のBSDFグリッドについて全画素の輝度リストを計算する（内部用）。"""
    uv_r2 = u_grid**2 + v_grid**2
    valid = uv_r2 <= 1.0
    du = abs(u_grid[1, 0] - u_grid[0, 0]) if u_grid.shape[0] > 1 else 1.0
    dv = abs(v_grid[0, 1] - v_grid[0, 0]) if v_grid.shape[1] > 1 else 1.0

    n_pix_u = max(1, int(1.0 / sin_half))
    luminances: list[float] = []

    for pu in range(-n_pix_u, n_pix_u + 1):
        for pv in range(-n_pix_u, n_pix_u + 1):
            u_center = pu * 2 * sin_half
            v_center = pv * 2 * sin_half
            pix_mask = (
                valid
                & (np.abs(u_grid - u_center) <= sin_half)
                & (np.abs(v_grid - v_center) <= sin_half)
            )
            if not np.any(pix_mask):
                continue
            cos_s = np.sqrt(np.maximum(1.0 - uv_r2[pix_mask], 0.0))
            lum = float(np.sum(bsdf[pix_mask] * cos_s) * du * dv * omega_pupil)
            luminances.append(lum)

    return luminances


def compute_sparkle(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    bsdf: np.ndarray,
    sparkle_config: dict,
    bsdf_per_wavelength: list[np.ndarray] | None = None,
) -> float:
    """ギラツキコントラスト Cs = σ/μ を計算する。

    ディスプレイ画素ごとにBSDFを積分し、輝度のばらつき（σ/μ）を計算する。
    RGB照明モード（wavelengths_um に3波長を指定）の場合は CIE 1931 輝度加重平均を行う。

    Args:
        u_grid: 方向余弦 u グリッド（2D）
        v_grid: 方向余弦 v グリッド（2D）
        bsdf: BSDF 値 [sr⁻¹]（2D）— 単波長モードで使用
        sparkle_config: 設定辞書（config.yaml の metrics.sparkle セクション）
        bsdf_per_wavelength: 複数波長のBSDFリスト（波長順、wavelengths_um と対応）。
            指定時は CIE 輝度加重でスパークルを合成する。

    Returns:
        ギラツキコントラスト Cs = σ/μ
    """
    # 設定の解決
    viewing = sparkle_config.get("viewing", {})
    display = sparkle_config.get("display", {})
    illumination = sparkle_config.get("illumination", {})

    distance_mm = float(viewing.get("distance_mm", 300.0))
    pupil_mm = float(viewing.get("pupil_diameter_mm", 3.0))
    pixel_pitch_mm = float(display.get("pixel_pitch_mm", 0.062))
    wavelengths_um = list(illumination.get("wavelengths_um", [0.55]))

    # 瞳孔の立体角 [sr]
    omega_pupil = np.pi * (pupil_mm / 2 / distance_mm) ** 2

    # 1画素の UV 空間での半角
    sin_half = np.sin(np.arctan(pixel_pitch_mm / 2 / distance_mm))

    # 複数波長モード（CIE 輝度加重平均）
    if bsdf_per_wavelength is not None and len(bsdf_per_wavelength) == len(wavelengths_um) > 1:
        # 波長に対応する CIE 輝度感度（標準値: 450/550/650nm ≒ B/G/R）
        cie_weights = np.array(_RGB_LUMINANCE_WEIGHTS[: len(wavelengths_um)], dtype=np.float64)
        cie_weights = cie_weights / cie_weights.sum()  # 正規化

        # 各波長の画素輝度リストを加重合成
        all_luminances_list = [
            _compute_sparkle_single(u_grid, v_grid, bsdf_w, omega_pupil, sin_half)
            for bsdf_w in bsdf_per_wavelength
        ]
        n_pixels = min(len(lst) for lst in all_luminances_list)
        if n_pixels < 2:
            return 0.0
        combined = sum(
            w * np.array(lst[:n_pixels])
            for w, lst in zip(cie_weights, all_luminances_list)
        )
        arr = np.asarray(combined)
    else:
        # 単波長モード
        luminances = _compute_sparkle_single(u_grid, v_grid, bsdf, omega_pupil, sin_half)
        if len(luminances) < 2:
            return 0.0
        arr = np.array(luminances)

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
    simulated: np.ndarray | None = None,
    measured: np.ndarray | None = None,
    config: dict | None = None,
    bsdf_floor: float = 1e-6,
) -> dict[str, float]:
    """すべての光学指標を一括計算する。

    Args:
        u_grid, v_grid: 方向余弦グリッド
        bsdf: BSDF 値 [sr⁻¹]
        simulated: Log-RMSE 計算用シミュレーション値（オプション）
        measured: Log-RMSE 計算用実測値（オプション）
        config: metrics セクションの設定辞書
        bsdf_floor: ノイズフロア [sr⁻¹]

    Returns:
        指標名と値の辞書
    """
    cfg = config or {}
    results: dict[str, float] = {}

    if cfg.get("haze", {}).get("enabled", True):
        half_angle = cfg.get("haze", {}).get("half_angle_deg", 2.5)
        results["haze"] = compute_haze(u_grid, v_grid, bsdf, half_angle)

    if cfg.get("gloss", {}).get("enabled", True):
        angle = cfg.get("gloss", {}).get("angle_deg", 60.0)
        results["gloss"] = compute_gloss(u_grid, v_grid, bsdf, angle)

    if cfg.get("doi", {}).get("enabled", True):
        results["doi"] = compute_doi(u_grid, v_grid, bsdf)

    if cfg.get("sparkle", {}).get("enabled", True):
        results["sparkle"] = compute_sparkle(u_grid, v_grid, bsdf, cfg.get("sparkle", {}))

    if simulated is not None and measured is not None:
        results["log_rmse"] = compute_log_rmse(simulated, measured, bsdf_floor)

    return results
