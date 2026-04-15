"""表面形状指標の計算。

準拠規格:
  - ISO 25178-2: 面の表面性状 — 面パラメータ（S-パラメータ）
  - JIS B 0601 / ISO 4287: 輪郭曲線方式 — プロファイルパラメータ（R-パラメータ）

物理単位は μm 統一。角度・無次元量はそれぞれ rad・無次元で返す。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ..models.base import HeightMap


# ─────────────────────────────────────────────────────────────────────────────
# 内部ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def _demean(data: np.ndarray) -> np.ndarray:
    """平均高さを差し引いてゼロ平均化する。"""
    return data - data.mean()


def _row_profiles(hm: HeightMap) -> np.ndarray:
    """行・列両方向のプロファイルをゼロ平均化して返す。shape: (2N, N)

    ISO 4287 では測定方向を指定するが、面粗さ評価では等方性を仮定して
    行方向と列方向の両プロファイルを平均することが一般的。
    """
    data = hm.data.astype(np.float64)
    rows = data - data.mean(axis=1, keepdims=True)          # 行プロファイル: (N, N)
    cols = data.T - data.T.mean(axis=1, keepdims=True)      # 列プロファイル: (N, N)
    return np.vstack([rows, cols])                           # (2N, N)


# ─────────────────────────────────────────────────────────────────────────────
# ISO 25178-2 面パラメータ（S-パラメータ）
# ─────────────────────────────────────────────────────────────────────────────

def compute_sq(hm: HeightMap) -> float:
    """二乗平均平方根高さ Sq [μm]（ISO 25178-2）。

    Sq = sqrt(1/A ∫∫ z(x,y)² dA)
    """
    z = _demean(hm.data.astype(np.float64))
    return float(np.sqrt(np.mean(z**2)))


def compute_sa(hm: HeightMap) -> float:
    """算術平均高さ Sa [μm]（ISO 25178-2）。

    Sa = 1/A ∫∫ |z(x,y)| dA
    """
    z = _demean(hm.data.astype(np.float64))
    return float(np.mean(np.abs(z)))


def compute_sp(hm: HeightMap) -> float:
    """最大山高さ Sp [μm]（ISO 25178-2）。

    Sp = max(z)  （平均面基準）
    """
    z = _demean(hm.data.astype(np.float64))
    return float(np.max(z))


def compute_sv(hm: HeightMap) -> float:
    """最大谷深さ Sv [μm]（ISO 25178-2）。

    Sv = |min(z)|  （平均面基準、正値で返す）
    """
    z = _demean(hm.data.astype(np.float64))
    return float(-np.min(z))


def compute_sz(hm: HeightMap) -> float:
    """最大高さ Sz [μm]（ISO 25178-2）。

    Sz = Sp + Sv = max(z) - min(z)
    """
    z = _demean(hm.data.astype(np.float64))
    return float(np.max(z) - np.min(z))


def compute_ssk(hm: HeightMap) -> float:
    """スキューネス Ssk [-]（ISO 25178-2）。

    Ssk = (1/Sq³) · (1/A ∫∫ z³ dA)

    - Ssk > 0: 突起が多い（峰が高い）
    - Ssk < 0: くぼみが多い（谷が深い）
    - Ssk ≈ 0: ガウス分布（対称）
    """
    z = _demean(hm.data.astype(np.float64))
    sq = float(np.sqrt(np.mean(z**2)))
    if sq < 1e-30:
        return 0.0
    return float(np.mean(z**3) / sq**3)


def compute_sku(hm: HeightMap) -> float:
    """クルトシス Sku [-]（ISO 25178-2）。

    Sku = (1/Sq⁴) · (1/A ∫∫ z⁴ dA)

    - Sku = 3: ガウス分布
    - Sku > 3: 急峻な突起・谷（レプトクルティック）
    - Sku < 3: なだらか（プラチクルティック）
    """
    z = _demean(hm.data.astype(np.float64))
    sq = float(np.sqrt(np.mean(z**2)))
    if sq < 1e-30:
        return 0.0
    return float(np.mean(z**4) / sq**4)


def compute_sdq(hm: HeightMap) -> float:
    """二乗平均平方根傾斜 Sdq [rad]（ISO 25178-2）。

    Sdq = sqrt(mean((∂z/∂x)² + (∂z/∂y)²))
    """
    dx = hm.pixel_size_um
    z = hm.data.astype(np.float64)
    dzdx = np.gradient(z, dx, axis=0)
    dzdy = np.gradient(z, dx, axis=1)
    return float(np.sqrt(np.mean(dzdx**2 + dzdy**2)))


def compute_sdr(hm: HeightMap) -> float:
    """界面展開面積比 Sdr [%]（ISO 25178-2）。

    Sdr = (実面積 - 投影面積) / 投影面積 × 100

    実面積 = Σ sqrt(1 + (∂z/∂x)² + (∂z/∂y)²) · Δx · Δy
    投影面積 = (N · Δx)²
    """
    dx = hm.pixel_size_um
    z = hm.data.astype(np.float64)
    dzdx = np.gradient(z, dx, axis=0)
    dzdy = np.gradient(z, dx, axis=1)
    actual_area = np.sum(np.sqrt(1.0 + dzdx**2 + dzdy**2)) * dx**2
    projected_area = (hm.grid_size * dx) ** 2
    return float((actual_area - projected_area) / projected_area * 100.0)


def compute_sal(hm: HeightMap, acf_threshold: float = 0.2) -> float:
    """自己相関長 Sal [μm]（ISO 25178-2）。

    正規化自己相関関数（NACF）が全方向で acf_threshold 以下となる
    最小ラグ距離。最速減衰方向の相関長に相当する。

    Args:
        hm: HeightMap
        acf_threshold: NACF のしきい値（デフォルト: 0.2）

    Returns:
        Sal [μm]
    """
    return _compute_sal_str(hm, acf_threshold)[0]


def compute_str(hm: HeightMap, acf_threshold: float = 0.2) -> float:
    """テクスチャアスペクト比 Str [-]（ISO 25178-2）。

    Str = Sal_最短 / Sal_最長

    - Str → 1: 等方性テクスチャ
    - Str → 0: 一方向性（異方性）テクスチャ

    Args:
        hm: HeightMap
        acf_threshold: NACF のしきい値（デフォルト: 0.2）

    Returns:
        Str [-]（0〜1）
    """
    sal_min, sal_max = _compute_sal_str(hm, acf_threshold)
    if sal_max < 1e-30:
        return 0.0
    return float(sal_min / sal_max)


def _compute_sal_str(
    hm: HeightMap, acf_threshold: float = 0.2
) -> tuple[float, float]:
    """Sal と Str の共通計算（内部関数）。

    Returns:
        (Sal_min, Sal_max) [μm]  各方向の相関長の最小・最大
    """
    N = hm.grid_size
    dx = hm.pixel_size_um
    z = _demean(hm.data.astype(np.float64))
    sq2 = np.mean(z**2)
    if sq2 < 1e-30:
        return 0.0, 0.0

    # 2D 自己相関関数を FFT で計算
    Z = np.fft.fft2(z)
    acf2d = np.real(np.fft.ifft2(np.abs(Z) ** 2)) / (N * N * sq2)
    acf2d = np.fft.fftshift(acf2d)  # DC を中央へ

    # 各方向（0°〜175° を 5°刻み）でしきい値を下回る最初のラグを求める
    cx, cy = N // 2, N // 2
    n_angles = 36
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    max_lag_px = N // 2 - 1
    sal_per_angle: list[float] = []

    for theta in angles:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        found = False
        for lag_px in range(1, max_lag_px + 1):
            ix = cx + round(lag_px * cos_t)
            iy = cy + round(lag_px * sin_t)
            if not (0 <= ix < N and 0 <= iy < N):
                break
            if acf2d[ix, iy] < acf_threshold:
                sal_per_angle.append(lag_px * dx)
                found = True
                break
        if not found:
            sal_per_angle.append(max_lag_px * dx)

    sal_min = float(min(sal_per_angle))
    sal_max = float(max(sal_per_angle))
    return sal_min, sal_max


# ─────────────────────────────────────────────────────────────────────────────
# JIS B 0601 / ISO 4287 プロファイルパラメータ（R-パラメータ）
# ─────────────────────────────────────────────────────────────────────────────
# 高さマップの全行をプロファイルとして評価し、各行の値を平均する。
# ─────────────────────────────────────────────────────────────────────────────

def compute_rq(hm: HeightMap) -> float:
    """プロファイル二乗平均平方根粗さ Rq [μm]（JIS B 0601）。

    全行プロファイルの Rq を平均する。
    """
    profiles = _row_profiles(hm)
    return float(np.mean(np.sqrt(np.mean(profiles**2, axis=1))))


def compute_ra(hm: HeightMap) -> float:
    """プロファイル算術平均粗さ Ra [μm]（JIS B 0601）。

    全行プロファイルの Ra を平均する。
    """
    profiles = _row_profiles(hm)
    return float(np.mean(np.mean(np.abs(profiles), axis=1)))


def compute_rz(hm: HeightMap) -> float:
    """プロファイル最大高さ Rz [μm]（JIS B 0601）。

    全行プロファイルの (max - min) を平均する。
    """
    profiles = _row_profiles(hm)
    return float(np.mean(np.max(profiles, axis=1) - np.min(profiles, axis=1)))


def compute_rp(hm: HeightMap) -> float:
    """プロファイル最大山高さ Rp [μm]（JIS B 0601）。

    全行プロファイルの max(z) を平均する（平均線基準）。
    """
    profiles = _row_profiles(hm)
    return float(np.mean(np.max(profiles, axis=1)))


def compute_rv(hm: HeightMap) -> float:
    """プロファイル最大谷深さ Rv [μm]（JIS B 0601）。

    全行プロファイルの |min(z)| を平均する（平均線基準、正値で返す）。
    """
    profiles = _row_profiles(hm)
    return float(np.mean(-np.min(profiles, axis=1)))


def compute_rsk(hm: HeightMap) -> float:
    """プロファイルスキューネス Rsk [-]（JIS B 0601）。

    Rsk = (1/Rq³) · mean(z³)  ← 全行プロファイルの平均

    - Rsk > 0: 突起が多い
    - Rsk < 0: くぼみが多い
    """
    profiles = _row_profiles(hm)
    rq_per_row = np.sqrt(np.mean(profiles**2, axis=1))
    mask = rq_per_row > 1e-30
    if not np.any(mask):
        return 0.0
    rsk_per_row = np.where(
        mask,
        np.mean(profiles**3, axis=1) / np.where(mask, rq_per_row**3, 1.0),
        0.0,
    )
    return float(np.mean(rsk_per_row[mask]))


def compute_rku(hm: HeightMap) -> float:
    """プロファイルクルトシス Rku [-]（JIS B 0601）。

    Rku = (1/Rq⁴) · mean(z⁴)  ← 全行プロファイルの平均

    - Rku = 3: ガウス分布
    - Rku > 3: 急峻な突起
    - Rku < 3: なだらか
    """
    profiles = _row_profiles(hm)
    rq_per_row = np.sqrt(np.mean(profiles**2, axis=1))
    mask = rq_per_row > 1e-30
    if not np.any(mask):
        return 0.0
    rku_per_row = np.where(
        mask,
        np.mean(profiles**4, axis=1) / np.where(mask, rq_per_row**4, 1.0),
        0.0,
    )
    return float(np.mean(rku_per_row[mask]))


def compute_rsm(hm: HeightMap, height_fraction: float = 0.1) -> float:
    """プロファイル要素の平均幅 Rsm [μm]（JIS B 0601）。

    プロファイル要素（山と隣接する谷の1組）の平均幅。
    平均線を挟む正→負→正の交差を1要素として数える。

    Args:
        hm: HeightMap
        height_fraction: 有効な山・谷の高さ判定に使う Rz に対する割合
                         （デフォルト: 10%、JIS B 0601 推奨）

    Returns:
        Rsm [μm]
    """
    profiles = _row_profiles(hm)
    dx = hm.pixel_size_um
    rsm_list: list[float] = []

    for row in profiles:
        rz_row = float(np.max(row) - np.min(row))
        threshold = height_fraction * rz_row

        # 平均線（=0）の正→負交差点を検出
        sign = np.sign(row)
        sign[sign == 0] = 1  # ゼロは正として扱う
        crossings = np.where(np.diff(sign) < 0)[0]  # 正→負交差のインデックス

        # 有効な山がある要素のみカウント（高さ判定）
        valid_crossings = [
            i for i in crossings
            if np.max(row[max(0, i - 5): i + 1]) >= threshold
        ]

        n_elements = len(valid_crossings)
        if n_elements > 0:
            profile_len = len(row) * dx
            rsm_list.append(profile_len / n_elements)

    if not rsm_list:
        return float(hm.physical_size_um)
    return float(np.mean(rsm_list))


def compute_rc(hm: HeightMap) -> float:
    """プロファイル要素の平均高さ Rc [μm]（JIS B 0601）。

    Rc = mean(山高さ + 谷深さ) の各要素平均。
    山ピーク高さと隣接谷深さの和の平均。

    Returns:
        Rc [μm]
    """
    profiles = _row_profiles(hm)
    rc_list: list[float] = []

    for row in profiles:
        # ピーク（山）の検出
        peaks, _ = find_peaks(row, height=0)
        # 谷（valleys）の検出
        valleys, _ = find_peaks(-row, height=0)

        if len(peaks) == 0 or len(valleys) == 0:
            continue

        # 各ピークとその最近傍の谷の高さの和
        element_heights: list[float] = []
        for pk in peaks:
            nearest_valley = valleys[np.argmin(np.abs(valleys - pk))]
            zt = float(row[pk] - row[nearest_valley])
            if zt > 0:
                element_heights.append(zt)

        if element_heights:
            rc_list.append(float(np.mean(element_heights)))

    if not rc_list:
        return 0.0
    return float(np.mean(rc_list))


# ─────────────────────────────────────────────────────────────────────────────
# 統合計算関数
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_surface_metrics(
    hm: HeightMap,
    verbose: bool = False,
) -> dict[str, float]:
    """全 JIS/ISO 表面形状指標を一括計算する。

    Args:
        hm: 高さマップ
        verbose: True のとき tqdm プログレスバーを表示する

    Returns:
        指標名と値の辞書。キーの接尾辞: _um=μm, _pct=%, _rad=rad, 無次元は省略。

    ISO 25178-2 面パラメータ（S-パラメータ）:
        Sq, Sa, Sp, Sv, Sz  ← 振幅
        Ssk, Sku            ← 統計
        Sdq                 ← 傾斜（ハイブリッド）
        Sdr                 ← 展開面積（ハイブリッド）
        Sal                 ← 自己相関長（空間）
        Str                 ← テクスチャアスペクト比（空間）

    JIS B 0601 / ISO 4287 プロファイルパラメータ（R-パラメータ）:
        Rq, Ra, Rz          ← 振幅
        Rp, Rv              ← 山・谷
        Rsk, Rku            ← 統計
        Rsm                 ← 間隔
        Rc                  ← 要素高さ
    """
    steps: list[tuple[str, object]] = [
        # ── ISO 25178-2 S-パラメータ ──────────────────────────────────────────
        ("sq_um",   compute_sq),
        ("sa_um",   compute_sa),
        ("sp_um",   compute_sp),
        ("sv_um",   compute_sv),
        ("sz_um",   compute_sz),
        ("ssk",     compute_ssk),
        ("sku",     compute_sku),
        ("sdq_rad", compute_sdq),
        ("sdr_pct", compute_sdr),
        ("sal_um",  compute_sal),
        ("str",     compute_str),
        # ── JIS B 0601 / ISO 4287 R-パラメータ ───────────────────────────────
        ("rq_um",   compute_rq),
        ("ra_um",   compute_ra),
        ("rz_um",   compute_rz),
        ("rp_um",   compute_rp),
        ("rv_um",   compute_rv),
        ("rsk",     compute_rsk),
        ("rku",     compute_rku),
        ("rsm_um",  compute_rsm),
        ("rc_um",   compute_rc),
    ]

    if verbose:
        try:
            from tqdm import tqdm
            steps = tqdm(steps, desc="  表面形状指標", unit="指標", leave=False)
        except ImportError:
            pass

    return {key: fn(hm) for key, fn in steps}
