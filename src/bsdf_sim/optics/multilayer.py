"""多層BSDF合成（Adding-Doubling法）。

spec_main.md Section 4 の仕様に従い実装する。
各層のBSDFを散乱行列に変換し、Adding-Doubling法で合成する。

物理単位は μm 統一。
"""

from __future__ import annotations

import numpy as np


# ── 離散化プリセット ──────────────────────────────────────────────────────────

_PRECISION_PRESETS: dict[str, dict[str, int]] = {
    "fast":     {"n_theta": 32,  "m_phi": 8},
    "standard": {"n_theta": 128, "m_phi": 18},
    "high":     {"n_theta": 256, "m_phi": 36},
}


def _get_quadrature(n_theta: int) -> tuple[np.ndarray, np.ndarray]:
    """ガウス-ルジャンドル求積点と重みを返す（仰角 0〜π/2 に変換済み）。

    Args:
        n_theta: 求積点数

    Returns:
        theta_quad: 仰角 [rad]（shape: n_theta）
        weights: 重み（shape: n_theta）
    """
    # [-1, 1] のガウス-ルジャンドル点 → [0, π/2] に変換
    x, w = np.polynomial.legendre.leggauss(n_theta)
    theta_quad = np.arccos((1 - x) / 2)  # [0, π/2] に写像
    weights = w / 2  # 積分区間の変換に伴う因子
    return theta_quad, weights


def _hg_phase(cos_angle: np.ndarray, g: float) -> np.ndarray:
    """Henyey-Greenstein 位相関数。

    p(cos θ) = (1 - g²) / (1 + g² - 2g*cos θ)^(3/2) / (4π)

    Args:
        cos_angle: 散乱角の余弦（shape: arbitrary）
        g: 非対称パラメータ（-1〜1）

    Returns:
        位相関数値（shape: cos_angle と同形状）
    """
    denom = (1 + g**2 - 2 * g * cos_angle) ** 1.5
    return (1 - g**2) / (4 * np.pi * np.maximum(denom, 1e-30))


def _build_scatter_matrix_from_bsdf(
    bsdf_2d: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    theta_quad: np.ndarray,
    weights: np.ndarray,
    m_phi: int,
) -> np.ndarray:
    """2次元 BSDF から散乱行列を構築する。

    方位角フーリエ展開（m_phi モード）と仰角ガウス求積（n_theta 点）で離散化する。

    Args:
        bsdf_2d: BSDF グリッド（2D, sr⁻¹）
        u_grid: 方向余弦 u グリッド
        v_grid: 方向余弦 v グリッド
        theta_quad: 仰角求積点 [rad]
        weights: 求積重み
        m_phi: 方位角フーリエモード数

    Returns:
        散乱行列 S（shape: n_theta × n_theta）[簡略版: 軸対称近似]
    """
    from scipy.interpolate import RegularGridInterpolator

    n_theta = len(theta_quad)
    u_axis = u_grid[:, 0]
    v_axis = v_grid[0, :]

    interp = RegularGridInterpolator(
        (u_axis, v_axis),
        bsdf_2d,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # 軸対称近似（方位角平均）で散乱行列を構築
    S = np.zeros((n_theta, n_theta), dtype=np.float64)
    n_phi_samples = max(m_phi * 4, 36)
    phi_samples = np.linspace(0, 2 * np.pi, n_phi_samples, endpoint=False)

    for i, theta_i in enumerate(theta_quad):
        u_i = np.sin(theta_i)
        for j, theta_s in enumerate(theta_quad):
            sin_s = np.sin(theta_s)
            u_s = sin_s * np.cos(phi_samples)
            v_s = sin_s * np.sin(phi_samples)
            pts = np.column_stack([u_s, v_s])
            bsdf_vals = interp(pts)
            S[i, j] = np.mean(bsdf_vals) * weights[j] * np.sin(theta_s)

    return S


def _build_bulk_scatter_matrix(
    theta_quad: np.ndarray,
    weights: np.ndarray,
    g: float,
    mu_s: float,
    thickness_um: float,
) -> np.ndarray:
    """内部媒質（ボリューム散乱）の散乱行列を構築する（HG位相関数）。

    Args:
        theta_quad: 仰角求積点 [rad]
        weights: 求積重み
        g: HG 非対称パラメータ
        mu_s: 散乱係数 [μm⁻¹]
        thickness_um: 層厚 [μm]

    Returns:
        散乱行列 S（shape: n_theta × n_theta）
    """
    n = len(theta_quad)
    S = np.zeros((n, n), dtype=np.float64)
    optical_depth = mu_s * thickness_um

    for i, theta_i in enumerate(theta_quad):
        for j, theta_s in enumerate(theta_quad):
            cos_angle = np.cos(theta_i) * np.cos(theta_s) + np.sin(theta_i) * np.sin(theta_s)
            phase = _hg_phase(np.array([cos_angle]), g)[0]
            # 単層 Beer-Lambert 透過と散乱の組み合わせ
            S[i, j] = (
                phase
                * (1 - np.exp(-optical_depth))
                * weights[j]
                * np.sin(theta_s)
            )

    return S


def adding_step(
    S_a: np.ndarray,
    S_b: np.ndarray,
) -> np.ndarray:
    """2つの散乱行列を Adding 法で合成する。

    S_ab = S_b + S_a @ inv(I - S_b @ S_a) @ S_b  （簡略実装）
    多重反射の寄与を考慮した合成。

    Args:
        S_a: 上層の散乱行列（n_theta × n_theta）
        S_b: 下層の散乱行列（n_theta × n_theta）

    Returns:
        合成散乱行列（n_theta × n_theta）
    """
    n = S_a.shape[0]
    I = np.eye(n)
    # 多重散乱を Neumann 級数で近似（1次）
    M = I - S_b @ S_a
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        M_inv = np.eye(n)

    return S_b + S_a @ M_inv @ S_b


class MultiLayerBSDF:
    """多層BSDF合成（Adding-Doubling法）クラス。

    Args:
        precision: 離散化プリセット（'fast' / 'standard' / 'high'）
        n_theta: 仰角ガウス求積点数（None の場合はプリセット値）
        m_phi: 方位角フーリエモード数（None の場合はプリセット値）
    """

    def __init__(
        self,
        precision: str = "standard",
        n_theta: int | None = None,
        m_phi: int | None = None,
    ) -> None:
        preset = _PRECISION_PRESETS.get(precision, _PRECISION_PRESETS["standard"])
        self.n_theta = n_theta if n_theta is not None else preset["n_theta"]
        self.m_phi = m_phi if m_phi is not None else preset["m_phi"]
        self._theta_quad, self._weights = _get_quadrature(self.n_theta)
        self._layers: list[dict] = []

    def add_surface_layer(
        self,
        bsdf_2d: np.ndarray,
        u_grid: np.ndarray,
        v_grid: np.ndarray,
    ) -> None:
        """FFT/PSD法で計算した表面BSDFを層として追加する。

        Args:
            bsdf_2d: BSDF グリッド [sr⁻¹]
            u_grid: 方向余弦 u グリッド
            v_grid: 方向余弦 v グリッド
        """
        S = _build_scatter_matrix_from_bsdf(
            bsdf_2d, u_grid, v_grid,
            self._theta_quad, self._weights, self.m_phi,
        )
        self._layers.append({"type": "surface", "matrix": S})

    def add_bulk_layer(
        self,
        g: float,
        scattering_coeff_um: float,
        thickness_um: float,
    ) -> None:
        """内部ヘイズ層（ボリューム散乱）を追加する。

        Args:
            g: Henyey-Greenstein 非対称パラメータ（-1〜1）
            scattering_coeff_um: 散乱係数 [μm⁻¹]
            thickness_um: 層厚 [μm]
        """
        S = _build_bulk_scatter_matrix(
            self._theta_quad, self._weights,
            g, scattering_coeff_um, thickness_um,
        )
        self._layers.append({"type": "bulk", "matrix": S})

    def compute(self) -> tuple[np.ndarray, np.ndarray]:
        """Adding-Doubling法で全層を合成し、トータル散乱行列を返す。

        Returns:
            theta_quad: 仰角求積点 [rad]
            S_total: 合成散乱行列（n_theta × n_theta）
        """
        if not self._layers:
            raise ValueError("少なくとも1つの層が必要。")

        S_total = self._layers[0]["matrix"]
        for layer in self._layers[1:]:
            S_total = adding_step(S_total, layer["matrix"])

        return self._theta_quad, S_total

    def to_bsdf_1d(self) -> tuple[np.ndarray, np.ndarray]:
        """合成散乱行列から法線入射時の1次元BSDFプロファイルを取得する。

        Returns:
            theta_deg: 散乱角 [deg]
            bsdf_1d: BSDF 値 [sr⁻¹]
        """
        theta_quad, S_total = self.compute()
        # 法線入射（最初の求積点に最も近い）からの行を取得
        normal_idx = np.argmin(np.abs(theta_quad))
        bsdf_row = S_total[normal_idx, :]
        theta_deg = np.rad2deg(theta_quad)
        return theta_deg, bsdf_row
