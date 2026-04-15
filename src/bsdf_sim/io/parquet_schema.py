"""Parquet スキーマの定義・読み書き（long format）。

spec_main.md Section 6.2 の仕様:
- 1行 = 1測定/計算点 × 1手法
- method カテゴリ: 'FFT' / 'PSD' / 'MultiLayer' / 'measured'
- UV 座標（方向余弦）を主キーとし、角度値も逆引き用に保持
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── スキーマ定義 ──────────────────────────────────────────────────────────────

# カテゴリカラムの有効値
VALID_POLARIZATIONS = ["S", "P", "Unpolarized"]
VALID_MODES = ["BRDF", "BTDF"]
VALID_METHODS = ["FFT", "PSD", "MultiLayer", "measured"]

# Parquet カラム名と dtype
SCHEMA_DTYPES: dict[str, str] = {
    "u":             "float32",   # 方向余弦 sinθ_s·cosφ_s
    "v":             "float32",   # 方向余弦 sinθ_s·sinφ_s
    "theta_s_deg":   "float32",   # 散乱天頂角 [deg]（逆引き用）
    "phi_s_deg":     "float32",   # 散乱方位角 [deg]（逆引き用）
    "theta_i_deg":   "float32",   # 入射天頂角 [deg]
    "phi_i_deg":     "float32",   # 入射方位角 [deg]
    "wavelength_um": "float32",   # 波長 [μm]
    "polarization":  "category",  # 'S' / 'P' / 'Unpolarized'
    "mode":          "category",  # 'BRDF' / 'BTDF'
    "method":        "category",  # 'FFT' / 'PSD' / 'MultiLayer' / 'measured'
    "bsdf":          "float32",   # BSDF 値 [sr⁻¹]
    "is_measured":   "bool",      # method='measured' のとき True
    "log_rmse":      "float32",   # Log-RMSE（NaN の場合は未計算）
}


def _uv_to_angles(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """方向余弦 (u, v) から散乱角 (θ_s, φ_s) に変換する。"""
    uv_r = np.sqrt(np.clip(u**2 + v**2, 0, 1))
    theta_s = np.rad2deg(np.arcsin(uv_r))
    phi_s = np.rad2deg(np.arctan2(v, u)) % 360.0
    return theta_s.astype(np.float32), phi_s.astype(np.float32)


def build_dataframe(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    bsdf: np.ndarray,
    method: str,
    theta_i_deg: float,
    phi_i_deg: float,
    wavelength_um: float,
    polarization: str,
    is_btdf: bool = False,
    log_rmse: float | None = None,
) -> pd.DataFrame:
    """BSDF グリッドから Parquet 用 DataFrame を構築する。

    半球内（u² + v² ≤ 1）の点のみを保持する。

    Args:
        u_grid, v_grid: 方向余弦グリッド（2D）
        bsdf: BSDF 値 [sr⁻¹]（2D）
        method: 計算手法（'FFT' / 'PSD' / 'MultiLayer' / 'measured'）
        theta_i_deg, phi_i_deg: 入射角 [deg]
        wavelength_um: 波長 [μm]
        polarization: 'S' / 'P' / 'Unpolarized'
        is_btdf: True の場合 BTDF モード
        log_rmse: Log-RMSE 値（計算済みの場合）

    Returns:
        Parquet スキーマ準拠の DataFrame
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method は {VALID_METHODS} のいずれかでなければならない。値={method}")
    if polarization not in VALID_POLARIZATIONS:
        raise ValueError(f"polarization は {VALID_POLARIZATIONS} のいずれかでなければならない。")

    # 半球内の点のみ抽出
    uv_r2 = u_grid**2 + v_grid**2
    valid = uv_r2 <= 1.0

    u_flat = u_grid[valid].astype(np.float32)
    v_flat = v_grid[valid].astype(np.float32)
    bsdf_flat = bsdf[valid].astype(np.float32)

    theta_s, phi_s = _uv_to_angles(u_flat, v_flat)

    mode = "BTDF" if is_btdf else "BRDF"
    n = len(u_flat)

    df = pd.DataFrame({
        "u":             u_flat,
        "v":             v_flat,
        "theta_s_deg":   theta_s,
        "phi_s_deg":     phi_s,
        "theta_i_deg":   np.full(n, theta_i_deg, dtype=np.float32),
        "phi_i_deg":     np.full(n, phi_i_deg,   dtype=np.float32),
        "wavelength_um": np.full(n, wavelength_um, dtype=np.float32),
        "polarization":  pd.Categorical([polarization] * n, categories=VALID_POLARIZATIONS),
        "mode":          pd.Categorical([mode] * n, categories=VALID_MODES),
        "method":        pd.Categorical([method] * n, categories=VALID_METHODS),
        "bsdf":          bsdf_flat,
        "is_measured":   np.full(n, method == "measured", dtype=bool),
        "log_rmse":      np.full(n, log_rmse if log_rmse is not None else float("nan"), dtype=np.float32),
    })

    return df


def build_measured_dataframe(
    theta_s_deg: np.ndarray,
    phi_s_deg: np.ndarray,
    bsdf_values: np.ndarray,
    theta_i_deg: float,
    phi_i_deg: float,
    wavelength_nm: float,
    polarization: str,
    is_btdf: bool | None = None,
) -> pd.DataFrame:
    """実測データから Parquet 用 DataFrame を構築する。

    実測データの wavelength_nm [nm] を wavelength_um [μm] に変換する。

    Args:
        theta_s_deg: 散乱天頂角 [deg]（1D）
        phi_s_deg: 散乱方位角 [deg]（1D）
        bsdf_values: BSDF 実測値 [sr⁻¹]（1D）
        theta_i_deg: 入射天頂角 [deg]
        phi_i_deg: 入射方位角 [deg]
        wavelength_nm: 波長 [nm]（内部で μm に変換）
        polarization: 'S' / 'P' / 'Unpolarized'
        is_btdf: True → BTDF、False → BRDF、None（デフォルト）→ theta_i_deg > 90° で自動判定

    Returns:
        Parquet スキーマ準拠の DataFrame
    """
    wavelength_um = wavelength_nm / 1000.0

    theta_s_rad = np.deg2rad(theta_s_deg)
    phi_s_rad = np.deg2rad(phi_s_deg)
    u = (np.sin(theta_s_rad) * np.cos(phi_s_rad)).astype(np.float32)
    v = (np.sin(theta_s_rad) * np.sin(phi_s_rad)).astype(np.float32)

    if is_btdf is None:
        is_btdf = theta_i_deg > 90.0
    mode = "BTDF" if is_btdf else "BRDF"
    n = len(u)

    df = pd.DataFrame({
        "u":             u,
        "v":             v,
        "theta_s_deg":   theta_s_deg.astype(np.float32),
        "phi_s_deg":     phi_s_deg.astype(np.float32),
        "theta_i_deg":   np.full(n, theta_i_deg,   dtype=np.float32),
        "phi_i_deg":     np.full(n, phi_i_deg,     dtype=np.float32),
        "wavelength_um": np.full(n, wavelength_um, dtype=np.float32),
        "polarization":  pd.Categorical([polarization] * n, categories=VALID_POLARIZATIONS),
        "mode":          pd.Categorical([mode] * n, categories=VALID_MODES),
        "method":        pd.Categorical(["measured"] * n, categories=VALID_METHODS),
        "bsdf":          bsdf_values.astype(np.float32),
        "is_measured":   np.ones(n, dtype=bool),
        "log_rmse":      np.full(n, float("nan"), dtype=np.float32),
    })

    return df


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """DataFrame を Parquet ファイルとして保存する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Parquet ファイルから DataFrame を読み込む。"""
    return pd.read_parquet(path, engine="pyarrow")


def merge_sim_and_measured(
    sim_df: pd.DataFrame,
    meas_df: pd.DataFrame,
    bsdf_floor: float = 1e-6,
) -> pd.DataFrame:
    """シミュレーション結果と実測データを結合し、Log-RMSE を計算する。

    実測データの UV 座標に最も近いシミュレーション点を補間して比較する。

    Args:
        sim_df: シミュレーション結果 DataFrame
        meas_df: 実測データ DataFrame
        bsdf_floor: ノイズフロア [sr⁻¹]

    Returns:
        Log-RMSE が計算された結合 DataFrame
    """
    from scipy.interpolate import griddata

    combined = pd.concat([sim_df, meas_df], ignore_index=True)

    # 実測データの UV 座標でシミュレーション値を補間
    for method in sim_df["method"].unique():
        method_mask = sim_df["method"] == method
        sim_sub = sim_df[method_mask]

        sim_bsdf_at_meas = griddata(
            points=sim_sub[["u", "v"]].values,
            values=sim_sub["bsdf"].values,
            xi=meas_df[["u", "v"]].values,
            method="linear",
            fill_value=0.0,
        )

        meas_bsdf = meas_df["bsdf"].values
        valid_mask = meas_bsdf > bsdf_floor
        if np.any(valid_mask):
            log_sim = np.log10(np.maximum(sim_bsdf_at_meas[valid_mask], bsdf_floor))
            log_meas = np.log10(meas_bsdf[valid_mask])
            rmse = float(np.sqrt(np.mean((log_sim - log_meas) ** 2)))
        else:
            rmse = float("nan")

        # シミュレーション行の log_rmse を更新
        combined.loc[combined["method"] == method, "log_rmse"] = rmse

    return combined
