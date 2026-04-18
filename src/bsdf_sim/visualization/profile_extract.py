"""BSDF 2D グリッドから phi≈0 方向の 1D プロファイルを抽出するヘルパー。

BSDF の (u_grid, v_grid, bsdf) 2D 配列は `np.fft.fftfreq` 由来で
u, v 軸が `[0, +1/N, +2/N, ..., +1/(2dx), -1/(2dx), ..., -1/N]` の並び。
このため単純に `|u|` を取って Curve に渡すと `0 → Nyq → 0` を折り返す
「二重曲線」として描画されるバグ（BUG-009）が発生する。

各所で個別に書かれていたスライス・ソート・下限クリップを本モジュールに集約し、
dashboard / matplotlib デモ / visualize レポートで共通のロジックを使う。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .constants import BSDF_LOG_FLOOR_DEFAULT

if TYPE_CHECKING:
    import pandas as pd


def slice_phi0(
    u: np.ndarray,
    v: np.ndarray,
    bsdf: np.ndarray,
    *,
    mode: str = "positive",
    v_band_bins: int = 0,
    floor: float = BSDF_LOG_FLOOR_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """phi≈0 プロファイルを抽出して (u_sorted, bsdf_sorted) を返す。

    Args:
        u: 方向余弦 u の 2D グリッド（fftfreq 並び。shape=(N, N)）
        v: 方向余弦 v の 2D グリッド（u と同形状）
        bsdf: BSDF 2D グリッド（u と同形状）
        mode:
            'positive' → `u ∈ [0, 1]` の片側のみ返す（phi=0 方向・対称性利用）
            'signed'   → `u ∈ [-1, 1]` 全体を符号付きで返す（両側比較用）
        v_band_bins:
            0 → `v=0` の 1 列のみをスライスに使う（既定）
            >0 → `|v|` が N bins 以内の帯を平均してスペックル低減
        floor: 下限値。log スケール描画での `log(0)` 発散を防ぐ。

    Returns:
        u_sorted: 昇順ソートされた u 軸（1D）
        bsdf_sorted: 対応する BSDF 値（≥ floor、1D）

    Raises:
        ValueError: mode が 'positive' / 'signed' 以外の場合
    """
    if mode not in ("positive", "signed"):
        raise ValueError(f"mode must be 'positive' or 'signed', got {mode!r}")

    if v_band_bins > 0:
        v_axis = v[0, :]
        v_order = np.argsort(v_axis)
        center = int(np.argmin(np.abs(v_axis[v_order])))
        i0 = max(0, center - v_band_bins)
        i1 = min(len(v_order), center + v_band_bins + 1)
        cols = v_order[i0:i1]
        row = bsdf[:, cols].mean(axis=1)
    else:
        row = bsdf[:, 0]

    u_axis = u[:, 0]

    if mode == "positive":
        mask = (u_axis >= 0) & (u_axis <= 1.0)
    else:
        mask = (u_axis >= -1.0) & (u_axis <= 1.0)

    u_sel = u_axis[mask]
    b_sel = row[mask]
    order = np.argsort(u_sel)
    return u_sel[order], np.maximum(b_sel[order], floor)


def sort_and_floor(
    df_sub: "pd.DataFrame",
    *,
    floor: float = BSDF_LOG_FLOOR_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """条件フィルタ済み long-format BSDF DataFrame から (θ_s, bsdf) を取り出す。

    `theta_s_deg` 昇順に並べ、BSDF 値を `floor` でクリップして返す。log スケール
    プロット用の共通前処理を単一実装にまとめる（dashboard の実測プロファイル
    抽出と visualize レポートのインライン処理で重複していたロジック）。

    Args:
        df_sub: `theta_s_deg` / `bsdf` 列を持つ DataFrame（条件フィルタ済み）
        floor: BSDF 下限クリップ値（log(0) 回避）

    Returns:
        (theta_s_deg, bsdf) の 1D ndarray ペア。空入力時は `(array([]), array([]))`
    """
    if df_sub is None or len(df_sub) == 0:
        return np.array([]), np.array([])
    sorted_df = df_sub.sort_values("theta_s_deg")
    return (
        sorted_df["theta_s_deg"].to_numpy(),
        np.maximum(sorted_df["bsdf"].to_numpy(), floor),
    )
