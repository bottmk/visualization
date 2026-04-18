"""ギラツキ Cs の実測校正（calibration）フレームワーク。

本実装の L1/L3/L4/L5 が返す Cs は角度ビニング/窓付き FFT 由来の系統誤差により
SEMI D63/IDMS 実測値と比較して桁で乖離する（L1: 100–10000×、L5: 4–60×）。
相対順位は正しいが絶対値は信頼できないため、実測データでの校正を行う。

校正モード:

- **None**（既定）: 校正なし、raw Cs を返す
- **scale**: `cs_cal = scale * cs_sim`（比例定数 1 つ）
- **polynomial**: `cs_cal = a * cs_sim**b + c`（3 パラメータ冪乗 + バイアス）

config.yaml 例::

    metrics:
      sparkle:
        level: 'L5'
        color: 'G'
        calibration:
          mode: 'scale'
          scale: 0.05      # L5 (G) 緑点灯 × 0.05 ≈ 実測 Cs

詳細: `docs/sparkle_calibration.md` を参照。
"""

from __future__ import annotations

import numpy as np


def apply_calibration(cs_sim: float, calibration_config: dict | None) -> float:
    """シミュレーション Cs に校正を適用する。

    Args:
        cs_sim: シミュレーションの raw Cs 値
        calibration_config: config.yaml の `metrics.sparkle.calibration` 辞書。
            None または {} の場合は校正なしで cs_sim をそのまま返す。

    Returns:
        校正済み Cs 値

    Raises:
        ValueError: 不正な mode または必須パラメータ不足
    """
    if not calibration_config:
        return float(cs_sim)

    mode = str(calibration_config.get("mode", "")).lower()
    if mode in ("", "none", "null"):
        return float(cs_sim)

    if mode == "scale":
        if "scale" not in calibration_config:
            raise ValueError(
                "sparkle.calibration.mode='scale' には scale 値が必要です。"
            )
        scale = float(calibration_config["scale"])
        return float(scale * cs_sim)

    if mode == "polynomial":
        if "polynomial" not in calibration_config:
            raise ValueError(
                "sparkle.calibration.mode='polynomial' には "
                "polynomial=[a, b, c] が必要です。"
            )
        poly = calibration_config["polynomial"]
        if len(poly) != 3:
            raise ValueError(
                f"polynomial は [a, b, c] の 3 要素である必要があります。"
                f" 受け取った値={poly}"
            )
        a, b, c = (float(x) for x in poly)
        val = a * np.power(max(cs_sim, 0.0), b) + c
        return float(val)

    raise ValueError(
        f"sparkle.calibration.mode={mode!r} は未知です。"
        f" 'scale' / 'polynomial' / 'none' のいずれかを指定してください。"
    )


def fit_scale(cs_sim_arr: list[float], cs_measured_arr: list[float]) -> float:
    """最小二乗で scale 校正係数をフィットする。

    実測・シミュレーションのペアから `cs_meas ≈ k * cs_sim` の k をフィット。
    ゼロ除算回避のため cs_sim > 1e-12 のペアのみ使用する。

    Args:
        cs_sim_arr: シミュレーション Cs のリスト
        cs_measured_arr: 実測 Cs のリスト（同じ長さ）

    Returns:
        フィットされた scale 係数
    """
    sim = np.asarray(cs_sim_arr, dtype=np.float64)
    meas = np.asarray(cs_measured_arr, dtype=np.float64)
    if len(sim) != len(meas):
        raise ValueError(
            f"cs_sim_arr と cs_measured_arr の長さが異なる: "
            f"{len(sim)} vs {len(meas)}"
        )
    valid = sim > 1e-12
    if not np.any(valid):
        raise ValueError("有効なサンプル（cs_sim > 1e-12）が 0 件")
    # k = sum(meas * sim) / sum(sim^2) が最小二乗解
    k = float(np.sum(meas[valid] * sim[valid]) / np.sum(sim[valid] ** 2))
    return k


def fit_polynomial(
    cs_sim_arr: list[float], cs_measured_arr: list[float]
) -> tuple[float, float, float]:
    """log 空間での 2 点フィットで polynomial 校正係数 [a, b, c] を求める。

    `cs_meas = a * cs_sim**b + c` を最小二乗フィット。c は外れ値補正用のバイアス。
    `scipy.optimize.curve_fit` を使用する。

    Args:
        cs_sim_arr: シミュレーション Cs のリスト
        cs_measured_arr: 実測 Cs のリスト（同じ長さ、>= 3 サンプル推奨）

    Returns:
        (a, b, c) フィット結果
    """
    from scipy.optimize import curve_fit

    sim = np.asarray(cs_sim_arr, dtype=np.float64)
    meas = np.asarray(cs_measured_arr, dtype=np.float64)
    if len(sim) != len(meas):
        raise ValueError(
            f"cs_sim_arr と cs_measured_arr の長さが異なる: "
            f"{len(sim)} vs {len(meas)}"
        )
    if len(sim) < 3:
        raise ValueError(
            f"polynomial フィットには 3 サンプル以上が必要です。受け取った={len(sim)}"
        )

    def _model(x, a, b, c):
        return a * np.power(np.maximum(x, 1e-12), b) + c

    # 初期値: k=mean(meas/sim), b=1, c=0
    k0 = float(np.mean(meas / np.maximum(sim, 1e-12)))
    p0 = [k0, 1.0, 0.0]
    popt, _ = curve_fit(_model, sim, meas, p0=p0, maxfev=5000)
    return float(popt[0]), float(popt[1]), float(popt[2])
