"""フレネル係数の計算（r_s, r_p, t_s, t_p）。

スネルの法則から透過角を導出し、S/P 偏光のフレネル反射・透過係数を計算する。
物理単位は μm に統一。角度は度数法 [deg] で入力し、内部で rad に変換する。
"""

from __future__ import annotations

import numpy as np


def snell_angle(theta_i_deg: float, n1: float, n2: float) -> float:
    """スネルの法則から透過角を計算する。

    n1 * sin(theta_i) = n2 * sin(theta_t)

    Args:
        theta_i_deg: 入射角 [deg]
        n1: 入射側媒質の屈折率
        n2: 透過側媒質の屈折率

    Returns:
        透過角 [deg]

    Raises:
        ValueError: 全反射条件（sin(theta_t) > 1）の場合
    """
    theta_i_rad = np.deg2rad(theta_i_deg)
    sin_t = n1 * np.sin(theta_i_rad) / n2
    if abs(sin_t) > 1.0:
        raise ValueError(
            f"全反射条件。theta_i={theta_i_deg}°, n1={n1}, n2={n2} では透過しない。"
        )
    theta_t_rad = np.arcsin(sin_t)
    return float(np.rad2deg(theta_t_rad))


def fresnel_rs(theta_i_deg: float, n1: float, n2: float) -> complex:
    """S偏光（TE）フレネル反射係数 r_s を計算する。

    r_s = (n1*cos(theta_i) - n2*cos(theta_t)) / (n1*cos(theta_i) + n2*cos(theta_t))

    Args:
        theta_i_deg: 入射角 [deg]
        n1: 入射側媒質の屈折率
        n2: 透過側媒質の屈折率

    Returns:
        複素反射係数 r_s
    """
    theta_i_rad = np.deg2rad(theta_i_deg)
    cos_i = np.cos(theta_i_rad)
    sin_t = n1 * np.sin(theta_i_rad) / n2
    # 全反射の場合は複素 cos_t
    cos_t = np.sqrt(np.complex128(1.0 - sin_t**2))
    return complex((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t))


def fresnel_rp(theta_i_deg: float, n1: float, n2: float) -> complex:
    """P偏光（TM）フレネル反射係数 r_p を計算する。

    r_p = (n1*cos(theta_t) - n2*cos(theta_i)) / (n1*cos(theta_t) + n2*cos(theta_i))

    Born-Wolf 規約に従い、法線入射では r_p = r_s = (n1-n2)/(n1+n2) となる。

    Args:
        theta_i_deg: 入射角 [deg]
        n1: 入射側媒質の屈折率
        n2: 透過側媒質の屈折率

    Returns:
        複素反射係数 r_p
    """
    theta_i_rad = np.deg2rad(theta_i_deg)
    cos_i = np.cos(theta_i_rad)
    sin_t = n1 * np.sin(theta_i_rad) / n2
    cos_t = np.sqrt(np.complex128(1.0 - sin_t**2))
    return complex((n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i))


def fresnel_ts(theta_i_deg: float, n1: float, n2: float) -> complex:
    """S偏光（TE）フレネル透過係数 t_s を計算する。

    t_s = 2*n1*cos(theta_i) / (n1*cos(theta_i) + n2*cos(theta_t))

    Args:
        theta_i_deg: 入射角 [deg]
        n1: 入射側媒質の屈折率
        n2: 透過側媒質の屈折率

    Returns:
        複素透過係数 t_s
    """
    theta_i_rad = np.deg2rad(theta_i_deg)
    cos_i = np.cos(theta_i_rad)
    sin_t = n1 * np.sin(theta_i_rad) / n2
    cos_t = np.sqrt(np.complex128(1.0 - sin_t**2))
    return complex(2 * n1 * cos_i / (n1 * cos_i + n2 * cos_t))


def fresnel_tp(theta_i_deg: float, n1: float, n2: float) -> complex:
    """P偏光（TM）フレネル透過係数 t_p を計算する。

    t_p = 2*n1*cos(theta_i) / (n2*cos(theta_i) + n1*cos(theta_t))

    Args:
        theta_i_deg: 入射角 [deg]
        n1: 入射側媒質の屈折率
        n2: 透過側媒質の屈折率

    Returns:
        複素透過係数 t_p
    """
    theta_i_rad = np.deg2rad(theta_i_deg)
    cos_i = np.cos(theta_i_rad)
    sin_t = n1 * np.sin(theta_i_rad) / n2
    cos_t = np.sqrt(np.complex128(1.0 - sin_t**2))
    return complex(2 * n1 * cos_i / (n2 * cos_i + n1 * cos_t))


def fresnel_all(
    theta_i_deg: float, n1: float, n2: float
) -> dict[str, complex]:
    """すべてのフレネル係数を一括計算する。

    Args:
        theta_i_deg: 入射角 [deg]
        n1: 入射側媒質の屈折率
        n2: 透過側媒質の屈折率

    Returns:
        {'rs': r_s, 'rp': r_p, 'ts': t_s, 'tp': t_p} の辞書
    """
    return {
        "rs": fresnel_rs(theta_i_deg, n1, n2),
        "rp": fresnel_rp(theta_i_deg, n1, n2),
        "ts": fresnel_ts(theta_i_deg, n1, n2),
        "tp": fresnel_tp(theta_i_deg, n1, n2),
    }
