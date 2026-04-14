"""表面形状指標の計算（Rq, Ra, Rz, Sdq）。"""

from __future__ import annotations

import numpy as np

from ..models.base import HeightMap


def compute_rq(hm: HeightMap) -> float:
    """RMS粗さ Rq/Sq [μm]。高さの二乗平均平方根。"""
    return float(np.sqrt(np.mean(hm.data**2)))


def compute_ra(hm: HeightMap) -> float:
    """算術平均粗さ Ra/Sa [μm]。高さの絶対値の平均。"""
    return float(np.mean(np.abs(hm.data)))


def compute_rz(hm: HeightMap) -> float:
    """最大断面高さ Rz/Sz [μm]。Peak-to-Valley 値。"""
    return float(np.max(hm.data) - np.min(hm.data))


def compute_sdq(hm: HeightMap) -> float:
    """RMS傾斜角 Sdq [rad]。局所的な傾き（スロープ）の二乗平均平方根。

    Sdq = sqrt(mean((dh/dx)² + (dh/dy)²))
    """
    dx = hm.pixel_size_um
    dhdx = np.gradient(hm.data, dx, axis=0)
    dhdy = np.gradient(hm.data, dx, axis=1)
    slope2 = dhdx**2 + dhdy**2
    return float(np.sqrt(np.mean(slope2)))


def compute_all_surface_metrics(hm: HeightMap) -> dict[str, float]:
    """すべての表面形状指標を一括計算する。

    Returns:
        {'rq_um': ..., 'ra_um': ..., 'rz_um': ..., 'sdq_rad': ...}
    """
    return {
        "rq_um":   compute_rq(hm),
        "ra_um":   compute_ra(hm),
        "rz_um":   compute_rz(hm),
        "sdq_rad": compute_sdq(hm),
    }
