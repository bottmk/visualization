"""実測データモデル（MeasuredSurface）。

実測高さデータを読み込み、欠損値補間・リサンプリング・レベリングを行う。
生ファイルフォーマットは装置ごとに異なるため、ローダーはサブクラスで実装する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import zoom

from .base import BaseSurfaceModel, HeightMap


class MeasuredSurface(BaseSurfaceModel):
    """実測高さデータ表面モデル。

    外部から NumPy 配列として高さデータを受け取り、
    欠損値補間・指定グリッドサイズへのリサンプリング・レベリングを行う。

    Args:
        height_data: 実測高さ配列 [μm]（任意の形状）
        source_pixel_size_um: 実測データのピクセルサイズ [μm]
        grid_size: 本計算用グリッドサイズ（デフォルト: 4096）
        pixel_size_um: 出力ピクセルサイズ [μm]（デフォルト: 0.25μm）
        leveling: True の場合、傾き・うねり成分を除去する（デフォルト: True）
    """

    def __init__(
        self,
        height_data: np.ndarray,
        source_pixel_size_um: float,
        grid_size: int = 4096,
        pixel_size_um: float = 0.25,
        leveling: bool = True,
    ) -> None:
        super().__init__(grid_size=grid_size, pixel_size_um=pixel_size_um)
        self.source_pixel_size_um = source_pixel_size_um
        self.leveling = leveling
        self._processed = self._preprocess(height_data)

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """欠損値補間・レベリングを行い前処理済み配列を返す。"""
        data = data.astype(np.float64)

        # 欠損値（NaN）補間
        if np.any(np.isnan(data)):
            data = self._interpolate_nan(data)

        # レベリング（最小二乗平面フィット → 除去）
        if self.leveling:
            data = self._level(data)

        return data.astype(np.float32)

    @staticmethod
    def _interpolate_nan(data: np.ndarray) -> np.ndarray:
        """NaN を周辺の有効値で補間する。"""
        h, w = data.shape
        yy, xx = np.mgrid[0:h, 0:w]
        valid = ~np.isnan(data)
        if not np.any(valid):
            return np.zeros_like(data)
        interpolated = griddata(
            points=np.column_stack([xx[valid], yy[valid]]),
            values=data[valid],
            xi=(xx, yy),
            method="linear",
            fill_value=0.0,
        )
        return interpolated

    @staticmethod
    def _level(data: np.ndarray) -> np.ndarray:
        """最小二乗平面フィットにより傾き・うねり成分を除去する。"""
        h, w = data.shape
        yy, xx = np.mgrid[0:h, 0:w]
        # 平面 z = ax + by + c の係数を最小二乗フィット
        A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(h * w)])
        z = data.ravel()
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        plane = (coeffs[0] * xx + coeffs[1] * yy + coeffs[2]).astype(np.float32)
        return data - plane

    def _generate(self, grid_size: int, pixel_size_um: float) -> np.ndarray:
        """前処理済みデータを指定グリッドサイズにリサンプリングして返す。"""
        current_h, current_w = self._processed.shape

        # 目標ピクセルサイズから zoom 倍率を計算
        # source_pixel_size_um → pixel_size_um への変換
        zoom_factor = (self.source_pixel_size_um / pixel_size_um)

        # まず元データを目標ピクセルサイズにリサンプリング
        resampled = zoom(self._processed, zoom_factor, order=3)

        # 次に目標グリッドサイズにクロップまたはパディング
        rh, rw = resampled.shape
        if rh >= grid_size and rw >= grid_size:
            # 中央からクロップ
            cy, cx = rh // 2, rw // 2
            half = grid_size // 2
            result = resampled[cy - half:cy - half + grid_size, cx - half:cx - half + grid_size]
        else:
            # 不足分をゼロパディング
            result = np.zeros((grid_size, grid_size), dtype=np.float32)
            min_h = min(rh, grid_size)
            min_w = min(rw, grid_size)
            result[:min_h, :min_w] = resampled[:min_h, :min_w]

        return result[:grid_size, :grid_size].astype(np.float32)

    @classmethod
    def from_numpy(
        cls,
        data: np.ndarray,
        source_pixel_size_um: float,
        **kwargs,
    ) -> "MeasuredSurface":
        """NumPy 配列から MeasuredSurface を生成する。

        Args:
            data: 高さ配列 [μm]
            source_pixel_size_um: 元データのピクセルサイズ [μm]
            **kwargs: BaseSurfaceModel のその他引数
        """
        return cls(
            height_data=data,
            source_pixel_size_um=source_pixel_size_um,
            **kwargs,
        )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        source_pixel_size_um: float,
        height_unit: str = "um",
        skiprows: int = 0,
        **kwargs,
    ) -> "MeasuredSurface":
        """CSV ファイルから MeasuredSurface を生成する（汎用ローダー）。

        装置固有のヘッダ情報がある場合は skiprows で行数を指定する。
        装置ごとの詳細ローダーは実ファイル提供後に追加予定。

        Args:
            path: CSV ファイルパス
            source_pixel_size_um: 元データのピクセルサイズ [μm]
            height_unit: 高さの単位（'um' / 'nm' / 'm'）
            skiprows: スキップするヘッダ行数
            **kwargs: BaseSurfaceModel のその他引数
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つからない: {path}")

        data = np.loadtxt(path, delimiter=",", skiprows=skiprows)

        # 単位変換 → μm
        unit_factors = {"m": 1e6, "um": 1.0, "nm": 1e-3}
        if height_unit not in unit_factors:
            raise ValueError(f"height_unit は 'm' / 'um' / 'nm' のいずれかでなければならない。値={height_unit}")
        data = data * unit_factors[height_unit]

        return cls(
            height_data=data,
            source_pixel_size_um=source_pixel_size_um,
            **kwargs,
        )
