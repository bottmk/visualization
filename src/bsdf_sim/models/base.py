"""基底クラス：HeightMap dataclass と BaseSurfaceModel。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class HeightMap:
    """高さマップとスケール情報を一体化したデータクラス。

    Attributes:
        data: 高さ配列 [μm]、shape: (N, N)
        pixel_size_um: ピクセルサイズ [μm]
    """

    data: np.ndarray
    pixel_size_um: float

    def __post_init__(self) -> None:
        if self.data.ndim != 2 or self.data.shape[0] != self.data.shape[1]:
            raise ValueError(f"data は正方形の2次元配列でなければならない。shape={self.data.shape}")
        if self.pixel_size_um <= 0:
            raise ValueError(f"pixel_size_um は正の値でなければならない。値={self.pixel_size_um}")

    @property
    def grid_size(self) -> int:
        """グリッドサイズ N。"""
        return self.data.shape[0]

    @property
    def physical_size_um(self) -> float:
        """物理サイズ [μm] = grid_size × pixel_size_um。"""
        return self.grid_size * self.pixel_size_um

    @property
    def rq_um(self) -> float:
        """RMS粗さ [μm]。"""
        return float(np.sqrt(np.mean(self.data**2)))

    def resample(self, target_grid_size: int) -> "HeightMap":
        """指定グリッドサイズにリサンプリングした新しい HeightMap を返す。

        pixel_size_um は変更しない（物理サイズが変わる）。
        """
        from scipy.ndimage import zoom

        factor = target_grid_size / self.grid_size
        resampled = zoom(self.data, factor, order=3)
        # zoom で端数が出る場合があるためトリミング
        resampled = resampled[:target_grid_size, :target_grid_size]
        return HeightMap(data=resampled, pixel_size_um=self.pixel_size_um)


class BaseSurfaceModel(ABC):
    """すべての表面形状モデルの基底クラス。

    物理単位は μm に統一する。
    本計算用の get_height_map() とプレビュー用の get_preview_height_map() を提供する。
    """

    def __init__(
        self,
        grid_size: int = 4096,
        pixel_size_um: float = 0.25,
    ) -> None:
        """
        Args:
            grid_size: 本計算用グリッドサイズ（デフォルト: 4096）
            pixel_size_um: ピクセルサイズ [μm]（デフォルト: 0.25μm）
                物理サイズ = grid_size × pixel_size_um（デフォルト: 1024μm）
        """
        if grid_size <= 0:
            raise ValueError(f"grid_size は正の整数でなければならない。値={grid_size}")
        if pixel_size_um <= 0:
            raise ValueError(f"pixel_size_um は正の値でなければならない。値={pixel_size_um}")
        self.grid_size = grid_size
        self.pixel_size_um = pixel_size_um

    @property
    def physical_size_um(self) -> float:
        """物理サイズ [μm]。"""
        return self.grid_size * self.pixel_size_um

    @abstractmethod
    def _generate(self, grid_size: int, pixel_size_um: float) -> np.ndarray:
        """指定サイズの高さ配列 [μm] を生成する内部メソッド。

        サブクラスはこのメソッドを実装する。

        Args:
            grid_size: 生成するグリッドサイズ
            pixel_size_um: ピクセルサイズ [μm]

        Returns:
            shape (grid_size, grid_size) の高さ配列 [μm]
        """

    def get_height_map(self) -> HeightMap:
        """本計算用の高さマップを返す。

        grid_size=4096, pixel_size_um=0.25（デフォルト）で生成する。
        物理サイズ: 1024μm、散乱角上限: 90°。

        Returns:
            HeightMap: 本計算用高さマップ
        """
        data = self._generate(self.grid_size, self.pixel_size_um)
        return HeightMap(data=data, pixel_size_um=self.pixel_size_um)

    def get_preview_height_map(
        self,
        mode: Literal["reduced_area", "reduced_resolution"] = "reduced_area",
        preview_grid_size: int = 512,
    ) -> HeightMap:
        """DynamicMap 用プレビュー高さマップを返す。

        Args:
            mode:
                'reduced_area'（アプローチA）:
                    pixel_size_um を固定し grid_size を縮小する。
                    物理サイズは縮小されるが（例: 128μm）、散乱角上限は 90° まで維持。
                    ヘイズ・広角散乱の定性確認に適する。

                'reduced_resolution'（アプローチB）:
                    physical_size_um を固定し pixel_size_um を拡大する。
                    pixel_size_um = physical_size_um / preview_grid_size（例: 2.0μm）
                    散乱角の上限が制限される（λ=0.55μm の場合 約 7.9°）。
                    BSDFピーク位置・形状の確認に適する。ヘイズ計算には使用不可。

            preview_grid_size: プレビュー用グリッドサイズ（デフォルト: 512）

        Returns:
            HeightMap: プレビュー用高さマップ
        """
        if mode == "reduced_area":
            # pixel_size_um 固定、grid_size 縮小
            data = self._generate(preview_grid_size, self.pixel_size_um)
            return HeightMap(data=data, pixel_size_um=self.pixel_size_um)

        elif mode == "reduced_resolution":
            # physical_size_um 固定、pixel_size_um 拡大
            preview_pixel_size = self.physical_size_um / preview_grid_size
            data = self._generate(preview_grid_size, preview_pixel_size)
            return HeightMap(data=data, pixel_size_um=preview_pixel_size)

        else:
            raise ValueError(f"mode は 'reduced_area' または 'reduced_resolution' でなければならない。値={mode}")
