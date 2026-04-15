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


# ── パディング方式 ─────────────────────────────────────────────────────────────

VALID_PADDINGS: tuple[str, ...] = ("zeros", "tile", "reflect", "smooth_tile")


def _pad_zeros(data: np.ndarray, grid_size: int) -> np.ndarray:
    """中央ゼロパディング。

    データを中央に配置し、周囲を 0（平坦面）で埋める。
    レベリング後は平均≒0 のため、データ端で段差が生じる場合がある。
    """
    data_size = data.shape[0]
    result = np.zeros((grid_size, grid_size), dtype=np.float32)
    offset = (grid_size - data_size) // 2
    result[offset:offset + data_size, offset:offset + data_size] = data
    return result


def _pad_tile(data: np.ndarray, grid_size: int) -> np.ndarray:
    """タイリング（周期的繰り返し）。

    データを周期的に繰り返して grid_size × grid_size を埋める。
    FFT の周期境界仮定と整合し、統計的に一様な表面（AG フィルム等）に適する。
    データの端値が異なる場合は境界で不連続が生じる。
    """
    data_size = data.shape[0]
    n = (grid_size + data_size - 1) // data_size + 1
    tiled = np.tile(data, (n, n))
    return tiled[:grid_size, :grid_size].astype(np.float32)


def _pad_reflect(data: np.ndarray, grid_size: int) -> np.ndarray:
    """ミラー反転タイリング。

    データを鏡像で反転しながら敷き詰める。境界で C0 連続（段差なし）。
    ただし人工的な対称性が生まれ、スペクトルに偽ピークが現れる場合がある。
    任意のパディング量に対応（numpy の pad より大きなパディングも可能）。
    """
    data_size = data.shape[0]
    n = (grid_size + data_size - 1) // data_size + 1

    rows = []
    for i in range(n):
        tile_row = data if i % 2 == 0 else data[::-1, :]
        cols = []
        for j in range(n):
            tile = tile_row if j % 2 == 0 else tile_row[:, ::-1]
            cols.append(tile)
        rows.append(np.concatenate(cols, axis=1))
    big = np.concatenate(rows, axis=0)
    return big[:grid_size, :grid_size].astype(np.float32)


def _pad_smooth_tile(data: np.ndarray, grid_size: int, blend_ratio: float = 0.1) -> np.ndarray:
    """タイリング＋FFT 周期境界スムーズブレンド。

    タイリングした後、FFT の周期境界（grid_size の折り返し点）で
    コサイン窓によるクロスフェードを適用する。
    右端は左端へ、下端は上端へ滑らかに接続されるため、
    境界で C∞ 連続となりスペクトルリークを最小化する。

    Args:
        data: 正方形の高さ配列（data_size ≤ grid_size を前提）
        grid_size: 目標グリッドサイズ
        blend_ratio: ブレンド幅の割合（grid_size 基準、デフォルト 10%）
                     ただし data_size の 1/4 を上限とする。
    """
    data_size = data.shape[0]

    # タイリングで grid_size を埋める
    n = (grid_size + data_size - 1) // data_size + 1
    tiled = np.tile(data, (n, n))
    result = tiled[:grid_size, :grid_size].copy().astype(np.float64)

    # ブレンドサイズ: grid_size の blend_ratio 以下かつ data_size の 1/4 以下
    B = min(max(4, int(grid_size * blend_ratio)), data_size // 4)

    # コサイン重み: i=0 で w≈0（タイル値保持）、i=B-1 で w≈1（折り返し値へ移行）
    i_arr = np.arange(B)
    alpha = (i_arr + 1) / (B + 1)
    w = 0.5 * (1.0 - np.cos(np.pi * alpha))  # 0 → 1

    # 列方向ブレンド: 右端 B 列 → 左端 B 列へ折り返し接続
    col_j = grid_size - B + i_arr          # ブレンド対象列
    target_cols = result[:, :B].copy()     # 折り返し後の値（グリッド先頭）
    result[:, col_j] = (
        (1.0 - w)[np.newaxis, :] * result[:, col_j]
        + w[np.newaxis, :] * target_cols
    )

    # 行方向ブレンド: 下端 B 行 → 上端 B 行へ折り返し接続
    row_i = grid_size - B + i_arr
    target_rows = result[:B, :].copy()
    result[row_i, :] = (
        (1.0 - w)[:, np.newaxis] * result[row_i, :]
        + w[:, np.newaxis] * target_rows
    )

    return result.astype(np.float32)


def _apply_padding(data: np.ndarray, grid_size: int, mode: str) -> np.ndarray:
    """指定方式で data を grid_size × grid_size にパディングする。

    Args:
        data: 正方形の高さ配列（data_size ≤ grid_size を前提）
        grid_size: 目標グリッドサイズ
        mode: パディング方式（VALID_PADDINGS のいずれか）

    Returns:
        shape (grid_size, grid_size) の高さ配列 [μm]
    """
    if mode == "zeros":
        return _pad_zeros(data, grid_size)
    if mode == "tile":
        return _pad_tile(data, grid_size)
    if mode == "reflect":
        return _pad_reflect(data, grid_size)
    if mode == "smooth_tile":
        return _pad_smooth_tile(data, grid_size)
    raise ValueError(
        f"padding は {VALID_PADDINGS} のいずれかでなければならない。値={mode!r}"
    )


# ── MeasuredSurface ──────────────────────────────────────────────────────────

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
        padding: データ < grid_size 時のパディング方式（デフォルト: 'tile'）
            'zeros'       — 中央ゼロパディング
            'tile'        — タイリング（周期的繰り返し）
            'reflect'     — ミラー反転タイリング（C0 連続）
            'smooth_tile' — タイリング＋FFT 周期境界コサインブレンド（C∞ 連続）
    """

    def __init__(
        self,
        height_data: np.ndarray,
        source_pixel_size_um: float,
        grid_size: int = 4096,
        pixel_size_um: float = 0.25,
        leveling: bool = True,
        padding: str = "tile",
    ) -> None:
        super().__init__(grid_size=grid_size, pixel_size_um=pixel_size_um)
        if padding not in VALID_PADDINGS:
            raise ValueError(
                f"padding は {VALID_PADDINGS} のいずれかでなければならない。値={padding!r}"
            )
        self.source_pixel_size_um = source_pixel_size_um
        self.leveling = leveling
        self.padding = padding
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
        A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(h * w)])
        z = data.ravel()
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        plane = (coeffs[0] * xx + coeffs[1] * yy + coeffs[2]).astype(np.float32)
        return data - plane

    def _generate(self, grid_size: int, pixel_size_um: float) -> np.ndarray:
        """ズーム → 正方形中央クロップ → パディング/クロップ。

        処理フロー:
            1. source_pixel_size_um → pixel_size_um へズーム（zoom_factor = src/tgt）
            2. 非正方形データを短辺基準で中央クロップ（HeightMap の正方形制約）
            3. データ > grid_size → 中央クロップ
               データ < grid_size → self.padding 方式でパディング
        """
        # 1. ピクセルサイズ変換
        zoom_factor = self.source_pixel_size_um / pixel_size_um
        if abs(zoom_factor - 1.0) < 1e-9:
            resampled = self._processed.copy()
        else:
            resampled = zoom(self._processed, zoom_factor, order=3)

        # 2. 正方形クロップ（HeightMap の正方形制約）
        rh, rw = resampled.shape
        sq = min(rh, rw)
        r0 = (rh - sq) // 2
        c0 = (rw - sq) // 2
        square = resampled[r0:r0 + sq, c0:c0 + sq]

        # 3. grid_size へのクロップまたはパディング
        if sq >= grid_size:
            s = (sq - grid_size) // 2
            return square[s:s + grid_size, s:s + grid_size].astype(np.float32)
        else:
            return _apply_padding(square, grid_size, self.padding)

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
            **kwargs: MeasuredSurface のその他引数（grid_size, pixel_size_um, leveling, padding）
        """
        return cls(
            height_data=data,
            source_pixel_size_um=source_pixel_size_um,
            **kwargs,
        )

    @classmethod
    def from_config(cls, config: dict) -> "MeasuredSurface":
        """config.yaml の設定辞書から MeasuredSurface を生成する。

        設定例::

            surface:
              model: 'MeasuredSurface'
              grid_size: 4096
              pixel_size_um: 0.25
              measured:
                path: 'data/surface.csv'
                source_pixel_size_um: 1.0
                height_unit: 'nm'
                skiprows: 3
                leveling: true
                padding: 'tile'   # 'zeros'/'tile'/'reflect'/'smooth_tile'

        Args:
            config: config.yaml 全体の設定辞書

        Returns:
            MeasuredSurface インスタンス
        """
        surface_cfg = config.get("surface", {})
        measured_cfg = surface_cfg.get("measured", {})

        path = measured_cfg.get("path")
        if path is None:
            raise ValueError("surface.measured.path が設定されていない。")

        source_pixel_size_um = float(measured_cfg.get("source_pixel_size_um", 1.0))
        height_unit = measured_cfg.get("height_unit", "um")
        skiprows = int(measured_cfg.get("skiprows", 0))
        leveling = bool(measured_cfg.get("leveling", True))
        padding = measured_cfg.get("padding", "tile")

        return cls.from_csv(
            path=path,
            source_pixel_size_um=source_pixel_size_um,
            height_unit=height_unit,
            skiprows=skiprows,
            grid_size=int(surface_cfg.get("grid_size", 4096)),
            pixel_size_um=float(surface_cfg.get("pixel_size_um", 0.25)),
            leveling=leveling,
            padding=padding,
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
        装置ごとの詳細ローダーは custom_surfaces/ サブクラスで実装する。

        Args:
            path: CSV ファイルパス
            source_pixel_size_um: 元データのピクセルサイズ [μm]
            height_unit: 高さの単位（'um' / 'nm' / 'm'）
            skiprows: スキップするヘッダ行数
            **kwargs: MeasuredSurface のその他引数（grid_size, pixel_size_um, leveling, padding）
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
