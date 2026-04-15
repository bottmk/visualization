"""装置XYZ 固有フォーマットのローダー（プラグイン例）。

フォーマット仕様:
  - タブ区切り（\\t）
  - 先頭 5 行: コメントヘッダ（# で始まる）
  - データ部: 行 × 列 = グリッド行数 × グリッド列数（高さ値 [nm]）
  - 高さ単位: nm（内部で μm に変換）

サンプルファイル: sample_inputs/device_xyz_sample.csv

使用方法（config.yaml）::

    surface:
      model: 'DeviceXyzSurface'
      grid_size: 4096
      pixel_size_um: 0.25
      measured:
        path: 'sample_inputs/device_xyz_sample.csv'
        source_pixel_size_um: 0.5   # 計測ピクセルサイズ [μm]
        leveling: true              # 傾き除去（デフォルト: true）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bsdf_sim.models.measured import MeasuredSurface


class DeviceXyzSurface(MeasuredSurface):
    """装置XYZ 固有フォーマット CSVローダー。

    MeasuredSurface を継承し、from_config() のみオーバーライドする。
    前処理（NaN補間・レベリング・リサンプリング）は親クラスに委譲。
    """

    @classmethod
    def from_config(cls, config: dict) -> "DeviceXyzSurface":
        """config.yaml の設定辞書から DeviceXyzSurface を生成する。

        Args:
            config: config.yaml 全体の設定辞書

        Returns:
            DeviceXyzSurface インスタンス
        """
        surface_cfg = config.get("surface", {})
        measured_cfg = surface_cfg.get("measured", {})

        path = measured_cfg.get("path")
        if path is None:
            raise ValueError("surface.measured.path が設定されていない。")

        source_pixel_size_um = float(measured_cfg.get("source_pixel_size_um", 0.5))
        leveling = bool(measured_cfg.get("leveling", True))
        grid_size = int(surface_cfg.get("grid_size", 4096))
        pixel_size_um = float(surface_cfg.get("pixel_size_um", 0.25))

        return cls.from_device_xyz(
            path=path,
            source_pixel_size_um=source_pixel_size_um,
            grid_size=grid_size,
            pixel_size_um=pixel_size_um,
            leveling=leveling,
        )

    @classmethod
    def from_device_xyz(
        cls,
        path: str | Path,
        source_pixel_size_um: float,
        **kwargs,
    ) -> "DeviceXyzSurface":
        """装置XYZ 固有フォーマットの CSV を読み込む。

        Args:
            path: CSV ファイルパス
            source_pixel_size_um: 計測ピクセルサイズ [μm]
            **kwargs: MeasuredSurface のその他引数（grid_size, pixel_size_um, leveling）

        Returns:
            DeviceXyzSurface インスタンス
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つからない: {path}")

        # 先頭 5 行のコメントヘッダをスキップ、タブ区切り、nm 単位
        data = np.loadtxt(path, delimiter="\t", skiprows=5, dtype=np.float64)
        data_um = data * 1e-3  # nm → μm

        return cls(
            height_data=data_um,
            source_pixel_size_um=source_pixel_size_um,
            **kwargs,
        )
