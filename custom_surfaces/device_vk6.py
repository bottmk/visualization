"""Keyence VK-X シリーズ（VK-6000/VK-X1000 等）CSV ローダー（プラグイン）。

フォーマット仕様:
  - エンコード: Shift-JIS
  - ヘッダ 15 行（13 行メタデータ + 1 行空行 + 1 行「高さ」ラベル）
  - データ部: カンマ区切りクォート文字列、高さ値 [μm]
  - ピクセルサイズ・単位はヘッダの「XYキャリブレーション」「単位」行から自動取得

ヘッダ構造:
  行 1:  測定日時
  行 2:  機種（例: VK-X1000 Series）
  行 3:  ファイル種別
  行 4:  ファイル バージョン
  行 5:  測定データ名
  行 6:  倍率
  行 7:  XYキャリブレーション, <pixel_size>, <unit>  ← ピクセルサイズ自動取得
  行 8:  出力画像データ
  行 9:  横（列数）
  行 10: 縦（行数）
  行 11: 最小値
  行 12: 最大値
  行 13: 単位  ← 高さ単位自動取得
  行 14: （空行）
  行 15: 「高さ」ラベル
  行 16+: データ

サンプルファイル: sample_inputs/device_vk6_sample.csv

使用方法（config.yaml）::

    surface:
      model: 'DeviceVk6Surface'
      grid_size: 4096
      pixel_size_um: 0.25
      measured:
        path: 'sample_inputs/device_vk6_sample.csv'
        # source_pixel_size_um は省略可能（ヘッダから自動取得）
        leveling: true
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

from bsdf_sim.models.measured import MeasuredSurface

# ヘッダ行数（データ開始前にスキップする行数）
_HEADER_ROWS = 15

# ヘッダ内の各キー（Shift-JIS デコード後の文字列）
_KEY_XY_CALIB = "XYキャリブレーション"
_KEY_UNIT     = "単位"

# 単位 → μm 換算係数
_UNIT_FACTORS: dict[str, float] = {
    "μm": 1.0,
    "nm": 1e-3,
    "mm": 1e3,
    "m":  1e6,
}


def _parse_header(lines: list[str]) -> tuple[float, float]:
    """ヘッダ行からピクセルサイズ [μm] と高さ単位係数を返す。

    Args:
        lines: ヘッダ 15 行（文字列リスト、クォート含む生テキスト）

    Returns:
        (source_pixel_size_um, height_unit_factor)
    """
    source_pixel_size_um = None
    height_unit_factor = 1.0  # デフォルト: μm

    for line in lines:
        # CSV パース（クォート対応）
        try:
            row = next(csv.reader([line]))
        except StopIteration:
            continue

        if not row:
            continue

        key = row[0].strip()

        if key == _KEY_XY_CALIB and len(row) >= 3:
            try:
                pixel_val = float(row[1])
                pixel_unit = row[2].strip()
                factor = _UNIT_FACTORS.get(pixel_unit, 1.0)
                source_pixel_size_um = pixel_val * factor
            except (ValueError, IndexError):
                pass

        elif key == _KEY_UNIT and len(row) >= 2:
            unit = row[1].strip()
            height_unit_factor = _UNIT_FACTORS.get(unit, 1.0)

    if source_pixel_size_um is None:
        raise ValueError(
            f"ヘッダに '{_KEY_XY_CALIB}' 行が見つからなかった。"
            "VK-X シリーズの CSV ファイルか確認してください。"
        )

    return source_pixel_size_um, height_unit_factor


class DeviceVk6Surface(MeasuredSurface):
    """Keyence VK-X シリーズ CSV ローダー。

    MeasuredSurface を継承し、from_config() のみオーバーライドする。
    前処理（NaN 補間・レベリング・リサンプリング）は親クラスに委譲。
    ピクセルサイズ・高さ単位はヘッダから自動取得するため、
    config.yaml での source_pixel_size_um 指定は省略可能。
    """

    @classmethod
    def from_config(cls, config: dict) -> "DeviceVk6Surface":
        """config.yaml の設定辞書から DeviceVk6Surface を生成する。

        Args:
            config: config.yaml 全体の設定辞書

        Returns:
            DeviceVk6Surface インスタンス
        """
        surface_cfg = config.get("surface", {})
        measured_cfg = surface_cfg.get("measured", {})

        path = measured_cfg.get("path")
        if path is None:
            raise ValueError("surface.measured.path が設定されていない。")

        leveling   = bool(measured_cfg.get("leveling", True))
        grid_size  = int(surface_cfg.get("grid_size", 4096))
        pixel_size_um = float(surface_cfg.get("pixel_size_um", 0.25))

        # source_pixel_size_um が config にあれば優先、なければヘッダから自動取得
        override_pixel_size = measured_cfg.get("source_pixel_size_um")

        return cls.from_vk6_csv(
            path=path,
            source_pixel_size_um=float(override_pixel_size) if override_pixel_size is not None else None,
            grid_size=grid_size,
            pixel_size_um=pixel_size_um,
            leveling=leveling,
        )

    @classmethod
    def from_vk6_csv(
        cls,
        path: str | Path,
        source_pixel_size_um: float | None = None,
        **kwargs,
    ) -> "DeviceVk6Surface":
        """Keyence VK-X CSV を読み込む。

        Args:
            path: CSV ファイルパス
            source_pixel_size_um: ピクセルサイズ [μm]。
                省略時はヘッダの XYキャリブレーション行から自動取得。
            **kwargs: MeasuredSurface のその他引数（grid_size, pixel_size_um, leveling）

        Returns:
            DeviceVk6Surface インスタンス
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つからない: {path}")

        # Shift-JIS でファイル全体を読み込む
        raw_text = path.read_bytes().decode("shift-jis", errors="replace")
        lines = raw_text.splitlines()

        # ヘッダ解析
        header_lines = lines[:_HEADER_ROWS]
        auto_pixel_size, height_unit_factor = _parse_header(header_lines)

        if source_pixel_size_um is None:
            source_pixel_size_um = auto_pixel_size

        # データ部をパース（行 16 以降）
        data_lines = lines[_HEADER_ROWS:]
        rows = []
        for line in data_lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = [float(v) for v in next(csv.reader([line]))]
            except (ValueError, StopIteration):
                continue
            if row:
                rows.append(row)

        if not rows:
            raise ValueError(f"データ行が見つからなかった: {path}")

        data = np.array(rows, dtype=np.float64) * height_unit_factor  # → μm

        return cls(
            height_data=data,
            source_pixel_size_um=source_pixel_size_um,
            **kwargs,
        )
