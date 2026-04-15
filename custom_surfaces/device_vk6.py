"""Keyence VK-X シリーズ（VK-6000/VK-X1000 等）CSV ローダー（プラグイン）。

フォーマット仕様:
  - エンコード: Shift-JIS
  - ヘッダ 15 行（13 行メタデータ + 1 行空行 + 1 行「高さ」ラベル）
  - データ部: カンマ区切りクォート文字列、高さ値 [μm]
  - ピクセルサイズ・高さ単位・グリッドサイズはヘッダから自動取得

ヘッダ構造:
  行 1:  測定日時
  行 2:  機種（例: VK-X1000 Series）
  行 3:  ファイル種別
  行 4:  ファイル バージョン
  行 5:  測定データ名
  行 6:  倍率
  行 7:  XYキャリブレーション, <pixel_size>, <unit>  ← ピクセルサイズ自動取得
  行 8:  出力画像データ
  行 9:  横（列数）                                   ← データ幅自動取得
  行 10: 縦（行数）                                   ← データ高さ自動取得
  行 11: 最小値
  行 12: 最大値
  行 13: 単位                                         ← 高さ単位自動取得
  行 14: （空行）
  行 15: 「高さ」ラベル
  行 16+: データ

サンプルファイル: sample_inputs/device_vk6_sample.csv

使用方法（config.yaml）::

    surface:
      model: 'DeviceVk6Surface'
      # grid_size: 省略時はヘッダから自動算出（2^n 切り捨て）
      # pixel_size_um: 省略時はヘッダの XYキャリブレーション値を使用（zoom なし）
      measured:
        path: 'sample_inputs/device_vk6_sample.csv'
        padding: 'tile'   # 'zeros'/'tile'/'reflect'/'smooth_tile'（省略時: 'tile'）
        leveling: true

    # 手動指定の例:
    surface:
      model: 'DeviceVk6Surface'
      grid_size: 2048          # 明示時はその値を使用
      pixel_size_um: 0.25      # 明示時はズームして変換
      measured:
        path: '...'
        padding: 'smooth_tile'
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from bsdf_sim.models.measured import MeasuredSurface

logger = logging.getLogger(__name__)

# ── ヘッダ定数 ────────────────────────────────────────────────────────────────

_HEADER_ROWS = 15

_KEY_XY_CALIB = "XYキャリブレーション"
_KEY_UNIT     = "単位"
_KEY_WIDTH    = "横"
_KEY_HEIGHT   = "縦"

_UNIT_FACTORS: dict[str, float] = {
    "μm": 1.0,
    "nm": 1e-3,
    "mm": 1e3,
    "m":  1e6,
}


# ── ヘッダ解析 ────────────────────────────────────────────────────────────────

def _parse_header(lines: list[str]) -> tuple[float, float, int, int]:
    """ヘッダ行からピクセルサイズ・単位係数・幅・高さを返す。

    Args:
        lines: ヘッダ 15 行（生テキスト、クォート含む）

    Returns:
        (source_pixel_size_um, height_unit_factor, file_width, file_height)
    """
    source_pixel_size_um: float | None = None
    height_unit_factor: float = 1.0
    file_width: int | None = None
    file_height: int | None = None

    for line in lines:
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

        elif key == _KEY_WIDTH and len(row) >= 2:
            try:
                file_width = int(row[1])
            except ValueError:
                pass

        elif key == _KEY_HEIGHT and len(row) >= 2:
            try:
                file_height = int(row[1])
            except ValueError:
                pass

    if source_pixel_size_um is None:
        raise ValueError(
            f"ヘッダに '{_KEY_XY_CALIB}' 行が見つからなかった。"
            "VK-X シリーズの CSV ファイルか確認してください。"
        )
    if file_width is None or file_height is None:
        raise ValueError(
            f"ヘッダに '{_KEY_WIDTH}' または '{_KEY_HEIGHT}' 行が見つからなかった。"
        )

    return source_pixel_size_um, height_unit_factor, file_width, file_height


def _floor_pow2(n: int) -> int:
    """n 以下で最大の 2 の冪乗を返す。"""
    return 2 ** int(np.floor(np.log2(n)))


# ── DeviceVk6Surface ─────────────────────────────────────────────────────────

class DeviceVk6Surface(MeasuredSurface):
    """Keyence VK-X シリーズ CSV ローダー。

    MeasuredSurface を継承し、from_config() のみオーバーライドする。
    前処理（NaN 補間・レベリング・リサンプリング・パディング）は親クラスに委譲。

    pixel_size_um と grid_size は config.yaml で省略可能:
      - 省略時はヘッダから自動算出
      - 明示時はその値を優先
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

        leveling = bool(measured_cfg.get("leveling", True))
        padding  = measured_cfg.get("padding", "tile")

        # None → from_vk6_csv 内でヘッダから自動取得
        pixel_size_raw        = surface_cfg.get("pixel_size_um")
        grid_size_raw         = surface_cfg.get("grid_size")
        source_pixel_size_raw = measured_cfg.get("source_pixel_size_um")

        return cls.from_vk6_csv(
            path=path,
            source_pixel_size_um=(
                float(source_pixel_size_raw) if source_pixel_size_raw is not None else None
            ),
            pixel_size_um=(
                float(pixel_size_raw) if pixel_size_raw is not None else None
            ),
            grid_size=(
                int(grid_size_raw) if grid_size_raw is not None else None
            ),
            leveling=leveling,
            padding=padding,
        )

    @classmethod
    def from_vk6_csv(
        cls,
        path: str | Path,
        source_pixel_size_um: float | None = None,
        pixel_size_um: float | None = None,
        grid_size: int | None = None,
        **kwargs,
    ) -> "DeviceVk6Surface":
        """Keyence VK-X CSV を読み込む。

        Args:
            path: CSV ファイルパス
            source_pixel_size_um: 計測ピクセルサイズ [μm]。
                省略時はヘッダの XYキャリブレーション行から自動取得。
            pixel_size_um: 出力ピクセルサイズ [μm]。
                省略時はヘッダ値をそのまま使用（zoom なし）。
            grid_size: 出力グリッドサイズ。
                省略時はデータサイズ（zoom 後）の短辺を 2^n 切り捨てで自動設定。
            **kwargs: MeasuredSurface のその他引数（leveling, padding）

        Returns:
            DeviceVk6Surface インスタンス
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つからない: {path}")

        # Shift-JIS でファイル全体を読み込む
        raw_text = path.read_bytes().decode("shift-jis", errors="replace")
        lines = raw_text.splitlines()

        # ── ヘッダ解析 ──
        header_lines = lines[:_HEADER_ROWS]
        auto_pixel_size, height_unit_factor, file_width, file_height = _parse_header(
            header_lines
        )

        if source_pixel_size_um is None:
            source_pixel_size_um = auto_pixel_size

        # pixel_size_um が省略された場合はヘッダ値をそのまま使用（zoom_factor=1）
        if pixel_size_um is None:
            pixel_size_um = auto_pixel_size
            logger.info(
                f"pixel_size_um を自動設定: {pixel_size_um:.4f} μm"
                f"（ヘッダ XYキャリブレーション値、zoom なし）"
            )

        # grid_size が省略された場合は zoom 後データサイズの短辺を 2^n 切り捨て
        if grid_size is None:
            zoom_factor = source_pixel_size_um / pixel_size_um
            data_size_after_zoom = int(min(file_width, file_height) * zoom_factor)
            if data_size_after_zoom < 2:
                raise ValueError(
                    f"zoom 後データサイズ {data_size_after_zoom} px が小さすぎる。"
                    "pixel_size_um または grid_size を明示してください。"
                )
            grid_size = _floor_pow2(data_size_after_zoom)
            logger.info(
                f"grid_size を自動設定: ファイル {file_width}×{file_height} px"
                f" → zoom後 {data_size_after_zoom} px"
                f" → {grid_size} (2^n 切り捨て、物理サイズ"
                f" {grid_size * pixel_size_um:.1f} μm)"
            )

        # ── データ部パース（行 16 以降）──
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
            pixel_size_um=pixel_size_um,
            grid_size=grid_size,
            **kwargs,
        )
