"""BSDF 実測ファイル読み込みプラグインシステム。

custom_bsdf_readers/ フォルダに置いた .py ファイルを動的ロードし、
拡張子やマジックバイトに基づいてリーダーを自動選択する。

使用例::

    from bsdf_sim.io.bsdf_reader import load_bsdf_readers, read_bsdf_file

    load_bsdf_readers()               # custom_bsdf_readers/ を自動探索
    dfs = read_bsdf_file("data.bsdf") # 適切なリーダーを自動選択
    df = pd.concat(dfs)               # 全ブロックを結合

プラグイン追加方法:
    1. custom_bsdf_readers/<name>.py に BaseBsdfFileReader サブクラスを実装
    2. load_bsdf_readers() を呼ぶだけで自動登録される
"""

from __future__ import annotations

import importlib.util
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── レジストリ ─────────────────────────────────────────────────────────────────

_READER_REGISTRY: dict[str, type[BaseBsdfFileReader]] = {}


# ── 基底クラス ─────────────────────────────────────────────────────────────────

class BaseBsdfFileReader(ABC):
    """BSDF ファイルリーダーの基底クラス。

    サブクラスは can_read() と read() を実装する。
    ファイル形式の判定は can_read() で行い、read() でデータを返す。
    """

    @classmethod
    @abstractmethod
    def can_read(cls, path: Path) -> bool:
        """このリーダーがファイルを読み込めるかを判定する。

        Args:
            path: ファイルパス

        Returns:
            True → このリーダーで読み込み可能
        """

    @classmethod
    @abstractmethod
    def read(cls, path: Path) -> list[pd.DataFrame]:
        """BSDF ファイルを読み込み、Parquet スキーマ準拠の DataFrame リストを返す。

        各ブロック（AOI/波長/モードの組み合わせ）が 1 つの DataFrame になる。

        Args:
            path: ファイルパス

        Returns:
            DataFrame のリスト（各要素 = 1 測定ブロック）
        """


# ── プラグインローダー ─────────────────────────────────────────────────────────

def load_bsdf_readers(plugin_dir: str | Path = "custom_bsdf_readers") -> None:
    """custom_bsdf_readers/ フォルダからリーダープラグインを動的ロードする。

    フォルダ内の .py ファイルを検索し、BaseBsdfFileReader のサブクラスを
    _READER_REGISTRY に自動登録する。

    Args:
        plugin_dir: プラグインフォルダのパス（デフォルト: 'custom_bsdf_readers'）
    """
    plugin_path = Path(plugin_dir)
    if not plugin_path.exists():
        logger.debug(f"BSDF リーダープラグインフォルダが見つからない: {plugin_path}")
        return

    for py_file in sorted(plugin_path.glob("*.py")):
        module_name = f"custom_bsdf_readers.{py_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.warning(f"BSDF リーダープラグインのロードに失敗: {py_file} ({e})")
            continue

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseBsdfFileReader)
                and attr is not BaseBsdfFileReader
                and attr_name not in _READER_REGISTRY
            ):
                _READER_REGISTRY[attr_name] = attr
                logger.info(f"BSDF リーダープラグインを登録: {attr_name} ({py_file})")


def register_reader(reader_class: type[BaseBsdfFileReader]) -> None:
    """リーダークラスを手動登録する（テスト・組み込み用）。

    Args:
        reader_class: BaseBsdfFileReader サブクラス
    """
    _READER_REGISTRY[reader_class.__name__] = reader_class


def list_readers() -> list[str]:
    """登録済みリーダー名の一覧を返す。"""
    return list(_READER_REGISTRY.keys())


# ── 自動検出・読み込み ─────────────────────────────────────────────────────────

def get_conditions(dfs: list[pd.DataFrame]) -> list[dict]:
    """実測 DataFrame リストから光学条件の一覧を抽出する。

    各 DataFrame は単一の (wavelength_um, theta_i_deg, mode) を持つ前提。
    先頭行の値を代表値として取り出す。

    Args:
        dfs: `read_bsdf_file()` が返す DataFrame リスト

    Returns:
        dict リスト。各要素のキー: wavelength_um, theta_i_deg, mode, phi_i_deg
    """
    conds: list[dict] = []
    for df in dfs:
        if len(df) == 0:
            continue
        conds.append({
            "wavelength_um": float(df["wavelength_um"].iloc[0]),
            "theta_i_deg":   float(df["theta_i_deg"].iloc[0]),
            "mode":          str(df["mode"].iloc[0]),
            "phi_i_deg":     float(df["phi_i_deg"].iloc[0]),
        })
    return conds


def select_block(
    dfs: list[pd.DataFrame],
    wavelength_um: float,
    theta_i_deg: float,
    mode: str,
    tolerance_deg: float = 1.0,
    tolerance_nm: float = 5.0,
) -> pd.DataFrame | None:
    """tolerance 内で最も近い実測ブロックを返す。

    mode（'BRDF'/'BTDF'）は厳密一致必須。tolerance を超える候補は除外し、
    残った候補の中で (正規化距離)² が最小のブロックを返す。

    Args:
        dfs: `read_bsdf_file()` が返す DataFrame リスト
        wavelength_um: 目標波長 [μm]
        theta_i_deg:   目標入射角 [deg]
        mode:          'BRDF' または 'BTDF'
        tolerance_deg: 入射角の許容誤差 [deg]
        tolerance_nm:  波長の許容誤差 [nm]

    Returns:
        最近傍のブロック DataFrame、該当なしなら None
    """
    best: pd.DataFrame | None = None
    best_dist = float("inf")
    target_wl_nm = wavelength_um * 1000.0

    for df in dfs:
        if len(df) == 0:
            continue
        if str(df["mode"].iloc[0]) != mode:
            continue

        df_wl_nm = float(df["wavelength_um"].iloc[0]) * 1000.0
        df_theta = float(df["theta_i_deg"].iloc[0])

        dw_nm = abs(df_wl_nm - target_wl_nm)
        dt_deg = abs(df_theta - theta_i_deg)

        if dw_nm > tolerance_nm or dt_deg > tolerance_deg:
            continue

        # 正規化距離（各許容値を単位 1 として扱う）
        dist = (dw_nm / tolerance_nm) ** 2 + (dt_deg / tolerance_deg) ** 2
        if dist < best_dist:
            best_dist = dist
            best = df

    return best


def read_bsdf_file(
    path: str | Path,
    reader_name: str | None = None,
) -> list[pd.DataFrame]:
    """BSDF ファイルを読み込む。リーダーは自動選択または名前指定。

    Args:
        path: BSDF ファイルパス
        reader_name: リーダークラス名（None → can_read() で自動選択）

    Returns:
        DataFrame のリスト（各要素 = 1 測定ブロック）

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: 対応するリーダーが見つからない場合
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つからない: {path}")

    if reader_name is not None:
        if reader_name not in _READER_REGISTRY:
            raise ValueError(
                f"未知のリーダー: '{reader_name}'。"
                f"有効なリーダー: {list(_READER_REGISTRY.keys())}"
            )
        return _READER_REGISTRY[reader_name].read(path)

    # 自動選択: can_read() が True を返す最初のリーダーを使用
    for name, reader_class in _READER_REGISTRY.items():
        try:
            if reader_class.can_read(path):
                logger.info(f"BSDF リーダーを自動選択: {name} for {path}")
                return reader_class.read(path)
        except Exception as e:
            logger.debug(f"can_read() 失敗 ({name}): {e}")

    raise ValueError(
        f"対応する BSDF リーダーが見つからない: {path}\n"
        f"登録済みリーダー: {list(_READER_REGISTRY.keys())}\n"
        "custom_bsdf_readers/ にプラグインを追加してください。"
    )
