"""表面形状モデルパッケージ。custom_surfaces/ からプラグインを動的ロードする。"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from .base import BaseSurfaceModel, HeightMap
from .measured import MeasuredSurface
from .random_rough import RandomRoughSurface
from .spherical_array import SphericalArraySurface

logger = logging.getLogger(__name__)

# 標準モデルの登録
_MODEL_REGISTRY: dict[str, type[BaseSurfaceModel]] = {
    "RandomRoughSurface": RandomRoughSurface,
    "SphericalArraySurface": SphericalArraySurface,
    "MeasuredSurface": MeasuredSurface,
}


def load_plugins(plugin_dir: str | Path = "custom_surfaces") -> None:
    """custom_surfaces/ フォルダからプラグインを動的ロードする。

    フォルダ内の .py ファイルを検索し、BaseSurfaceModel のサブクラスを
    自動登録する。メインプログラムを改修せずに新しい形状モデルを追加できる。

    Args:
        plugin_dir: プラグインフォルダのパス
    """
    plugin_path = Path(plugin_dir)
    if not plugin_path.exists():
        return

    for py_file in plugin_path.glob("*.py"):
        module_name = f"custom_surfaces.{py_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.warning(f"プラグインのロードに失敗: {py_file} ({e})")
            continue

        # BaseSurfaceModel のサブクラスを検索して登録
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseSurfaceModel)
                and attr is not BaseSurfaceModel
                and attr_name not in _MODEL_REGISTRY
            ):
                _MODEL_REGISTRY[attr_name] = attr
                logger.info(f"プラグインモデルを登録: {attr_name} ({py_file})")


def get_model_class(name: str) -> type[BaseSurfaceModel]:
    """モデル名からクラスを取得する。

    Args:
        name: モデル名（例: 'RandomRoughSurface'）

    Returns:
        BaseSurfaceModel のサブクラス

    Raises:
        KeyError: 未登録のモデル名の場合
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"未知の表面形状モデル: '{name}'。"
            f"有効なモデル: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name]


def create_model_from_config(config: dict) -> BaseSurfaceModel:
    """設定辞書から表面形状モデルを生成する。

    Args:
        config: 設定辞書（config.yaml の内容）

    Returns:
        BaseSurfaceModel インスタンス
    """
    model_name = config.get("surface", {}).get("model", "RandomRoughSurface")
    model_class = get_model_class(model_name)

    if hasattr(model_class, "from_config"):
        return model_class.from_config(config)

    raise NotImplementedError(f"{model_name} に from_config() が実装されていない。")


__all__ = [
    "HeightMap",
    "BaseSurfaceModel",
    "RandomRoughSurface",
    "SphericalArraySurface",
    "MeasuredSurface",
    "load_plugins",
    "get_model_class",
    "create_model_from_config",
]
