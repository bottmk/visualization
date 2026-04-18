"""評価指標パッケージ。custom_metrics/ からプラグインを動的ロードする。"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Callable

import numpy as np

from .optical import (
    compute_all_optical_metrics,
    compute_doi_astm,
    compute_doi_comb,
    compute_doi_nser,
    compute_gloss,
    compute_haze,
    compute_log_rmse,
    compute_sparkle,
)
from .surface import compute_all_surface_metrics, compute_ra, compute_rq, compute_rz, compute_sdq

logger = logging.getLogger(__name__)

# 指標関数レジストリ: 名前 → 計算関数
_METRIC_REGISTRY: dict[str, Callable] = {
    "haze":     compute_haze,
    "gloss":    compute_gloss,
    "doi_nser": compute_doi_nser,
    "doi_comb": compute_doi_comb,
    "doi_astm": compute_doi_astm,
    "sparkle":  compute_sparkle,
    "log_rmse": compute_log_rmse,
    "rq":       compute_rq,
    "ra":       compute_ra,
    "rz":       compute_rz,
    "sdq":      compute_sdq,
}


def load_metric_plugins(plugin_dir: str | Path = "custom_metrics") -> None:
    """custom_metrics/ フォルダからプラグインを動的ロードする。

    フォルダ内の .py ファイルを検索し、`compute_` で始まる関数を自動登録する。

    Args:
        plugin_dir: プラグインフォルダのパス
    """
    plugin_path = Path(plugin_dir)
    if not plugin_path.exists():
        return

    for py_file in plugin_path.glob("*.py"):
        module_name = f"custom_metrics.{py_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.warning(f"指標プラグインのロードに失敗: {py_file} ({e})")
            continue

        for attr_name in dir(module):
            if attr_name.startswith("compute_") and callable(getattr(module, attr_name)):
                metric_name = attr_name[len("compute_"):]
                if metric_name not in _METRIC_REGISTRY:
                    _METRIC_REGISTRY[metric_name] = getattr(module, attr_name)
                    logger.info(f"指標プラグインを登録: {metric_name} ({py_file})")


def get_metric_names() -> list[str]:
    """登録済みの指標名一覧を返す。"""
    return list(_METRIC_REGISTRY.keys())


__all__ = [
    "compute_haze",
    "compute_gloss",
    "compute_doi_nser",
    "compute_doi_comb",
    "compute_doi_astm",
    "compute_sparkle",
    "compute_log_rmse",
    "compute_all_optical_metrics",
    "compute_rq",
    "compute_ra",
    "compute_rz",
    "compute_sdq",
    "compute_all_surface_metrics",
    "load_metric_plugins",
    "get_metric_names",
]
