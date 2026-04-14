"""BSDF シミュレーション・最適化パッケージ。

AGフィルム等の表面形状データを入力とし、FFT/PSD法でBSDFをシミュレーションする。
Optuna で最適化、MLflow で実験管理、HoloViews で可視化する。
"""

from .models import HeightMap, BaseSurfaceModel, RandomRoughSurface, SphericalArraySurface, MeasuredSurface
from .io import BSDFConfig

__version__ = "0.1.0"

__all__ = [
    "HeightMap",
    "BaseSurfaceModel",
    "RandomRoughSurface",
    "SphericalArraySurface",
    "MeasuredSurface",
    "BSDFConfig",
    "__version__",
]
