"""可視化モジュール共通の定数。

`BSDF_LOG_FLOOR_DEFAULT` は log スケール描画時に `log(0)` や極小値による
対数発散を回避するための下限クリップ値。装置ノイズフロアを表す
`config.error_metrics.bsdf_floor`（既定 1e-6、Log-RMSE 計算用）とは
**役割が異なる**ので混同しないこと:

- `BSDF_LOG_FLOOR_DEFAULT`: 描画安定化。値は物理的意味を持たない。
- `config.error_metrics.bsdf_floor`: 誤差計算のマスク閾値。装置 DR に合わせる。
"""

from __future__ import annotations

BSDF_LOG_FLOOR_DEFAULT: float = 1e-10
