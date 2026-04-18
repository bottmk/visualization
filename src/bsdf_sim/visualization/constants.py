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

# phi_s≈0 プロファイルと見なす方位角の許容幅 [deg]。
# 実測データ（LightTools 等）は phi_s=0 ちょうどではなく±数度の点を持つことが
# 多いため、ある程度の許容幅が必要。dashboard と visualize レポートで同じ値を
# 使うよう定数化（以前は両方 6.0 で重複）。
MEASURED_PHI_S_TOL_DEG: float = 6.0
