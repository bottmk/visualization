# ギラツキ $C_s$ の実測校正（Calibration）

本ドキュメントは、シミュレーション Cs と SEMI D63 / IDMS 実測 Cs の乖離を補正する **校正フレームワーク** の使い方と、**乖離原因の定量分析結果**、および **推奨ワークフロー** を記載する。

前提: `docs/sparkle_calculation.md`（基本計算方法）、`docs/sparkle_approximation_levels.md`（L1/L3/L4/L5 定義）
関連コード: `src/bsdf_sim/metrics/sparkle_calibrator.py`

---

## 1. 背景: なぜ校正が必要か

本実装の L1/L3/L4/L5 が返す Cs は、SEMI D63/IDMS 実測値と **桁で乖離** する（raw 値の典型比較）:

| 実装 | 典型 Cs（medium AG） | 実測 ≈ 0.05 との比 |
|---|---:|---:|
| L1 | 36.0 | 720× |
| L3 | 25.7 | 514× |
| L4 | 15.8 | 316× |
| **L5 (緑点灯)** | **0.58** | **11.5×** |

L5 は角度ビニング由来のアーティファクトを回避するため他より 1–2 桁改善するが、それでも実測値の 5–20× 程度に残る。この乖離は実測カメラ系の MTF や光学系モデル化不足などが原因で、純粋な実装バグではなく **物理モデルの簡略化の必然的な結果** である。

**校正の役割**: 実測データ数点を使ってシミュレーション値を実測スケールへマッピングし、**絶対値評価と相対比較の両方を可能にする**。

---

## 2. 測定プロトコル（緑点灯）

### 2.1. 推奨構成

本プロジェクトでの運用は **緑点灯（L5、color='G'）** を基準とする:

- **理由 1**: 実機ギラツキ測定装置（Instrument Systems DMS 803 等）では典型的に白点灯測定だが、緑単色測定もサポートされる。緑単色は R/G/B 合成の非線形性を排除できるため校正が安定
- **理由 2**: 緑の V(λ)（明所視応答）が大きく ~0.79、人間視覚の感度と整合しやすい
- **理由 3**: L4（白点灯）は サブピクセル間の非線形 σ/μ により校正が複雑

### 2.2. 必要な測定データ

| 項目 | 内容 |
|---|---|
| AG サンプル | 低/中/高 sparkle レンジの 5–10 枚 |
| 表面高さマップ | 共焦点顕微鏡 / 干渉計 / AFM（視野 0.5–2 mm 角、分解能 0.25–1 μm） |
| 実測 Cs 値 | SEMI D63 / IDMS 準拠装置、**緑単色点灯**（波長 525 nm 付近）|
| ディスプレイ仕様 | 画素ピッチ（実測 AG を使う実機ディスプレイの値）|
| 観察条件 | 距離・瞳孔径（測定装置固定値、規格準拠ならデフォルト 300mm/3mm） |

### 2.3. シミュレーション側の設定

`config.yaml`:

```yaml
metrics:
  sparkle:
    enabled: true
    level: 'L5'
    color: 'G'
    viewing:
      preset: 'smartphone'
    display:
      preset: 'fhd_smartphone'
    calibration:
      mode: null     # 初期は校正なしで raw Cs を収集
```

---

## 3. 乖離原因の定量分析結果

ベンチマーク AG (`Rq=0.3μm, Lc=5μm, N=512, dx=0.25μm`) で各要因の影響を測定。

### 3.1. 結果サマリ

| # | 要因 | 影響度 | 変動幅 | 対応 |
|---|---|---|---:|---|
| 1 | **窓幅 `window_size_factor`** | **大** | **4.6×（2x→5x）** | パラメータチューニング可 |
| 2 | **グリッドサイズ `N`** | **大** | **未収束（N=512 で不安定）** | **N ≥ 1024 推奨** |
| 3 | AG 相関長 $L_c$ | 中 | ~25% | AG 依存、コントロール不可 |
| 4 | 窓関数種（Hann vs rect） | 小 | ~1% | Hann で十分 |
| 5 | pupil 立体角積分 | 小 | 0%（通常設定）〜 8% | デフォルト ON で OK |
| 6 | 単波長 vs 狭帯域分光 | 小 | ~7%（510–540nm）| narrowband 近似で十分 |
| 7 | 観察方向 off-axis | 小 | < 2% 推定 | 実装省略可 |

### 3.2. 窓幅の影響（大）

```
window_size_factor  Cs
           2.0x     0.7019
           3.0x     0.5773   ← 既定
           4.0x     0.4358
           5.0x     0.1534
```

窓幅が大きいほど角度分解能が細かくなり、pupil 積分が有効になるので Cs が低下する。**校正は窓幅を固定した上で行う必要がある**。推奨は `window_size_factor=3.0`（既定）を使い、それに対する校正係数をフィット。

### 3.3. グリッドサイズの影響（大・要注意）

```
N         Cs
256      0.0000   ← 画素サンプル数 1² → σ/μ 計算不能
384      1.1558   ← 画素サンプル数 2² → 統計不十分
512      0.5773   ← 画素サンプル数 ~4² → 下限
768      0.7500   ← 画素サンプル数 ~7² → 未収束
```

**N=512 は sparkle 統計の信頼性として不十分**。絶対値評価時は `N=1024` 以上を推奨。最適化用途の相対比較では N=512 でも順位は保たれる。

### 3.4. 窓関数の影響（小）

```
Hann 窓:         Cs = 0.5773
Rectangular 窓:  Cs = 0.5702
```

窓関数の選択による差は 1% 程度で無視できる。Hann 窓（既定）で十分。

### 3.5. pupil 積分の影響（条件依存）

smartphone 標準設定（$u_\text{pupil} = 0.005$、FFT 分解能 $du \approx 0.008$）では **pupil がサブサンプル** になるため DC 1 点と一致。大窓幅で pupil > du となる条件下では最大 -8% の補正効果あり。

### 3.6. 分光幅の影響（小）

```
λ = 510 nm  Cs = 0.5968
λ = 525 nm  Cs = 0.5773
λ = 540 nm  Cs = 0.5567
5 波長平均:  Cs = 0.5768
```

緑帯域 ±15 nm で ±7% の変動。narrowband 近似（525 nm 単波長）と 5 波長分光積分の差は 0.1% 以下。**分光積分の実装優先度は低い**。

### 3.7. AG 相関長 $L_c$ の依存性（参考）

```
Lc = 2.0 μm   Cs = 0.528
Lc = 5.0 μm   Cs = 0.577   ← ベンチマーク
Lc = 10.0 μm  Cs = 0.470
Lc = 20.0 μm  Cs = 0.424
```

AG 固有のパラメータで、校正では固定できない。**同じ AG グレードで校正した係数は、同グレードの新規 AG に対してのみ有効**。AG の種類を跨ぐと校正係数を再フィット。

### 3.8. L1/L3/L4/L5 レベルの比較

```
L1 (白点灯一様):    Cs = 36.0   実測比 720×
L3 (緑単色+ビニング): Cs = 25.7   実測比 514×
L4 (白点灯3色合成):   Cs = 15.8   実測比 316×
L5 (緑単色+空間分解): Cs = 0.58   実測比 11.5×
```

L5 が他より 2 桁近く実測に近く、**校正誤差も最小**。運用推奨レベル。

---

## 4. 校正フレームワーク

### 4.1. API

```python
from bsdf_sim.metrics.sparkle_calibrator import (
    apply_calibration, fit_scale, fit_polynomial,
)

# 実測・シミュレーションペアから校正係数をフィット
k = fit_scale(cs_sim_list, cs_meas_list)  # scale のみ
a, b, c = fit_polynomial(cs_sim_list, cs_meas_list)  # a*x^b + c

# 単一 Cs に校正を適用（通常は compute_all_optical_metrics 内部で自動）
cs_cal = apply_calibration(cs_raw, {"mode": "scale", "scale": k})
```

### 4.2. config.yaml 記法

**スケール校正** (1 パラメータ):

```yaml
metrics:
  sparkle:
    level: 'L5'
    color: 'G'
    calibration:
      mode: 'scale'
      scale: 0.087      # 例: L5 緑点灯ベンチマーク AG で fit
```

**多項式校正** (3 パラメータ、より高精度):

```yaml
calibration:
  mode: 'polynomial'
  polynomial: [0.087, 1.0, 0.0]   # cs_cal = 0.087 * cs_sim^1.0 + 0.0
```

**校正なし**（既定、raw Cs を返す）:

```yaml
calibration:
  mode: null
```

### 4.3. 適用タイミング

`compute_all_optical_metrics` 内で各レベル計算直後に `apply_calibration` が自動適用される。MLflow メトリクス `sparkle_l5_fft_525_0_t` として記録される値は **校正後** の値。

---

## 5. 推奨ワークフロー

### 5.1. Phase 1: 初期実装（実装済）

- 校正フレームワーク `sparkle_calibrator.py` を実装
- config.yaml `sparkle.calibration` セクションを整備
- `mode: null` の状態で運用開始（raw Cs を収集）

### 5.2. Phase 2: スケール校正（実測データ入手後、1–2 日）

1. **実測 AG サンプル 3–5 枚** を用意（低/中/高 sparkle）
2. 各サンプルの高さマップを取得
3. 同じ config で `compute_sparkle_l5(color='G')` を実行、raw Cs を得る
4. 実測装置（DMS 803 等）で緑単色 Cs を測定
5. `fit_scale()` で係数 k をフィット
6. `config.yaml` に `calibration: {mode: scale, scale: k}` を設定

```python
from bsdf_sim.metrics.sparkle_calibrator import fit_scale

# 5 サンプルでの例
cs_sim  = [0.12, 0.35, 0.58, 0.82, 1.20]
cs_meas = [0.012, 0.035, 0.055, 0.078, 0.110]
k = fit_scale(cs_sim, cs_meas)
print(f"scale = {k:.4f}")  # e.g., 0.095
```

### 5.3. Phase 3: 多項式校正（実測 10–20 点、1 週間）

1. サンプル数を 10–20 に拡張
2. `fit_polynomial()` で `(a, b, c)` をフィット
3. 交差検証で外挿精度を評価（Rq や Lc のレンジ外で誤差増大に注意）

```python
from bsdf_sim.metrics.sparkle_calibrator import fit_polynomial

a, b, c = fit_polynomial(cs_sim, cs_meas)
# config.yaml:
# calibration: {mode: polynomial, polynomial: [a, b, c]}
```

### 5.4. Phase 4: 物理モデル改良（長期）

Phase 2/3 の校正誤差が実用精度（±10–20%）を超える場合、以下を検討:

- カメラ MTF モデリング（実機レンズ・検出器応答の畳み込み）
- Off-axis 観察方向の厳密実装
- 完全分光積分（`compute_sparkle_l4` の narrowband 解除）
- 窓幅・グリッドサイズの収束性確認（N=1024, 2048 で再フィット）

---

## 6. 運用上の注意

### 6.1. 校正係数の適用範囲

校正係数はフィットに使った **AG グレード・ディスプレイ仕様・観察条件** に対してのみ有効。以下が変わる場合は再フィット:

- **AG の種類**（粒子分散系 → 表面刻印系 等）
- **ディスプレイ画素ピッチ**（スマホ → モニタ）
- **観察距離・瞳孔径**（smartphone → tablet プリセット変更）
- **波長・色**（緑点灯 → 赤点灯）
- **評価レベル**（L5 → L1）

### 6.2. 校正しても解消しない問題

- **AG の非均一性（不均一欠陥）の検出**: 校正は平均的スケーリングのみ、局所的異常は別途検出が必要
- **Phase 2 の係数は絶対値保証ではなく推定**: サンプル数が少ないと外挿誤差大
- **Sparkle 値は人間視覚と 1:1 対応しない**: 知覚閾値は Cs≈0.02 とされるが、個人差や照明条件で変動

### 6.3. 相対比較用途

Optuna 最適化など相対比較目的では **校正なしでも実用的**。raw Cs の順位が保たれていれば設計比較には十分。絶対値を規格値と比較する場合のみ校正必須。

---

## 7. まとめ

| 段階 | 必要データ | 期待精度 | 工数 |
|---|---|---|---|
| Phase 1 | なし | raw Cs のみ（相対比較可） | 実装済み |
| Phase 2 | 実測 3–5 サンプル | ±30–50% | 測定 1 日 |
| Phase 3 | 実測 10–20 サンプル | ±10–20% | 測定 1 週間 |
| Phase 4 | 実機仕様書 + 数十サンプル | ±5–10% | 1–2 ヶ月 |

本実装は Phase 1 まで完了。Phase 2 は実測データ入手後すぐ運用可能。

---

## 関連ドキュメント

- `docs/sparkle_calculation.md`: 基本計算アルゴリズム
- `docs/sparkle_approximation_levels.md`: L1/L3/L4/L5 近似階層と数式
- `spec_main.md` Section 5.2: 仕様書での sparkle 計算概要
- `src/bsdf_sim/metrics/sparkle_calibrator.py`: 校正関数の実装
