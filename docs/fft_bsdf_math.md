# `compute_bsdf_fft()` の数式

`src/bsdf_sim/optics/fft_bsdf.py` に実装されている FFT 法（スカラー回折理論）による BSDF 計算の数式リファレンス。

実装フロー:

```
compute_bsdf_fft()        ← src/bsdf_sim/optics/fft_bsdf.py
 ├─ (1) 位相分布 φ(x, y) の生成
 ├─ (2) np.fft.fft2(U) で複素振幅を周波数空間へ
 ├─ (3) 空間周波数 → 方向余弦 (u, v) 空間へ写像
 └─ (4) BSDF [sr⁻¹] へ換算
        ▼
 (u, v, bsdf) 2D グリッド
```

---

## 基本前提

スカラー回折理論。表面高さを $h(x, y)$（μm 単位）、波長を $\lambda$ [μm]、入射天頂角 $\theta_i$、入射方位角 $\phi_i$ として、波数は

$$k = \frac{2\pi}{\lambda} \quad [\mathrm{μm^{-1}}]$$

偏光は扱わない（スカラー近似）。偏光依存 BSDF が必要な場合は PSD 法を使用する。

---

## (1) 位相分布 $\varphi(x, y)$ の生成

表面高さ起因の位相差。BRDF / BTDF で式が異なる。

### BRDF（反射）モード

光が $h$ だけ進み、反射で $h$ だけ戻るので往復 $2h$：

$$\varphi_{\text{surface}}(x, y) \;=\; \frac{4\pi}{\lambda}\, n_1 \, h(x, y) \, \cos\theta_i$$

### BTDF（透過）モード

スネルの法則で透過角 $\theta_t$ を求め：

$$\sin\theta_t \;=\; \frac{n_1}{n_2}\sin\theta_i$$

$$\varphi_{\text{surface}}(x, y) \;=\; \frac{2\pi}{\lambda}\, \bigl(n_2 \cos\theta_t - n_1 \cos\theta_i\bigr)\, h(x, y)$$

### 斜入射の傾き項

平面波を FFT 座標系に乗せるためのシフト（シフト不変性の活用）：

$$\varphi_{\text{tilt}}(x, y) \;=\; k\, n_1 \, \sin\theta_i \,\bigl(\cos\phi_i \cdot x + \sin\phi_i \cdot y\bigr)$$

### 複素振幅

$$U(x, y) \;=\; \exp\!\Bigl[\, j\bigl(\varphi_{\text{surface}} + \varphi_{\text{tilt}}\bigr)\Bigr]$$

---

## (2) 2 次元 FFT とパワースペクトル

$$\widetilde U(f_x, f_y) \;=\; \mathrm{FFT}_{2D}\bigl[U(x, y)\bigr]$$

$$I(f_x, f_y) \;=\; \bigl|\widetilde U(f_x, f_y)\bigr|^2$$

実装: `U_fft = np.fft.fft2(U)` / `I_fft = np.abs(U_fft) ** 2`

---

## (3) 空間周波数 → 方向余弦 (UV) への写像

グレーティング方程式 $\sin\theta_s = f\,\lambda$ から：

$$u \;=\; f_x \, \lambda \;=\; \sin\theta_s \cos\phi_s$$

$$v \;=\; f_y \, \lambda \;=\; \sin\theta_s \sin\phi_s$$

実装: `freq_x = np.fft.fftfreq(N, d=dx)` → `u_grid = fx * wavelength_um`

**物理的に有効な領域**（半球内）:

$$u^2 + v^2 \;\le\; 1$$

この条件を満たさない格子点（エバネッセント波領域）は `bsdf = 0` として扱う。

---

## (4) BSDF への換算

### 散乱角余弦

$$\cos\theta_s \;=\; \sqrt{1 - u^2 - v^2}$$

### 正規化係数

サンプル全面積（物理面積）:

$$A \;=\; (N \cdot dx)^2 \quad [\mathrm{μm^{2}}]$$

実装: `normalization = (N * dx) ** 2`

### 最終式

$$\boxed{\;\mathrm{BSDF}(u, v) \;=\; \frac{I(f_x, f_y)}{A \cdot \cos\theta_s \cdot \lambda^2} \quad [\mathrm{sr}^{-1}]\;}$$

各因子の意味:

| 因子 | 役割 |
|---|---|
| $I \,/\, A$ | FFT パワー密度 → 物理面積で割って空間パワースペクトル密度に |
| $1 \,/\, \lambda^2$ | UV 空間微小面積 $du\,dv$ と立体角 $d\Omega$ の変換ヤコビアン |
| $1 \,/\, \cos\theta_s$ | BSDF 定義に含まれる投影立体角補正 |

有効領域外（$u^2 + v^2 > 1$）は `bsdf = 0`。

---

## 座標変換まとめ

| 変数 | 式 | 単位 | コード |
|---|---|---|---|
| 波数 $k$ | $2\pi / \lambda$ | $\mathrm{μm^{-1}}$ | `k = 2 * np.pi / wavelength_um` |
| 空間周波数 $f_x, f_y$ | — | $\mathrm{μm^{-1}}$ | `np.fft.fftfreq(N, d=dx)` |
| 方向余弦 $u, v$ | $f \cdot \lambda$ | 無次元 | `fx * wavelength_um` |
| 散乱天頂角 $\theta_s$ | $\arcsin\sqrt{u^2 + v^2}$ | rad | — |
| 立体角微小体 $d\Omega$ | $du\,dv \,/\, (\lambda^2 \cos\theta_s)$ | sr | — |

---

## 注意事項

- **広角近似の限界**: 散乱角 $\theta_s > 30°$ で偏光依存の誤差が大きくなる。`_WIDE_ANGLE_WARNING_DEG = 30.0` で警告が出る。厳密な偏光依存 BSDF は PSD 法を使用すること。
- **単位**: 実装内部ではすべて μm 統一。
- **FFT 順序**: `np.fft.fftfreq` は `[0, +1/N, ..., +0.5, -0.5, ..., -1/N]` の並びで返るため、連続した θ_s としてプロットする場合は `fftshift` または `argsort` で並び替えが必要（dashboard での二重線バグ BUG-009 の原因もこれ）。

---

## 3 つの `fft_mode` オプション

`compute_bsdf_fft(..., fft_mode=...)` で挙動を切替可能。既定値は `'tilt'`（後方互換）。

### mode = 'tilt'（既定・現行方式）
入力に $\varphi_\text{tilt}$ を加えて specular を (sin θ_i cos φ_i, sin θ_i sin φ_i) に配置。

- ✅ 全半球カバー（θ_i に依らず $u \in [-\lambda/2dx, +\lambda/2dx)$）
- ❌ $\sin\theta_i \cdot N \cdot dx / \lambda$ が非整数で **DFT スペクトル漏れ** が発生

### mode = 'output_shift'（漏れなし）
tilt を使わず、出力 `u_grid` ラベルを $u_\text{spec}$ だけシフト。

- ✅ tilt 漏れが構造的に発生しない（前方散乱側で精度高）
- ❌ $u$ の格子範囲が $[-\lambda/2dx + u_\text{spec},\ +\lambda/2dx + u_\text{spec})$ にシフトし、**θ_i が大きいと後方散乱側が欠落**
- 条件: $dx \leq \lambda / \bigl(2(1 + |\sin\theta_i|)\bigr)$

### mode = 'zero'（垂直入射近似・θ_i 非依存）
$\cos\theta_i \to 1$, $\sin\theta_i \to 0$ を代入。θ_i によらず 1 回の FFT で計算。

- ✅ 1 回計算で θ_i 無限通り使える（最速）、漏れなし、全半球カバー
- ❌ specular は常に (0, 0)。θ_i 依存は物理的に失われる
- 用途: パターン形状の参考表示、小角度（< 10°）の近似、教育目的

### 比較サマリ

| 観点 | tilt | output_shift | zero |
|---|---|---|---|
| specular 位置 | 物理的に正しい | 物理的に正しい | 常に (0, 0) |
| 漏れ | あり | なし | なし |
| 半球カバー | 常に完全 | θ_i 大で片側欠損 | 常に完全 |
| θ_i 依存性 | 厳密 | 厳密 | なし（近似） |
| 必要 $dx$ | $\leq \lambda/2$ | $\leq \lambda/\bigl(2(1+\|\sin\theta_i\|)\bigr)$ | $\leq \lambda/2$ |
| 再利用 | 条件ごと計算 | 条件ごと計算 | 1 回で全 θ_i |

### 使い分けガイド

| 用途 | 推奨 mode |
|---|---|
| 後方互換（既存コード） | `tilt`（省略可） |
| 前方散乱・specular 近傍の精密比較 | `output_shift` |
| パターン形状のみ必要・最速計算 | `zero` |
| 実測との定量一致（小粗さ） | PSD 法と併用 |

### 実装参照
- 実装: `src/bsdf_sim/optics/fft_bsdf.py`
- デモ: `outputs/_demo_fft_modes.py` → `outputs/fft_modes_comparison.png`
- テスト: `tests/test_fft_bsdf.py::TestFFTModes`

---

## 関連ファイル

- 実装: `src/bsdf_sim/optics/fft_bsdf.py`
- 仕様書: `spec_main.md` Section 3.2
- サンプリング関数（角度点補間）: `sample_bsdf_at_angles()`（同ファイル内）
