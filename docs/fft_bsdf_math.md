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

すべてのモードで共通の最終式は

$$\mathrm{BSDF}(u, v) \;=\; \frac{I(f_x, f_y)}{A \cdot \cos\theta_s \cdot \lambda^2} \quad [\mathrm{sr}^{-1}]$$

違いは **(1) 複素振幅 $U$ の作り方** と **(2) 出力 $u_\text{grid}$ のラベル** の 2 点のみ。
config.yaml からは `fft.mode` で指定できる（`simulate` / `dashboard` 双方で有効）。
以下、BRDF モードで示します（BTDF も位相変換式が別になるだけで構造は同じ）。

### mode = 'tilt'（既定・現行方式）

**表面位相**
$$\varphi_\text{surface}(x, y) \;=\; \frac{4\pi}{\lambda}\, n_1 \, h(x, y) \, \cos\theta_i$$

**傾き位相**（斜入射を shift invariance で normal-incidence 形式に揃える線形ランプ）
$$\varphi_\text{tilt}(x, y) \;=\; k\, n_1 \sin\theta_i \bigl(\cos\phi_i\, x + \sin\phi_i\, y\bigr)$$

**複素振幅**
$$U(x, y) \;=\; \exp\!\Bigl[\, j\bigl(\varphi_\text{surface} + \varphi_\text{tilt}\bigr)\Bigr]$$

**出力格子**
$$u_\text{grid} = f_x \lambda, \quad v_\text{grid} = f_y \lambda$$

**性質**: 傾き項の空間周波数は $f_\text{tilt} = n_1 \sin\theta_i / \lambda$。これが FFT 格子点 $m/(N\cdot dx)$ と一致するためには

$$\frac{\sin\theta_i \cdot N \cdot dx}{\lambda} \in \mathbb{Z}$$

を満たす必要がある。満たさない場合、DFT の巡回周期性が破れて sinc 型の**スペクトル漏れ**が生じる。

- ✅ 全半球カバー（$u \in [-\lambda/2dx,\, +\lambda/2dx)$、θ_i 非依存）
- ❌ 一般の θ_i で漏れ発生（小粗さほど相対的に支配的）

### mode = 'output_shift'（漏れなし・格子シフト）

**表面位相**（tilt と同じ）
$$\varphi_\text{surface}(x, y) \;=\; \frac{4\pi}{\lambda}\, n_1 \, h(x, y) \, \cos\theta_i$$

**傾き項は使わない**
$$U(x, y) \;=\; \exp(j\varphi_\text{surface})$$

**出力格子にだけシフトを加える**
$$u_\text{grid} = f_x \lambda + n_1 \sin\theta_i \cos\phi_i$$
$$v_\text{grid} = f_y \lambda + n_1 \sin\theta_i \sin\phi_i$$

**等価性の証明**: convolution theorem より

$$\mathcal{F}\bigl[e^{j\varphi_\text{surface}} \cdot e^{j\varphi_\text{tilt}}\bigr](f) \;=\; \mathcal{F}\bigl[e^{j\varphi_\text{surface}}\bigr](f - f_\text{tilt})$$

従って、tilt 方式で周波数 $f$ にある値は output_shift 方式で $f - f_\text{tilt}$ にある値と等しい。$u$ 座標系ではこれは「出力ラベルを $+u_\text{spec}$ だけシフト」に相当する。**連続関数的には同一の物理量**を指す。

**なぜ漏れなくなるのか**: tilt は $e^{j\varphi_\text{tilt}}$ を離散サンプリングする段階で、非整数周波数のため標本終端に位相不連続（段差）ができる。output_shift は $U = e^{j\varphi_\text{surface}}$ のみで、$h(x, y)$ は周期境界が整う（FFT の周期延長と両立）ため段差が生じない。

**覆う範囲のシフト**: FFT 格子 $f \in [-1/(2dx),\, +1/(2dx))$ を $u$ 空間に写像すると

$$u \in \left[-\frac{\lambda}{2dx} + u_\text{spec},\ +\frac{\lambda}{2dx} + u_\text{spec}\right)$$

物理的に有効な $u \in [-1, +1]$ 全体を覆うためには

$$dx \;\leq\; \frac{\lambda}{2(1 + |\sin\theta_i|)}$$

を満たす必要がある。満たせない場合、**後方散乱側が格子外**になり BSDF が計算されない。

| θ_i | dx の上限 (λ=0.55) |
|---|---|
| 0° | 0.275 μm |
| 20° | 0.205 |
| 45° | 0.161 |
| 60° | 0.147 |
| 80° | 0.141 |

- ✅ 漏れ構造的にゼロ、specular も物理的に正しい位置
- ❌ 大 θ_i で後方散乱欠損（dx を小さくする以外に解消策なし）

### mode = 'zero'（垂直入射近似・θ_i 非依存）

**全ての θ_i で normal-incidence 位相を使用**

BRDF:
$$\varphi_\text{surface}^{(0)}(x, y) \;=\; \frac{4\pi}{\lambda}\, n_1 \, h(x, y)$$

BTDF:
$$\varphi_\text{surface}^{(0)}(x, y) \;=\; \frac{2\pi}{\lambda}\, (n_2 - n_1)\, h(x, y)$$

**傾き項もオフセットもなし**
$$U(x, y) \;=\; \exp(j\varphi_\text{surface}^{(0)})$$
$$u_\text{grid} = f_x \lambda,\quad v_\text{grid} = f_y \lambda$$

**近似の本質**:

- $\cos\theta_i \to 1$: 表面粗さの $\cos\theta_i$ 倍投影効果を無視
- $\sin\theta_i \to 0$: specular を常に原点に固定（tilt ラベルシフトも無し）

Rayleigh-Rice 線形域では、tilt の結果を $\cos^2\theta_i$ で割って $(u_\text{spec}, v_\text{spec})$ だけシフトすると zero の結果とほぼ一致する（形状として）。

**誤差の上限**: 小粗さ（$Rq \cos\theta_i \ll \lambda/8$）では $\cos^2\theta_i$ 倍の振幅差のみ → log スケールで $\log_{10}\cos^2\theta_i$ のシフト（θ_i=60° で −0.6 dec）。大粗さでは非線形項 $O(\varphi^4)$ が効き、形状そのものが乖離する。

- ✅ θ_i に依らず結果が同じ → **1 回の FFT で全 θ_i に使い回せる**（最速）
- ✅ 漏れなし、全半球カバー
- ❌ θ_i 依存が完全に失われる（物理的に θ_i ≠ 0 では誤り）
- ❌ specular 方向が (0, 0) に固定（実測との角度一致比較に不適）

### 並列比較

| 観点 | tilt | output_shift | zero |
|---|---|---|---|
| 表面位相の $\cos\theta_i$ | あり | あり | **なし** ($=1$) |
| 傾き項 $\varphi_\text{tilt}$ | あり | なし | なし |
| $u_\text{grid}$ オフセット | 0 | $+u_\text{spec}$ | 0 |
| specular 位置 | $(u_\text{spec}, v_\text{spec})$ | 同左 | **$(0, 0)$** 固定 |
| 漏れ | あり（非整数 θ_i で） | なし | なし |
| 半球カバー | 常に完全 | θ_i 大で片側欠損 | 常に完全 |
| θ_i 依存性 | 厳密 | 厳密 | **なし**（近似） |
| 必要 $dx$ | $\leq \lambda/2$ | $\leq \lambda/\bigl(2(1+\|\sin\theta_i\|)\bigr)$ | $\leq \lambda/2$ |
| 再利用 | 条件ごと FFT | 条件ごと FFT | **1 回で全 θ_i** |

### 使い分けガイド

| 用途 | 推奨 mode | 理由 |
|---|---|---|
| 後方互換（既存コード） | `tilt` | 既定値・API 互換 |
| 前方散乱・specular 近傍の精密比較 | `output_shift` | 漏れなし、specular 正確 |
| 後方散乱 (retro-reflection) が必要 | `tilt` | output_shift は格子外で取れない |
| パターン形状のみ必要・最速計算 | `zero` | 1 FFT で全 θ_i 共有 |
| 実測との定量一致（小粗さ） | PSD 法と併用 | FFT 系は Fresnel・振幅に限界 |
| Optuna 最適化の高速評価 | `zero` | trial 数を稼げる |

### 実装参照

- 実装: `src/bsdf_sim/optics/fft_bsdf.py`
- config 経由の指定: `config.yaml` の `fft.mode`（`simulate`, `dashboard` 共通）
- デモ: `outputs/_demo_fft_modes.py` → `outputs/fft_modes_comparison.png`
  （RandomRough + SphericalArray の 2 行で視覚比較）
- テスト: `tests/test_fft_bsdf.py::TestFFTModes`, `tests/test_config_loader.py::TestFFTMode`

---

## 関連ファイル

- 実装: `src/bsdf_sim/optics/fft_bsdf.py`
- 仕様書: `spec_main.md` Section 3.2
- サンプリング関数（角度点補間）: `sample_bsdf_at_angles()`（同ファイル内）
