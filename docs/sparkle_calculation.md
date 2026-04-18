# ギラツキ（Sparkle）計算方法

AG（アンチグレア）フィルムをディスプレイ前面に貼ったときに発生する **ギラツキ（sparkle / display mura / scintillation）** を、BSDF から算出する手順と、その背景にある規格・文献を整理する。

関連コード: `src/bsdf_sim/metrics/optical.py`（`compute_sparkle`, `_compute_sparkle_single`）
関連 spec: `spec_main.md` Section 5.2 / 5.3

---

## 1. ギラツキとは

AG フィルムはディスプレイ表面反射を散乱させて映り込みを軽減するが、その凹凸がディスプレイ画素からの透過光を観察者の瞳孔上で **画素ごとに不均一にリダイレクト** するため、高精細ディスプレイ上では **粒状の輝度ムラ（sparkle / 微細なキラキラ感）** として知覚される。画素ピッチが瞳孔径に対して十分小さく、かつ AG フィルムの表面凹凸スケールが画素ピッチと同程度のときに顕在化する、ディスプレイと AG フィルムの相互作用で生じる 2 次的な光学現象である。

業界では **GD 値**（Glare / sparkle Display value）あるいは **Sparkle Contrast** と呼ばれることが多い。

---

## 2. 準拠規格・参考文献

本実装が準拠・参照した規格と文献を挙げる。

### 2.1. 規格

| 規格 | 発行団体 | 内容 |
|---|---|---|
| **SEMI D63-0709** | SEMI（国際半導体製造装置材料協会） | "Test Method for Measurement of Display Sparkle" — AG ディスプレイのスパークル測定手順を規定。カメラ系による画素輝度分布の撮像と、コントラスト $C_s=\sigma/\mu$ による定量化を定めた最初期の標準。 |
| **IDMS（Information Display Measurements Standard）v1.1** | ICDM（International Committee for Display Metrology, SID 配下） | Section 17.9 "Sparkle, Grain and Glitter" — SEMI D63 をベースに、観察距離・瞳孔径・画素ピッチをパラメータとした標準化された測定条件を規定。ハードウェア依存のない幾何学的定義。 |

本実装は **IDMS が規定する幾何配置（観察者瞳孔－ディスプレイ面－単画素領域）を光学シミュレーション側で再現** し、実測カメラの代わりに BSDF から瞳孔に届く輝度を数値積分する。

### 2.2. 主要文献

| 文献 | 内容 |
|---|---|
| E. F. Kelley, "Display sparkle measurement and human response," *SID Symposium Digest*, vol. 45, pp. 808–811 (2014) | NIST の Kelley による、sparkle の計測とヒト視覚応答の定量化。観察距離 300 mm・瞳孔径 3 mm の「スマートフォン観察条件」を標準化し、$C_s=\sigma/\mu$ を人間の知覚閾値と結びつけた基礎文献。 |
| K. Käläntär, "A directional-view measurement system for sparkle evaluation of anti-glare films on high-resolution LCDs," *J. SID*, vol. 21, pp. 499–507 (2013) | AG フィルムのスパークル測定方法と、画素ピッチ × 観察距離 × 瞳孔径が sparkle 値に与える影響を定量化。 |
| V. Becker et al., "Perception-based sparkle evaluation," *SID Symposium Digest*, vol. 50, pp. 473–476 (2019) | 心理物理実験との相関を含む評価手法。 |
| 日本学術振興会 光学薄膜第 180 委員会・AG 関連資料 | 国内における AG フィルムのギラツキ評価の実務的な解説。 |

本実装のパラメータ（smartphone=300 mm/3 mm, tablet=350 mm/3 mm, monitor=600 mm/3 mm）は Kelley (2014) および IDMS v1.1 の観察条件をベースにした値である。

---

## 3. 計算原理

### 3.1. 物理モデル

**用語の明確化**: 本ドキュメントで「画素」は原則 **ディスプレイ画素**（AG フィルム背面の発光素子）を指す。SEMI D63 / IDMS の実測では別途「カメラ画素（imaging sensor pixel）」が登場するが、本実装ではカメラを使わず、**ディスプレイ画素 1 個が観察者から張る角度領域** を方向余弦空間上のビンとして扱う。混同を避けるため以下では「ディスプレイ画素」「(角度)ビン」という呼び分けを使う。

**幾何配置**:

```
  [AG フィルム + ディスプレイ]        [観察者の瞳孔]
     ─────────────                          ○  径 d_p
     │ │ │ │ │ │ │    ─────────── 距離 D ──────────►
     pitch p
```

観察者はディスプレイから距離 $D$ の位置に瞳孔径 $d_p$ で配置されている。ディスプレイ上の 1 ディスプレイ画素（ピッチ $p$）は観察者から見て

$$2\,\sin\alpha_\text{half} = 2\,\sin\!\left[\arctan\!\left(\frac{p}{2D}\right)\right]$$

の角度幅（方向余弦空間での幅）を占める。一方、観察者の瞳孔は立体角

$$\Omega_\text{pupil} = \pi\left(\frac{d_p}{2D}\right)^2 \quad [\mathrm{sr}]$$

でディスプレイから光を受け取る。

**BSDF との対応**: AG フィルム背面で均一に発光するディスプレイを仮定すると、観察者から見たあるディスプレイ画素の見かけ輝度は、そのディスプレイ画素が占める角度領域に入る BSDF $f_s(u,v)$ を瞳孔立体角と合わせて積分した値に比例する（観察者の角度座標 ↔ ディスプレイ上の空間座標 が小角近似で 1 対 1 対応するため、角度ビンでの BSDF 分布がそのままディスプレイ上のディスプレイ画素輝度分布になる）。AG フィルムの散乱分布が完全に一様なら全ディスプレイ画素が同じ輝度になるが、実際には凹凸がディスプレイ画素ごとに異なる方向余弦成分を瞳孔に向けるため、**ディスプレイ画素ごとの積分輝度に σ のばらつき** が生じる。これが sparkle である。

### 3.2. 定義式

方向余弦空間を、1 ディスプレイ画素の角度幅（$2\sin\alpha_\text{half}$）で格子状にビン分割する。ビン $k$（= ディスプレイ画素 $k$ に対応）の領域を $A_\text{pix}^{(k)}$ として、このビン内のグリッド点を瞳孔立体角で積分した輝度を

$$L_k \;=\; \int_{A_\text{pix}^{(k)}} f_s(u, v)\,\cos\theta_s\;\Omega_\text{pupil}\; du\, dv$$

とする。ギラツキコントラスト $C_s$ は

$$C_s \;=\; \frac{\sigma(L_k)}{\mu(L_k)} \quad \text{(IDMS / SEMI D63 準拠)}$$

で定義される。SEMI D63 / IDMS の実測ではカメラ撮像後の **カメラ画素** 輝度分布 $\{L_k\}$ に同じ式を適用するが、撮像系の光学分解能を「個々のディスプレイ画素が分解可能」な水準に設定する規格であるため、カメラ画素輝度分布 ≒ ディスプレイ画素輝度分布 となる。本実装では BSDF グリッド点をディスプレイ画素サイズの角度ビンへ振り分けることで、この分布を直接数値的に再現する（カメラを介さない）。

---

## 4. 実装アルゴリズム

`compute_sparkle()` と `_compute_sparkle_single()` の処理を順に示す。

### 4.1. 幾何パラメータの導出

| パラメータ | 式 | デフォルト値 |
|---|---|---|
| 瞳孔立体角 $\Omega_\text{pupil}$ [sr] | $\pi(d_p / 2D)^2$ | $\pi(1.5/300)^2 \approx 7.85\times 10^{-5}$ |
| 1 ディスプレイ画素の方向余弦半幅 $\sin\alpha_\text{half}$ | $\sin[\arctan(p/2D)]$ | $\sin[\arctan(0.031/300)] \approx 1.03\times 10^{-4}$ |

プリセット一覧:

| プリセット | $D$ [mm] | $d_p$ [mm] | 代表用途 |
|---|---|---|---|
| smartphone | 300 | 3.0 | FHD スマートフォン観察 |
| tablet | 350 | 3.0 | タブレット観察 |
| monitor | 600 | 3.0 | デスクトップモニタ観察 |

| ディスプレイプリセット | 画素ピッチ $p$ [mm] | 代表用途 |
|---|---|---|
| fhd_smartphone | 0.062 | FHD（1080p）6 インチ相当 |
| qhd_monitor | 0.124 | QHD 24 インチ相当 |
| 4k_monitor | 0.160 | 4K 27 インチ相当 |

### 4.2. BSDF グリッド点 → ディスプレイ画素ビンへの割り当て

方向余弦グリッド $(u, v)$ 上の各点を、最近傍のディスプレイ画素ビンに振り分ける。ビンインデックスは

$$
\text{bin}_u = \mathrm{round}\!\left(\frac{u}{2\sin\alpha_\text{half}}\right),\quad
\text{bin}_v = \mathrm{round}\!\left(\frac{v}{2\sin\alpha_\text{half}}\right)
$$

で求める（$\text{bin}_u, \text{bin}_v$ の組 1 つが 1 ディスプレイ画素に対応）。$u^2 + v^2 > 1$（propagating でない領域）は除外する。

### 4.3. ディスプレイ画素ごとの輝度合算

各グリッド点が寄与する微小輝度

$$\Delta L \;=\; f_s(u, v)\,\cos\theta_s\,\Omega_\text{pupil}\,du\,dv$$

を、同一ビンインデックスに属するグリッド点について合算する（`np.bincount` によるベクトル化実装）。

$\cos\theta_s = \sqrt{1 - u^2 - v^2}$ は立体角要素 $d\Omega = \cos\theta_s\,du\,dv$ の重みである。

### 4.4. コントラストの算出

ディスプレイ画素ごとの輝度配列 $\{L_k\}$ に対し

$$C_s = \frac{\sigma(\{L_k\})}{\mu(\{L_k\})}$$

を算出する。$\mu < 10^{-30}$（実質的にゼロ輝度）の場合は 0 を返す。

---

## 5. 波長依存性と多波長計算

### 5.1. 波長依存性

SEMI D63 / IDMS は **白色バックライト × 明所視応答 V(λ)** を光源条件として規定している。AG フィルムの表面凹凸スケールは可視光波長より十分大きい（凹凸 ≫ λ）ため、一般に sparkle の波長依存性は弱い（典型的に 1〜5% 程度の差）。

本実装は **単波長ごとに独立計算** を行う方針を採用している。`simulation.wavelength_um` に list を与えると波長ごとに BSDF が独立計算され、それぞれに対して `compute_sparkle()` が呼ばれる。

### 5.2. メトリクスキーの命名

各波長 × 入射角 × mode の組合せで

```
sparkle_<method>_<nm>_<deg>_<mode>
```

の形式で記録される。

- `<method>` = `fft` / `psd` / `ml`
- `<nm>` = 波長 [nm]（整数）
- `<deg>` = 入射角（整数）
- `<mode>` = `t`（BTDF）/ `r`（BRDF）

例: `sparkle_fft_555_0_t`（555 nm, θ_i=0°, BTDF, FFT 法）

---

## 6. 実装上の注意と既知の近似

### 6.1. 採用している近似

1. **スカラー回折・非偏光**: BSDF 側の仮定を継承。偏光依存 sparkle は扱わない。
2. **単色光近似**: SEMI D63 の白色光＋V(λ) 重み付けではなく、単一波長で評価する（5.1 参照）。多波長設定時は波長ごとに独立計算し、加重合成はユーザー側で行う。
3. **矩形ディスプレイ画素・サブピクセル非考慮**: `_compute_sparkle_single` は 1 ディスプレイ画素を方向余弦空間での正方形ビンとして扱う。RGB ストライプ等のサブピクセル配列は現状では積分領域に反映しない（config の `subpixel_layout` は将来拡張用として保持）。
4. **最近傍割り当て**: グリッド点 → ディスプレイ画素ビンの割り当ては round で最近傍 1 ビンに寄与する。ビン境界にかかるグリッド点の部分面積寄与は無視する。BSDF グリッドがビンに対して十分密（smartphone プリセットで $\sin\alpha_\text{half}\approx10^{-4}$、grid_size=512 で $du\approx4\times10^{-3}$）ならばこの誤差は小さい。
5. **照明立体角**: 瞳孔立体角 $\Omega_\text{pupil}$ は全グリッド点に一様に掛けている（$\theta_s$ 依存の foreshortening は `cos_s` 項のみで扱う）。

### 6.2. 照明（illumination）設定の廃止

過去には `sparkle.illumination` セクションで多波長 RGB 加重を指定する実装があったが、simulate ループからは到達不能な dead code であったため削除した（CHANGELOG 参照）。多波長評価は `simulation.wavelength_um` のリスト化で行う。

---

## 7. 参考: 実測との対応

実機で sparkle を測定する場合、典型的には以下の装置が用いられる。

- **Display-Measurement Inc. SMS-1000 シリーズ**: SEMI D63 準拠のカメラ系装置
- **Instrument Systems DMS 803 / DMS 505**: IDMS 準拠の輝度分布測定
- **Westboro Photonics Sparkle Measurement System**

本実装の $C_s$ は、これら装置が測定する $\sigma/\mu$ と **同一定義** であり、直接比較可能である（前提: 同一の観察距離・瞳孔径・画素ピッチ）。

---

## 8. 設定例

```yaml
metrics:
  sparkle:
    enabled: true
    viewing:
      preset: 'smartphone'      # 300 mm / 3 mm
      distance_mm: null          # null ならプリセット値
      pupil_diameter_mm: null
    display:
      preset: 'fhd_smartphone'  # 0.062 mm / rgb_stripe
      pixel_pitch_mm: null
      subpixel_layout: null
```

カスタム値を優先したい場合は `distance_mm: 250` のように数値を入れる（プリセットとの混在可）。

---

## 9. 用語対応表

| 本ドキュメント | 英語表記 | 業界別名 |
|---|---|---|
| ギラツキ | Sparkle | GD 値 / Display Sparkle / Scintillation / Glittering |
| ギラツキコントラスト $C_s$ | Sparkle Contrast | σ/μ / GD 値 / Sparkle Index |
| 瞳孔立体角 | Pupil solid angle | — |
| ディスプレイ画素 | Display pixel | 発光素子、subpixel（RGB 個別の場合） |
| カメラ画素 | Camera / sensor pixel | 実測時の撮像素子のピクセル（本実装では使用しない） |
| ディスプレイ画素ピッチ | Display pixel pitch | Subpixel pitch（サブピクセルの場合） |
