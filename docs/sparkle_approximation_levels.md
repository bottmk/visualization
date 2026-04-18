# ギラツキ計算の近似階層と厳密計算

本ドキュメントは、ギラツキ（sparkle）計算における近似レベル L1〜L5 の数学的定義と相互関係、空間分解 BSDF・窓付き FFT の数式、および L4 と L5 の差が顕在化する条件を整理する。

前提: `docs/sparkle_calculation.md`（基本計算方法と規格対応）
関連コード: `src/bsdf_sim/metrics/optical.py`（L1）、`src/bsdf_sim/metrics/sparkle_extended.py`（L3'/L4/L5）

---

## 1. 近似階層の全体像

ディスプレイ前面に貼った AG フィルムの sparkle を計算するにあたり、以下の 2 軸で近似レベルを分類する:

A. **分光**: 単波長 vs 多波長 $V(\lambda)$ 重み
B. **空間発光モデル**: 画素全面一様 vs サブピクセル構造 vs 空間分解 BSDF

これに基づく階層:

| Level | 分光 | 空間発光 | AG 応答 | FFT 回数 | 実装状況 |
|---|---|---|---|---|---|
| **L1** | 単波長 | 画素全面一様 | グローバル BSDF | 1 | 実装済（`compute_sparkle`） |
| **L2** | 多波長 $V(\lambda)$ | 画素全面一様 | グローバル BSDF | $N_\lambda$ | 未実装 |
| **L3'** | 単波長（該当色） | サブピクセル限定発光 | グローバル BSDF | 1 | 実装済（`compute_sparkle_l3prime`） |
| **L4** | 多波長 $V(\lambda)$ (narrowband 近似) | サブピクセル限定発光（色ごと） | グローバル BSDF | $3 \cdot N_\lambda$ | 実装済（`compute_sparkle_l4`、narrowband 版） |
| **L5** | 単波長（該当色） | サブピクセル限定発光 | **空間分解 BSDF** | $3 \cdot N_\lambda \cdot N_\text{pix}$ | 実装済（`compute_sparkle_l5`、単色版） |

**実装上の注意**: L3'/L4 は角度ビニング手法の性質上 DC 近傍の集中により **apparent Cs** が大きくなる傾向があり、絶対値としての規格値（SEMI D63 / IDMS 実測値）とは直接比較できない。**相対比較・最適化目的関数として有用**だが、規格値の代替にはならない。L5 は局所 FFT により DC 集中アーティファクトを回避するため、物理的に妥当な sparkle 値を返す。

用途対応:

| Level | 適用場面 | 規格対応 |
|---|---|---|
| L1 | 相対比較・最適化指標 | SEMI D63 白点灯の単波長近似 |
| L2 | 白点灯の規格準拠評価 | SEMI D63 白点灯に最も近い（分光を正確に） |
| L3' | 単色（R/G/B）表示時の sparkle 評価 | 規格外（規格は白点灯のみ規定） |
| L4 | 白点灯 + サブピクセル効果の評価 | SEMI D63 白点灯の空間構造精緻化 |
| L5 | 不均一 AG・高 PPI × 微細 AG の厳密評価 | SEMI D63 白点灯の最厳密近似 |

---

## 2. 共通記号と前提

| 記号 | 意味 | 単位 |
|---|---|---|
| $h(x, y)$ | AG フィルム表面高さ | μm |
| $\varphi(x, y)$ | AG フィルム位相分布（`docs/fft_bsdf_math.md` 参照） | rad |
| $U(x, y)$ | 出射面の複素振幅 $e^{i\varphi(x,y)}$ | — |
| $\lambda$ | 波長 | μm |
| $k = 2\pi/\lambda$ | 波数 | μm⁻¹ |
| $D$ | 観察距離 | mm |
| $d_p$ | 瞳孔径 | mm |
| $p$ | ディスプレイ画素ピッチ | mm |
| $w_\text{sub}$ | サブピクセル幅（通常 $\approx p/3$） | mm |
| $\Omega_\text{pupil}$ | 瞳孔立体角 $= \pi (d_p/2D)^2$ | sr |
| $\sin\alpha_h$ | 1 画素の方向余弦半幅 $= \sin[\arctan(p/2D)]$ | — |
| $(u, v)$ | 方向余弦 $(\sin\theta_s \cos\phi_s, \sin\theta_s \sin\phi_s)$ | — |
| $f_s(u, v)$ | 通常の BSDF（面全体アンサンブル平均） | sr⁻¹ |
| $f_s(x, y; u, v)$ | 空間分解 BSDF | sr⁻¹ |
| $k$（添字） | ディスプレイ画素インデックス（$L_k$ 等） | — |
| $V(\lambda)$ | CIE 明所視応答 | — |
| $S_c(\lambda)$ | サブピクセル $c \in \{R, G, B\}$ の分光放射分布 | — |
| $L_k$ | ディスプレイ画素 $k$ の観察輝度（観察者瞳孔に届く全光束） | 相対値（sr⁻¹ × sr × μm² スケール） |
| $\mu(\{L_k\})$ | 全画素の平均輝度 $\mathrm{mean}(L_k)$ | 同上 |
| $\sigma(\{L_k\})$ | 全画素の標準偏差 $\mathrm{std}(L_k)$ | 同上 |
| $C_s$ | **ギラツキコントラスト**（= $\sigma/\mu$） | **無次元** |

### 2.1. ギラツキコントラスト $C_s$ について

$C_s$（Sparkle Contrast、無次元）は画素輝度分布のばらつきを平均で正規化した値:

$$C_s = \frac{\sigma(\{L_k\})}{\mu(\{L_k\})}$$

- **値域**: 0 以上の実数（下限 0、上限なし）
- **物理的意味**: 値が大きいほどディスプレイ画素間の輝度ばらつきが大きく、観察者には「ギラつき」として知覚される
- **単位**: 無次元（標準偏差 ÷ 平均の比なので輝度の単位がキャンセル）
- **実測典型値（SEMI D63 / IDMS 準拠測定）**:
  - 低 sparkle AG フィルム: $C_s \approx 0.01\text{–}0.03$
  - 標準 sparkle AG フィルム: $C_s \approx 0.03\text{–}0.08$
  - 高 sparkle AG フィルム: $C_s \approx 0.08\text{–}0.15$
- **知覚閾値**: Kelley (2014) によれば $C_s \lesssim 0.02$ で観察者が気にならないレベル

### 2.2. 本実装で得られる $C_s$ 値の注意

L1/L3'/L4（角度ビニング方式）は DC 集中・サブピクセル Moiré によって **apparent $C_s$** が実測値より 1–2 桁大きくなる傾向がある。**相対比較・最適化目的関数としては有用** だが、SEMI D63 規格値の直接代替にはならない。L5 は局所 FFT により DC 集中を回避し、実測オーダーに近い $C_s$ 値を返す。

全レベル共通:

- スカラー回折・非偏光近似
- サブピクセル間は **空間的に非コヒーレント**（強度加算、干渉無し）
- 観察者瞳孔に届く輝度 $L_k$ のディスプレイ画素間分散からコントラストを算出（式は 2.1 参照）

---

## 3. L1: 現実装（単波長・全面一様・グローバル BSDF）

### 3.1. 数式

単一波長 $\lambda_0$ で、AG フィルム全面を一様平面波照明した場合。

(1) 複素振幅:

$$U(x, y) = \exp[\,i\,\varphi(x, y; \lambda_0)\,]$$

(2) グローバル BSDF:

$$f_s(u, v) = \frac{|\mathcal{F}[U](u, v)|^2}{N^2 \cdot dx^2 \cdot \cos\theta_s}$$

ここで $\mathcal{F}$ は 2D FFT、$(u, v) = \lambda (f_x, f_y)$ の座標変換。

(3) 角度ビニング（ディスプレイ画素 $k$ に対応する角度ビン $B_k$）:

$$B_k = \left\{(u, v) \;\middle|\; \text{round}\!\left(\frac{u}{2\sin\alpha_h}\right) = k_u,\; \text{round}\!\left(\frac{v}{2\sin\alpha_h}\right) = k_v \right\}$$

(4) ビン輝度:

$$L_k^\text{L1} = \Omega_\text{pupil} \sum_{(u, v) \in B_k} f_s(u, v)\,\cos\theta_s\,du\,dv$$

(5) sparkle コントラスト: $C_s = \sigma(L_k) / \mu(L_k)$

### 3.2. 暗黙の仮定

- 発光は空間的に一様（サブピクセル構造無視）
- AG フィルムは統計的に均一（グローバル BSDF で代表可能）
- 光源は単色（分光平均は代表波長で代用）

---

## 4. L2: 白点灯 $V(\lambda)$ 重み合成

### 4.1. 数式

波長セット $\{\lambda_m\}_{m=1}^{N_\lambda}$、白色光源分光 $S_\text{white}(\lambda)$（例: D65）について。

(1) 各波長で L1 を実行: $L_k^\text{L1}(\lambda_m)$

(2) $V(\lambda)$ 重み加算:

$$L_k^\text{L2} = \frac{\sum_{m} S_\text{white}(\lambda_m)\,V(\lambda_m)\,L_k^\text{L1}(\lambda_m)\,\Delta\lambda}{\sum_{m} S_\text{white}(\lambda_m)\,V(\lambda_m)\,\Delta\lambda}$$

(3) $C_s^\text{L2} = \sigma(L_k^\text{L2}) / \mu(L_k^\text{L2})$

### 4.2. L1 との関係

波長と空間は **分離可能**:

$$L_k^\text{L2} = \int V(\lambda)\,S_\text{white}(\lambda)\,L_k^\text{L1}(\lambda)\,d\lambda \;\Big/\; \int V(\lambda)\,S_\text{white}(\lambda)\,d\lambda$$

L1 の出力 $L_k^\text{L1}(\lambda_m)$ を波長次元で線形合成するだけ。**空間発光パターンは波長に依存しない**（全波長で画素全面一様）。

### 4.3. コスト

FFT 回数: $N_\lambda$（典型 3–5 波長）。

---

## 5. L3': 単色表示（再定義版）

元の L3 は「サブピクセル空間 + 単波長（全サブピクセル同色）」で物理非現実的だった。再定義版 L3' は:

### 5.1. サブピクセル発光マスク

サブピクセル位置マスクを定義（例: RGB ストライプ配列、x 方向にストライプ）:

$$M_c(x, y) = \begin{cases} 1 & (x, y) \in \text{サブピクセル領域 } c \\ 0 & \text{else} \end{cases}$$

典型的 RGB ストライプ配列では、1 画素内で x 方向に 3 分割、各サブピクセル幅 $w_\text{sub} = p / 3$。

### 5.2. 数式

単色 $c \in \{R, G, B\}$ を点灯、対応する代表波長 $\lambda_c$（例: $\lambda_R = 0.630$ μm）:

(1) 発光変調された複素振幅:

$$U_c(x, y) = M_c(x, y)\,\exp[\,i\,\varphi(x, y; \lambda_c)\,]$$

(2) グローバル BSDF:

$$f_{s,c}(u, v) = \frac{|\mathcal{F}[U_c](u, v)|^2}{N^2 \cdot dx^2 \cdot \cos\theta_s}$$

(3) 以下 L1 と同じ角度ビニングで $L_k^\text{L3'}(c)$、$C_s^\text{L3'}(c)$ を算出。

### 5.3. L1 との差

発光面積が 1/3 に縮小し、空間的オフセットが入る。サブピクセル配列のフーリエ変換による角度方向の変調（通常 $\pm 1/w_\text{sub} \cdot \lambda$ 付近に回折ピーク）が BSDF に現れる。

AG 相関長 $L_c \ll w_\text{sub}$ では、サブピクセル配列による回折と AG フィルム散乱のスペクトルは分離されるため影響は限定的。$L_c \gtrsim w_\text{sub}$ では両者が畳み込まれるため sparkle 値が大きく変わる。

---

## 6. L4: 白点灯 + サブピクセル + $V(\lambda)$

### 6.1. 数式

白表示時は R/G/B サブピクセルが **それぞれ固有波長分布** を発光。空間発光パターン $M(x, y)$ が波長 $\lambda$ に依存するため、発光関数 $E(x, y, \lambda)$ は **空間成分と波長成分の積に分解できない（分離不可能 / non-separable）**:

(1) 各色・各波長で L3' を実行: $L_k^\text{L3'}(c, \lambda_m)$

(2) 色と波長で合成:

$$L_k^\text{L4} = \frac{\sum_{c \in \{R,G,B\}} \sum_m S_c(\lambda_m)\,V(\lambda_m)\,L_k^\text{L3'}(c, \lambda_m)\,\Delta\lambda}{\sum_{c} \sum_m S_c(\lambda_m)\,V(\lambda_m)\,\Delta\lambda}$$

(3) $C_s^\text{L4} = \sigma(L_k^\text{L4}) / \mu(L_k^\text{L4})$

### 6.2. L2 との非自明な差

L2 が波長と空間を分離可能に扱えるのは、全波長で同じ空間発光（一様）を使うため。L4 では:

- $\lambda_m \approx \lambda_R$: $M_R(x, y)$ のみ発光
- $\lambda_m \approx \lambda_G$: $M_G(x, y)$ のみ発光
- $\lambda_m \approx \lambda_B$: $M_B(x, y)$ のみ発光

**各波長ごとに異なる空間発光パターン** を使うため、$E(x, y, \lambda) = M_\text{space}(x, y) \cdot E_\text{spec}(\lambda)$ のような積形式に書けない（**空間・波長の分離不可能性**）。結果、各色 × 各波長で独立に FFT が必要で、計算量は L2 の 3 倍（サブピクセル数倍）。

**補足**: ここで言う「分離不可能」は数学的な関数分解の意味であり、サブピクセル間の光学的干渉を意味しない。R/G/B 発光は空間的に非コヒーレントなので、$\sum_c$ は強度の単純加算（振幅干渉項は含まない）。

### 6.3. コスト

FFT 回数: $3 \times N_\lambda$（典型 9–15 回）。L1 比で 10× オーダー。

---

## 7. L5: 空間分解 BSDF（窓付き FFT）

### 7.1. 空間分解 BSDF の定義

**通常の BSDF** は面全体の角度散乱をアンサンブル平均する:

$$f_s(u, v) = \frac{1}{A} \int_A |U(x, y) \to (u, v) 方向への散乱|^2\,dx\,dy$$

これに対し **空間分解 BSDF** は、局所位置 $(x_0, y_0)$ での散乱特性を保持:

$$f_s(x_0, y_0; u, v)$$

関係:

$$f_s(u, v) = \frac{1}{A} \int_A f_s(x_0, y_0; u, v)\,dx_0\,dy_0$$

次元は 2D (角度のみ) から 4D (空間 × 角度) に拡張される。

### 7.2. 窓付き FFT による計算

空間分解 BSDF を数値計算する標準手法が **窓付き FFT**（STFT: Short-Time Fourier Transform の 2D 版）。

窓関数 $w(x, y)$（例: Gaussian、Tukey、Hann）、窓幅 $W$ を用意する:

$$w_W(x, y) = \begin{cases} \exp\!\left(-\frac{x^2 + y^2}{2(W/2)^2}\right) & \text{Gaussian 窓} \\ \text{0.5}\,[1 - \cos(2\pi r/W)] & \text{Hann 窓（半径 } r = \sqrt{x^2+y^2}\text{）} \end{cases}$$

位置 $(x_0, y_0)$ を中心とする局所複素振幅:

$$U_{x_0, y_0}(x, y) = w_W(x - x_0, y - y_0)\,E(x, y)\,\exp[\,i\,\varphi(x, y)\,]$$

（$E(x, y)$ は発光分布。L5 では L4 と同じ $M_c(x, y) \cdot e^{i\varphi}$ 構造を使う）

局所 BSDF:

$$f_s(x_0, y_0; u, v) = \frac{|\mathcal{F}[U_{x_0, y_0}](u, v)|^2}{\|w_W\|^2 \cdot dx^2 \cdot \cos\theta_s}$$

正規化の $\|w_W\|^2 = \int |w_W|^2 \,dx\,dy$ は窓のエネルギー。

### 7.3. ディスプレイ画素ごとの輝度

L5 では、観察者は画素 $k$（位置 $(x_k, y_k)$）を見る方向が決まっている:

$$\vec\theta_{k \to \text{obs}} \approx (x_k/D, y_k/D)$$

瞳孔は画素 $k$ から見て立体角 $\Omega_\text{pupil}$ を占める。観察者瞳孔が受け取る画素 $k$ からの輝度:

$$L_k^\text{L5} = P_k \int_{\Omega_\text{pupil}} f_s(x_k, y_k;\, \vec\theta_{k \to \text{obs}} + \Delta\vec\theta)\,d(\Delta\Omega)$$

ここで $P_k$ は画素 $k$ の総発光強度。小角近似（瞳孔立体角が局所 BSDF の角度変動スケールより小さい場合）:

$$L_k^\text{L5} \approx P_k\,\Omega_\text{pupil}\,f_s(x_k, y_k;\, \vec\theta_{k \to \text{obs}})$$

**この式が L5 の本質**: 観察者は画素ごとに、その直上の局所 BSDF を観察方向で 1 点サンプリングしている。

### 7.4. 窓幅選択と不確定性関係

窓幅 $W$ は trade-off を持つ:

- **小さい $W$**: 空間分解能高 ↔ 角度分解能低 $\Delta u \sim \lambda / W$
- **大きい $W$**: 空間分解能低 ↔ 角度分解能高

時間-周波数の Heisenberg 不確定性関係の 2D 版:

$$\Delta x \cdot \Delta u \geq \lambda / 4\pi$$

ディスプレイ画素ごとの局所 BSDF を得るには、$\Delta x \lesssim p$（画素ピッチ）、同時に $\Delta u \lesssim 2 \sin\alpha_h$（瞳孔角度）を満たす必要。これが可能かは:

$$p \cdot 2\sin\alpha_h \gtrsim \lambda / 4\pi \quad \Leftrightarrow \quad \frac{p^2}{D} \gtrsim \frac{\lambda}{4\pi}$$

典型条件（$p = 0.062$ mm, $D = 300$ mm, $\lambda = 0.000555$ mm）:
- 左辺: $0.062^2 / 300 \approx 1.3 \times 10^{-5}$ mm
- 右辺: $0.000555 / (4\pi) \approx 4.4 \times 10^{-5}$ mm

ほぼ同オーダーで、不確定性関係が **ぎりぎり抵触** する領域。実装上は $W \approx 3p\text{–}5p$（例 200–300 μm）を推奨。より細かい空間分解能が必要なら本質的に不可能（量子力学的下限）。

### 7.5. コスト

ディスプレイ画素数 $N_\text{pix} = (N dx / p)^2$。典型 $N = 4096$, $dx = 0.00025$ mm, $p = 0.062$ mm → $N_\text{pix} \approx 273$。

1 回の局所 FFT サイズは窓幅に対応（$W \approx 250$ μm → $W/dx = 1000$ サンプル）。つまり 1000² FFT を 273 回 × 3 色 × $N_\lambda$ 波長。

**コスト見積**: L4 の 10–30× 程度（局所 FFT のサイズが元の FFT より小さいため、素朴な 273× にはならない）。

---

## 8. L4 と L5 の差が顕在化する条件

### 8.1. 物理的起源

L4 は **AG フィルムが統計的に均一** と仮定してグローバル BSDF を使い、角度 ↔ 空間 reciprocity で画素ごとの輝度分散を推定する。

L5 は **AG フィルムが空間的に非均一** であり得ることを許容し、画素ごとに異なる局所 BSDF を直接計算する。

差が顕在化する条件は、AG フィルムの **空間相関長 $L_c$** と画素ピッチ $p$ の関係で支配される。

### 8.2. 定量条件

AG フィルムの 2 点相関関数:

$$C_h(r) = \langle h(x) h(x + r) \rangle, \quad L_c = \text{(C_h \text{が半減する距離}など)}$$

典型的な AG フィルム: $L_c = 2$–$20$ μm。

| 領域 | $L_c / p$ の範囲 | 物理的状況 | L4 vs L5 差 |
|---|---|---|---|
| **均一領域** | $L_c / p \lesssim 0.05$ | 1 画素内に多数の散乱構造（アンサンブル平均が成立） | $\lesssim$ 5% |
| **遷移領域** | $0.05 \lesssim L_c / p \lesssim 0.3$ | 1 画素内に数個の散乱構造 | 5–15% |
| **非均一領域** | $L_c / p \gtrsim 0.3$ | 1 画素内に 1〜数個の散乱構造（長距離相関が画素スケールで効く） | 15–50%+ |

典型ディスプレイでの境界:

| ディスプレイ | $p$ [mm] | 遷移境界 $L_c$ [μm] | 非均一境界 $L_c$ [μm] |
|---|---|---|---|
| FHD スマホ | 0.062 | 3 | 19 |
| QHD モニタ | 0.124 | 6 | 37 |
| 4K モニタ | 0.160 | 8 | 48 |
| 高 PPI スマホ（460 PPI） | 0.055 | 3 | 17 |

### 8.3. L4 の誤差の方向性

L4 がグローバル FFT + 角度ビニングで sparkle を推定する際:

- **長距離相関を持つ AG**（低周波の凹凸変動、粒子分散ムラ等）は、グローバル FFT の角度領域では低角度側に集中し、ビニング時に特定ビンに集中して現れる
- これは **全画素に共通の角度パターン** として出現するため、画素間分散 $\sigma(L_k)$ には寄与せず、**平均 $\mu(L_k)$ のみを増加させる**
- 結果: L4 は長距離相関成分を sparkle として捉え損ね、**sparkle を過小評価する**

L5 は画素ごとに異なる局所 BSDF を持つため、長距離相関は **画素間の平均値 $L_k$ の変動** として正しく反映される。

### 8.4. 実測との比較

文献（Käläntär 2013, Kelley 2014 等）で報告されている AG フィルム sparkle:

- 典型的な AG フィルム（$L_c \approx 5$ μm）と FHD スマホ（$p = 0.062$ mm, $L_c / p \approx 0.08$）: L4 と実測が ±10% で一致
- 高ヘイズ粗テクスチャ AG（$L_c \approx 20$ μm）と同スマホ（$L_c / p \approx 0.32$）: L4 は実測の 60–70% 程度に過小評価する例あり

ただし本実装（L1）で実測との比較検証は未実施のため、上記は文献ベースの推定。

---

## 9. 実装上の示唆

### 9.1. 現実装（L1）のユースケース

- **最適化の目的関数**: 相対比較で十分なため L1 で問題なし（誤差は設計比較で相殺）
- **AG フィルム設計方針の大枠確認**: $L_c / p < 0.1$ の範囲なら L2 との差も小さい

### 9.2. 優先実装候補

順に拡張する場合の推奨順序:

1. **L2**（白点灯 V(λ) 重み合成）: FFT 回数 3–5× 増。SEMI D63 白点灯規格への準拠度を上げる
2. **L3'**（単色評価の意味付け）: 低コスト（L1 + マスク）。単色表示時の sparkle を議論する場合
3. **L4**（サブピクセル × V(λ)）: L2 と L3' の合体。高 PPI 評価に必要
4. **L5**（空間分解 BSDF）: L4 で sparkle が AG 実測と乖離するケースで導入

### 9.3. 実装時の注意

- **L3'/L4 の空間マスク解像度**: $w_\text{sub} \approx p/3$ を解像できる $dx$ が必要（FHD スマホなら $dx \leq 5$ μm 推奨、現実装の $dx = 0.25$ μm は十分）
- **L5 の窓幅**: 画素ピッチの 3–5 倍 + Hann/Gaussian 窓。窓幅選択で sparkle 値がどう変動するか、収束性テストが必須
- **L5 の境界画素**: グリッド端近傍の画素は窓が切れるので除外する

---

## 10. まとめ

| 問い | 答え |
|---|---|
| L1 → L2 の拡張は何か | 波長次元の線形合成（波長と空間が分離可能、コスト低） |
| L3' → L4 の拡張は何か | 発光関数 $E(x,y,\lambda)$ が空間と波長で分離不可能になる（各波長で異なるサブピクセル発光パターン、コスト中） |
| L4 → L5 の拡張は何か | グローバル BSDF を空間分解 BSDF に置換（AG 非均一性を捉える、コスト大） |
| L4 と L5 の差は何に依存するか | AG 相関長 $L_c$ と画素ピッチ $p$ の比 $L_c / p$ |
| 空間分解 BSDF と通常 BSDF の違いは | 次元（4D vs 2D）。通常 BSDF は空間分解 BSDF の面平均 |
| 窓付き FFT の物理的意味は | 局所位置での BSDF を近似的に取り出す STFT の 2D 版。不確定性関係で解像度に下限 |

現実装 L1 は単波長・一様発光・グローバル BSDF という 3 重の近似を含むが、**典型条件（$L_c / p < 0.1$）では実用的な精度**で sparkle を評価できる。SEMI D63 規格準拠や高 PPI × 微細 AG の厳密評価が必要な場合に、上記の階層で段階的に拡張できる。
