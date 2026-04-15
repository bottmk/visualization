# 光学散乱（BSDF）統合シミュレーション・最適化プログラム 仕様書（最新改訂版）

## 0. 共通規約

### 0.1. 物理単位系
本プログラム内部で使用する物理単位はすべて **μm（マイクロメートル）** に統一する。

| 物理量 | 単位 |
|---|---|
| 高さ $h$、粗さ $Rq$、相関長 $L_c$、ピクセルサイズ | μm |
| 波長 $\lambda$ | μm（例: 0.55μm） |
| 空間周波数 $f$ | μm⁻¹ |
| BSDF | sr⁻¹ |
| 観察距離、画素ピッチ | mm |

外部入力（実測データ等）の単位が異なる場合は、取り込み時に μm へ変換する（例: 波長 nm → μm は ÷1000）。

### 0.2. 設定ファイル
すべてのパラメータは **YAML 形式**の設定ファイル（`config.yaml`）で管理する。プリセット値と個別数値指定の両方に対応し、数値が明示された場合はプリセットより優先する。

```yaml
# config.yaml 全体構造（概要）
simulation:       # 光学条件
surface:          # 表面形状モデル
adding_doubling:  # 多層BSDF合成
error_metrics:    # 誤差計算
sparkle:          # ギラツキ評価
optuna:           # 最適化
dynamicmap:       # プレビュー設定
```

### 0.3. CLI インターフェース
```bash
bsdf simulate  --config config.yaml                  # シミュレーション単体実行
bsdf optimize  --config config.yaml --trials 100     # Optuna 最適化実行
bsdf visualize --run-id <mlflow_run_id>              # 結果可視化
bsdf report    --run-ids <id1>,<id2>                 # 複数 Run の比較レポート生成
```

---

## 1. システムの目的
本プログラムは、AGフィルム等の表面形状データ（数理モデルによる生成、または実測データ）を入力とし、FFTベースのスカラー回折理論およびPSDベースの散乱モデルを用いてBSDF（双方向散乱分布関数）をシミュレーションする。さらに「多層BSDF合成」を用いてフィルム全体の光学特性を再現する。
最適化エンジンに **Optuna**、実験管理・データ保管に **MLflow** を用い、シミュレーション結果と実測データとの誤差を最小化する表面形状パラメータを自動探索する。結果は **HoloViews** を用いてインタラクティブなHTMLグラフとして可視化・一元管理する。

---

## 2. 入力仕様：表面形状モデル（プラグイン・ポリモーフィズム設計）

### 2.1. HeightMap データクラス
すべての形状モデルは `HeightMap` データクラスを返す。スケール情報を配列と常に一体化させることで、下流処理での単位変換ミスを防ぐ。

```python
@dataclass
class HeightMap:
    data: np.ndarray      # shape: (N, N)、単位: μm
    pixel_size_um: float  # ピクセルサイズ [μm]

    @property
    def physical_size_um(self) -> float:
        return self.data.shape[0] * self.pixel_size_um

    @property
    def grid_size(self) -> int:
        return self.data.shape[0]
```

### 2.2. BaseSurfaceModel
すべての形状モデルは基底クラス `BaseSurfaceModel` を継承し、以下の共通インターフェースを実装する。

```python
class BaseSurfaceModel:
    def __init__(self, grid_size: int = 4096, pixel_size_um: float = 0.25):
        self.grid_size = grid_size          # 本計算用グリッドサイズ（デフォルト: 4096）
        self.pixel_size_um = pixel_size_um  # ピクセルサイズ [μm]（デフォルト: 0.25μm）
        # 物理サイズ = 4096 × 0.25 = 1024μm（デフォルト）

    @property
    def physical_size_um(self) -> float:
        return self.grid_size * self.pixel_size_um

    def get_height_map(self) -> HeightMap:
        """本計算用（grid_size=4096, pixel_size_um=0.25 → 物理サイズ 1024μm）"""
        raise NotImplementedError

    def get_preview_height_map(
        self,
        mode: Literal['reduced_area', 'reduced_resolution'],
        preview_grid_size: int = 512,
    ) -> HeightMap:
        """
        DynamicMap用プレビュー。2つのモードを提供する。

        'reduced_area'（アプローチA）:
            pixel_size_um を固定し grid_size を縮小する。
            物理サイズは縮小されるが（例: 128μm）、散乱角の上限は最大 90° まで維持される。
            ヘイズ（2.5°以上）・広角散乱の定性確認に適する。

        'reduced_resolution'（アプローチB）:
            physical_size_um を固定し pixel_size_um を拡大する。
            pixel_size_um = physical_size_um / preview_grid_size（例: 2.0μm）
            散乱角の上限が制限される（λ=0.55μm の場合 約 7.9°）ため、
            ヘイズ計算には使用不可。BSDFピーク位置・形状の確認に適する。

        注意: DynamicMap のデフォルトは 'reduced_area' を推奨する。
        """
        raise NotImplementedError
```

**【プレビューモード比較】**

| 比較項目 | reduced_area（A） | reduced_resolution（B） |
|---|---|---|
| 固定値 | pixel_size_um=0.25μm | physical_size_um=1024μm |
| 散乱角上限 | 最大 90°（ヘイズ確認可） | 約 7.9°（ヘイズ確認不可） |
| 角度分解能 | 粗くなる | 維持される |
| 用途 | ヘイズ・広角散乱の確認 | ピーク位置・DOI の確認 |

**【拡張性（プラグインアーキテクチャ）】**
指定フォルダ（`custom_surfaces/`）に新しい形状モデルの `.py` ファイルを配置するだけで、メインプログラムを改修することなくシステムが動的に読み込み、Optunaの探索候補として自動登録する。

### 2.3. 標準実装する形状モデル
- **ランダム粗面モデル（`RandomRoughSurface`）**
  - パラメータ: RMS粗さ（$Rq$ [μm]）、相関長（$L_c$ [μm]）、フラクタル次元
- **球面アレイモデル（`SphericalArraySurface`）**
  - パラメータ: レンズ曲率半径（$R$ [μm]）、ベース高さ [μm]
  - 配置アルゴリズム（外部関数注入）: `Grid`, `Hexagonal`, `Random`, `PoissonDisk`
  - 重なり（干渉）処理: `Maximum`（Max値採用）, `Additive`（加算）
- **実測データモデル（`MeasuredSurface`）**
  - 処理: 欠損値補間、指定グリッドサイズへのリサンプリング、レベリング（傾き・うねり成分の除去）
  - 汎用ローダー `from_csv()`: カンマ区切り、`height_unit`（`'um'` / `'nm'` / `'m'`）、`skiprows` 対応
  - 装置固有ローダー（プラグイン方式）: `MeasuredSurface` のサブクラスを `custom_surfaces/<name>.py` に置くと `load_plugins()` が自動登録する。`config.yaml` で `model: '<ClassName>'` と指定するだけで使用可能。
  - 実装例:
    - `custom_surfaces/device_xyz.py` → `DeviceXyzSurface`（タブ区切り・5行ヘッダ・nm 単位）
    - `custom_surfaces/device_vk6.py` → `DeviceVk6Surface`（Keyence VK-X シリーズ、Shift-JIS、ピクセルサイズ・単位をヘッダから自動取得）
  - サンプルファイル・設定: `sample_inputs/` フォルダに格納

---

## 3. 光学シミュレーション処理（計算モジュール）
光学条件（波長 $\lambda$ [μm]、入射の天頂角 $\theta_i$・方位角 $\phi_i$、散乱の天頂角 $\theta_s$・方位角 $\phi_s$、媒質の屈折率 $n$）および偏光状態（$S, P, \text{Unpolarized}$）は、形状パラメータとは分離し計算モジュール側で一括管理する。

### 3.1. 測定座標系とBRDF/BTDFの自動判定
実測データの座標系定義に基づき、プログラム内で反射と透過の物理モデルを自動的に切り替える。
- **座標系の定義**: 表面の法線を $0°$、裏面の法線を $180°$ とし、散乱（受光）は常に表面側の半球（$0° \le \theta_s < 90°$）で行われるものとする。
- **BRDFモード（表面入射）**: 入射角 $\theta_i < 90°$ の場合、光が表面で反射するBRDFとして扱い、反射の位相変化を適用する。
- **BTDFモード（裏面入射）**: 入射角 $\theta_i > 90°$ の場合、裏面からの透過光（BTDF）として扱う。入射角を表面側の座標系に換算（$|180° - \theta_i|$）した上で、透過の位相変化を適用する。
- **境界条件**: $\theta_i = 90°$ は未定義とし、エラーを発生させる。

### 3.2. FFT法（スカラー回折理論 / 波動伝搬法）ルート

#### 位相変換式
高さ $h(x,y)$ [μm] から位相分布 $\phi(x,y)$ を計算し、複素振幅 $U(x,y) = \exp(i\phi(x,y))$ を生成する。**FFT法はスカラー近似であり、偏光依存性は扱わない**（偏光依存BSDFはPSD法のQ因子に委ねる）。

**BRDFモード（反射）:**
$$\phi(x,y) = \frac{4\pi}{\lambda} n_1 h(x,y) \cos\theta_i$$

**BTDFモード（透過）:**
$$\phi(x,y) = \frac{2\pi}{\lambda} \left( n_2 \cos\theta_t - n_1 \cos\theta_i \right) h(x,y)$$

ここで $\theta_t$ はスネルの法則 $n_1 \sin\theta_i = n_2 \sin\theta_t$ から導出する。

**斜入射時のx方向傾き項（位相シフト）:**
$$\phi_{\text{tilt}}(x) = \frac{2\pi}{\lambda} n_1 \sin\theta_i \cdot x$$

#### 処理フロー
1. **位相変換**: 上記の式に従い $U(x,y)$ を生成する。
2. **フーリエ変換**: 2次元FFTを実行し、遠方界の光強度分布 $I(f_x, f_y)$ を算出する。
3. **余弦空間（方向余弦）でのマッピングとBSDF導出**:
   - FFTで得られた空間周波数 $(f_x, f_y)$ を余弦空間（$u = \sin\theta_s \cos\phi_s,\ v = \sin\theta_s \sin\phi_s$）にマッピングする。受光角 $\theta_s$ が常に $0° \sim 90°$ に固定されるため、同一のUV座標上でシミュレーション結果と実測データを直接補間・比較できる。
   - 斜入射の場合は、角度スペクトルの中心を $\sin\theta_i / \lambda$ だけシフトさせることで、垂直入射時と同一のアルゴリズムで処理する（シフト不変性の活用）。
   - 実測データとシミュレーション結果を比較する際も、このUV座標上で補間（サンプリング）を行うことで、天頂（極）付近での座標の歪みを防ぎ、高精度なマッチングを実現する。
4. **警告システム**: 偏光（S/P）が指定され、かつ広角（約30°以上）への散乱を計算する場合、スカラー近似の限界を超えているため誤差が大きくなる旨の警告をロギングする。

### 3.3. PSD法（Rayleigh-Rice理論近似）ルート
1. $h(x,y)$ の2次元FFTからパワースペクトル密度 $PSD(f_x, f_y)$ を算出する。
2. BRDFモード・BTDFモードおよび偏光状態（S/P/Unpolarized）に応じた偏光因子 $Q$ を算出し、BSDFを導出する。$Q$ の完全形はAppendix Aに記載する。

$$\text{BSDF}(\theta_i, \theta_s) = \frac{16\pi^2}{\lambda^4} \cos\theta_i \cos^2\theta_s \cdot Q \cdot PSD(f_x, f_y)$$

3. ここでも余弦空間を介して空間周波数と散乱角の変換を行う。
4. `approx_mode: bool = False` パラメータにより、計算速度が必要な場合は簡略形に切り替え可能（詳細はAppendix A）。

### 3.4. 対数（Log）スケールでの誤差計算と評価

BSDFの値は正反射のピーク付近と広角の散乱裾野とで数桁にわたって変化するため、**「実測値の $\log_{10}(\text{BSDF})$」と「計算値の $\log_{10}(\text{BSDF})$」の差分（Log-RMSE）** を最適化の評価値とする。

**処理ロジック:**
```python
# config.yaml で設定
# error_metrics:
#   bsdf_floor: 1.0e-6   # sr⁻¹ 装置のノイズフロアに合わせて変更

# 実測データが有効な点のみを誤差計算対象とする（マスク処理）
mask = measured > bsdf_floor

log_rmse = np.sqrt(np.mean(
    (np.log10(np.maximum(simulated[mask], bsdf_floor))
   - np.log10(measured[mask])) ** 2
))
```

- `bsdf_floor` のデフォルト値は `1e-6 sr⁻¹`（散乱計の典型的なダイナミックレンジ 120dB に対応）。
- 装置のノイズフロアに合わせて `config.yaml` で変更可能とする。
- マスク範囲を試行間で統一するため、`bsdf_floor` は最適化セッション中に固定する。

---

## 4. 多層BSDFの合成（Adding-Doubling法）
実際のAGフィルム構造（表面AG凹凸 ＋ 内部ヘイズ ＋ 裏面平滑）をシミュレーションするため、各層のBSDFを合成するパイプラインを実装する。

### 4.1. 離散化パラメータ
散乱行列を数値計算するための方向基底の離散数を以下の設定で管理する。

```yaml
adding_doubling:
  precision: 'standard'   # 'fast' / 'standard' / 'high'
  n_theta: null           # null のときはプリセット値を使用（数値指定でオーバーライド）
  m_phi: null             # null のときはプリセット値を使用（数値指定でオーバーライド）

# プリセット定義（測定装置: 天頂1°刻み・方位10°刻みを基準に設定）
# fast:     n_theta=32,  m_phi=8   ← DynamicMapプレビュー用
# standard: n_theta=128, m_phi=18  ← 装置分解能（天頂1°・方位10°）に対応
# high:     n_theta=256, m_phi=36  ← 2倍オーバーサンプリング
```

- `n_theta`: 仰角方向のガウス求積点数
- `m_phi`: 方位角のフーリエ展開モード数（装置の方位10°刻みに対するナイキスト限界: 18モード）
- 数値が明示された場合はプリセットより優先する。両方未指定の場合はエラーとする。
- Adding-Doubling の出力は装置の測定グリッド（天頂1°、方位10°）へ補間してから実測データと比較する。

### 4.2. 処理フロー
1. **方向基底への展開と離散化**: 計算されたBSDFを方位角のフーリエ級数と仰角のコサインを用いた方向基底に展開し、散乱特性を行列化する。
2. **層ごとの散乱行列計算**:
   - 境界層（表面・裏面）: FFT法等で求めたBSDFから散乱行列を抽出する。
   - 内部媒質層: 内部ヘイズ（微粒子）が存在する場合、Henyey-Greenstein（HG）位相関数等を用いてボリューム散乱行列をモデリングする。
3. **Adding-Doubling法による合成**: 放射伝達理論における「加算方程式」を用い、表面・内部・裏面の散乱行列を反復的に倍増（Doubling）させながら結合（Adding）し、多重反射・多重透過を考慮したフィルム全体のトータルBSDFを計算する。
4. **スパース性の活用**: データサイズを抑えつつ高周波な散乱特性を高速合成するため、行列のスパース性（疎性）を活用する。
5. **2次元BSDFへの展開（`to_bsdf_2d`）**: 合成散乱行列から法線入射の1次元BSDFプロファイルを取り出し、参照グリッド（FFT/PSD法の UV グリッド）上に `scipy.interpolate.interp1d` で補間展開して2次元BSDFを生成する。これにより「MultiLayer」メソッドとして Parquet に保存し、FFT/PSD 結果と同一グリッド上で比較可能にする。

---

## 5. 評価指標（光学・形状指標）と評価アルゴリズム
シミュレーション結果（BSDFや高さマップ）から、最適化の目的関数となる各種指標を算出する。

**【拡張性（プラグインアーキテクチャ）】**
指定フォルダ（`custom_metrics/`）に評価指標を計算する `.py` ファイルを追加するだけで、新しい光学指標やJIS/ISO規格の形状パラメータをシステムに動的に組み込み、Optunaの多目的最適化ターゲットとして選択可能にする。

### 5.1. 標準実装する評価指標
- **光学指標（BSDF・光伝搬から算出）**
  - ヘイズ（Haze）: 2.5°以上の広角へ散乱した透過光の割合。
  - グロス（Gloss）: 規定角（60°等）における正反射方向のピーク強度。
  - 写像性（DOI / Clarity）: 中心の直進光と周辺の小角散乱光のエネルギー比、または光学くしを通したMTFコントラスト。
  - ギラツキ（Sparkle / GD値）: ピクセルレベル積分評価（下記参照）により算出。
- **表面形状指標（高さ配列 $h(x,y)$ から算出）**

  準拠規格: **ISO 25178-2**（面パラメータ）、**JIS B 0601 / ISO 4287**（プロファイルパラメータ）。
  プロファイルパラメータは行・列両方向のプロファイルを平均して算出する。

  **ISO 25178-2 S-パラメータ（面粗さ）**

  | 記号 | 名称 | 定義 |
  |---|---|---|
  | $Sq$ | 二乗平均平方根高さ | $\sqrt{\frac{1}{A}\iint z^2\,dA}$ |
  | $Sa$ | 算術平均高さ | $\frac{1}{A}\iint |z|\,dA$ |
  | $Sp$ | 最大山高さ | $\max(z)$ |
  | $Sv$ | 最大谷深さ | $|\min(z)|$ |
  | $Sz$ | 最大高さ | $Sp + Sv$ |
  | $Ssk$ | スキューネス | $\frac{1}{Sq^3}\cdot\frac{1}{A}\iint z^3\,dA$ |
  | $Sku$ | クルトシス | $\frac{1}{Sq^4}\cdot\frac{1}{A}\iint z^4\,dA$ |
  | $Sdq$ | 二乗平均平方根傾斜 | $\sqrt{\text{mean}\!\left[(\partial z/\partial x)^2+(\partial z/\partial y)^2\right]}$ |
  | $Sdr$ | 界面展開面積比 | $(\text{実面積}-\text{投影面積})/\text{投影面積}\times 100\%$ |
  | $Sal$ | 自己相関長 | NACFが全方向でしきい値(0.2)を下回る最小ラグ距離 |
  | $Str$ | テクスチャアスペクト比 | $Sal_{\min} / Sal_{\max}$（0→異方性、1→等方性） |

  **JIS B 0601 / ISO 4287 R-パラメータ（プロファイル粗さ）**

  | 記号 | 名称 | 内容 |
  |---|---|---|
  | $Rq$ | 二乗平均平方根粗さ | プロファイルのRMS値（行・列平均） |
  | $Ra$ | 算術平均粗さ | プロファイルの絶対値平均 |
  | $Rz$ | 最大高さ | プロファイルのPV値平均 |
  | $Rp$ | 最大山高さ | $\max(z)$ のプロファイル平均 |
  | $Rv$ | 最大谷深さ | $|\min(z)|$ のプロファイル平均 |
  | $Rsk$ | スキューネス | $\text{mean}(z^3)/Rq^3$ のプロファイル平均 |
  | $Rku$ | クルトシス | $\text{mean}(z^4)/Rq^4$ のプロファイル平均 |
  | $Rsm$ | 輪郭曲線要素の平均幅 | 評価長 / プロファイル要素数 |
  | $Rc$ | 輪郭曲線要素の平均高さ | 山と隣接谷の高さ差の平均 |

### 5.2. 評価アルゴリズム詳細：ギラツキ（Sparkle）

ギラツキコントラスト $C_s = \sigma / \mu$（輝度の標準偏差 / 平均）を以下の設定で算出する。

```yaml
sparkle:
  # ── グループ1：観察条件 ──────────────────────────────────
  viewing:
    preset: 'smartphone'      # 'smartphone' / 'tablet' / 'monitor' / 'custom'
    distance_mm: null         # null のときはプリセット値を使用
    pupil_diameter_mm: null
  # プリセット: smartphone=300mm/3mm, tablet=350mm/3mm, monitor=600mm/3mm

  # ── グループ2：ディスプレイ仕様 ──────────────────────────
  display:
    preset: 'fhd_smartphone'  # 'fhd_smartphone' / 'qhd_monitor' / '4k_monitor' / 'custom'
    pixel_pitch_mm: null      # null のときはプリセット値を使用
    subpixel_layout: null     # 'rgb_stripe' / 'bgr_stripe' / 'pentile'
  # プリセット: fhd_smartphone=0.062mm/rgb_stripe, qhd_monitor=0.124mm/rgb_stripe

  # ── グループ3：照明条件 ──（廃止・`illumination` は読み込まれない）
  # 多波長解析は `simulation.wavelength_um` を list にする方法で行う。
  # 各波長で独立に Sparkle が計算され、`sparkle_fft_wl525nm_aoi0_brdf`
  # のようにサフィックス付きで記録される。

  # ── グループ4：評価指標（固定） ────────────────────────────
  metrics:
    integration_area: 'single_pixel'   # 1画素領域で積分（固定）
    sparkle_index: 'sigma_over_mu'     # Cs = σ/μ（固定）
```

**プリセット優先ルール**: 数値が指定されている場合（null でない）は数値を優先。数値が null の場合はプリセット値を使用。プリセットも未指定の場合はエラーとする。

---

### 5.3. 規格対応と波長依存性

**結論先出し**：本実装では **Haze・Gloss・DOI は代表波長 1 つだけで計算**し、`simulation.wavelength_um` の list 設定に影響されない。一方 **Sparkle・Log-RMSE・BSDF 本体**は波長ごとに独立計算される。

#### 5.3.1. 各指標の規格と光源規定

業界規格では Haze・Gloss・DOI・Sparkle は **いずれも CIE 標準光源（白色）× 明所視応答 V(λ)** で規定されている：

| 指標 | 主な規格 | 光源 | 検出器応答 |
|---|---|---|---|
| **Haze** | JIS K 7136 / ISO 14782 / ASTM D1003 | CIE Illuminant D65 / C | CIE 1931 Y（明所視 V(λ)） |
| **Gloss** | JIS Z 8741 / ISO 2813 / ASTM D523 | CIE Illuminant C | CIE 1931 Y |
| **DOI（写像性）** | JIS K 7374 / ASTM D5767 | 白色光源（A または C） | 明所視 V(λ) |
| **Sparkle（GD 値）** | SEMI D63 / IDMS | 白色バックライト（D65 等） | 明所視 V(λ) |

本実装は **代表波長 1 つでの単波長近似**を採用する。デフォルトは **555 nm（V(λ) ピーク）**。凹凸 ≫ 波長の AG フィルム等では近似誤差は典型的に 1〜5%（波長依存が弱い）。

#### 5.3.2. 代表波長の設定

```yaml
metrics:
  representative_wavelength_um: 0.555   # デフォルト（V(λ) ピーク付近）
```

**挙動ルール:**
- `metrics.representative_wavelength_um` が省略された場合は **0.555 μm** を使用する。
- `simulation.wavelength_um` が list でも、代表波長のみで Haze/Gloss/DOI を計算する。
- 代表波長が `wavelength_um` のリストに含まれる場合は、既存計算の BSDF を再利用（追加計算なし）。
- 含まれない場合は、主条件（list 先頭の θ_i, mode）で代表波長の BSDF を **1 条件だけ追加計算**する。この BSDF は Parquet にも `wavelength_um=0.555` の行として保存される。
- Haze/Gloss/DOI が全て `enabled: false` の場合は代表波長の追加計算自体がスキップされる（無駄な sim なし）。

#### 5.3.3. `wavelength_um` リストが影響する指標・しない指標

| 項目 | `wavelength_um: [...]` の影響 | 記録形式 |
|---|---|---|
| **BSDF 本体** | **あり** — 波長ごとに独立計算 | Parquet 行に `wavelength_um` 列、method で区別 |
| **Haze** | **なし** — 代表波長のみ | `haze_fft` / `haze_psd`（1 つ固定） |
| **DOI** | **なし** — 代表波長のみ | `doi_fft` / `doi_psd`（1 つ固定） |
| **Gloss** | **なし** — 代表波長のみ | `gloss_fft` / `gloss_psd`（1 つ固定） |
| **Sparkle** | **あり** — 各波長で独立計算 | `sparkle_fft_wl<X>nm_aoi<Y>_<mode>` |
| **Log-RMSE** | **あり** — 各条件で独立計算 | `log_rmse_<method>_wl<X>nm_aoi<Y>_<mode>` |

**なぜ Sparkle だけ波長依存を残すか:** Sparkle はディスプレイの RGB サブピクセル発光パターンの評価にも使われるため、波長別の内訳を残しておくと色依存の解析が可能。規格値は従来の単波長近似でも得られる（代表波長のみの条件で記録される）。

#### 5.3.4. Log-RMSE の定義

Log-RMSE は実測 BSDF とシミュレーション BSDF の **対数空間 RMS 誤差**で、BSDF の 5〜6 桁の動的レンジ全体で誤差を均等に評価する。

$$\text{Log-RMSE} = \sqrt{\frac{1}{N_\text{valid}} \sum_{i \in \text{valid}} \left( \log_{10}\max(\text{BSDF}_{\text{sim}, i}, F) - \log_{10}(\text{BSDF}_{\text{meas}, i}) \right)^2}$$

- $F$ = `error_metrics.bsdf_floor`（ノイズフロア、デフォルト $10^{-6}$ sr⁻¹）
- $\text{valid}$ = 実測 > $F$ の点のみ
- シミュレーション値はゼロ割り防止のため $F$ でクリップ

**解釈:**
| Log-RMSE 値 | 物理的意味 |
|---|---|
| 0.0 | 完全一致 |
| 0.3 | 平均して約 2 倍の差 |
| 1.0 | 平均して 10 倍の差 |
| 2.0 | 平均して 100 倍の差 |

**計算方法:**
1. `merge_sim_and_measured()` が sim DataFrame の各 `(method, wavelength_um, theta_i_deg, mode)` に対して、実測 DataFrame 内で `tolerance_deg` / `tolerance_nm` 内の一致ブロックを検索
2. 実測ブロックの UV 座標に sim 側を線形補間（`scipy.interpolate.griddata`）
3. 対数空間で RMS を計算
4. 結果を sim 行の `log_rmse` 列に書き込む

`metric.representative_wavelength_um` の設定は Log-RMSE には**無関係**（全条件で独立に計算される）。

---

## 6. 最適化要件（Optuna）とデータ管理（MLflow）

### 6.1. Optunaによる自動最適化と拡張
- **多目的最適化（MOTPE）**: 「ギラツキの最小化」と「ヘイズの目標値追従」など、複数の評価指標のパレートフロントを探索する。
- **タグチメソッドとの融合**: 過去の実験データ（直交表など）を `study.add_trial()` 等でナレッジベースとして注入し、TPEによる探索を効率化する。
- **探索制御**: `enqueue_trial` による優先実行キューの設定、および過去の履歴と照合した重複試行のスキップ（枝刈り）を実装する。

**重複試行スキップ（正規化ユークリッド距離方式）:**
```yaml
optuna:
  duplicate_skip:
    enabled: true
    distance_threshold: 0.01   # 正規化パラメータ空間でのユークリッド距離閾値
```

各パラメータを探索範囲 [low, high] で 0〜1 に正規化した上でユークリッド距離を計算し、閾値以内であればスキップする。パラメータが増えても閾値の管理は1つで済む。

### 6.2. MLflowによる一元管理ルール
プロジェクトの肥大化を防ぐため、Experiment（実験）を以下の順序で階層管理する。
1. **01_BSDF_Raw_Data（データ貯蔵庫）**: 1 Run = 1 形状。Optunaの各Trialのパラメータ、メトリクス（各種評価指標）、および生データ（Parquet、2Dヒートマップ）をArtifactとして保存する。
2. **02_Analysis_Reports（比較レポート）**: 1 Run = 1 解析タスク。複数の形状データを引き出して重ね合わせたインタラクティブなBSDF比較グラフ（HTML）を保存する。
3. **03_GenAI_Insights（AI考察）**: LLMを用いて、測定値やメトリクスに基づく設計改善の考察レポートを自動生成・記録する。

**Parquet スキーマ（01_BSDF_Raw_Data）:**

1行 = 1測定/計算点 × 1手法。`method` カラムで手法を区別することで、新しい計算手法を追加してもスキーマ変更が不要。

```
┌─────────────────┬──────────┬──────────────────────────────────────────────┐
│ カラム名         │ 型       │ 内容                                          │
├─────────────────┼──────────┼──────────────────────────────────────────────┤
│ u               │ float32  │ sinθ_s·cosφ_s（方向余弦）                    │
│ v               │ float32  │ sinθ_s·sinφ_s（方向余弦）                    │
│ theta_s_deg     │ float32  │ 散乱天頂角 [deg]（逆引き用）                  │
│ phi_s_deg       │ float32  │ 散乱方位角 [deg]（逆引き用）                  │
│ theta_i_deg     │ float32  │ 入射天頂角 [deg]                              │
│ phi_i_deg       │ float32  │ 入射方位角 [deg]                              │
│ wavelength_um   │ float32  │ 波長 [μm]                                     │
│ polarization    │ category │ 'S' / 'P' / 'Unpolarized'                     │
│ mode            │ category │ 'BRDF' / 'BTDF'（theta_i から自動判定）       │
│ method          │ category │ 'FFT' / 'PSD' / 'MultiLayer' / 'measured'     │
│ bsdf            │ float32  │ BSDF 値 [sr⁻¹]                               │
│ is_measured     │ bool     │ method='measured' のとき True                 │
│ log_rmse        │ float32  │ Log-RMSE（同一条件の measured と計算値の差）  │
└─────────────────┴──────────┴──────────────────────────────────────────────┘
```

**実測データの取り込みスキーマ（生ファイル → Parquet 変換）:**

```
# measured_bsdf_raw.csv（装置生データ）
┌──────────────┬─────────┬──────────────────────────────────────────┐
│ カラム名      │ 型      │ 内容                                      │
├──────────────┼─────────┼──────────────────────────────────────────┤
│ theta_s_deg  │ float32 │ 散乱天頂角 [deg]（0〜89°、1°刻み）       │
│ phi_s_deg    │ float32 │ 散乱方位角 [deg]（0〜350°、10°刻み）     │
│ theta_i_deg  │ float32 │ 入射天頂角 [deg]（<90°=BRDF、>90°=BTDF）│
│ phi_i_deg    │ float32 │ 入射方位角 [deg]                          │
│ wavelength_nm│ float32 │ 波長 [nm]（取り込み時に μm へ変換）       │
│ polarization │ str     │ 'S' / 'P' / 'Unpolarized'                 │
│ bsdf         │ float32 │ 実測 BSDF 値 [sr⁻¹]                      │
└──────────────┴─────────┴──────────────────────────────────────────┘
```

装置ごとのヘッダ情報を含む生ファイルフォーマットは、実ファイル提供時に `MeasuredSurface` のローダー仕様として追記する。

---

## 7. 可視化仕様（HoloViews）
MLflowのArtifactsに保存する、またはJupyter上でインタラクティブに操作するためのグラフ仕様。

- **3者オーバーレイ比較**:
  - FFT計算値（青・実線）、PSD計算値（オレンジ・破線）、実測データ（黒・散布図）を重ねてプロット。
  - シミュレーションで得られた高解像度な計算結果を、実測データのサンプリングポイント（角度）に合わせて補間（抽出）し、実測生データと直接比較する。
- **スケール切替UI**: Panelウィジェットを活用し、ブラウザ上でグラフの軸スケールを「リニア」「片対数」「両対数」に動的に切り替える機能を実装する。
- **リアルタイム・ダッシュボード（DynamicMap）**:

```yaml
dynamicmap:
  preview_grid_size_drag: 128    # ドラッグ中の解像度（応答性優先）
  preview_grid_size_idle: 512    # 停止後の解像度（精度優先）
  cache_size: 256                # LRUキャッシュのエントリ数
  random_seed: 42                # プレビュー用乱数シード（固定）
  default_preview_mode: 'reduced_area'  # DynamicMap のデフォルトプレビューモード
```

  粗さやピッチなどの形状パラメータのスライダー操作に連動して、リアルタイムにBSDF分布が再計算・更新されるGUIを提供する。以下の3段階の解像度で操作感と精度を両立する:

  | フェーズ | grid_size | 処理時間の目安 | 用途 |
  |---|---|---|---|
  | ドラッグ中 | 128 | ~0.1秒 | リアルタイム応答 |
  | ドラッグ停止後 | 512 | ~0.6秒 | パラメータ確認 |
  | 本計算ボタン | 4096 | ~数分 | Optuna 最適化用 |

  - `RandomRoughSurface` 等、乱数を使用する形状モデルではプレビュー用乱数シードを `random_seed` で固定し、LRUキャッシュを有効にする。本計算では乱数シードを変えてアンサンブル平均を取る。
  - **注意**: `reduced_resolution` モードでは散乱角上限が約7.9°（λ=0.55μm 時）となり、ヘイズの確認が不可能になる。ヘイズを含む評価は `reduced_area` モードを使用すること。

---

## 8. 依存パッケージ管理

`pyproject.toml` に範囲指定、`requirements.lock` に完全固定版を記載する二層構造で管理する。

```toml
# pyproject.toml（範囲指定）
[project.dependencies]
numpy = ">=1.24,<3.0"
scipy = ">=1.10,<2.0"
optuna = ">=3.0,<5.0"
mlflow = ">=2.0,<3.0"
holoviews = ">=1.17,<2.0"
panel = ">=1.0,<2.0"
```

```
# requirements.lock（完全固定・CI/CD 用）
numpy==1.26.4
scipy==1.12.0
optuna==3.6.1
mlflow==2.13.0
holoviews==1.18.3
panel==1.4.2
```

---

## 9. 品質保証・テスト計画

### 9.1. テスト方針

| レベル | ツール | 実行タイミング |
|---|---|---|
| ユニットテスト | pytest | コード変更のたびに実行 |
| 統合テスト（CLI） | pytest + subprocess | リリース前 |
| 物理検証 | 手動・スクリプト | 式変更・モデル追加時 |

合格基準: **全ユニットテスト pass、物理検証の相対誤差 5% 以内**。

---

### 9.2. ユニットテスト一覧

#### 9.2.1. 表面形状モデル（`tests/test_models.py`）

| テスト名 | 検証内容 |
|---|---|
| `test_basic_creation` | HeightMap の grid_size / physical_size_um / pixel_size_um プロパティ |
| `test_invalid_non_square` | 非正方形データで ValueError |
| `test_invalid_pixel_size` | 負の pixel_size_um で ValueError |
| `test_rq_property` | Rq プロパティの数値確認 |
| `test_resample` | リサンプル後の grid_size 変化・pixel_size_um 不変 |
| `test_rq_normalization` | RandomRoughSurface の Rq が指定値に一致（相対誤差 5%） |
| `test_preview_reduced_area` | `reduced_area` モードで pixel_size_um 固定・grid_size 縮小 |
| `test_preview_reduced_resolution` | `reduced_resolution` モードで physical_size_um 固定・pixel_size_um 拡大 |
| `test_grid_placement` | SphericalArraySurface（Grid）高さが非負 |
| `test_hexagonal_placement` | SphericalArraySurface（Hexagonal）高さが非負 |
| `test_maximum_overlap` | Overlap=Maximum で高さが非負 |
| `test_additive_overlap` | Overlap=Additive の最大値 ≥ Maximum モードの最大値 |

#### 9.2.2. FFT法・フレネル係数（`tests/test_fft_bsdf.py`）

| テスト名 | 検証内容 |
|---|---|
| `test_normal_incidence_rs` | 法線入射 $r_s = (n_1-n_2)/(n_1+n_2)$ |
| `test_normal_incidence_rp` | 法線入射 $r_p = (n_1-n_2)/(n_1+n_2)$（Born-Wolf規約） |
| `test_energy_conservation` | $R_s + T_s = 1$（エネルギー保存） |
| `test_snell_angle` | $n_1 \sin\theta_i = n_2 \sin\theta_t$ |
| `test_total_reflection` | 臨界角超えで ValueError |
| `test_output_shape` | BSDF グリッドが (N, N) |
| `test_bsdf_non_negative` | BSDF 値が非負 |
| `test_brdf_mode` | BRDF モードで max > 0 |
| `test_btdf_mode` | BTDF モードで max > 0 |
| `test_sample_at_angles` | 指定角度でのサンプリング形状・非負性 |
| `test_uv_range` | 有効散乱点が UV 半球内（半径 ≤ 1） |

#### 9.2.3. PSD法（`tests/test_psd_bsdf.py`）

| テスト名 | 検証内容 |
|---|---|
| `test_psd_shape` | PSD グリッドが (N, N) |
| `test_psd_non_negative` | PSD 値が非負 |
| `test_parseval_theorem` | PSD 総和 = $Rq^2 \times$ 物理面積（パーセバルの定理、相対誤差 10%） |
| `test_output_shape` | compute_bsdf_psd の出力形状 |
| `test_bsdf_non_negative` | PSD法 BSDF が非負 |
| `test_complete_vs_simplified_brdf` | 法線入射で完全形と簡略形の比率が 1.0（係数 10 以内） |
| `test_polarization_s` | S偏光 BSDF が非負 |
| `test_polarization_p` | P偏光 BSDF が非負 |
| `test_btdf_mode` | BTDF モードで max > 0 |

#### 9.2.4. 表面形状・光学指標（`tests/test_metrics.py`）

**TestSurfaceMetrics（既存パラメータ）**

| テスト名 | 検証内容 |
|---|---|
| `test_rq_flat` | 平坦面の Rq = 0 |
| `test_rq_sine` | 正弦波の Rq（行列両方向平均） |
| `test_ra_positive` | Ra ≥ 0 |
| `test_rz_flat` | 平坦面の Rz = 0 |
| `test_rz_positive` | 正弦波の Rz > 0 |
| `test_sdq_flat` | 平坦面の Sdq = 0 |
| `test_sdq_sine` | 正弦波の Sdq > 0 |
| `test_all_metrics_keys` | compute_all_surface_metrics の全キー集合 |

**TestISO25178Metrics（ISO 25178-2 S-パラメータ）**

| テスト名 | 検証内容 |
|---|---|
| `test_sq_flat` | 平坦面の Sq = 0 |
| `test_sq_sine` | 正弦波の Sq = amplitude/√2（相対誤差 5%） |
| `test_sa_flat` | 平坦面の Sa = 0 |
| `test_sa_positive` | 正弦波の Sa > 0 |
| `test_sp_positive` | 正弦波の Sp > 0 |
| `test_sv_positive` | 正弦波の Sv > 0 |
| `test_sz_equals_sp_plus_sv` | Sz = Sp + Sv が成立する |
| `test_ssk_sine_near_zero` | 正弦波（対称分布）の Ssk ≈ 0 |
| `test_ssk_flat` | 平坦面の Ssk = 0 |
| `test_sku_sine` | 正弦波の Sku = 1.5（理論値） |
| `test_sdr_flat` | 平坦面の Sdr = 0% |
| `test_sdr_positive` | 起伏面の Sdr > 0% |
| `test_sal_positive` | Sal > 0 |
| `test_str_range` | Str ∈ [0, 1] |

**TestJISB0601Metrics（JIS B 0601 R-パラメータ）**

| テスト名 | 検証内容 |
|---|---|
| `test_rp_positive` | 正弦波の Rp > 0 |
| `test_rv_positive` | 正弦波の Rv > 0 |
| `test_rp_rv_relation` | Rp + Rv ≤ Rz（プロファイル平均の性質） |
| `test_rsk_sine_near_zero` | 正弦波（対称分布）の Rsk ≈ 0 |
| `test_rku_sine` | 正弦波の Rku ≈ 1.5（理論値） |
| `test_rsm_positive` | 正弦波の Rsm > 0 |
| `test_rsm_flat` | 平坦面の Rsm = 物理サイズ（要素なし） |
| `test_rc_positive` | 正弦波の Rc > 0 |
| `test_rc_flat` | 平坦面の Rc = 0 |

**TestLogRMSE・TestOpticalMetrics**

| テスト名 | 検証内容 |
|---|---|
| `test_perfect_match` | Log-RMSE = 0（完全一致） |
| `test_floor_masking` | フロア以下の点が誤差計算から除外される |
| `test_order_of_magnitude_error` | 10倍差 → Log-RMSE = 1.0 |
| `test_all_below_floor` | 全点フロア以下 → inf |
| `test_haze_range` | Haze ∈ [0, 1] |
| `test_haze_narrow_beam` | 法線集中ビームのヘイズ < 0.5 |
| `test_gloss_positive` | Gloss ≥ 0 |
| `test_doi_range` | DOI ∈ [0, 1] |
| `test_doi_narrow_beam` | 法線集中ビームの DOI > 0.5 |

#### 9.2.5. 設定読み込み・バリデーション（`tests/test_config_loader.py`）

| テスト名 | 検証内容 |
|---|---|
| `test_wavelength` / `test_theta_i` / `test_n1_n2` | BSDFConfig プロパティの値取得 |
| `test_is_brdf` / `test_is_btdf` | BRDF/BTDF モード判定 |
| `test_theta_i_effective_btdf` | BTDF モード（150°）の有効入射角 = 30° |
| `test_theta_90_raises` | theta_i=90° で ValueError |
| `test_invalid_polarization_raises` | 未知の polarization で ValueError |
| `test_zero_bsdf_floor_raises` / `test_negative_bsdf_floor_raises` | 非正の bsdf_floor で ValueError |
| `test_smartphone_preset_distance` / `test_tablet_preset_distance` | Sparkle プリセット解決の数値確認 |
| `test_green_illumination_wavelength` | green プリセット → wavelengths_um=[0.55] |
| `test_unknown_preset_raises` | 未知プリセットで ValueError |
| `test_manual_override_takes_priority` | 個別数値がプリセットより優先される |
| `test_from_file_not_found` | 存在しないパスで FileNotFoundError |
| `test_from_file_loads_yaml` | YAML ファイルから BSDFConfig を生成 |

#### 9.2.6. 最適化ユーティリティ（`tests/test_optimization.py`）

| テスト名 | 検証内容 |
|---|---|
| `test_lower_bound_maps_to_zero` / `test_upper_bound_maps_to_one` | `_normalize_params` の境界値 |
| `test_midpoint` | 中点 → 0.5 |
| `test_clamps_out_of_range` | 探索範囲外 → クリップ |
| `test_multiple_params_order` | 複数パラメータの順序保持 |
| `test_empty_history_is_not_duplicate` | 履歴なし → 重複でない |
| `test_identical_params_is_duplicate` | 同一パラメータ → 重複 |
| `test_distant_params_is_not_duplicate` | 遠いパラメータ → 重複でない |
| `test_just_below_threshold_is_not_duplicate` | 閾値未満 → 重複 |
| `test_threshold_zero_means_only_exact_match` | 閾値=0 → 完全一致のみ重複 |
| `test_multiple_params` | 多次元正規化距離の確認 |
| `test_runs_n_trials` | n_trials=5 で完了試行数=5 |
| `test_duplicate_skip_reduces_completed` | 重複スキップが機能し完了試行数 ≤ n_trials |
| `test_best_trials_summary_single_objective` | best_trials_summary の返り値キー確認 |

#### 9.2.7. 実測データモデル（`tests/test_measured.py`）

| テスト名 | 検証内容 |
|---|---|
| `test_generates_correct_grid_size` | grid_size が正しく設定される |
| `test_leveling_removes_tilt` | レベリング後の平均 ≈ 0 |
| `test_nan_removed_after_preprocessing` | NaN が補間で消去される |
| `test_all_nan_returns_zero` | 全 NaN → ゼロ配列 |
| `test_from_numpy_basic` / `test_from_numpy_pixel_size` | from_numpy の基本動作 |
| `test_from_csv_um` | μm 単位の CSV 読み込み |
| `test_from_csv_nm_converts_to_um` | nm → μm 単位変換 |
| `test_from_csv_invalid_height_unit` | 未知の height_unit で ValueError |
| `test_from_csv_file_not_found` | 存在しないパスで FileNotFoundError |
| `test_from_csv_skiprows` | ヘッダ行をスキップして読み込み |
| `test_from_config_basic` | from_config の基本動作（config 辞書から生成） |
| `test_from_config_no_path_raises` | path 未指定で ValueError |
| `test_from_config_height_unit_nm` | from_config で nm 単位の CSV を読み込み |
| `TestPaddingFunctions` | 4 パディング方式（zeros/tile/reflect/smooth_tile）の形状・値・dtype 検証 |
| `TestMeasuredSurfacePadding` | MeasuredSurface.padding 引数の受け入れ・拒否・実動作検証 |
| `TestDeviceVk6AutoDetect` | DeviceVk6Surface の pixel_size_um / grid_size 自動算出・明示オーバーライド |

#### 9.2.8. CLI コマンドスモークテスト（`tests/test_cli.py`）

| テスト名 | 検証内容 |
|---|---|
| `test_version` | `bsdf --version` で 0.1.0 が含まれる |
| `test_simulate_fft_only` | `--method fft` で正常終了（exit_code=0） |
| `test_simulate_psd_only` | `--method psd` で正常終了 |
| `test_simulate_both_methods` | `--method both` で正常終了 |
| `test_simulate_saves_parquet` | `--save-parquet` で Parquet ファイルが生成される |
| `test_simulate_missing_config_fails` | 存在しない config で終了コード≠0 |
| `test_simulate_btdf_mode` | `theta_i=150°`（BTDF）で正常終了 |

#### 9.2.9. BSDF 実測ファイルリーダー（`tests/test_bsdf_reader.py`）

| テスト名 | 検証内容 |
|---|---|
| `TestBuildMeasuredDataframeIsbtdf` | `build_measured_dataframe` の `is_btdf` 引数（None/True/False）の挙動・mode 列・wavelength 変換 |
| `TestRegistry::test_register_reader_adds_to_registry` | `register_reader` でレジストリに登録される |
| `TestRegistry::test_read_bsdf_file_uses_can_read` | `can_read()=True` のリーダーが自動選択される |
| `TestRegistry::test_read_bsdf_file_file_not_found` | 存在しないファイルで FileNotFoundError |
| `TestRegistry::test_read_bsdf_file_no_reader_raises` | 対応リーダーなしで ValueError |
| `TestRegistry::test_read_bsdf_file_by_name` | リーダー名指定で読み込み |
| `TestLightToolsBsdfReaderCanRead` | MiniDiff ヘッダ検出・非対応ファイル・欠損ファイルの判定 |
| `TestLightToolsBsdfReaderRead` | ブロック数・カラム構成・BRDF/BTDF モード・AOI・波長・行数・BSDF 値・UV 座標 |
| `TestLightToolsBsdfReaderInvalidFile` | ScatterAzimuth なし・DataBegin なしで ValueError |
| `TestLoadBsdfReadersPlugin` | プラグインフォルダから自動登録・存在しないフォルダで例外なし |
| `TestReadBsdfFileWithRealSample` | 実ファイル 24 ブロック・BRDF/BTDF 各 12・行数 91×361×24・非負 BSDF |
| `TestGetConditions` | `get_conditions()` で実測 DataFrame リストから条件一覧を抽出 |
| `TestSelectBlock` | `select_block()` の厳密一致・BRDF/BTDF 分離・tolerance 内最近傍・圏外 None |
| `TestMergePerCondition` | `merge_sim_and_measured()` の条件ごと Log-RMSE（完全一致 0・10倍差 1.0・BRDF/BTDF 独立） |

#### 9.2.10. 多条件シミュレーション・実測 BSDF 統合（`tests/test_config_loader.py`, `tests/test_cli.py`）

| テスト名 | 検証内容 |
|---|---|
| `TestMultiConditionSupport::test_wavelength_{scalar,list}_*` | `wavelength_um` のスカラ/list 両対応（案 2-B） |
| `TestMultiConditionSupport::test_theta_{scalar,list}_*` | `theta_i_deg` のスカラ/list 両対応 |
| `TestMultiConditionSupport::test_mode_*` | `mode` のスカラ/list 両対応・invalid で ValueError |
| `TestMultiConditionSupport::test_legacy_btdf_auto_detect` | mode 未指定で `theta_i > 90°` → BTDF 自動判定（旧互換） |
| `TestMultiConditionSupport::test_explicit_mode_cartesian_product` | theta_i × mode 直積展開 |
| `TestMultiConditionSupport::test_full_grid_24_conditions` | 3λ × 4AOI × 2mode = 24 条件の直積 |
| `TestMeasuredBsdfConfig` | `measured_bsdf` セクション（path / match_measured / tolerance）の読み込み |
| `TestCLIMultiCondition::test_multi_{wavelength,theta_i,mode}` | 多条件 `simulate` が Parquet に全条件を保存 |
| `TestCLIMultiCondition::test_parquet_contains_multiple_conditions` | Parquet 内に 2λ × 2θ = 4 条件が保存される |
| `TestCLIMeasuredBsdfReal::test_simulate_with_measured_bsdf_cli_option` | `--measured-bsdf` で実測読み込み・log_rmse 計算・Parquet に measured 行 |
| `TestCLIMeasuredBsdfReal::test_simulate_match_measured_auto_conditions` | `match_measured: true` で実測 24 条件を sim に自動採用 |
| `TestCLIMultiCondition::test_representative_wavelength_added_for_standards_metrics` | Haze 有効 × 多波長 → 代表波長 0.555μm の追加 sim が走る |
| `TestCLIMultiCondition::test_representative_wavelength_reused_when_in_list` | `wavelength_um` に 0.555 が含まれる場合は追加 sim なし（既存 BSDF 再利用） |
| `TestCLIMultiCondition::test_representative_wavelength_custom` | `metrics.representative_wavelength_um: 0.500` 指定で代表波長を変更可能 |
| `TestCLIMultiCondition::test_haze_unaffected_by_wavelength_list` | 多波長設定でも Haze は代表波長でのみ計算される |

#### 9.2.11. visualize 実測オーバーレイ・多条件レポート（`tests/test_visualize_overlay.py`）

| テスト名 | 検証内容 |
|---|---|
| `TestOverlay1D::test_default_single_condition` | 引数省略で単条件 1D オーバーレイが描画される |
| `TestOverlay1D::test_auto_select_multi_condition` | 多条件 df でも自動選択で "データなし" にならない |
| `TestOverlay1D::test_mode_filter_brdf_vs_btdf` | 同じ λ・θ でも BRDF/BTDF を分離して描画 |
| `TestOverlay1D::test_empty_df_returns_placeholder` | 空 df でプレースホルダを返す |
| `TestOverlay1D::test_measured_overlay_present` | `method='measured'` 行が Scatter 要素として追加される |
| `TestBsdfReportMultiCondition::test_multi_condition_uses_tabs` | 多条件 df で `pn.Tabs` が生成される |
| `TestBsdfReportMultiCondition::test_tab_count_matches_conditions` | Tab 数 = 3λ × 2θ × 2mode = 12 |
| `TestBsdfReportMultiCondition::test_metrics_table_includes_log_rmse` | `log_rmse_*` メトリクスが Comparison カテゴリで表示 |
| `TestVisualizeCLI::test_visualize_with_measured_overlay` | simulate → Parquet 読込 → report 生成が HTML を出力 |
| `TestSparkleMultiWavelength::test_sparkle_multi_wavelength_keys_in_parquet` | 多波長 × Sparkle 有効で 3 波長の BSDF が Parquet に保存 |
| `TestSparkleMultiWavelength::test_sparkle_metric_suffix_helper` | メトリクスキーのサフィックス生成規則 `_<method>_wl<X>nm_aoi<Y>_<mode>` |

#### 9.2.12. リアルタイム BSDF ダッシュボード（`tests/test_dashboard.py`）

| テスト名 | 検証内容 |
|---|---|
| `TestFactoryDispatch::test_random_rough_dispatches_correctly` | `surface.model='RandomRoughSurface'` → `RandomRoughDynamicMap` |
| `TestFactoryDispatch::test_spherical_array_dispatches_correctly` | `surface.model='SphericalArraySurface'` → `SphericalArrayDynamicMap` |
| `TestFactoryDispatch::test_measured_surface_dispatches_correctly` | `DeviceVk6Surface` 等 → `MeasuredSurfaceDynamicMap` |
| `TestCreateDashboard::test_random_rough_creates_dashboard` | RandomRough ダッシュボードが Panel Column を生成 |
| `TestCreateDashboard::test_spherical_array_creates_dashboard` | SphericalArray ダッシュボードが Panel Column を生成 |
| `TestMeasuredOverlay::test_measured_blocks_loaded_in_dashboard` | `measured_bsdf.path` 指定時に 24 ブロック読み込み + 条件一致マッチング |
| `TestMeasuredOverlay::test_measured_profile_extraction` | `_measured_profile()` が phi≈0° の (θ_s, BSDF) 1D ペアを返す |
| `TestMeasuredOverlay::test_no_measured_file_returns_none_profile` | 実測ファイル未指定時は None を返す |
| `TestMake1DOverlay::test_sim_only` | sim のみの 1D オーバーレイ（Scatter なし） |
| `TestMake1DOverlay::test_sim_with_measured` | sim + 実測の 1D オーバーレイ（黒点 Scatter 追加） |
| `TestDashboardCLI::test_dashboard_command_in_help` | `bsdf --help` に dashboard が表示 |
| `TestDashboardCLI::test_dashboard_subcommand_help` | `bsdf dashboard --help` に `--config` / `--port` |
| `TestDashboardCLI::test_dashboard_missing_config_fails` | 存在しない config で exit_code ≠ 0 |

---

### 9.3. 物理検証項目

コードの数値が物理的に正しいことを確認するための検証。ユニットテストに含まれないが、式変更・モデル追加時に手動で確認する。

| 項目 | 確認方法 | 合格基準 |
|---|---|---|
| フレネル反射率 | $R = \|r_s\|^2$、$R + T = 1$ | 全角度で相対誤差 < 0.01% |
| PSD パーセバルの定理 | $\sum PSD \cdot \Delta f^2 = Rq^2$ | 相対誤差 < 10% |
| BSDF エネルギー保存 | $\int \text{BSDF} \cos\theta_s \, d\Omega \leq 1$ | 常に ≤ 1.0 |
| Q因子対称性 | 法線入射で $Q_s = Q_p$ | 相対誤差 < 0.1% |
| Adding-Doubling 対称性 | 表面 A+B と B+A の結果が等しい | 相対誤差 < 1% |

---

### 9.4. 未テスト項目（今後の課題）

以下の項目は現時点でテストが存在しない。機能追加・安定化の際に対応する。

| 項目 | 優先度 | 備考 |
|---|---|---|
| Adding-Doubling 多層合成の数値検証 | 高 | 単層 = 表面のみ計算と一致するか |
| Parquet スキーマの読み書き往復 | 中 | 保存 → 読み込みでデータが一致するか |
| Sparkle RGB 多波長モードの数値検証 | 中 | `bsdf_per_wavelength` 使用時の輝度加重が正しいか |
| DynamicMap / Panel 可視化 | 低 | UIテストは手動確認 |
| Optuna 最適化収束 | 低 | 既知パラメータへの収束確認 |

**対応済みの項目（Phase B で実装）:**

| 項目 | 対応ファイル |
|---|---|
| CLI エンドツーエンド（`bsdf simulate`） | `tests/test_cli.py`（9.2.8） |
| MeasuredSurface CSV 読み込み・NaN補間 | `tests/test_measured.py`（9.2.7） |
| BSDFConfig バリデーション・プリセット解決 | `tests/test_config_loader.py`（9.2.5） |
| 重複スキップ・BSDFOptimizer | `tests/test_optimization.py`（9.2.6） |

---

### 9.5. フレネル係数の符号規約

本プログラムは **Born-Wolf 規約** を採用する（`src/bsdf_sim/optics/fresnel.py`）。

| 量 | 式 | 法線入射値（n₁=1.0, n₂=1.5） |
|---|---|---|
| $r_s$ | $(n_1\cos\theta_i - n_2\cos\theta_t)/(n_1\cos\theta_i + n_2\cos\theta_t)$ | −0.200 |
| $r_p$ | $(n_1\cos\theta_t - n_2\cos\theta_i)/(n_1\cos\theta_t + n_2\cos\theta_i)$ | −0.200 |

法線入射では $r_p = r_s$ が成立する。Q因子計算は $|r|^2$ 型のため、Hecht規約との BSDF 出力差は生じない。

---

## Appendix A. PSD法の偏光因子 Q 定義

### A.1. 基本式
Rayleigh-Rice 理論による BSDF:
$$\text{BSDF}(\theta_i, \theta_s) = \frac{16\pi^2}{\lambda^4} \cos\theta_i \cos^2\theta_s \cdot Q \cdot PSD(f_x, f_y)$$

### A.2. 完全形（デフォルト: `approx_mode=False`）

Elson-Bennett 形式。$\theta_i$ と $\theta_s$（透過時は $\theta_t$）の両方でフレネル係数を評価する。

**BRDFモード（反射）:**
$$Q_{s,\text{refl}} = \frac{|r_s(\theta_i)\cos\theta_s + r_s(\theta_s)\cos\theta_i|^2}{(2\cos\theta_i)^2}$$
$$Q_{p,\text{refl}} = \frac{|r_p(\theta_i)\cos\theta_s + r_p(\theta_s)\cos\theta_i|^2}{(2\cos\theta_i)^2}$$
$$Q_{u,\text{refl}} = \frac{Q_{s,\text{refl}} + Q_{p,\text{refl}}}{2}$$

**BTDFモード（透過）:**

エネルギー保存係数 $E = \dfrac{n_2 \cos\theta_t}{n_1 \cos\theta_i}$、$\theta_t$ はスネルの法則から導出。

$$Q_{s,\text{trans}} = E \cdot \frac{|t_s(\theta_i)\cos\theta_t + t_s(\theta_t)\cos\theta_i|^2}{(2\cos\theta_i)^2}$$
$$Q_{p,\text{trans}} = E \cdot \frac{|t_p(\theta_i)\cos\theta_t + t_p(\theta_t)\cos\theta_i|^2}{(2\cos\theta_i)^2}$$
$$Q_{u,\text{trans}} = \frac{Q_{s,\text{trans}} + Q_{p,\text{trans}}}{2}$$

フレネル係数（$n_1$: 入射側, $n_2$: 透過側）— **Born-Wolf 規約**を採用:
$$r_s(\theta) = \frac{n_1\cos\theta - n_2\cos\theta_t}{n_1\cos\theta + n_2\cos\theta_t}, \quad r_p(\theta) = \frac{n_1\cos\theta_t - n_2\cos\theta}{n_1\cos\theta_t + n_2\cos\theta}$$
$$t_s(\theta) = \frac{2n_1\cos\theta}{n_1\cos\theta + n_2\cos\theta_t}, \quad t_p(\theta) = \frac{2n_1\cos\theta}{n_2\cos\theta + n_1\cos\theta_t}$$

> **注記**: Born-Wolf 規約では法線入射（$\theta = 0$）で $r_p = r_s = (n_1 - n_2)/(n_1 + n_2)$ が成立する。
> Q因子式は $|r_p|^2$ 型の項のみを含むため、符号規約の選択はBSDF計算結果に影響しない。

### A.3. 簡略形（高速モード: `approx_mode=True`）

Stover 近似。$\theta_i$ のみでフレネル係数を評価する（広角散乱 $\theta_s > 30°$ では誤差が増大）。

**BRDFモード（反射）:**
$$Q_{s,\text{refl}} = |r_s(\theta_i)|^2, \quad Q_{p,\text{refl}} = |r_p(\theta_i)|^2 \cos^2(\theta_i + \theta_s)$$

**BTDFモード（透過）:**
$$Q_{s,\text{trans}} = E \cdot |t_s(\theta_i)|^2, \quad Q_{p,\text{trans}} = E \cdot |t_p(\theta_i)|^2 \cos^2(\theta_i - \theta_t)$$

**注意**: 簡略形は完全形の近似であり、$\theta_s = 0$ 代入によって完全形から導出されるものではない。

---

## 10. 変更履歴・既知バグ記録

コードの重要な変更・バグ修正を時系列で記録する。git コミット履歴の補足として、**判断の根拠・影響範囲・同種バグの有無**を記述する。

---

### 10.1. バグ修正履歴

#### [BUG-001] フレネル係数 r_p の符号規約 — 修正済み
- **発見日**: 初期実装レビュー時
- **影響ファイル**: `src/bsdf_sim/optics/fresnel.py`
- **症状**: `test_normal_incidence_rp` FAILED。法線入射で `r_p = +0.2`（期待値: `−0.2`）
- **原因**: Hecht 規約（$r_p = (n_2\cos\theta_i - n_1\cos\theta_t)/(n_2\cos\theta_i + n_1\cos\theta_t)$）を誤って実装。Born-Wolf 規約と符号が逆になっていた。
- **修正**: Born-Wolf 規約（$r_p = (n_1\cos\theta_t - n_2\cos\theta_i)/(n_1\cos\theta_t + n_2\cos\theta_i)$）に統一。法線入射で $r_p = r_s$ が成立するよう修正（Appendix A.2 参照）。
- **BSDF への影響**: Q 因子は $|r_p|^2$ 型のため BSDF 数値への影響なし。
- **対応コミット**: `8d422d1`

---

#### [BUG-002] FFT 軸の非単調性による RegularGridInterpolator エラー — 修正済み
- **発見日**: 初期実装レビュー時
- **影響ファイル**: `src/bsdf_sim/optics/fft_bsdf.py` — `sample_bsdf_at_angles()`
- **症状**: `test_sample_at_angles` FAILED。`ValueError: points must be strictly ascending`
- **原因**: FFT で得られる周波数軸は `[0, ..., N/2-1, -N/2, ..., -1]` の非単調順。`RegularGridInterpolator` は単調増加を要求するため例外が発生。
- **修正**: `np.argsort(u_axis)` で軸をソートし、BSDF グリッドを `bsdf[np.ix_(u_sort, v_sort)]` で並び替えてから補間器に渡す。
- **対応コミット**: `8d422d1`

---

#### [BUG-003] `compute_all_optical_metrics` の enabled デフォルト値 — 修正済み
- **発見日**: `bsdf simulate` で sparkle をコメントアウトしても停止する問題の調査中
- **影響ファイル**: `src/bsdf_sim/metrics/optical.py`
- **症状**: YAML で `haze` / `gloss` / `doi` / `sparkle` セクションをコメントアウトしても、該当指標が実行され続ける（sparkle は後述 BUG-004 により実質フリーズ）。
- **原因**: `cfg.get("sparkle", {}).get("enabled", True)` のパターン。セクションが欠落すると `{}` にフォールバックし、`enabled` のデフォルト `True` が適用されていた。全4指標に同一のバグが存在。
- **修正**: セクションの存在を先に確認する2段階チェックに変更。
  ```python
  # 修正前（バグ）
  if cfg.get("sparkle", {}).get("enabled", True):

  # 修正後
  sparkle_cfg = cfg.get("sparkle")
  if sparkle_cfg is not None and sparkle_cfg.get("enabled", True):
  ```
- **影響指標**: haze, gloss, doi, sparkle の全4指標（同種バグ）。
- **対応コミット**: `d22ae09`

---

#### [BUG-004] `compute_sparkle` の計算量が天文学的なループ数 — 修正済み
- **発見日**: BUG-003 調査中（`bsdf simulate` が FFT 完了後に無応答になる根本原因）
- **影響ファイル**: `src/bsdf_sim/metrics/optical.py` — `_compute_sparkle_single()`
- **症状**: smartphone プリセットで `bsdf simulate` が FFT 完了後に数時間フリーズ。
- **原因**: 画素の反復方向（全 UV 半球を画素サイズで分割）でループを設計していた。
  ```
  sin_half ≈ 0.0001033（smartphone: pixel=0.062mm, distance=300mm）
  n_pix_u = int(1.0 / 0.0001033) = 9,680
  ループ数: (2×9680+1)² ≈ 3.7億回
  各ループで 512×512 マスク演算 → 実質無限
  ```
- **修正**: 逆方向の発想（各グリッド点が属する画素を求める）でベクトル化。
  ```python
  pix_u = np.round(u_grid / (2 * sin_half)).astype(np.int32)
  pix_v = np.round(v_grid / (2 * sin_half)).astype(np.int32)
  _, inverse = np.unique(pixel_key, return_inverse=True)
  return np.bincount(inverse, weights=power_flat)
  ```
  計算量 O(N²) → 0.013s（旧実装: 数時間）。
- **対応コミット**: `d22ae09`

---

#### [BUG-005] `dynamicmap.py` の lru_cache リサイズが無効 — 修正済み
- **発見日**: Phase A 監査時
- **影響ファイル**: `src/bsdf_sim/visualization/dynamicmap.py`
- **症状**: `cache_size` パラメータを変更してもキャッシュサイズが変わらない。
- **原因**: `_cached_bsdf.__wrapped__` に新しい lru_cache を代入しても、`_cached_bsdf` 本体は元の `maxsize=256` のまま。`__wrapped__` は参照用属性であり、上書きしても呼び出し経路は変わらない。
- **修正**: 生の計算関数 `_compute_bsdf_raw()` を分離し、`__init__` でインスタンスごとに `functools.lru_cache(maxsize=cache_size)(_compute_bsdf_raw)` を生成して `self._cached_bsdf` に保持。
- **対応コミット**: `90f352e`

---

#### [BUG-006] `cli/main.py` で `--method psd` 実行時に NameError — 修正済み
- **発見日**: Phase A 監査時
- **影響ファイル**: `src/bsdf_sim/cli/main.py`
- **症状**: `bsdf simulate --method psd` で `NameError: name 'bsdf_fft' is not defined`
- **原因**: `u_primary / bsdf_primary` の追跡変数がなく、`method='psd'` の場合に FFT 変数（`bsdf_fft`）を参照するコードパスが存在していた。
- **修正**: `u_primary, v_primary, bsdf_primary = None` を明示的に初期化し、FFT/PSD ブロックそれぞれで代入するよう変更。
- **対応コミット**: `90f352e`

---

### 10.2. 仕様変更履歴

| 日付 | 変更内容 | 対応コミット |
|---|---|---|
| 初期実装 | Python パッケージ全体・テスト 53件 | `8d422d1` |
| JIS/ISO 形状指標追加 | ISO 25178-2 S-パラメータ 11項目・JIS B 0601 R-パラメータ 9項目 | `27331e6` |
| spec/README 更新 | 形状指標仕様を spec_main.md Section 5.1・README に反映 | `4a7f9af` |
| Phase A〜D 修正一括 | BUG-005/006・to_bsdf_2d()・from_config()・sparkle RGB加重・テスト 76→131件 | `90f352e`, `a508446` |
| BUG-003/004 修正 | enabled デフォルト値バグ（4指標）・sparkle ベクトル化・プログレスバー追加 | `d22ae09` |
| 光学指標の手法別記録・目的関数統一 | simulate: haze_fft/haze_psd/haze_ml 形式で手法別記録。optimize: 全指標を統一記録＋objectives を config.yaml で指定可能に。_psd 指標指定時は PSD 自動計算 | — |
| visualize レポート強化 | bsdf visualize: 1D プロファイル＋2D ヒートマップ（手法別）＋指標テーブルの Panel HTML 出力。simulate --log-to-mlflow: 表面形状 PNG・2D BSDF PNG を artifacts/plots/ に保存 | — |
| 装置固有 CSV ローダー（プラグイン方式） | custom_surfaces/ に MeasuredSurface サブクラスを置くだけで自動登録。DeviceXyzSurface を実装例として追加。sample_inputs/ にサンプルファイル・config を格納 | `b35813b` |
| Keyence VK-X シリーズ対応 | DeviceVk6Surface を追加。Shift-JIS・ヘッダからピクセルサイズ・単位を自動取得。sample_inputs/device_vk6_sample.csv・config_device_vk6.yaml を同梱 | `67a1931` |
| 4 種パディング方式追加 | MeasuredSurface に zeros/tile/reflect/smooth_tile パディングを実装。DeviceVk6Surface の pixel_size_um / grid_size ヘッダ自動算出を実装。テスト 131→172件 | — |
| BSDF 実測ファイルリーダー（プラグイン方式） | custom_bsdf_readers/ に BaseBsdfFileReader サブクラスを置くだけで自動登録。LightToolsBsdfReader（LightTools/MiniDiff .bsdf）を実装。build_measured_dataframe に is_btdf 引数を追加。テスト 172→213件 | — |
| 多条件シミュレーション＋実測 BSDF 統合 | 1 run 内で多波長・多入射角・BRDF/BTDF を実行可能に（案 2-B: スカラ/list 両対応）。`simulation.wavelength_um` / `theta_i_deg` / `mode` に list を指定すると直積展開。`measured_bsdf` セクションで実測ファイルを紐づけ、`match_measured: true` で実測条件を sim に自動採用、`--measured-bsdf` CLI オプションで上書き。条件ごとに `merge_sim_and_measured()` が Log-RMSE を自動計算。`select_block` / `get_conditions` ヘルパー追加。テスト 213→255件 | — |
| visualize 実測オーバーレイ＋多条件対応 | `plot_bsdf_1d_overlay` で条件未指定時に df 先頭を自動選択（多条件 Parquet でも "データなし" にならない）。`mode='BRDF'/'BTDF'` フィルタ追加。`plot_bsdf_report` は多条件時に `pn.Tabs` で条件ごとに 1D+2D+Log-RMSE を切替表示。実測行（`method='measured'`）は 1D に黒点 Scatter で自動オーバーレイ。テスト 255→269件 | — |
| Sparkle illumination 削除 | `compute_sparkle` の `bsdf_per_wavelength` 引数と CIE 輝度加重分岐を削除（多条件 simulate ループから到達不能な dead code だった）。`_ILLUMINATION_PRESETS` / `_RGB_LUMINANCE_WEIGHTS` 定数を除去。config の `sparkle.illumination` セクションは読み込み時に silently 無視される（後方互換） | — |
| 代表波長で規格準拠 Haze/Gloss/DOI | `metrics.representative_wavelength_um`（デフォルト 0.555 μm, V(λ) ピーク）を追加。Haze/Gloss/DOI は `wavelength_um` list の影響を受けず代表波長 1 条件でのみ計算され、列は `haze_fft`/`gloss_fft`/`doi_fft` に固定（波長サフィックス無し）。代表波長が list に含まれない場合は追加 1 条件 sim を実行、含まれる場合は再利用。Sparkle/Log-RMSE は従来通り波長ごと。spec Section 5.3「規格対応と波長依存性」を追加。テスト 269→273件 | — |
| `bsdf dashboard` CLI + 多モデル対応 DynamicMap | 新 CLI サブコマンド `bsdf dashboard --config file.yaml --port 5006` でリアルタイムブラウザダッシュボードを起動。config.surface.model を判定して `RandomRoughDynamicMap` / `SphericalArrayDynamicMap` / `MeasuredSurfaceDynamicMap` のいずれかを起動（`create_dashboard_from_config()` ファクトリ）。`measured_bsdf.path` 指定時は `select_block()` で条件一致の実測ブロックを 1D プロファイルに黒点 Scatter で自動オーバーレイ。SphericalArraySurface は radius/pitch/placement/overlap_mode のスライダー＆セレクタ UI を追加、MeasuredSurface 系は固定表示。テスト 273→286件 | — |
