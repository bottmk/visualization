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
  - 生ファイルフォーマット: 装置ごとのヘッダ情報を含む形式。実ファイル提供時にローダー仕様を追記する。

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
  - RMS粗さ（$Rq$ / $Sq$）: 高さの二乗平均平方根。
  - 算術平均粗さ（$Ra$ / $Sa$）: 高さの絶対値の平均。
  - 最大断面高さ（$Rz$ / $Sz$）: Peak-to-Valley（PV値）。
  - RMS傾斜角（$Sdq$）: 局所的な傾き（スロープ）の二乗平均平方根。

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

  # ── グループ3：照明条件 ──────────────────────────────────
  illumination:
    preset: 'green'           # 'green' / 'rgb' / 'custom'
    wavelengths_um: null      # null のときはプリセット値を使用
                              # 単波長: [0.55] / RGB: [0.45, 0.55, 0.65]
  # rgb プリセット: 3波長を個別計算後、輝度加重平均で合成

  # ── グループ4：評価指標（固定） ────────────────────────────
  metrics:
    integration_area: 'single_pixel'   # 1画素領域で積分（固定）
    sparkle_index: 'sigma_over_mu'     # Cs = σ/μ（固定）
```

**プリセット優先ルール**: 数値が指定されている場合（null でない）は数値を優先。数値が null の場合はプリセット値を使用。プリセットも未指定の場合はエラーとする。

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

#### 9.2.4. 光学指標（`tests/test_metrics.py`）

| テスト名 | 検証内容 |
|---|---|
| `test_rq_flat` | 平坦面の Rq = 0 |
| `test_rq_sine` | 正弦波の Rq = amplitude/√2（相対誤差 5%） |
| `test_sdq_flat` | 平坦面の Sdq = 0 |
| `test_sdq_sine` | 正弦波の Sdq > 0 |
| `test_all_metrics_keys` | compute_all_surface_metrics のキー集合 |
| `test_perfect_match` | Log-RMSE = 0（完全一致） |
| `test_floor_masking` | フロア以下の点が誤差計算から除外される |
| `test_order_of_magnitude_error` | 10倍差 → Log-RMSE = 1.0 |
| `test_all_below_floor` | 全点フロア以下 → inf |
| `test_haze_range` | Haze ∈ [0, 1] |
| `test_haze_narrow_beam` | 法線集中ビームのヘイズ < 0.5 |
| `test_gloss_positive` | Gloss ≥ 0 |
| `test_doi_range` | DOI ∈ [0, 1] |
| `test_doi_narrow_beam` | 法線集中ビームの DOI > 0.5 |

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
| CLI エンドツーエンド（`bsdf simulate`） | 高 | subprocess でコマンド実行・出力確認 |
| Parquet スキーマの読み書き往復 | 中 | 保存 → 読み込みでデータが一致するか |
| MeasuredHeightMap CSV 読み込み | 中 | 単位変換・NaN補間の確認 |
| DynamicMap / Panel 可視化 | 低 | UIテストは手動確認 |
| Optuna 最適化収束 | 低 | 既知パラメータへの収束確認 |

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
