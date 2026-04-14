# bsdf-sim — AGフィルム光学散乱（BSDF）シミュレーション・最適化プログラム

表面形状データを入力とし、FFT法・PSD法でBSDF（双方向散乱分布関数）をシミュレーションする。
Optunaによる自動最適化、MLflowによる実験管理、HoloViewsによるインタラクティブ可視化をサポート。

---

## 必要環境

| ソフトウェア | バージョン |
|---|---|
| Python | 3.10 以上 |
| pip | 23.0 以上推奨 |

---

## インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/bottmk/visualization.git
cd visualization
```

### 2. 仮想環境の作成（推奨）

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. パッケージのインストール

```bash
pip install -e .
```

開発用ツール（pytest等）も含める場合：

```bash
pip install -e ".[dev]"
```

### 4. インストールの確認

```bash
bsdf --version
# bsdf, version 0.1.0
```

---

## クイックスタート

### シミュレーション単体実行

```bash
bsdf simulate --config config.yaml
```

実行結果は `outputs/bsdf_data.parquet` に保存される。

### MLflow に記録しながら実行

```bash
bsdf simulate --config config.yaml --log-to-mlflow
```

### Optuna による自動最適化

```bash
bsdf optimize --config config.yaml --trials 200
```

---

## CLIコマンド一覧

### `bsdf simulate` — シミュレーション単体実行

```
bsdf simulate [オプション]

オプション:
  -c, --config PATH          設定ファイルパス（YAML）        [必須]
  -o, --output-dir PATH      出力ディレクトリ                [デフォルト: outputs]
  -m, --method [fft|psd|both] 計算手法                       [デフォルト: both]
  --save-parquet / --no-save-parquet  Parquet 保存の有無     [デフォルト: 保存する]
  --log-to-mlflow / --no-log-to-mlflow  MLflow 記録の有無    [デフォルト: 記録しない]
```

**出力ファイル:**

| ファイル | 内容 |
|---|---|
| `outputs/bsdf_data.parquet` | 計算結果（ロング形式、1行=1散乱点） |

---

### `bsdf optimize` — Optuna による自動最適化

```
bsdf optimize [オプション]

オプション:
  -c, --config PATH    設定ファイルパス（YAML）    [必須]
  -n, --trials INT     試行回数（設定ファイルを上書き）
  --study-name TEXT    Optuna Study 名
```

最適化結果はMLflowに自動記録される。パレートフロントの試行がログに出力される。

---

### `bsdf visualize` — 結果の可視化

```
bsdf visualize [オプション]

オプション:
  --run-id TEXT                 MLflow の run_id              [必須]
  --tracking-uri TEXT           MLflow トラッキング URI      [デフォルト: mlruns]
  -o, --output PATH             出力 HTML パス               [デフォルト: report.html]
  --scale [linear|log]          BSDF 軸スケール              [デフォルト: log]
```

```bash
bsdf visualize --run-id abc123ef --output bsdf_plot.html
```

---

### `bsdf report` — 複数Runの比較レポート生成

```
bsdf report [オプション]

オプション:
  --run-ids TEXT                カンマ区切りの run_id リスト  [必須]
  --tracking-uri TEXT           MLflow トラッキング URI      [デフォルト: mlruns]
  -o, --output PATH             出力 HTML パス               [デフォルト: comparison_report.html]
  --log-to-mlflow / --no-log-to-mlflow  レポートをMLflow記録  [デフォルト: 記録する]
```

```bash
bsdf report --run-ids abc123,def456,ghi789 --output compare.html
```

---

## 設定ファイル（config.yaml）

`config.yaml` がすべてのパラメータを管理する。各セクションの主要項目を示す。

### simulation — 光学条件

```yaml
simulation:
  wavelength_um: 0.55       # 波長 [μm]（例: 550nm = 0.55μm）
  theta_i_deg: 0.0          # 入射天頂角 [deg]
                            #   0〜90°未満 → BRDF（反射）
                            #   90°超〜180° → BTDF（透過）
                            #   90° → エラー
  phi_i_deg: 0.0            # 入射方位角 [deg]
  n1: 1.0                   # 入射側媒質の屈折率（空気=1.0）
  n2: 1.5                   # 透過側媒質の屈折率（ガラス=1.5）
  polarization: 'Unpolarized'  # 'S' / 'P' / 'Unpolarized'
```

### surface — 表面形状モデル

3種類のモデルを選択できる。

```yaml
surface:
  model: 'RandomRoughSurface'   # モデル名
  grid_size: 4096               # 計算グリッドサイズ（物理サイズ = grid_size × pixel_size_um）
  pixel_size_um: 0.25           # ピクセルサイズ [μm]
```

| モデル名 | 説明 |
|---|---|
| `RandomRoughSurface` | FFTフィルタ法によるガウス統計ランダム粗面 |
| `SphericalArraySurface` | 球面要素アレイ（Grid/Hexagonal/Random/PoissonDisk配置） |
| `MeasuredSurface` | 実測高さデータのCSV読み込み |

**RandomRoughSurface パラメータ:**

```yaml
  random_rough:
    rq_um: 0.005        # RMS粗さ [μm]（例: 5nm = 0.005μm）
    lc_um: 2.0          # 相関長 [μm]
    fractal_dim: 2.5    # フラクタル次元（2.0〜3.0）
```

**SphericalArraySurface パラメータ:**

```yaml
  spherical_array:
    radius_um: 50.0           # 球曲率半径 [μm]
    pitch_um: 50.0            # 配置ピッチ [μm]
    placement: 'Hexagonal'    # 'Grid' / 'Hexagonal' / 'Random' / 'PoissonDisk'
    overlap_mode: 'Maximum'   # 'Maximum'（最大値採用） / 'Additive'（加算）
```

**MeasuredSurface パラメータ:**

```yaml
  measured:
    path: 'data/surface.csv'      # CSV ファイルパス
    source_pixel_size_um: 1.0     # 元データのピクセルサイズ [μm]
    height_unit: 'nm'             # 'um' / 'nm' / 'm'
    skiprows: 0                   # ヘッダ行のスキップ数
    leveling: true                # 傾き・うねり成分を除去する
```

CSV は数値のみの行列形式（ヘッダなし）を想定。装置固有のフォーマットがある場合は `skiprows` でスキップ行数を指定する。

### psd — PSD法オプション

```yaml
psd:
  approx_mode: false    # false=完全形（Elson-Bennett） / true=簡略形（Stover、高速）
```

### error_metrics — 誤差計算

```yaml
error_metrics:
  bsdf_floor: 1.0e-6    # ノイズフロア [sr⁻¹]（装置に合わせて変更）
```

### metrics — 評価指標

```yaml
metrics:
  haze:
    enabled: true
    half_angle_deg: 2.5         # ヘイズ境界角（2.5°以上を散乱光として扱う）
  gloss:
    enabled: true
    angle_deg: 60.0             # グロス測定角
  doi:
    enabled: true               # 写像性（0〜1）
  sparkle:
    enabled: true
    viewing:
      preset: 'smartphone'      # 'smartphone' / 'tablet' / 'monitor'
    display:
      preset: 'fhd_smartphone'  # 'fhd_smartphone' / 'qhd_monitor' / '4k_monitor'
    illumination:
      preset: 'green'           # 'green'（550nm） / 'rgb'（450/550/650nm）
```

プリセット値は `null` を設定すると適用される。数値を明示すると上書きされる。

---

### 表面形状指標（JIS/ISO準拠）

シミュレーション実行時に高さマップから自動計算される。

**ISO 25178-2 S-パラメータ（面粗さ）**

| 記号 | 名称 | 単位 |
|---|---|---|
| Sq | 二乗平均平方根高さ | μm |
| Sa | 算術平均高さ | μm |
| Sp | 最大山高さ | μm |
| Sv | 最大谷深さ | μm |
| Sz | 最大高さ（= Sp + Sv） | μm |
| Ssk | スキューネス（> 0: 突起多、< 0: くぼみ多） | 無次元 |
| Sku | クルトシス（= 3: ガウス、> 3: 急峻） | 無次元 |
| Sdq | 二乗平均平方根傾斜 | rad |
| Sdr | 界面展開面積比 | % |
| Sal | 自己相関長 | μm |
| Str | テクスチャアスペクト比（0: 異方性、1: 等方性） | 無次元 |

**JIS B 0601 / ISO 4287 R-パラメータ（プロファイル粗さ）**

行・列両方向のプロファイルを平均して算出する。

| 記号 | 名称 | 単位 |
|---|---|---|
| Rq | 二乗平均平方根粗さ | μm |
| Ra | 算術平均粗さ | μm |
| Rz | 最大高さ | μm |
| Rp | 最大山高さ | μm |
| Rv | 最大谷深さ | μm |
| Rsk | スキューネス | 無次元 |
| Rku | クルトシス | 無次元 |
| Rsm | 輪郭曲線要素の平均幅 | μm |
| Rc | 輪郭曲線要素の平均高さ | μm |

### optuna — 最適化

```yaml
optuna:
  n_trials: 200
  n_jobs: 1                 # 並列試行数
  sampler: 'MOTPE'          # 'MOTPE'（多目的） / 'TPE'（単目的）
  objectives:
    - metric: 'log_rmse'
      direction: 'minimize'
    - metric: 'sparkle'
      direction: 'minimize'
  duplicate_skip:
    enabled: true
    distance_threshold: 0.01   # 正規化ユークリッド距離閾値（重複試行をスキップ）
```

### mlflow — 実験管理

```yaml
mlflow:
  tracking_uri: 'mlruns'                  # ローカルディレクトリ or リモートURI
  experiment_name: '01_BSDF_Raw_Data'
```

MLflow UIの起動：

```bash
mlflow ui --backend-store-uri mlruns
# ブラウザで http://localhost:5000 を開く
```

---

## 出力データ形式

### Parquet（BSDFデータテーブル）

ロング形式（1行 = 1散乱点 × 1計算手法）で保存される。

| カラム名 | 型 | 内容 |
|---|---|---|
| `u` | float32 | 方向余弦 u = sin(θ_s)cos(φ_s) |
| `v` | float32 | 方向余弦 v = sin(θ_s)sin(φ_s) |
| `bsdf` | float32 | BSDF値 [sr⁻¹] |
| `method` | category | `'FFT'` / `'PSD'` / `'MultiLayer'` / `'measured'` |
| `theta_i_deg` | float32 | 入射天頂角 [deg] |
| `phi_i_deg` | float32 | 入射方位角 [deg] |
| `wavelength_um` | float32 | 波長 [μm] |
| `polarization` | category | `'S'` / `'P'` / `'Unpolarized'` |
| `is_btdf` | bool | BTDF フラグ |

---

## DynamicMap プレビュー（インタラクティブUI）

パラメータスライダーを動かしながらリアルタイムにBSDFプレビューを確認できる。

```python
from bsdf_sim.visualization.dynamicmap import RandomRoughDynamicMap

dm = RandomRoughDynamicMap(config_path="config.yaml")
dashboard = dm.create_dashboard()
dashboard.show()   # ブラウザで起動
```

| フェーズ | グリッドサイズ | 用途 |
|---|---|---|
| ドラッグ中 | 128 | リアルタイム応答（~0.1秒） |
| ドラッグ停止後 | 512 | パラメータ確認（~0.6秒） |
| 本計算ボタン | 4096 | Optuna最適化用（~数分） |

> **注意**: `reduced_resolution` プレビューモードでは散乱角上限が約7.9°となり、ヘイズの確認ができない。ヘイズを評価する場合は `reduced_area` モードを使用すること。

---

## プラグイン拡張

### カスタム表面モデルの追加

`custom_surfaces/` ディレクトリに Python ファイルを配置する。

```python
# custom_surfaces/my_surface.py
from bsdf_sim.models.base import BaseSurfaceModel
import numpy as np

class MySurface(BaseSurfaceModel):
    def _generate(self, grid_size: int, pixel_size_um: float) -> np.ndarray:
        # 任意の高さ配列を返す（shape: (grid_size, grid_size), 単位: μm）
        return np.zeros((grid_size, grid_size), dtype=np.float32)
```

config.yaml で指定：

```yaml
surface:
  model: 'MySurface'
```

### カスタム評価指標の追加

```python
# custom_metrics/my_metric.py
import numpy as np

def compute_my_metric(u_grid, v_grid, bsdf, **kwargs) -> float:
    return float(np.max(bsdf))
```

---

## テストの実行

```bash
# 全テスト実行
pytest tests/

# カバレッジ付き
pytest tests/ --cov=bsdf_sim --cov-report=html

# 特定ファイルのみ
pytest tests/test_fft_bsdf.py -v
```

現在のテスト数: **131件**（全 pass）

| テストファイル | 対象 | テスト数 |
|---|---|---|
| `test_models.py` | 表面形状モデル | 14 |
| `test_fft_bsdf.py` | Fresnel係数・FFT法BSDF | 11 |
| `test_psd_bsdf.py` | PSD・Q因子・BSDF | 9 |
| `test_metrics.py` | 表面形状指標（ISO 25178-2 / JIS B 0601）・光学指標 | 42 |
| `test_config_loader.py` | BSDFConfig 読み込み・プリセット解決・バリデーション | 17 |
| `test_optimization.py` | 重複スキップ・BSDFOptimizer | 13 |
| `test_measured.py` | MeasuredSurface 前処理・CSV・from_config | 17 |
| `test_cli.py` | CLI コマンドスモークテスト | 8 |
| 合計 | | **131** |

---

## 物理単位系

本プログラム内部はすべて **μm（マイクロメートル）** に統一されている。

| 外部入力 | 変換 |
|---|---|
| 波長 [nm] | ÷ 1000 → [μm] |
| 高さデータ [nm] | ÷ 1000 → [μm] |
| 観察距離・画素ピッチ [mm] | mm のまま（別系列） |

---

## ライセンス

本リポジトリのライセンスについてはリポジトリオーナーに確認すること。
