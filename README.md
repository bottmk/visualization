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
  --measured-bsdf PATH       BSDF 実測ファイル（.bsdf 等）
                             config.yaml の measured_bsdf.path を上書き
  --match-measured /         実測ファイルの条件を sim 条件に自動採用するか
    --no-match-measured
```

**多条件 simulate**:
`config.yaml` の `simulation.wavelength_um` / `theta_i_deg` / `mode` にスカラでもリストでも指定可能。
リスト指定時は直積で展開され、1 run で複数条件の BSDF を計算する。

```yaml
simulation:
  wavelength_um: [0.465, 0.525, 0.630]   # 3 波長
  theta_i_deg: [0, 20, 40, 60]            # 4 入射角
  mode: ['BRDF', 'BTDF']                  # 2 モード → 24 条件を実行
```

**出力ファイル:**

| ファイル | 内容 |
|---|---|
| `outputs/bsdf_data.parquet` | 計算結果（ロング形式、1行=1散乱点。実測 BSDF 指定時は `method='measured'` 行も含む） |

**`--log-to-mlflow` 有効時の artifacts**:
MLflow の当該 run に以下が自動保存される。
- `data/bsdf_data.parquet`
- `plots/surface.png` / `plots/surface.html`
- `plots/bsdf_2d_<method>[条件サフィックス].png`
- `plots/bsdf_report.html`（多条件時は Panel Tabs で条件別、実測データは黒点 Scatter オーバーレイ）

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

### `bsdf surface` — 表面形状の可視化

```
bsdf surface [オプション]

オプション:
  -c, --config PATH          設定ファイルパス（YAML）        [必須]
  -o, --output PATH          出力ファイルパス                [デフォルト: surface.html/png]
  --format [html|png]        出力形式                        [デフォルト: html]
  --unit [nm|um]             高さの表示単位                  [デフォルト: nm]
  --colormap TEXT            カラーマップ名                  [デフォルト: RdYlBu_r]
```

出力内容（4パネル構成）:

| パネル | 内容 |
|---|---|
| 2D カラーマップ | 高さ分布を色で表示（単位: nm または μm） |
| 高さ分布ヒストグラム | ピクセル高さの頻度分布 |
| X 断面プロファイル | y=中央での高さプロファイル |
| Y 断面プロファイル | x=中央での高さプロファイル |

```bash
# PNG（静止画）で保存
bsdf surface --config config.yaml --format png --unit nm

# インタラクティブ HTML で保存
bsdf surface --config config.yaml --format html --output surface.html
```

---

### `bsdf dashboard` — リアルタイム BSDF ダッシュボード

config.yaml の `surface.model` に応じたブラウザダッシュボードを起動する。
スライダーを動かすと BSDF が即時更新される（3 段階解像度、LRU キャッシュ）。
`measured_bsdf.path` 指定時は 1D プロファイルに実測データが黒点オーバーレイされる。

```
bsdf dashboard [オプション]

オプション:
  -c, --config PATH             設定ファイルパス（YAML）    [必須]
  -p, --port INTEGER            Panel サーバーポート        [デフォルト: 5006]
  --preview-grid INTEGER        プレビュー計算グリッドサイズ [デフォルト: 512]
  --no-browser                  起動時にブラウザを開かない
```

```bash
# RandomRoughSurface のパラメータ探索
bsdf dashboard --config config.yaml

# Keyence VK-X 実測形状 + LightTools 実測 BSDF 比較
bsdf dashboard --config sample_inputs/config_device_vk6.yaml
```

対応モデル: `RandomRoughSurface` / `SphericalArraySurface` / `MeasuredSurface` 系（`DeviceVk6Surface` 等）

---

### `bsdf runs list` — MLflow run 一覧表示

optimize で大量 run を生成したあと、メトリクス順で確認し `run_id` のショートカット（short_id など）をコピーするためのコマンド。

```
bsdf runs list [オプション]

オプション:
  --tracking-uri TEXT           MLflow トラッキング URI       [デフォルト: mlruns]
  -e, --experiment TEXT         実験名                       [デフォルト: 01_BSDF_Raw_Data]
  -s, --sort-by TEXT            並び順の基準メトリクス名
  --ascending/--descending      sort-by 指定時の昇順/降順     [デフォルト: ascending]
  -n, --limit INTEGER           最大表示件数                 [デフォルト: 20]
  -m, --metrics TEXT            表示列のカスタマイズ（カンマ区切り）
```

```bash
# haze_fft 昇順で上位 10 件
bsdf runs list --sort-by haze_fft --limit 10

# 表示メトリクスをカスタム指定
bsdf runs list -s sparkle_fft -m haze_fft,sparkle_fft --descending
```

---

### `bsdf visualize` — 結果の可視化

```
bsdf visualize [オプション]

オプション:
  --run-id TEXT                 MLflow の run_id              [必須]
                                完全 ID / latest / latest-N /
                                best:METRIC[:max] / 8文字以上のプレフィックス可
  --tracking-uri TEXT           MLflow トラッキング URI      [デフォルト: mlruns]
  -e, --experiment TEXT         run_id 解決時の実験名         [デフォルト: 01_BSDF_Raw_Data]
  -o, --output PATH             出力 HTML パス               [デフォルト: report.html]
  --scale [linear|log]          BSDF 軸スケール              [デフォルト: log]
  --log-to-mlflow /             生成 HTML を元 run の
    --no-log-to-mlflow           artifacts/plots/ に書き戻す  [デフォルト: no]
```

```bash
# 完全 run_id で
bsdf visualize --run-id abc123ef01234567890abcdef01234567 --output bsdf_plot.html

# ショートカット: 最新 / 最良 / プレフィックス
bsdf visualize --run-id latest --log-to-mlflow
bsdf visualize --run-id best:haze_fft --log-to-mlflow
bsdf visualize --run-id abc123ef --log-to-mlflow   # 8 文字以上
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

`--log-to-mlflow`（デフォルト有効）の場合、`02_Analysis_Reports` 実験に新規 run が作成され、
`params.source_run_ids`（カンマ区切りの比較対象 ID）と `artifacts/reports/comparison_report.html` が保存される。

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

**MeasuredSurface パラメータ（汎用CSVローダー）:**

```yaml
  measured:
    path: 'data/surface.csv'      # CSV ファイルパス
    source_pixel_size_um: 1.0     # 元データのピクセルサイズ [μm]
    height_unit: 'nm'             # 'um' / 'nm' / 'm'
    skiprows: 0                   # ヘッダ行のスキップ数
    leveling: true                # 傾き・うねり成分を除去する
```

CSV は数値のみのカンマ区切り行列形式を想定。

**装置固有フォーマット（プラグイン方式）:**

`MeasuredSurface` のサブクラスを `custom_surfaces/<name>.py` に実装すると自動登録される。

同梱されている実装例:

| クラス名 | ファイル | 対象装置 | 特徴 |
|---|---|---|---|
| `DeviceXyzSurface` | `custom_surfaces/device_xyz.py` | 装置XYZ（参考実装） | タブ区切り・5行ヘッダ・nm単位 |
| `DeviceVk6Surface` | `custom_surfaces/device_vk6.py` | Keyence VK-X シリーズ | Shift-JIS・ピクセルサイズ/単位をヘッダから自動取得 |

```yaml
# Keyence VK-X シリーズの例
surface:
  model: 'DeviceVk6Surface'   # custom_surfaces/device_vk6.py が自動ロードされる
  measured:
    path: 'sample_inputs/device_vk6_sample.csv'
    # source_pixel_size_um は省略可（ヘッダから自動取得）
    leveling: true
```

サンプルファイルと設定例は `sample_inputs/` フォルダを参照。

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
mlflow ui --backend-store-uri mlruns # ブラウザで http://localhost:5000 を開く
uv run mlflow server --backend-store-uri mlruns --host 0.0.0.0 --port 5000 # 他PCに公開
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

`custom_surfaces/` ディレクトリに Python ファイルを配置する。起動時に自動ロード・登録される。

**パターン1: 新しい形状生成モデル**

```python
# custom_surfaces/my_surface.py
from bsdf_sim.models.base import BaseSurfaceModel
import numpy as np

class MySurface(BaseSurfaceModel):
    def _generate(self, grid_size: int, pixel_size_um: float) -> np.ndarray:
        # 任意の高さ配列を返す（shape: (grid_size, grid_size), 単位: μm）
        return np.zeros((grid_size, grid_size), dtype=np.float32)
```

**パターン2: 装置固有CSVローダー（`MeasuredSurface` のサブクラス）**

```python
# custom_surfaces/my_device.py
from bsdf_sim.models.measured import MeasuredSurface
import numpy as np
from pathlib import Path

class MyDeviceSurface(MeasuredSurface):
    @classmethod
    def from_config(cls, config):
        surface_cfg = config.get("surface", {})
        measured_cfg = surface_cfg.get("measured", {})
        path = Path(measured_cfg["path"])
        data = np.loadtxt(path, delimiter="\t", skiprows=5) * 1e-3  # nm → μm
        return cls(
            height_data=data,
            source_pixel_size_um=float(measured_cfg.get("source_pixel_size_um", 0.5)),
            grid_size=int(surface_cfg.get("grid_size", 4096)),
            pixel_size_um=float(surface_cfg.get("pixel_size_um", 0.25)),
        )
```

config.yaml で指定：

```yaml
surface:
  model: 'MySurface'   # または 'MyDeviceSurface'
```

同梱の実装例: `DeviceXyzSurface`（参考実装）、`DeviceVk6Surface`（Keyence VK-X シリーズ）  
サンプルファイルと動作確認用 config は `sample_inputs/` を参照。

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

現在のテスト数: **311件**（全 pass）

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

[MIT License](LICENSE) © 2025 bottmk
