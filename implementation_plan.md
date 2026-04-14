# 実装計画書

## 作業全容

`spec_main.md` および `spec_decisions.md` に基づき、AGフィルム光学散乱（BSDF）シミュレーション・最適化プログラムを Python で実装する。

---

## ディレクトリ構成

```
/workspaces/visualization/
├── pyproject.toml                           # プロジェクト設定・依存パッケージ
├── requirements.lock                        # 完全固定バージョン（CI/CD用）
├── config.yaml                              # 設定ファイルサンプル（全セクション）
├── custom_surfaces/                         # プラグイン用（ユーザー追加形状モデル）
│   └── .gitkeep
├── custom_metrics/                          # プラグイン用（ユーザー追加評価指標）
│   └── .gitkeep
├── src/
│   └── bsdf_sim/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py                  # プラグイン動的ロード
│       │   ├── base.py                      # HeightMap dataclass + BaseSurfaceModel
│       │   ├── random_rough.py              # RandomRoughSurface
│       │   ├── spherical_array.py           # SphericalArraySurface
│       │   └── measured.py                  # MeasuredSurface
│       ├── optics/
│       │   ├── __init__.py
│       │   ├── fresnel.py                   # フレネル係数（r_s, r_p, t_s, t_p）
│       │   ├── fft_bsdf.py                  # FFT法（スカラー回折・BRDF/BTDF）
│       │   ├── psd_bsdf.py                  # PSD法（Q因子 完全形/簡略形）
│       │   └── multilayer.py                # Adding-Doubling法
│       ├── metrics/
│       │   ├── __init__.py                  # プラグイン動的ロード
│       │   ├── surface.py                   # Rq, Ra, Rz, Sdq
│       │   └── optical.py                   # Haze, Gloss, DOI, Sparkle（Cs=σ/μ）
│       ├── io/
│       │   ├── __init__.py
│       │   ├── config_loader.py             # YAML設定読み込み・バリデーション
│       │   └── parquet_schema.py            # Parquet読み書き（long format）
│       ├── optimization/
│       │   ├── __init__.py
│       │   ├── optuna_runner.py             # Optuna（多目的・重複スキップ）
│       │   └── mlflow_logger.py             # MLflow（3階層Experiment管理）
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── holoviews_plots.py           # 3者オーバーレイ・スケール切替UI
│       │   └── dynamicmap.py                # DynamicMap（3段階解像度・LRUキャッシュ）
│       └── cli/
│           ├── __init__.py
│           └── main.py                      # CLIエントリポイント（4サブコマンド）
└── tests/
    ├── __init__.py
    ├── test_models.py
    ├── test_fft_bsdf.py
    ├── test_psd_bsdf.py
    └── test_metrics.py
```

---

## フェーズ別作業計画

### フェーズ1：基盤（依存なし）
| # | ファイル | 内容 |
|---|---|---|
| 1 | `pyproject.toml` | プロジェクト設定、依存パッケージ範囲指定、CLIエントリポイント |
| 2 | `requirements.lock` | 完全固定バージョン |
| 3 | `config.yaml` | 全セクションのサンプル設定ファイル |
| 4 | `src/bsdf_sim/models/base.py` | `HeightMap` dataclass、`BaseSurfaceModel` 基底クラス |
| 5 | `src/bsdf_sim/io/config_loader.py` | YAML読み込み、プリセット解決、バリデーション |

### フェーズ2：表面形状モデル
| # | ファイル | 内容 |
|---|---|---|
| 6 | `src/bsdf_sim/models/__init__.py` | `custom_surfaces/` プラグイン動的ロード |
| 7 | `src/bsdf_sim/models/random_rough.py` | FFTフィルタ法でGaussian粗面生成（Rq, Lc, フラクタル次元） |
| 8 | `src/bsdf_sim/models/spherical_array.py` | レンズ配置（Grid/Hexagonal/Random/PoissonDisk）＋重なり処理 |
| 9 | `src/bsdf_sim/models/measured.py` | 実測高さデータのローダー基盤（欠損補間・リサンプリング・レベリング） |

### フェーズ3：光学計算
| # | ファイル | 内容 |
|---|---|---|
| 10 | `src/bsdf_sim/optics/fresnel.py` | r_s, r_p, t_s, t_p（スネル則含む） |
| 11 | `src/bsdf_sim/optics/fft_bsdf.py` | 位相変換・2DFFT・UV空間マッピング・BRDF/BTDF自動判定 |
| 12 | `src/bsdf_sim/optics/psd_bsdf.py` | PSD算出・Q因子（完全形/簡略形）・BSDF変換 |
| 13 | `src/bsdf_sim/optics/multilayer.py` | Adding-Doubling法（プリセット離散化・HGボリューム散乱） |
| 14 | `src/bsdf_sim/optics/__init__.py` | 公開API |

### フェーズ4：評価指標
| # | ファイル | 内容 |
|---|---|---|
| 15 | `src/bsdf_sim/metrics/surface.py` | Rq, Ra, Rz, Sdq |
| 16 | `src/bsdf_sim/metrics/optical.py` | Haze, Gloss, DOI, Sparkle（Cs=σ/μ、プリセット対応） |
| 17 | `src/bsdf_sim/metrics/__init__.py` | `custom_metrics/` プラグイン動的ロード |

### フェーズ5：データ入出力
| # | ファイル | 内容 |
|---|---|---|
| 18 | `src/bsdf_sim/io/parquet_schema.py` | long format Parquet読み書き、実測CSVインポート |
| 19 | `src/bsdf_sim/io/__init__.py` | 公開API |

### フェーズ6：最適化・実験管理
| # | ファイル | 内容 |
|---|---|---|
| 20 | `src/bsdf_sim/optimization/mlflow_logger.py` | 3階層Experiment管理、Parquet/HTML Artifact保存 |
| 21 | `src/bsdf_sim/optimization/optuna_runner.py` | 多目的最適化、正規化ユークリッド距離による重複スキップ |
| 22 | `src/bsdf_sim/optimization/__init__.py` | 公開API |

### フェーズ7：可視化
| # | ファイル | 内容 |
|---|---|---|
| 23 | `src/bsdf_sim/visualization/holoviews_plots.py` | FFT/PSD/実測の3者オーバーレイ、リニア/片対数/両対数切替 |
| 24 | `src/bsdf_sim/visualization/dynamicmap.py` | スライダー連動DynamicMap、3段階解像度、LRUキャッシュ |
| 25 | `src/bsdf_sim/visualization/__init__.py` | 公開API |

### フェーズ8：CLI
| # | ファイル | 内容 |
|---|---|---|
| 26 | `src/bsdf_sim/cli/main.py` | `simulate / optimize / visualize / report` 4サブコマンド |
| 27 | `src/bsdf_sim/cli/__init__.py` | 公開API |
| 28 | `src/bsdf_sim/__init__.py` | パッケージルート |

### フェーズ9：プラグインフォルダ・テスト
| # | ファイル | 内容 |
|---|---|---|
| 29 | `custom_surfaces/.gitkeep` | プラグイン用空フォルダ |
| 30 | `custom_metrics/.gitkeep` | プラグイン用空フォルダ |
| 31 | `tests/__init__.py` | テストパッケージ |
| 32 | `tests/test_models.py` | HeightMap・各形状モデルの基本動作確認 |
| 33 | `tests/test_fft_bsdf.py` | FFT法・フレネル係数・BRDF/BTDF判定のテスト |
| 34 | `tests/test_psd_bsdf.py` | PSD法・Q因子（完全形/簡略形）のテスト |
| 35 | `tests/test_metrics.py` | 表面粗さ・光学指標の計算テスト |

---

## 主要な実装仕様（spec_main.md より）

| 項目 | 仕様 |
|---|---|
| 物理単位 | μm 統一 |
| HeightMap | `data: np.ndarray` + `pixel_size_um: float` の dataclass |
| プレビュー | `reduced_area`（pixel_size固定）と `reduced_resolution`（面積固定）の両モード |
| FFT位相式 | BRDF: `φ=(4π/λ)n₁h cosθ_i`、BTDF: `φ=(2π/λ)(n₂cosθ_t - n₁cosθ_i)h` |
| PSD-Q因子 | 完全形（Elson-Bennett）デフォルト、`approx_mode=True` で簡略形（Stover） |
| Log-RMSE | `bsdf_floor`（デフォルト1e-6）でクリップ＋マスク処理 |
| Adding-Doubling | fast/standard/high プリセット（n_theta=32/128/256、m_phi=8/18/36） |
| Sparkle | 観察条件・ディスプレイ・照明をプリセット+数値入力、Cs=σ/μ固定 |
| Parquet | long format（`bsdf`列 + `method`カテゴリ列） |
| DynamicMap | drag=N128 / idle=N512 / 本計算=N4096、LRUキャッシュ、シード固定 |
| 重複スキップ | 正規化ユークリッド距離（threshold=0.01、config.yaml で変更可） |
| プラグイン | `custom_surfaces/` と `custom_metrics/` を動的ロード |
