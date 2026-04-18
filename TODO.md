# プロジェクト改善・不具合リスト

すべて下記の統一フォーマットで管理する:

- **x**: 対応済みなら `x`、未対応なら空欄
- **優先度**: 高 / 中 / 低
- **項目**: 短い要約
- **概要**: 詳細・背景
- **関連仕様**: 対応するコード / 仕様書セクション

---

## 🚨 緊急（バグ修正）

| x | 優先度 | 項目 | 概要 | 関連仕様 |
|---|---|---|---|---|
| x | 高 | dashboard Y軸範囲固定機能 | グラフの Y 軸範囲を固定する機能（チェックボックスなどで ON/OFF 切替） | `src/bsdf_sim/visualization/dynamicmap.py` |
| x | 高 | dashboard X軸目盛の自動設定 | リニア表示の補助目盛 5°刻み、主目盛（目盛り数値）10°刻みなど半端でない値にする | `src/bsdf_sim/visualization/dynamicmap.py` |
| x | 高 | `bsdf dashboard --host 0.0.0.0` 到達不能 | 起動時に自動ブラウザで "このページに到達できません" エラー。Chrome は `0.0.0.0` URL を解決不可 | spec_main.md Section 10.1 BUG-008 |
| x | 高 | dashboard X軸リニア／対数切り替え | 散乱角（X 軸）を linear / log で切り替えできる RadioButtonGroup を追加。log 時は `theta>0.05` のみ表示 | `src/bsdf_sim/visualization/dynamicmap.py` |
| x | 高 | mlflow artifacts surface.png の表示サイズ縮小 | `save_heightmap_png()` の figsize を (12,9) → (6,4.5) にして出力 PNG を 50% に。フォントも縮小して破綻を防止 | `src/bsdf_sim/visualization/holoviews_plots.py` |

---

## ✨ 改善・機能追加

### 仕様書で言及されているが未実装の機能

| x | 優先度 | 項目 | 概要 | 関連仕様 |
|---|---|---|---|---|
|   | 高 | Parquet 往復テスト | 保存 → 読み込みでデータが一致するかの回帰テスト | spec_main.md Section 9.4 「未テスト項目」 |
|   | 低 | 03_GenAI_Insights 実験 | LLM による測定値・メトリクスに基づく設計改善考察レポートの自動生成。`EXPERIMENT_GENAI` 定数のみ定義済み、`GenAILogger` クラスと LLM 呼び出しロジックは未実装 | spec_main.md Section 6.2 / `optimization/mlflow_logger.py` |
|   | 低 | DynamicMap プレビューモード連携 | `reduced_area` / `reduced_resolution` のプレビューモード（`get_preview_height_map()` のモード引数）を dashboard から選べるようにする。現状は常に `reduced_area` 固定 | spec_main.md Section 7 / `models/base.py:115` |

### 光学指標・規格対応（設計相談で検討したが未実装）

| x | 優先度 | 項目 | 概要 | 関連仕様 |
|---|---|---|---|---|
|   | 中 | Adding-Doubling 多条件対応 | 現状は多条件 simulate 実行時でも Adding-Doubling は最初の 1 条件にのみ適用（`cli/main.py` で警告ログ）。全条件で多層 BSDF を計算するには simulate ループ内で `MultiLayerBSDF` を繰り返し呼ぶ。`precision='high'` で 1 条件あたり数十秒〜数分のため opt-in オプション（`--multilayer-all-conditions` 等）で導入推奨。用途出現まで保留 | `cli/main.py` / `bsdf_sim/optics/adding_doubling.py` |
|   | 低 | CIE 輝度加重合成 Haze/Gloss/DOI | 現状は代表波長 1 つで近似。3 波長 × CIE V(λ) 加重の `haze_fft_photopic` / `gloss_fft_photopic` / `doi_fft_photopic` バリアントを追加すれば規格により近い値が得られる（案 2 として相談済み・保留） | spec_main.md Section 5.3 |
|   | 低 | 完全スペクトル積分モード | `simulation.illuminant: 'D65' / 'C' / 'A'` 等で 11 波長自動サンプリング + CIE V(λ) 積分で Haze/Gloss/DOI を計算。計算時間 5〜10 倍（案 3 として相談済み・保留） | spec_main.md Section 5.3 |
|   | 低 | Sparkle RGB 合成の再導入 | 旧 `bsdf_per_wavelength` + CIE 輝度加重分岐は dead code として削除済み。`sparkle.rgb_combine: true` オプションで復活し多波長時に `sparkle_fft_photopic` として記録する設計が可能（需要次第） | `metrics/optical.py` |
|   | 中 | Sparkle L2（多波長 V(λ)・グローバル BSDF）実装 | L1 の単波長をそのまま多波長展開した階層。画素全面一様発光 × 多波長 BSDF × $V(\lambda)$ 重みで輝度合成。L4 (サブピクセル) より簡単だが SEMI D63 白点灯の規格対応により近い。FFT 回数 $N_\lambda$ 回で計算可能 | `metrics/sparkle_extended.py` / docs/sparkle_approximation_levels.md 表 22 行目 |
| x | 中 | Sparkle L3/L4/L5 の CLI/simulate パイプライン統合 | `compute_sparkle_l3` / `compute_sparkle_l4` / `compute_sparkle_l5` を `config.yaml` の `sparkle.level: 'L1' / 'L3' / 'L4' / 'L5'` で切替可能にし、`cli/main.py` の simulate パイプラインから呼び出せるように統合。MLflow メトリクス名も `sparkle_l1_fft_525_0_r` / `sparkle_l3_fft_525_0_r` 等に統一（L1 も破壊的改名、案 B）。pytest 統合テスト追加 | spec_main.md 10.2 / `cli/main.py` / `metrics/__init__.py` |
|   | 低 | Sparkle L4 完全版（narrowband 近似脱却） | 現 `compute_sparkle_l4` は R/G/B 各色を単波長近似（narrowband）で計算。各色ピーク周辺の分光幅を積分する厳密版が未実装 | `metrics/sparkle_extended.py:295` docstring / docs/sparkle_approximation_levels.md Section 6 |

### Dashboard 機能拡張

| x | 優先度 | 項目 | 概要 | 関連仕様 |
|---|---|---|---|---|
|   | 中 | 多条件スライダー | `wavelength_um` / `theta_i_deg` / `mode` をスライダーまたはセレクタで対話的に切替。現状は config.yaml の値で固定 | `visualization/dynamicmap.py` |
|   | 中 | 指標リアルタイム表示 | Haze / Gloss / DOI / Sparkle の計算値をスライダー更新に合わせて即時表示 | `visualization/dynamicmap.py` / `metrics/optical.py` |
|   | 低 | Parquet ダウンロードボタン | 現在の条件で計算した BSDF を Parquet としてダウンロード | `visualization/dynamicmap.py` |
|   | 低 | 複数モデル並置比較 | 2 つの RandomRough パラメータセット、または RandomRough vs SphericalArray を左右に並べて比較 | `visualization/dynamicmap.py` |
|   | 低 | MeasuredSurface パディングセレクタ | `DeviceVk6Surface` 等で `padding: 'tile' / 'zeros' / 'reflect' / 'smooth_tile'` を UI から切替 | `models/measured.py` |

### 指標オーバーレイ可視化

| x | 優先度 | 項目 | 概要 | 関連仕様 |
|---|---|---|---|---|
|   | 低 | `overlay_doi_comb_1d` を `hv.VSpans` に置換 | 現状は縦帯を `hv.Rectangles([(x0, -1e30, x1, 1e30), ...])` で擬似実装。Y 軸 log 表示時に `log(-1e30)` 未定義でブラウザコンソール警告の懸念。HoloViews 1.18+ の `hv.VSpans` は y 範囲を軸全体に自動追従するため置換でクリーンになる | `src/bsdf_sim/visualization/metric_overlays.py:368` |
|   | 低 | 1D COMB overlay の specular 対応（log-x） | `xscale='log'` で `theta_s > 0.05°` フィルタが効くため、`theta_axis_deg=0` (specular) 付近の明スリット縞がほぼ全て非表示になる。BRDF (θ_i=20°/30° 等) では問題ないが、transmission/0° の 1D プロットで overlay が見えない点をドキュメント化するか、log-x 時に対数中心 (例 θ=0.1°) から展開する代替モードを追加 | `src/bsdf_sim/visualization/metric_overlays.py:323` |

### CLI / UI 拡張

| x | 優先度 | 項目 | 概要 | 関連仕様 |
|---|---|---|---|---|
|   | 低 | インタラクティブ run ピッカー | `bsdf visualize --pick` で矢印キー選択（`questionary` 依存）。非 tty 環境で動かない制約があるため `bsdf runs list` + ショートカット記法で代替中 | `cli/main.py` |
|   | 低 | MLflow リバースプロキシ | `bsdf mlflow-proxy --port 5001` で MLflow UI の artifact クリック時に HTML を lazy 生成・キャッシュ。現状は `simulate --log-to-mlflow` で事前生成、または `visualize --log-to-mlflow` で手動書き戻しで代替 | `cli/main.py` / `visualization/holoviews_plots.py` |
|   | 低 | `bsdf surface --log-to-mlflow` | surface コマンド単独で 01_BSDF_Raw_Data に run を作成し surface plot を artifact として保存。現状は simulate 経由でのみ記録 | `cli/main.py` |
|   | 低 | visualize で surface.html 再生成 | 現状 `visualize --log-to-mlflow` は `plots/bsdf_report.html` のみ書き戻す。surface 形状も params から再構成して `plots/surface.html` を一緒に生成する拡張が可能 | `cli/main.py` |

### テスト・品質

| x | 優先度 | 項目 | 概要 | 関連仕様 |
|---|---|---|---|---|
|   | 中 | DynamicMap 実サーブテスト | 現状 `pn.serve()` を呼ぶとハングするためスキップ。プロセス起動 + HTTP 疎通テスト（requests ライブラリ経由）の追加 | `tests/test_dashboard.py` |
|   | 中 | Adding-Doubling 数値検証 | 「単層 = 表面のみ計算と一致するか」の検証が未実装として記録 | spec_main.md Section 9.4 |
|   | 中 | Optuna 最適化収束テスト | 「既知パラメータへの収束確認」が未実装として記録 | spec_main.md Section 9.4 |
|   | 低 | Sparkle RGB 多波長モードの数値検証 | Section 9.4 で未実装と記録（該当コード自体が削除されたため項目ごと不要な可能性あり） | spec_main.md Section 9.4 |

---

## 💡 将来的なアイデア

| x | 優先度 | 項目 | 概要 | 関連仕様 |
|---|---|---|---|---|

---

## 運用ルール

- 新規項目が発生したら本ファイルに追記する（x 列は空欄）
- 実装が完了したら x 列に `x` を入れる
- 定期的に完了済み項目を spec_main.md Section 10.2 の変更履歴テーブルに移し、本ファイルから削除する
- バグ修正は spec_main.md Section 10.1（BUG-XXX）にも追記する
