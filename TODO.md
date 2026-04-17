# プロジェクト改善・不具合リスト

## 🚨 緊急（バグ修正）
- [x] dashboardのグラフのy軸範囲を固定する機能(チェックボックスなどでON/OFFできるようにする)
- [x] dashboard x軸角度のリニア表示の補助目盛が5度刻み、目盛り数値も10度刻み、など自動設定で半端でないようにする。
- [x] bsdf dashboard --config config.yaml --host 0.0.0.0  で起動したとき、ブラウザで"申し訳ございません。このページに到達できません"と表示される

## ✨ 改善・機能追加

### 仕様書で言及されているが未実装の機能

| 項目 | 概要 | 関連仕様 | 優先度 |
|---|---|---|---|
| 03_GenAI_Insights 実験 | LLM による測定値・メトリクスに基づく設計改善考察レポートの自動生成。`EXPERIMENT_GENAI` 定数のみ定義済み、`GenAILogger` クラスと LLM 呼び出しロジックは未実装 | spec_main.md Section 6.2 | 低 |
| DynamicMap プレビューモード連携 | `reduced_area` / `reduced_resolution` のプレビューモード（`get_preview_height_map()` のモード引数）を dashboard から選べるようにする。現状は常に `reduced_area` 固定 | spec_main.md Section 7 / `models/base.py:115` | 低 |
| Parquet 往復テスト | 保存 → 読み込みでデータが一致するかの回帰テスト | spec_main.md Section 9.4 の「未テスト項目」| 中 |

### 光学指標・規格対応（設計相談で検討したが未実装）

| 項目 | 概要 | 判断 |
|---|---|---|
| Adding-Doubling 多条件対応 | 現状は多条件 simulate 実行時でも Adding-Doubling は最初の 1 条件にのみ適用される（`cli/main.py` で警告ログ出力）。全条件で多層 BSDF を計算するには simulate ループ内で `MultiLayerBSDF` を繰り返し呼ぶ。`precision='high'` で 1 条件あたり数十秒〜数分のため、opt-in オプション（`--multilayer-all-conditions` 等）での導入推奨 | 用途出現まで保留 |
| CIE 輝度加重合成 Haze/Gloss/DOI | 現状は代表波長 1 つで近似（Section 5.3）。将来的には 3 波長 × CIE V(λ) 加重の `haze_fft_photopic` / `gloss_fft_photopic` / `doi_fft_photopic` サフィックス付きバリアントを追加することで、より規格に近い単一値が得られる | 案 2 として相談済み・保留 |
| 完全スペクトル積分モード | `simulation.illuminant: 'D65' / 'C' / 'A'` 等を指定すると 11 波長自動サンプリング + CIE V(λ) 積分で Haze/Gloss/DOI を計算。計算時間が 5〜10 倍になるため重い案件用 | 案 3 として相談済み・保留 |
| Sparkle RGB 合成の再導入 | 旧 `bsdf_per_wavelength` + CIE 輝度加重分岐は dead code として削除済み。必要であれば `sparkle.rgb_combine: true` オプションで復活し、多波長 simulate 時に `sparkle_fft_photopic` として記録する設計が可能 | 削除済み・復活は需要次第 |

### Dashboard 機能拡張

| 項目 | 概要 |
|---|---|
| 多条件スライダー | `wavelength_um` / `theta_i_deg` / `mode` をスライダーまたはセレクタで対話的に切替。現状は config.yaml の値で固定 |
| 指標リアルタイム表示 | Haze / Gloss / DOI / Sparkle の計算値をスライダー更新に合わせて即時表示 |
| Parquet ダウンロードボタン | 現在の条件で計算した BSDF を Parquet としてダウンロード |
| 複数モデル並置比較 | 2 つの RandomRough パラメータセット、または RandomRough vs SphericalArray を左右に並べて比較 |
| MeasuredSurface パディングセレクタ | `DeviceVk6Surface` 等で `padding: 'tile' / 'zeros' / 'reflect' / 'smooth_tile'` を UI から切替 |

### CLI / UI 拡張

| 項目 | 概要 |
|---|---|
| インタラクティブ run ピッカー | `bsdf visualize --pick` で矢印キー選択（`questionary` 依存）。非 tty 環境で動かない制約があるため `bsdf runs list` + ショートカット記法で代替中 |
| MLflow リバースプロキシ | `bsdf mlflow-proxy --port 5001` で MLflow UI の artifact クリック時に HTML を lazy 生成・キャッシュ。一度生成した後は次回から直接配信。現状は `simulate --log-to-mlflow` で事前生成、または `visualize --log-to-mlflow` で手動書き戻しで代替 |
| `bsdf surface --log-to-mlflow` | surface コマンド単独で 01_BSDF_Raw_Data に run を作成し、surface plot を artifact として保存。現状は simulate 経由でのみ記録される |
| visualize で surface.html 再生成 | 現状 `visualize --log-to-mlflow` は `plots/bsdf_report.html` のみ書き戻す。surface 形状も params から再構成して `plots/surface.html` を一緒に生成する拡張が可能 |

### テスト・品質

| 項目 | 概要 |
|---|---|
| DynamicMap 実サーブテスト | 現状 `pn.serve()` を呼ぶとハングするためスキップしている。プロセス起動 + HTTP 疎通テスト（requests ライブラリ経由）の追加 |
| Adding-Doubling 数値検証 | Section 9.4 で「単層 = 表面のみ計算と一致するか」の検証が未実装として記録されている |
| Sparkle RGB 多波長モードの数値検証 | Section 9.4 で未実装と記録（ただし 該当コード自体が削除されたため項目ごと不要な可能性あり） |
| Optuna 最適化収束テスト | Section 9.4 で「既知パラメータへの収束確認」が未実装として記録されている |

## 💡 将来的なアイデア
- [ ]

---

## 優先度の目安

- **高**: Parquet 往復テスト（データ完全性に関わる）
- **中**: Adding-Doubling 多条件対応、DynamicMap 実サーブテスト
- **低**: 03_GenAI_Insights、CIE スペクトル積分モード、インタラクティブ run ピッカー、MLflow リバースプロキシ

## 運用ルール

新規項目が発生したら本ファイルに追記する。実装が完了したら spec_main.md Section 10.2 の変更履歴テーブルに移し、本ファイルから削除する。
