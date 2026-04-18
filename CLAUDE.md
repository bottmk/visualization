# CLAUDE.md — bsdf-sim プロジェクト ガイドライン

Claude Code がこのプロジェクトで作業する際に必ず従うルール。

---

## プロジェクト概要

AGフィルム光学散乱（BSDF）のシミュレーション・最適化プログラム。
詳細仕様: `spec_main.md` / 設計決定記録: `spec_decisions.md`

---

## 作業開始時
作業にかかる推定時間を提示して作業に取り掛かる

## 出力フォーマット

- **丸数字（①②③…）を使わない**。読みにくいため、番号付けは `(1)(2)(3)` / `1. 2. 3.` / `A/B/C` 等を使う。表のヘッダ・箇条書き・コード内コメントすべてに適用。

## バグ修正・変更の記録ルール

### バグを修正したとき
`spec_main.md` の **Section 10.1「バグ修正履歴」** に追記する。

**フォーマット:**
```
#### [BUG-XXX] タイトル — 修正済み
- **発見日**: YYYY-MM-DD またはきっかけ
- **影響ファイル**: `src/...`
- **症状**: テスト名や実行時の挙動
- **原因**: 技術的な根本原因
- **修正**: 変更内容の要約
- **対応コミット**: `xxxxxxx`
```

現在の最新 ID: **BUG-009**。次は BUG-010 から。

### 機能追加・大規模変更をしたとき
- `spec_main.md` Section 10.2「仕様変更履歴」テーブルに行を追加する
- `spec_main.md` の該当セクション（仕様の変更箇所）を更新する
- `README.md` に影響がある場合は README も更新する

---

## コード変更時の確認事項

1. **テストを実行して全件 pass を確認する**
   ```bash
   python -m pytest tests/ -q
   ```
   現在のテスト数: 337件

2. **新機能・バグ修正にはテストを追加する**
   - テストファイルは `tests/` 以下
   - 追加後は `spec_main.md` Section 9.2 のテスト一覧も更新する

3. **物理単位は μm 統一**（コード内部はすべて μm）

---

## 設定ファイル（config.yaml）の注意点

- `metrics` 以下のセクションをコメントアウトすると、その指標は実行されない（enabled デフォルト値バグは BUG-003 で修正済み）
- `sparkle` を有効にすると `grid_size` が大きいほど計算時間が増加するが、ベクトル化済み（BUG-004 修正済み）なので通常は高速

---

## CLIコマンド一覧（主要）

| コマンド | 用途 |
|---|---|
| `bsdf simulate --config c.yaml -m fft` | BSDF シミュレーション実行 |
| `bsdf surface --config c.yaml --format png` | 表面形状の可視化（PNG/HTML） |
| `bsdf optimize --config c.yaml` | Optuna 最適化 |
| `bsdf dashboard --config c.yaml` | リアルタイム BSDF ダッシュボード起動（ローカルのみ） |
| `bsdf dashboard --config c.yaml --host 0.0.0.0` | 他 PC からもアクセス可能なダッシュボード起動 |
| `bsdf runs list --sort-by haze_fft` | MLflow の run 一覧をメトリクス順で表示 |
| `bsdf visualize --run-id <id>` | MLflow の run から BSDF プロット（`latest` / `best:METRIC` / 8 文字プレフィックスも可） |
| `bsdf report --run-ids id1,id2` | 複数 run の比較レポート |

## よく参照するファイル

| ファイル | 用途 |
|---|---|
| `spec_main.md` | 全仕様・設計決定・バグ修正履歴 |
| `spec_decisions.md` | 設計判断の背景・却下案 |
| `config.yaml` | パラメータの全項目サンプル |
| `src/bsdf_sim/metrics/optical.py` | Haze/Gloss/DOI/Sparkle 計算 |
| `src/bsdf_sim/metrics/surface.py` | ISO 25178-2 / JIS B 0601 指標 |
| `src/bsdf_sim/cli/main.py` | CLI エントリポイント |
