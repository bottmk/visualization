"""`resolve_run_id` ヘルパーと `bsdf runs list` CLI のテスト。

ショートカット形式:
- `latest` / `latest-N`
- `best:METRIC[:min|:max]`
- 8 文字以上のプレフィックスマッチ
- 完全な run_id

1 つの一時 MLflow tracking_uri に bare run を 3 件作り、各解決パスを検証。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from click.testing import CliRunner


# ── 共通ヘルパー ──────────────────────────────────────────────────────────────


def _make_bare_run(
    tracking_uri: str,
    run_name: str,
    metric_values: dict[str, float] | None = None,
) -> str:
    """最小の run を作成して run_id を返す。"""
    from bsdf_sim.io.parquet_schema import build_dataframe
    from bsdf_sim.optimization.mlflow_logger import RawDataLogger

    n = 9
    u1 = np.linspace(-0.5, 0.5, n)
    v1 = np.linspace(-0.5, 0.5, n)
    u_grid = np.broadcast_to(u1.reshape(-1, 1), (n, n)).copy()
    v_grid = np.broadcast_to(v1.reshape(1, -1), (n, n)).copy()
    bsdf = np.full_like(u_grid, 0.01, dtype=np.float32)
    df = build_dataframe(
        u_grid, v_grid, bsdf, "FFT",
        theta_i_deg=0.0, phi_i_deg=0.0,
        wavelength_um=0.55, polarization="Unpolarized",
        is_btdf=False,
    )
    logger_ = RawDataLogger(tracking_uri=tracking_uri)
    return logger_.log_trial(
        params={"run_name": run_name},
        metrics=metric_values or {"haze_fft": 0.1},
        df=df,
        run_name=run_name,
    )


@pytest.fixture
def three_runs(tmp_path):
    """3 件の run を作り (tracking_uri, [run_id1, run_id2, run_id3]) を返す。

    haze_fft の値を異なる値で記録する：
      run_a: haze_fft=0.15
      run_b: haze_fft=0.05 (最小)
      run_c: haze_fft=0.25 (最大)
    作成順の時系列は a < b < c。
    """
    import time
    tracking_uri = str(tmp_path / "mlruns")
    ids = []
    for name, metric in [
        ("run_a", 0.15),
        ("run_b", 0.05),
        ("run_c", 0.25),
    ]:
        rid = _make_bare_run(
            tracking_uri, name, metric_values={"haze_fft": metric}
        )
        ids.append(rid)
        time.sleep(0.01)  # start_time の順序を明確に
    return tracking_uri, ids


# ── resolve_run_id の単体テスト ──────────────────────────────────────────────


class TestResolveRunId:
    def test_full_run_id_passthrough(self, three_runs):
        from bsdf_sim.optimization.mlflow_logger import resolve_run_id

        tracking_uri, ids = three_runs
        full = ids[0]
        assert resolve_run_id(full, tracking_uri=tracking_uri) == full

    def test_latest_returns_most_recent(self, three_runs):
        from bsdf_sim.optimization.mlflow_logger import resolve_run_id

        tracking_uri, ids = three_runs
        # run_c が最後に作られた
        assert resolve_run_id("latest", tracking_uri=tracking_uri) == ids[2]

    def test_latest_N(self, three_runs):
        from bsdf_sim.optimization.mlflow_logger import resolve_run_id

        tracking_uri, ids = three_runs
        # latest-2 は 2 番目に新しい = run_b
        assert resolve_run_id("latest-2", tracking_uri=tracking_uri) == ids[1]
        # latest-3 は最も古い = run_a
        assert resolve_run_id("latest-3", tracking_uri=tracking_uri) == ids[0]

    def test_latest_N_too_large_raises(self, three_runs):
        from bsdf_sim.optimization.mlflow_logger import resolve_run_id

        tracking_uri, _ = three_runs
        with pytest.raises(ValueError, match="run が"):
            resolve_run_id("latest-10", tracking_uri=tracking_uri)

    def test_best_min_metric(self, three_runs):
        from bsdf_sim.optimization.mlflow_logger import resolve_run_id

        tracking_uri, ids = three_runs
        # haze_fft 最小は run_b (0.05)
        assert (
            resolve_run_id("best:haze_fft", tracking_uri=tracking_uri) == ids[1]
        )

    def test_best_min_explicit(self, three_runs):
        from bsdf_sim.optimization.mlflow_logger import resolve_run_id

        tracking_uri, ids = three_runs
        assert (
            resolve_run_id("best:haze_fft:min", tracking_uri=tracking_uri)
            == ids[1]
        )

    def test_best_max_metric(self, three_runs):
        from bsdf_sim.optimization.mlflow_logger import resolve_run_id

        tracking_uri, ids = three_runs
        # haze_fft 最大は run_c (0.25)
        assert (
            resolve_run_id("best:haze_fft:max", tracking_uri=tracking_uri)
            == ids[2]
        )

    def test_best_unknown_direction_raises(self, three_runs):
        from bsdf_sim.optimization.mlflow_logger import resolve_run_id

        tracking_uri, _ = three_runs
        with pytest.raises(ValueError, match="方向"):
            resolve_run_id("best:haze_fft:banana", tracking_uri=tracking_uri)

    def test_best_nonexistent_metric_raises(self, three_runs):
        from bsdf_sim.optimization.mlflow_logger import resolve_run_id

        tracking_uri, _ = three_runs
        with pytest.raises(ValueError, match="見つからない"):
            resolve_run_id("best:nonexistent_metric", tracking_uri=tracking_uri)

    def test_prefix_unique_match(self, three_runs):
        from bsdf_sim.optimization.mlflow_logger import resolve_run_id

        tracking_uri, ids = three_runs
        prefix = ids[0][:10]  # 10 文字のプレフィックス
        resolved = resolve_run_id(prefix, tracking_uri=tracking_uri)
        assert resolved == ids[0]

    def test_prefix_too_short_raises(self, three_runs):
        """7 文字以下は受理しない。"""
        from bsdf_sim.optimization.mlflow_logger import resolve_run_id

        tracking_uri, ids = three_runs
        with pytest.raises(ValueError, match="解決に失敗"):
            resolve_run_id(ids[0][:7], tracking_uri=tracking_uri)

    def test_prefix_no_match_raises(self, three_runs):
        from bsdf_sim.optimization.mlflow_logger import resolve_run_id

        tracking_uri, _ = three_runs
        with pytest.raises(ValueError, match="見つからない"):
            resolve_run_id("zzzzzzzz12345", tracking_uri=tracking_uri)

    def test_unknown_format_raises(self, three_runs):
        from bsdf_sim.optimization.mlflow_logger import resolve_run_id

        tracking_uri, _ = three_runs
        with pytest.raises(ValueError, match="解決に失敗"):
            resolve_run_id("abc", tracking_uri=tracking_uri)


# ── runs list CLI ────────────────────────────────────────────────────────────


class TestRunsListCLI:
    def test_empty_experiment_shows_message(self, tmp_path):
        from bsdf_sim.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "runs", "list",
            "--tracking-uri", str(tmp_path / "mlruns"),
            "--experiment", "NonExistent",
        ])
        assert result.exit_code == 0
        assert "見つからない" in result.output

    def test_list_shows_short_id_and_metrics(self, three_runs):
        from bsdf_sim.cli.main import cli

        tracking_uri, ids = three_runs
        runner = CliRunner()
        result = runner.invoke(cli, [
            "runs", "list",
            "--tracking-uri", tracking_uri,
            "--sort-by", "haze_fft",
        ])
        assert result.exit_code == 0, result.output
        # short_id 列ヘッダがある
        assert "short_id" in result.output
        # 3 件の short_id が全て含まれる
        for rid in ids:
            assert rid[:8] in result.output
        # 値 0.05 / 0.15 / 0.25 が見える
        assert "0.05" in result.output
        assert "0.15" in result.output

    def test_list_sort_order(self, three_runs):
        """haze_fft 昇順でソートすると run_b (0.05) が先頭に来る。"""
        from bsdf_sim.cli.main import cli

        tracking_uri, ids = three_runs
        runner = CliRunner()
        result = runner.invoke(cli, [
            "runs", "list",
            "--tracking-uri", tracking_uri,
            "--sort-by", "haze_fft",
            "--ascending",
        ])
        assert result.exit_code == 0, result.output
        out_lines = result.output.splitlines()
        # 最初の short_id がある行を探す
        data_lines = [
            L for L in out_lines
            if any(rid[:8] in L for rid in ids)
        ]
        assert len(data_lines) == 3
        # 最小 haze (run_b) が先頭
        assert ids[1][:8] in data_lines[0]
        assert ids[2][:8] in data_lines[2]

    def test_list_limit(self, three_runs):
        from bsdf_sim.cli.main import cli

        tracking_uri, ids = three_runs
        runner = CliRunner()
        result = runner.invoke(cli, [
            "runs", "list",
            "--tracking-uri", tracking_uri,
            "--limit", "2",
        ])
        assert result.exit_code == 0, result.output
        # 表示される short_id の数が 2 以下
        seen = sum(1 for rid in ids if rid[:8] in result.output)
        assert seen == 2

    def test_list_custom_metrics_column(self, three_runs):
        from bsdf_sim.cli.main import cli

        tracking_uri, _ = three_runs
        runner = CliRunner()
        result = runner.invoke(cli, [
            "runs", "list",
            "--tracking-uri", tracking_uri,
            "--metrics", "haze_fft",
        ])
        assert result.exit_code == 0, result.output
        assert "haze_fft" in result.output


# ── visualize + ショートカット統合 ───────────────────────────────────────────


class TestVisualizeWithShortcut:
    def test_visualize_with_latest(self, three_runs, tmp_path):
        from bsdf_sim.cli.main import cli

        tracking_uri, ids = three_runs
        output_html = tmp_path / "latest.html"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "visualize",
            "--run-id", "latest",
            "--tracking-uri", tracking_uri,
            "--output", str(output_html),
        ])
        assert result.exit_code == 0, result.output
        assert output_html.exists()

    def test_visualize_with_best_metric(self, three_runs, tmp_path):
        from bsdf_sim.cli.main import cli

        tracking_uri, ids = three_runs
        output_html = tmp_path / "best.html"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "visualize",
            "--run-id", "best:haze_fft",
            "--tracking-uri", tracking_uri,
            "--output", str(output_html),
        ])
        assert result.exit_code == 0, result.output
        assert output_html.exists()

    def test_visualize_with_prefix(self, three_runs, tmp_path):
        from bsdf_sim.cli.main import cli

        tracking_uri, ids = three_runs
        output_html = tmp_path / "prefix.html"
        prefix = ids[0][:10]
        runner = CliRunner()
        result = runner.invoke(cli, [
            "visualize",
            "--run-id", prefix,
            "--tracking-uri", tracking_uri,
            "--output", str(output_html),
        ])
        assert result.exit_code == 0, result.output
        assert output_html.exists()

    def test_visualize_with_bad_shortcut_fails(self, three_runs, tmp_path):
        from bsdf_sim.cli.main import cli

        tracking_uri, _ = three_runs
        runner = CliRunner()
        result = runner.invoke(cli, [
            "visualize",
            "--run-id", "best:nonexistent_metric",
            "--tracking-uri", tracking_uri,
            "--output", str(tmp_path / "out.html"),
        ])
        assert result.exit_code != 0
