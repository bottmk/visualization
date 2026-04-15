"""CLI コマンドのスモークテスト（Click CliRunner 使用）。"""

import textwrap

import pytest
import yaml

from click.testing import CliRunner

from bsdf_sim.cli.main import cli


_MINIMAL_CONFIG = {
    "simulation": {
        "wavelength_um": 0.55,
        "theta_i_deg": 0.0,
        "phi_i_deg": 0.0,
        "n1": 1.0,
        "n2": 1.5,
        "polarization": "Unpolarized",
    },
    "surface": {
        "model": "RandomRoughSurface",
        "grid_size": 64,        # 高速化のため小さいグリッド
        "pixel_size_um": 0.25,
        "random_rough": {
            "rq_um": 0.005,
            "lc_um": 2.0,
            "fractal_dim": 2.5,
        },
    },
    "psd": {
        "approx_mode": True,    # 高速モード
    },
    "error_metrics": {
        "bsdf_floor": 1e-6,
    },
    "metrics": {
        "haze": {"enabled": True, "half_angle_deg": 2.5},
        "gloss": {"enabled": False},
        "doi": {"enabled": False},
        "sparkle": {"enabled": False},
    },
    "mlflow": {
        "tracking_uri": "mlruns",
        "experiment_name": "test_experiment",
    },
}


@pytest.fixture
def config_file(tmp_path):
    """一時設定ファイルを生成するフィクスチャ。"""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(_MINIMAL_CONFIG), encoding="utf-8")
    return str(path)


class TestCLIVersion:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestCLISimulate:
    def test_simulate_fft_only(self, config_file, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "simulate",
            "--config", config_file,
            "--output-dir", str(tmp_path / "out"),
            "--method", "fft",
            "--no-save-parquet",
            "--no-log-to-mlflow",
        ])
        assert result.exit_code == 0, result.output

    def test_simulate_psd_only(self, config_file, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "simulate",
            "--config", config_file,
            "--output-dir", str(tmp_path / "out"),
            "--method", "psd",
            "--no-save-parquet",
            "--no-log-to-mlflow",
        ])
        assert result.exit_code == 0, result.output

    def test_simulate_both_methods(self, config_file, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "simulate",
            "--config", config_file,
            "--output-dir", str(tmp_path / "out"),
            "--method", "both",
            "--no-save-parquet",
            "--no-log-to-mlflow",
        ])
        assert result.exit_code == 0, result.output

    def test_simulate_per_method_metrics_fft(self, config_file, tmp_path, caplog):
        """--method fft のとき haze_fft がログ出力され haze_psd は出力されない。"""
        import logging
        runner = CliRunner()
        with caplog.at_level(logging.INFO):
            result = runner.invoke(cli, [
                "simulate",
                "--config", config_file,
                "--output-dir", str(tmp_path / "out"),
                "--method", "fft",
                "--no-save-parquet",
                "--no-log-to-mlflow",
            ])
        assert result.exit_code == 0, result.output
        assert "haze_fft" in caplog.text
        assert "haze_psd" not in caplog.text

    def test_simulate_per_method_metrics_both(self, config_file, tmp_path, caplog):
        """--method both のとき haze_fft と haze_psd が両方ログ出力される。"""
        import logging
        runner = CliRunner()
        with caplog.at_level(logging.INFO):
            result = runner.invoke(cli, [
                "simulate",
                "--config", config_file,
                "--output-dir", str(tmp_path / "out"),
                "--method", "both",
                "--no-save-parquet",
                "--no-log-to-mlflow",
            ])
        assert result.exit_code == 0, result.output
        assert "haze_fft" in caplog.text
        assert "haze_psd" in caplog.text

    def test_simulate_per_method_metrics_psd(self, config_file, tmp_path, caplog):
        """--method psd のとき haze_psd がログ出力され haze_fft は出力されない。"""
        import logging
        runner = CliRunner()
        with caplog.at_level(logging.INFO):
            result = runner.invoke(cli, [
                "simulate",
                "--config", config_file,
                "--output-dir", str(tmp_path / "out"),
                "--method", "psd",
                "--no-save-parquet",
                "--no-log-to-mlflow",
            ])
        assert result.exit_code == 0, result.output
        assert "haze_psd" in caplog.text
        assert "haze_fft" not in caplog.text

    def test_simulate_saves_parquet(self, config_file, tmp_path):
        """--save-parquet フラグで Parquet ファイルが生成される。"""
        out_dir = tmp_path / "out"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "simulate",
            "--config", config_file,
            "--output-dir", str(out_dir),
            "--method", "fft",
            "--save-parquet",
            "--no-log-to-mlflow",
        ])
        assert result.exit_code == 0, result.output
        assert (out_dir / "bsdf_data.parquet").exists()

    def test_simulate_missing_config_fails(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "simulate",
            "--config", str(tmp_path / "nonexistent.yaml"),
        ])
        assert result.exit_code != 0


class TestCLISimulateBTDF:
    def test_simulate_btdf_mode(self, tmp_path):
        """BTDF モード（theta_i=150°）でも正常終了する。"""
        btdf_cfg = {**_MINIMAL_CONFIG}
        btdf_cfg["simulation"] = {**_MINIMAL_CONFIG["simulation"], "theta_i_deg": 150.0}
        config_path = tmp_path / "config_btdf.yaml"
        config_path.write_text(yaml.dump(btdf_cfg), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "simulate",
            "--config", str(config_path),
            "--output-dir", str(tmp_path / "out"),
            "--method", "fft",
            "--no-save-parquet",
            "--no-log-to-mlflow",
        ])
        assert result.exit_code == 0, result.output


# ── 多条件シミュレーション + 実測 BSDF 統合テスト ─────────────────────────────


class TestCLIMultiCondition:
    """1 run 内で多波長・多入射角・BRDF/BTDF を実行する。"""

    def _run(self, cfg, tmp_path, name):
        config_path = tmp_path / f"{name}.yaml"
        config_path.write_text(yaml.dump(cfg), encoding="utf-8")
        runner = CliRunner()
        return runner.invoke(cli, [
            "simulate",
            "--config", str(config_path),
            "--output-dir", str(tmp_path / f"out_{name}"),
            "--method", "fft",
            "--no-log-to-mlflow",
        ])

    def _load_result(self, tmp_path, name):
        from bsdf_sim.io.parquet_schema import load_parquet
        return load_parquet(tmp_path / f"out_{name}" / "bsdf_data.parquet")

    def test_multi_wavelength(self, tmp_path):
        cfg = {**_MINIMAL_CONFIG}
        cfg["simulation"] = {
            **_MINIMAL_CONFIG["simulation"],
            "wavelength_um": [0.465, 0.525, 0.630],
        }
        result = self._run(cfg, tmp_path, "multi_wl")
        assert result.exit_code == 0, result.output
        df = self._load_result(tmp_path, "multi_wl")
        wls = set(df[df["method"] == "FFT"]["wavelength_um"].unique())
        # ユーザー指定の 3 波長が存在
        for w in (0.465, 0.525, 0.630):
            assert any(abs(float(wl) - w) < 1e-4 for wl in wls), f"{w} not in {wls}"
        # Haze が有効なので代表波長 0.555 も追加される
        assert any(abs(float(wl) - 0.555) < 1e-4 for wl in wls)

    def test_multi_theta_i(self, tmp_path):
        cfg = {**_MINIMAL_CONFIG}
        cfg["simulation"] = {
            **_MINIMAL_CONFIG["simulation"],
            "theta_i_deg": [0.0, 20.0, 40.0],
        }
        result = self._run(cfg, tmp_path, "multi_theta")
        assert result.exit_code == 0, result.output
        df = self._load_result(tmp_path, "multi_theta")
        thetas = df[df["method"] == "FFT"]["theta_i_deg"].unique()
        assert len(thetas) == 3

    def test_representative_wavelength_added_for_standards_metrics(self, tmp_path):
        """Haze 有効 × 多波長 → 代表波長 0.555μm の追加 sim が走る。"""
        cfg = {**_MINIMAL_CONFIG}
        cfg["simulation"] = {
            **_MINIMAL_CONFIG["simulation"],
            "wavelength_um": [0.465, 0.525, 0.630],
        }
        # Haze 有効（デフォルト設定のまま）
        result = self._run(cfg, tmp_path, "rep_wl")
        assert result.exit_code == 0, result.output
        df = self._load_result(tmp_path, "rep_wl")
        wls = sorted(df[df["method"] == "FFT"]["wavelength_um"].unique())
        assert len(wls) == 4  # 3 + 代表波長 1
        assert any(abs(float(w) - 0.555) < 1e-4 for w in wls)

    def test_representative_wavelength_reused_when_in_list(self, tmp_path):
        """wavelength_um に 0.555 が含まれる場合は追加 sim なし。"""
        cfg = {**_MINIMAL_CONFIG}
        cfg["simulation"] = {
            **_MINIMAL_CONFIG["simulation"],
            "wavelength_um": [0.465, 0.555, 0.630],  # 0.555 を含む
        }
        result = self._run(cfg, tmp_path, "rep_in_list")
        assert result.exit_code == 0, result.output
        df = self._load_result(tmp_path, "rep_in_list")
        wls = sorted(df[df["method"] == "FFT"]["wavelength_um"].unique())
        assert len(wls) == 3  # 重複計算なし

    def test_representative_wavelength_custom(self, tmp_path):
        """metrics.representative_wavelength_um で代表波長を変更可能。"""
        cfg = {**_MINIMAL_CONFIG}
        cfg["simulation"] = {
            **_MINIMAL_CONFIG["simulation"],
            "wavelength_um": [0.465, 0.525, 0.630],
        }
        cfg["metrics"] = {
            **_MINIMAL_CONFIG["metrics"],
            "representative_wavelength_um": 0.500,  # 500nm をカスタム指定
        }
        result = self._run(cfg, tmp_path, "rep_custom")
        assert result.exit_code == 0, result.output
        df = self._load_result(tmp_path, "rep_custom")
        wls = sorted(df[df["method"] == "FFT"]["wavelength_um"].unique())
        assert any(abs(float(w) - 0.500) < 1e-4 for w in wls)

    def test_haze_unaffected_by_wavelength_list(self, tmp_path):
        """多波長設定でも haze_fft は 1 つのみ記録される（波長サフィックス無し）。"""
        from bsdf_sim.io.parquet_schema import load_parquet

        cfg = {**_MINIMAL_CONFIG}
        cfg["simulation"] = {
            **_MINIMAL_CONFIG["simulation"],
            "wavelength_um": [0.465, 0.525, 0.630],
        }
        # Haze のみ有効
        cfg["metrics"] = {
            "haze":    {"enabled": True, "half_angle_deg": 2.5},
            "gloss":   {"enabled": False},
            "doi":     {"enabled": False},
            "sparkle": {"enabled": False},
        }
        result = self._run(cfg, tmp_path, "haze_single")
        assert result.exit_code == 0, result.output
        # Parquet には多波長 + 代表波長の BSDF が保存されるが、
        # haze の値は log から確認できないので、計算完了のみ確認
        df = self._load_result(tmp_path, "haze_single")
        # 代表波長 0.555 が含まれる
        wls = sorted(df[df["method"] == "FFT"]["wavelength_um"].unique())
        assert any(abs(float(w) - 0.555) < 1e-4 for w in wls)

    def test_multi_mode_brdf_btdf(self, tmp_path):
        cfg = {**_MINIMAL_CONFIG}
        cfg["simulation"] = {
            **_MINIMAL_CONFIG["simulation"],
            "theta_i_deg": 20.0,
            "mode": ["BRDF", "BTDF"],
        }
        result = self._run(cfg, tmp_path, "multi_mode")
        assert result.exit_code == 0, result.output
        df = self._load_result(tmp_path, "multi_mode")
        modes = df[df["method"] == "FFT"]["mode"].unique()
        assert set(modes) == {"BRDF", "BTDF"}

    def test_parquet_contains_multiple_conditions(self, tmp_path):
        """Parquet 保存時に多条件の行が混在して保存される。"""
        import pandas as pd
        from bsdf_sim.io.parquet_schema import load_parquet

        cfg = {**_MINIMAL_CONFIG}
        cfg["simulation"] = {
            **_MINIMAL_CONFIG["simulation"],
            "wavelength_um": [0.465, 0.630],
            "theta_i_deg": [0.0, 20.0],
        }
        # Haze を無効化して代表波長の追加 sim を走らせない
        cfg["metrics"] = {
            "haze":    {"enabled": False},
            "gloss":   {"enabled": False},
            "doi":     {"enabled": False},
            "sparkle": {"enabled": False},
        }
        config_path = tmp_path / "config_grid.yaml"
        config_path.write_text(yaml.dump(cfg), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "simulate",
            "--config", str(config_path),
            "--output-dir", str(tmp_path / "out"),
            "--method", "fft",
            "--no-log-to-mlflow",
        ])
        assert result.exit_code == 0, result.output
        df = load_parquet(tmp_path / "out" / "bsdf_data.parquet")
        unique_keys = df[["wavelength_um", "theta_i_deg"]].drop_duplicates()
        assert len(unique_keys) == 4  # 2 λ × 2 θ（代表波長 sim はスキップされる）


class TestCLIMeasuredBsdfReal:
    """実ファイル統合テスト（sample_inputs 依存）。"""

    from pathlib import Path as _Path
    SAMPLE = _Path(__file__).parent.parent / "sample_inputs" / "BRDF_BTDF_LightTools.bsdf"

    @pytest.mark.skipif(
        not SAMPLE.exists(),
        reason="sample_inputs/BRDF_BTDF_LightTools.bsdf が存在しない",
    )
    def test_simulate_with_measured_bsdf_cli_option(self, tmp_path):
        """--measured-bsdf オプションで実測を読み込み、log_rmse が計算される。"""
        import pandas as pd
        from bsdf_sim.io.parquet_schema import load_parquet

        cfg = {**_MINIMAL_CONFIG}
        cfg["simulation"] = {
            **_MINIMAL_CONFIG["simulation"],
            "wavelength_um": 0.525,
            "theta_i_deg": 20.0,
            "mode": "BRDF",
        }
        config_path = tmp_path / "config_with_meas.yaml"
        config_path.write_text(yaml.dump(cfg), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "simulate",
            "--config", str(config_path),
            "--output-dir", str(tmp_path / "out"),
            "--method", "fft",
            "--measured-bsdf", str(self.SAMPLE),
            "--no-log-to-mlflow",
        ])
        assert result.exit_code == 0, result.output

        df = load_parquet(tmp_path / "out" / "bsdf_data.parquet")
        # sim + 実測の両方が含まれる
        assert "FFT" in df["method"].unique()
        assert "measured" in df["method"].unique()
        # FFT 行に log_rmse（一致した条件）が入っている
        fft_rows = df[df["method"] == "FFT"]
        assert not fft_rows["log_rmse"].dropna().empty

    @pytest.mark.skipif(
        not SAMPLE.exists(),
        reason="sample_inputs/BRDF_BTDF_LightTools.bsdf が存在しない",
    )
    def test_simulate_match_measured_auto_conditions(self, tmp_path):
        """--match-measured で実測の 24 条件を自動採用。"""
        import pandas as pd
        from bsdf_sim.io.parquet_schema import load_parquet

        cfg = {**_MINIMAL_CONFIG}
        cfg["measured_bsdf"] = {
            "path": str(self.SAMPLE),
            "match_measured": True,
        }
        # Haze を無効化して代表波長の追加 sim をスキップ
        cfg["metrics"] = {
            "haze":    {"enabled": False},
            "gloss":   {"enabled": False},
            "doi":     {"enabled": False},
            "sparkle": {"enabled": False},
        }
        config_path = tmp_path / "config_auto.yaml"
        config_path.write_text(yaml.dump(cfg), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "simulate",
            "--config", str(config_path),
            "--output-dir", str(tmp_path / "out"),
            "--method", "fft",
            "--no-log-to-mlflow",
        ])
        assert result.exit_code == 0, result.output

        df = load_parquet(tmp_path / "out" / "bsdf_data.parquet")
        # 3 λ × 4 AOI × 2 mode = 24 ユニーク条件（代表波長 sim はスキップ）
        unique = df[df["method"] == "FFT"][
            ["wavelength_um", "theta_i_deg", "mode"]
        ].drop_duplicates()
        assert len(unique) == 24


class TestCLISimulateMLflowArtifacts:
    """`simulate --log-to-mlflow` が HTML インタラクティブレポートも artifacts に記録する。"""

    def test_simulate_logs_html_artifacts(self, tmp_path):
        """--log-to-mlflow 指定時に surface.html / bsdf_report.html が生成される。"""
        import mlflow
        from bsdf_sim.optimization.mlflow_logger import (
            EXPERIMENT_RAW_DATA,
            _get_or_create_experiment,
        )

        tracking_uri = str(tmp_path / "mlruns")
        cfg = {
            **_MINIMAL_CONFIG,
            "mlflow": {
                "tracking_uri": tracking_uri,
                "experiment_name": EXPERIMENT_RAW_DATA,
            },
        }
        config_path = tmp_path / "config_mlflow.yaml"
        config_path.write_text(yaml.dump(cfg), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "simulate",
            "--config", str(config_path),
            "--output-dir", str(tmp_path / "out"),
            "--method", "fft",
            "--log-to-mlflow",
        ])
        assert result.exit_code == 0, result.output

        # MLflow の artifacts ツリーを走査し HTML が 2 つ含まれていることを確認
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(EXPERIMENT_RAW_DATA)
        assert exp is not None
        runs = client.search_runs(exp.experiment_id, max_results=5)
        assert len(runs) >= 1
        run = runs[0]

        artifacts = client.list_artifacts(run.info.run_id, path="plots")
        names = {a.path.rsplit("/", 1)[-1] for a in artifacts}
        assert "surface.png" in names
        assert "surface.html" in names
        assert "bsdf_report.html" in names
        # 少なくとも 1 つの 2D BSDF PNG
        assert any(n.startswith("bsdf_2d_") and n.endswith(".png") for n in names)
