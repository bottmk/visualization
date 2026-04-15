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
