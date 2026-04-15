"""リアルタイム BSDF ダッシュボードのテスト。

create_dashboard_from_config のモデル判定、各 DynamicMap サブクラスの
ダッシュボード生成、実測 BSDF オーバーレイ、CLI サブコマンドを検証する。
`pn.serve()` は実際には呼ばない（サーバーが立ち上がってハングするため）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml


SAMPLE_BSDF = Path(__file__).parent.parent / "sample_inputs" / "BRDF_BTDF_LightTools.bsdf"


@pytest.fixture
def base_cfg():
    return {
        "simulation": {
            "wavelength_um": 0.525,
            "theta_i_deg": 0.0,
            "phi_i_deg": 0.0,
            "n1": 1.0,
            "n2": 1.5,
            "polarization": "Unpolarized",
        },
        "error_metrics": {"bsdf_floor": 1e-6},
        "metrics": {
            "haze": {"enabled": False},
            "gloss": {"enabled": False},
            "doi": {"enabled": False},
            "sparkle": {"enabled": False},
        },
    }


def _write_cfg(tmp_path: Path, cfg: dict) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


# ── create_dashboard_from_config のモデル判定 ────────────────────────────────


class TestFactoryDispatch:
    """config.surface.model に応じた DynamicMap クラスの自動選択。"""

    def test_random_rough_dispatches_correctly(self, base_cfg, tmp_path):
        from bsdf_sim.visualization.dynamicmap import (
            RandomRoughDynamicMap, create_dashboard_from_config,
        )
        base_cfg["surface"] = {
            "model": "RandomRoughSurface",
            "grid_size": 128, "pixel_size_um": 0.25,
            "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
        }
        cfg_path = _write_cfg(tmp_path, base_cfg)
        dash = create_dashboard_from_config(cfg_path, preview_grid_size_idle=64)
        assert isinstance(dash, RandomRoughDynamicMap)
        assert dash.wavelength_um == pytest.approx(0.525)

    def test_spherical_array_dispatches_correctly(self, base_cfg, tmp_path):
        from bsdf_sim.visualization.dynamicmap import (
            SphericalArrayDynamicMap, create_dashboard_from_config,
        )
        base_cfg["surface"] = {
            "model": "SphericalArraySurface",
            "grid_size": 128, "pixel_size_um": 0.25,
            "spherical_array": {
                "radius_um": 50.0,
                "pitch_um": 50.0,
                "placement": "Hexagonal",
                "overlap_mode": "Maximum",
            },
        }
        cfg_path = _write_cfg(tmp_path, base_cfg)
        dash = create_dashboard_from_config(cfg_path, preview_grid_size_idle=64)
        assert isinstance(dash, SphericalArrayDynamicMap)

    def test_measured_surface_dispatches_correctly(self, base_cfg, tmp_path):
        """DeviceVk6Surface などの実測系プラグインは MeasuredSurfaceDynamicMap へ。"""
        from bsdf_sim.visualization.dynamicmap import (
            MeasuredSurfaceDynamicMap, create_dashboard_from_config,
        )
        vk6_sample = Path(__file__).parent.parent / "sample_inputs" / "device_vk6_sample.csv"
        if not vk6_sample.exists():
            pytest.skip("sample_inputs/device_vk6_sample.csv が存在しない")

        base_cfg["surface"] = {
            "model": "DeviceVk6Surface",
            "measured": {
                "path": str(vk6_sample),
                "padding": "tile",
                "leveling": True,
            },
        }
        cfg_path = _write_cfg(tmp_path, base_cfg)
        dash = create_dashboard_from_config(cfg_path, preview_grid_size_idle=64)
        assert isinstance(dash, MeasuredSurfaceDynamicMap)


# ── ダッシュボード生成（実際にサーブはしない）────────────────────────────────


class TestCreateDashboard:
    """`create_dashboard()` が Panel オブジェクトを返せることの確認。"""

    def test_random_rough_creates_dashboard(self, base_cfg, tmp_path):
        import panel as pn
        from bsdf_sim.visualization.dynamicmap import create_dashboard_from_config

        base_cfg["surface"] = {
            "model": "RandomRoughSurface",
            "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
        }
        cfg_path = _write_cfg(tmp_path, base_cfg)
        dash = create_dashboard_from_config(cfg_path, preview_grid_size_idle=64)
        layout = dash.create_dashboard()
        assert isinstance(layout, pn.Column)

    def test_spherical_array_creates_dashboard(self, base_cfg, tmp_path):
        import panel as pn
        from bsdf_sim.visualization.dynamicmap import create_dashboard_from_config

        base_cfg["surface"] = {
            "model": "SphericalArraySurface",
            "spherical_array": {
                "radius_um": 50.0, "pitch_um": 50.0,
                "placement": "Hexagonal", "overlap_mode": "Maximum",
            },
        }
        cfg_path = _write_cfg(tmp_path, base_cfg)
        dash = create_dashboard_from_config(cfg_path, preview_grid_size_idle=64)
        layout = dash.create_dashboard()
        assert isinstance(layout, pn.Column)


# ── 実測 BSDF オーバーレイ ───────────────────────────────────────────────────


class TestMeasuredOverlay:
    """config.measured_bsdf で実測ファイルを指定すると 1D にオーバーレイされる。"""

    @pytest.mark.skipif(
        not SAMPLE_BSDF.exists(),
        reason="sample_inputs/BRDF_BTDF_LightTools.bsdf が存在しない",
    )
    def test_measured_blocks_loaded_in_dashboard(self, base_cfg, tmp_path):
        from bsdf_sim.visualization.dynamicmap import create_dashboard_from_config

        base_cfg["surface"] = {
            "model": "RandomRoughSurface",
            "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
        }
        # LightTools 実測の (λ=0.525, θ_i=0, BRDF) と一致する条件
        base_cfg["simulation"]["wavelength_um"] = 0.525
        base_cfg["simulation"]["theta_i_deg"] = 0.0
        base_cfg["measured_bsdf"] = {"path": str(SAMPLE_BSDF)}
        cfg_path = _write_cfg(tmp_path, base_cfg)

        dash = create_dashboard_from_config(cfg_path, preview_grid_size_idle=64)
        # 24 ブロックが読み込まれる
        assert len(dash.measured_dfs) == 24
        # λ=525nm / θ_i=0° / BRDF に一致するブロックが選ばれる
        assert dash._matched_meas_df is not None
        assert str(dash._matched_meas_df["mode"].iloc[0]) == "BRDF"
        assert abs(float(dash._matched_meas_df["wavelength_um"].iloc[0]) - 0.525) < 1e-3

    @pytest.mark.skipif(
        not SAMPLE_BSDF.exists(),
        reason="sample_inputs/BRDF_BTDF_LightTools.bsdf が存在しない",
    )
    def test_measured_profile_extraction(self, base_cfg, tmp_path):
        """_measured_profile() が phi≈0° の (theta_s, bsdf) ペアを返す。"""
        from bsdf_sim.visualization.dynamicmap import create_dashboard_from_config

        base_cfg["surface"] = {
            "model": "RandomRoughSurface",
            "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
        }
        base_cfg["simulation"]["wavelength_um"] = 0.525
        base_cfg["measured_bsdf"] = {"path": str(SAMPLE_BSDF)}
        cfg_path = _write_cfg(tmp_path, base_cfg)

        dash = create_dashboard_from_config(cfg_path, preview_grid_size_idle=64)
        theta_s, bsdf_vals = dash._measured_profile()
        assert len(theta_s) > 0
        assert len(bsdf_vals) == len(theta_s)
        # theta_s は昇順 (0°〜90°)
        assert np.all(np.diff(theta_s) >= 0)
        # BSDF は非負（極小値クリップ後）
        assert np.all(bsdf_vals >= 1e-10)

    def test_no_measured_file_returns_none_profile(self, base_cfg, tmp_path):
        from bsdf_sim.visualization.dynamicmap import create_dashboard_from_config

        base_cfg["surface"] = {
            "model": "RandomRoughSurface",
            "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
        }
        cfg_path = _write_cfg(tmp_path, base_cfg)
        dash = create_dashboard_from_config(cfg_path, preview_grid_size_idle=64)
        assert dash.measured_dfs == []
        assert dash._measured_profile() is None


# ── 1D オーバーレイヘルパーの直接テスト ──────────────────────────────────────


class TestMake1DOverlay:
    def test_sim_only(self):
        import holoviews as hv
        from bsdf_sim.visualization.dynamicmap import _make_1d_overlay

        n = 33
        u = np.linspace(-0.5, 0.5, n).reshape(-1, 1)
        v = np.linspace(-0.5, 0.5, n).reshape(1, -1)
        u_grid = np.broadcast_to(u, (n, n)).copy()
        v_grid = np.broadcast_to(v, (n, n)).copy()
        bsdf = np.ones((n, n), dtype=np.float32) * 0.01

        result = _make_1d_overlay(u_grid, v_grid, bsdf, "log", "Test")
        assert isinstance(result, hv.Overlay)
        labels = [str(el.label) for el in result]
        assert "FFT 計算" in labels
        assert "実測データ" not in labels

    def test_sim_with_measured(self):
        import holoviews as hv
        from bsdf_sim.visualization.dynamicmap import _make_1d_overlay

        n = 33
        u = np.linspace(-0.5, 0.5, n).reshape(-1, 1)
        v = np.linspace(-0.5, 0.5, n).reshape(1, -1)
        u_grid = np.broadcast_to(u, (n, n)).copy()
        v_grid = np.broadcast_to(v, (n, n)).copy()
        bsdf = np.ones((n, n), dtype=np.float32) * 0.01

        meas_profile = (
            np.array([0.0, 10.0, 20.0, 30.0]),
            np.array([0.02, 0.015, 0.01, 0.005]),
        )
        result = _make_1d_overlay(
            u_grid, v_grid, bsdf, "log", "Test", measured_profile=meas_profile,
        )
        assert isinstance(result, hv.Overlay)
        labels = [str(el.label) for el in result]
        assert "FFT 計算" in labels
        assert "実測データ" in labels


# ── CLI サブコマンド ─────────────────────────────────────────────────────────


class TestDashboardCLI:
    def test_dashboard_command_in_help(self):
        from click.testing import CliRunner
        from bsdf_sim.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "dashboard" in result.output

    def test_dashboard_subcommand_help(self):
        from click.testing import CliRunner
        from bsdf_sim.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--port" in result.output

    def test_dashboard_missing_config_fails(self):
        from click.testing import CliRunner
        from bsdf_sim.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", "--config", "/nonexistent.yaml"])
        assert result.exit_code != 0
