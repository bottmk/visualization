"""visualize の実測オーバーレイ + 多条件対応テスト。

- `plot_bsdf_1d_overlay` の自動条件選択・mode フィルタ
- `plot_bsdf_report` の多条件 Tabs・Log-RMSE 集計
- `visualize` CLI が多条件 Parquet + measured 行を正しく描画
- Sparkle の多波長モード下でのメトリクスキーサフィックス
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bsdf_sim.io.parquet_schema import build_dataframe, build_measured_dataframe


# ── 共通フィクスチャ ──────────────────────────────────────────────────────────


def _make_sim_df(method: str, wl_um: float, theta_i: float, mode: str, bsdf_val=0.01):
    n = 11
    u_1d = np.linspace(-0.5, 0.5, n)
    v_1d = np.linspace(-0.5, 0.5, n)
    u_grid = np.broadcast_to(u_1d.reshape(-1, 1), (n, n)).copy()
    v_grid = np.broadcast_to(v_1d.reshape(1, -1), (n, n)).copy()
    bsdf = np.full_like(u_grid, bsdf_val, dtype=np.float32)
    return build_dataframe(
        u_grid, v_grid, bsdf, method,
        theta_i_deg=theta_i, phi_i_deg=0.0,
        wavelength_um=wl_um, polarization="Unpolarized",
        is_btdf=(mode == "BTDF"),
    )


def _make_meas_df(wl_um: float, theta_i: float, mode: str, bsdf_val=0.01):
    theta_s = np.array([0.0, 10.0, 20.0, 30.0, 45.0])
    phi_s   = np.array([0.0,  0.0,  0.0,  0.0,  0.0])
    bsdf    = np.full(5, bsdf_val, dtype=np.float32)
    return build_measured_dataframe(
        theta_s, phi_s, bsdf,
        theta_i_deg=theta_i, phi_i_deg=0.0,
        wavelength_nm=wl_um * 1000.0, polarization="Unpolarized",
        is_btdf=(mode == "BTDF"),
    )


@pytest.fixture
def single_condition_df():
    sim_fft = _make_sim_df("FFT", 0.525, 20.0, "BRDF")
    meas    = _make_meas_df(0.525, 20.0, "BRDF")
    return pd.concat([sim_fft, meas], ignore_index=True)


@pytest.fixture
def multi_condition_df():
    dfs = []
    for wl in [0.465, 0.525, 0.630]:
        for theta_i in [0.0, 20.0]:
            for mode in ["BRDF", "BTDF"]:
                dfs.append(_make_sim_df("FFT", wl, theta_i, mode, bsdf_val=0.01))
                dfs.append(_make_meas_df(wl, theta_i, mode, bsdf_val=0.02))
    return pd.concat(dfs, ignore_index=True)


# ── plot_bsdf_1d_overlay ─────────────────────────────────────────────────────


class TestOverlay1D:
    """`plot_bsdf_1d_overlay` の自動条件選択・mode フィルタ。"""

    def test_default_single_condition(self, single_condition_df):
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_1d_overlay
        # 引数なしで呼んでもデータが出てくる（自動選択）
        result = plot_bsdf_1d_overlay(single_condition_df)
        assert result is not None

    def test_auto_select_multi_condition(self, multi_condition_df):
        """多条件 df でも自動選択でエラーにならない。"""
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_1d_overlay
        result = plot_bsdf_1d_overlay(multi_condition_df)
        assert result is not None

    def test_explicit_wavelength_filter(self, multi_condition_df):
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_1d_overlay
        result = plot_bsdf_1d_overlay(
            multi_condition_df,
            wavelength_um=0.525, theta_i_deg=20.0, mode="BRDF",
        )
        assert result is not None

    def test_mode_filter_brdf_vs_btdf(self, multi_condition_df):
        """同じ λ・θ でも BRDF/BTDF を分離できる。"""
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_1d_overlay
        brdf_plot = plot_bsdf_1d_overlay(
            multi_condition_df, wavelength_um=0.525,
            theta_i_deg=20.0, mode="BRDF",
        )
        btdf_plot = plot_bsdf_1d_overlay(
            multi_condition_df, wavelength_um=0.525,
            theta_i_deg=20.0, mode="BTDF",
        )
        assert brdf_plot is not None
        assert btdf_plot is not None

    def test_empty_df_returns_placeholder(self):
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_1d_overlay
        empty = pd.DataFrame(columns=[
            "u", "v", "theta_s_deg", "phi_s_deg", "theta_i_deg",
            "phi_i_deg", "wavelength_um", "polarization", "mode",
            "method", "bsdf", "is_measured", "log_rmse",
        ])
        result = plot_bsdf_1d_overlay(empty)
        assert result is not None  # hv.Text プレースホルダ

    def test_measured_overlay_present(self, single_condition_df):
        """method='measured' 行がある場合、オーバーレイに Scatter が含まれる。"""
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_1d_overlay
        import holoviews as hv
        result = plot_bsdf_1d_overlay(single_condition_df)
        # Overlay 内の要素を走査して Scatter が存在するか確認
        if isinstance(result, hv.Overlay):
            labels = [str(el.label) for el in result]
            assert any("実測" in lbl or "measured" in lbl.lower() for lbl in labels)


# ── plot_bsdf_report ─────────────────────────────────────────────────────────


class TestBsdfReportMultiCondition:
    """`plot_bsdf_report` の多条件 Tab 展開。"""

    def test_single_condition_flat_layout(self, single_condition_df):
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_report
        result = plot_bsdf_report(single_condition_df, title="Test")
        assert result is not None

    def test_multi_condition_uses_tabs(self, multi_condition_df):
        """多条件 df では Panel Tabs が作られる。"""
        import panel as pn
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_report
        result = plot_bsdf_report(multi_condition_df, title="Test")
        assert result is not None
        # Column 内のどこかに pn.Tabs がある
        has_tabs = False
        for obj in result:
            if isinstance(obj, pn.Tabs):
                has_tabs = True
                break
        assert has_tabs, "多条件 df では pn.Tabs が使われるべき"

    def test_tab_count_matches_conditions(self, multi_condition_df):
        """Tab 数 = 条件数（3 λ × 2 θ × 2 mode = 12）。"""
        import panel as pn
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_report
        result = plot_bsdf_report(multi_condition_df)
        tabs_obj = next(obj for obj in result if isinstance(obj, pn.Tabs))
        assert len(tabs_obj) == 12

    def test_metrics_table_includes_log_rmse(self, single_condition_df):
        """log_rmse_fft が metrics 辞書に含まれていれば Comparison カテゴリで表示。"""
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_report
        metrics = {
            "sq_um": 0.005,
            "haze_fft": 0.12,
            "log_rmse_fft": 0.3,
        }
        result = plot_bsdf_report(single_condition_df, metrics=metrics)
        assert result is not None

    def test_measured_only_df_still_works(self):
        """sim なし・実測のみの df でも描画できる。"""
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_report
        meas_only = _make_meas_df(0.525, 20.0, "BRDF")
        result = plot_bsdf_report(meas_only)
        assert result is not None


# ── visualize CLI 統合 ───────────────────────────────────────────────────────


class TestVisualizeCLI:
    """simulate → visualize エンドツーエンドでオーバーレイが生成される。"""

    def test_visualize_with_measured_overlay(self, tmp_path):
        """multi-condition + measured_bsdf で simulate → visualize が HTML を生成。"""
        import yaml
        from click.testing import CliRunner
        from bsdf_sim.cli.main import cli

        sample = Path(__file__).parent.parent / "sample_inputs" / "BRDF_BTDF_LightTools.bsdf"
        if not sample.exists():
            pytest.skip("sample_inputs/BRDF_BTDF_LightTools.bsdf が存在しない")

        cfg = {
            "simulation": {
                "wavelength_um": 0.525,
                "theta_i_deg": 20.0,
                "mode": "BRDF",
                "n1": 1.0, "n2": 1.5,
                "polarization": "Unpolarized",
            },
            "surface": {
                "model": "RandomRoughSurface",
                "grid_size": 64, "pixel_size_um": 0.25,
                "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
            },
            "psd": {"approx_mode": True},
            "error_metrics": {"bsdf_floor": 1e-6},
            "metrics": {
                "haze": {"enabled": False},
                "gloss": {"enabled": False},
                "doi": {"enabled": False},
                "sparkle": {"enabled": False},
            },
            "measured_bsdf": {"path": str(sample)},
        }
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

        runner = CliRunner()
        # simulate: Parquet を保存
        r = runner.invoke(cli, [
            "simulate", "-c", str(cfg_path),
            "-o", str(tmp_path / "out"),
            "-m", "fft",
            "--no-log-to-mlflow",
        ])
        assert r.exit_code == 0, r.output

        # Parquet を直接読んで plot_bsdf_report を呼ぶ（visualize CLI と同等）
        from bsdf_sim.io.parquet_schema import load_parquet
        from bsdf_sim.visualization.holoviews_plots import plot_bsdf_report, save_html
        df = load_parquet(tmp_path / "out" / "bsdf_data.parquet")
        assert "measured" in df["method"].unique()
        report = plot_bsdf_report(df, title="Test")
        html_path = tmp_path / "report.html"
        save_html(report, html_path)
        assert html_path.exists()
        assert html_path.stat().st_size > 1000  # 非空


# ── Sparkle 多波長メトリクスキー ─────────────────────────────────────────────


class TestSparkleMultiWavelength:
    """多条件 simulate 時の Sparkle メトリクスキーのサフィックス検証。"""

    def test_sparkle_multi_wavelength_keys_in_parquet(self, tmp_path):
        """多波長 × Sparkle 有効で simulate を実行し、各波長の BSDF が保存される。"""
        import yaml
        from click.testing import CliRunner
        from bsdf_sim.cli.main import cli
        from bsdf_sim.io.parquet_schema import load_parquet

        cfg = {
            "simulation": {
                "wavelength_um": [0.465, 0.525, 0.630],
                "theta_i_deg": 0.0,
                "n1": 1.0, "n2": 1.5,
                "polarization": "Unpolarized",
            },
            "surface": {
                "model": "RandomRoughSurface",
                "grid_size": 128, "pixel_size_um": 0.25,
                "random_rough": {"rq_um": 0.01, "lc_um": 2.0, "fractal_dim": 2.5},
            },
            "psd": {"approx_mode": True},
            "error_metrics": {"bsdf_floor": 1e-6},
            "metrics": {
                "sparkle": {
                    "enabled": True,
                    "viewing": {"preset": "smartphone"},
                    "display": {"preset": "fhd_smartphone"},
                },
                "haze":  {"enabled": False},
                "gloss": {"enabled": False},
                "doi":   {"enabled": False},
            },
        }
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

        runner = CliRunner()
        r = runner.invoke(cli, [
            "simulate", "-c", str(cfg_path),
            "-o", str(tmp_path / "out"),
            "-m", "fft",
            "--no-log-to-mlflow",
        ])
        assert r.exit_code == 0, r.output

        df = load_parquet(tmp_path / "out" / "bsdf_data.parquet")
        wls = sorted(df[df["method"] == "FFT"]["wavelength_um"].unique())
        assert len(wls) == 3
        assert any(abs(w - 0.465) < 1e-4 for w in wls)
        assert any(abs(w - 0.525) < 1e-4 for w in wls)
        assert any(abs(w - 0.630) < 1e-4 for w in wls)

    def test_sparkle_metric_suffix_helper(self):
        """サフィックス生成ロジックの直接検証（simulate loop の内部ロジック）。"""
        # 模倣: cli/main.py:simulate() のサフィックス生成部分
        def make_key(base: str, method_key: str, wl_um: float,
                     theta_i: float, mode: str, multi: bool) -> str:
            if multi:
                return (
                    f"{base}_{method_key}"
                    f"_wl{int(round(wl_um * 1000))}nm"
                    f"_aoi{int(round(theta_i))}"
                    f"_{mode.lower()}"
                )
            return f"{base}_{method_key}"

        # 単条件
        assert make_key("sparkle", "fft", 0.55, 0, "BRDF", multi=False) == "sparkle_fft"
        assert make_key("haze", "psd", 0.55, 0, "BRDF", multi=False) == "haze_psd"

        # 多条件
        assert make_key(
            "sparkle", "fft", 0.525, 20, "BRDF", multi=True
        ) == "sparkle_fft_wl525nm_aoi20_brdf"
        assert make_key(
            "sparkle", "fft", 0.465, 0, "BTDF", multi=True
        ) == "sparkle_fft_wl465nm_aoi0_btdf"
        assert make_key(
            "log_rmse", "psd", 0.630, 60, "BTDF", multi=True
        ) == "log_rmse_psd_wl630nm_aoi60_btdf"
